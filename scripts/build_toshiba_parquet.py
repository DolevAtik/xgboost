"""Convert the Toshiba xlsx export (105 workbooks, 197 columns) into one parquet file.

Reading the export back through pandas.read_excel every time costs ~5 minutes and
holds a full float64 copy of every column in RAM. One parquet file loads the same
3.2M drive-days in seconds and lets the notebook read only the columns it needs.

Output:
  data/toshiba_drivestats.parquet   all models, all 197 columns, zstd, one row group per window
  data/toshiba_parquet/             the same data as one file per model (hive-style), for
                                    per-model queries that shouldn't touch the other models

The 105 workbooks share one header signature, so the schema below is pinned explicitly:
a column that happens to be all-null in one window would otherwise land as float64 there
and int64 elsewhere, and the streaming writer rejects a mid-file schema change.
"""
import os
import sys
import glob
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Paths are relative to the project root; run this as `python scripts/build_toshiba_parquet.py`.
DATA_DIR = "data"
SRC_DIR = os.path.join(DATA_DIR, "Toshiba")
OUT_FILE = os.path.join(DATA_DIR, "toshiba_drivestats.parquet")
OUT_DIR = os.path.join(DATA_DIR, "toshiba_parquet")
# Per-workbook intermediates, deleted at the end. Override with TOSHIBA_TMP to keep
# them off a synced drive.
TMP_DIR = os.environ.get("TOSHIBA_TMP", os.path.join(tempfile.gettempdir(), "toshiba_xlsx_parts"))

STRING_COLS = ["serial_number", "model", "datacenter"]
INT_COLS = ["capacity_bytes", "cluster_id", "vault_id", "pod_id", "pod_slot_num"]


def build_schema(columns):
    fields = []
    for c in columns:
        if c == "date":
            fields.append(pa.field(c, pa.timestamp("us")))
        elif c in STRING_COLS:
            fields.append(pa.field(c, pa.string()))
        elif c == "failure":
            fields.append(pa.field(c, pa.int8()))
        elif c == "is_legacy_format":
            fields.append(pa.field(c, pa.bool_()))
        elif c in INT_COLS:
            fields.append(pa.field(c, pa.int64()))
        else:
            # every smart_*_raw / _normalized column: float64, not float32 --
            # smart_241_raw (total LBAs written) runs to ~1e12, past the point where
            # float32 stops representing integers exactly, which would corrupt deltas.
            fields.append(pa.field(c, pa.float64()))
    return pa.schema(fields)


def convert_one(args):
    """xlsx -> temp parquet. Runs in a worker process; returns (model, window, path, rows)."""
    src, tmp_path, schema_ser = args
    schema = pa.ipc.read_schema(pa.py_buffer(schema_ser))
    df = pd.read_excel(src, engine="calamine")
    model = os.path.basename(os.path.dirname(src))
    window = os.path.splitext(os.path.basename(src))[0]

    df["date"] = pd.to_datetime(df["date"])
    for c in STRING_COLS:
        df[c] = df[c].astype("string")
    df["failure"] = pd.to_numeric(df["failure"], errors="coerce").fillna(0).astype("int8")
    if "is_legacy_format" in df.columns:
        df["is_legacy_format"] = df["is_legacy_format"].fillna(False).astype(bool)
    for c in INT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in df.columns:
        if c.startswith("smart_"):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
    pq.write_table(table, tmp_path, compression="zstd", compression_level=3)
    return model, window, tmp_path, len(df)


def main():
    files = sorted(glob.glob(os.path.join(SRC_DIR, "*", "*.xlsx")))
    if not files:
        sys.exit(f"No workbooks found under {SRC_DIR}/")

    import python_calamine as pc
    header = pc.CalamineWorkbook.from_path(files[0]).get_sheet_by_index(0).to_python(nrows=1)[0]
    schema = build_schema([str(c) for c in header])
    schema_ser = schema.serialize().to_pybytes()

    os.makedirs(TMP_DIR, exist_ok=True)
    jobs = [
        (f, os.path.join(TMP_DIR, f"{i:03d}.parquet"), schema_ser)
        for i, f in enumerate(files)
    ]

    print(f"Converting {len(files)} workbooks with {min(8, os.cpu_count())} workers ...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as ex:
        for i, res in enumerate(ex.map(convert_one, jobs), 1):
            results.append(res)
            print(f"  [{i:3d}/{len(files)}] {res[0]} {res[1]}  {res[3]:,} rows", flush=True)

    results.sort(key=lambda r: (r[0], r[1]))

    # Single combined file, streamed one window at a time so no more than one
    # window's worth of columns is ever resident.
    total = 0
    with pq.ParquetWriter(OUT_FILE, schema, compression="zstd", compression_level=3) as w:
        for model, window, path, rows in results:
            w.write_table(pq.read_table(path, schema=schema))
            total += rows
    print(f"\nWrote {OUT_FILE}: {total:,} rows, {len(schema.names)} columns, "
          f"{os.path.getsize(OUT_FILE)/1e6:.1f} MB")

    # Per-model files for anything that only wants one model.
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)
    for model in sorted({r[0] for r in results}):
        parts = [r for r in results if r[0] == model]
        dest = os.path.join(OUT_DIR, f"model={model}")
        os.makedirs(dest, exist_ok=True)
        with pq.ParquetWriter(os.path.join(dest, "data.parquet"), schema,
                              compression="zstd", compression_level=3) as w:
            for _, _, path, _ in parts:
                w.write_table(pq.read_table(path, schema=schema))
        print(f"  {model}: {sum(p[3] for p in parts):,} rows")

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    print("\nDone.")


if __name__ == "__main__":
    main()
