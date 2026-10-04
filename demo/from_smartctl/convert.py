"""Turn a smartctl-style telemetry log into a file the prediction app can score.

    python demo/from_smartctl/convert.py smartctl_log.csv -o ready_to_score.csv
    python demo/from_smartctl/convert.py mylog.csv -o fleet.xlsx --id-column device

Run it from anywhere; paths are resolved against your working directory.

The app needs three things that a raw scrape usually does not have:

1.  a column literally called `date` and one called `serial_number` (it also accepts
    `time`/`timestamp`, and `serial`/`id`/`drive_id`, but nothing else);
2.  the SMART attributes named `smart_<N>_raw`, not `Reallocated_Sector_Ct`;
3.  one row per drive per day, holding the *raw* attribute value rather than the
    normalised 0-253 one.

This script does 1 and 2, and collapses multiple readings in a day for 3. It is
deliberately small and readable -- the point is that you can retarget it at whatever
your own scraper writes by editing the two tables at the top.

It never invents data. Attributes your log does not carry are simply absent, and the app
fills them with the training-set median and says so on screen.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import pandas as pd

# smartctl's attribute names -> what the model calls them. Add to this if your scraper
# writes a name that is not here; the match is case-insensitive and ignores spaces,
# hyphens and underscores, so "Current Pending Sector" finds the same entry.
ATTRIBUTE_MAP = {
    "Raw_Read_Error_Rate": "smart_1_raw",
    "Spin_Up_Time": "smart_3_raw",
    "Start_Stop_Count": "smart_4_raw",
    "Reallocated_Sector_Ct": "smart_5_raw",
    "Reallocated_Sectors_Count": "smart_5_raw",
    "Seek_Error_Rate": "smart_7_raw",
    "Power_On_Hours": "smart_9_raw",
    "Power_On_Hours_and_Msec": "smart_9_raw",
    "Spin_Retry_Count": "smart_10_raw",
    "Power_Cycle_Count": "smart_12_raw",
    "Power-Off_Retract_Count": "smart_192_raw",
    "Load_Cycle_Count": "smart_193_raw",
    "Temperature_Celsius": "smart_194_raw",
    "Airflow_Temperature_Cel": "smart_194_raw",
    "Current_Pending_Sector": "smart_197_raw",
    "Offline_Uncorrectable": "smart_198_raw",
    "UDMA_CRC_Error_Count": "smart_199_raw",
    "Head_Flying_Hours": "smart_240_raw",
}

# Whatever your log calls the timestamp and the drive identity, in preference order.
DATE_CANDIDATES = ("date", "timestamp", "time", "day", "collected_at")
ID_CANDIDATES = ("serial_number", "serial", "serialnumber", "drive_id", "wwn", "id",
                 "device")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


LOOKUP = {_key(k): v for k, v in ATTRIBUTE_MAP.items()}


def convert(df: pd.DataFrame, id_column: str | None = None,
            date_column: str | None = None, aggregate: str = "last") -> pd.DataFrame:
    """Rename, coerce and collapse to one row per drive-day."""
    # Searched in the candidate list's order, not the file's: a log carrying both
    # `serial` and `device` should key on the serial, which follows the disk when it is
    # moved to another bay, rather than on the bay it happens to sit in today.
    def first_match(candidates: tuple[str, ...]) -> str | None:
        by_key = {_key(c): c for c in reversed(list(df.columns))}
        return next((by_key[_key(want)] for want in candidates
                     if _key(want) in by_key), None)

    date_col = date_column or first_match(DATE_CANDIDATES)
    id_col = id_column or first_match(ID_CANDIDATES)
    if date_col is None:
        raise SystemExit(f"no timestamp column found. Columns: {list(df.columns)}\n"
                         f"Pass --date-column to name it explicitly.")
    if id_col is None:
        raise SystemExit(f"no drive identity column found. Columns: {list(df.columns)}\n"
                         f"Pass --id-column to name it explicitly.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    out["serial_number"] = df[id_col].astype(str).str.strip()

    for passthrough in ("model", "failure"):
        hit = next((c for c in df.columns if _key(c) == passthrough), None)
        if hit:
            out[passthrough] = df[hit]

    mapped, skipped = {}, []
    for col in df.columns:
        if col in (date_col, id_col):
            continue
        # An already-canonical name passes straight through.
        if re.fullmatch(r"smart_\d+_raw", str(col).strip().lower()):
            mapped[str(col).strip().lower()] = col
            continue
        target = LOOKUP.get(_key(col))
        if target and target not in mapped:
            mapped[target] = col
        elif not target and _key(col) not in ("model", "failure"):
            skipped.append(col)

    order = sorted(mapped.items(), key=lambda kv: int(re.search(r"\d+", kv[0]).group()))
    for target, source in order:
        out[target] = pd.to_numeric(df[source], errors="coerce")

    before = len(out)
    out = out.dropna(subset=["date", "serial_number"])
    dropped = before - len(out)

    # smartctl run more than once a day gives several rows per drive-day; the app needs
    # exactly one. `last` keeps the final reading, which is what these counters mean.
    dupes = int(out.duplicated(["serial_number", "date"]).sum())
    if dupes:
        numeric = set(out.select_dtypes("number").columns)
        agg = {c: (aggregate if c in numeric else "last")
               for c in out.columns if c not in ("serial_number", "date")}
        out = out.groupby(["serial_number", "date"], as_index=False).agg(agg)

    out = out.sort_values(["serial_number", "date"]).reset_index(drop=True)

    print(f"  timestamp column : {date_col!r} -> date")
    print(f"  identity column  : {id_col!r} -> serial_number")
    print(f"  attributes mapped: {len(mapped)}")
    for target, source in order:
        print(f"      {str(source):<28} -> {target}")
    if skipped:
        print(f"  columns ignored  : {', '.join(map(str, skipped[:10]))}"
              + (" ..." if len(skipped) > 10 else ""))
    if dropped:
        print(f"  rows dropped     : {dropped:,} with no usable date or serial")
    if dupes:
        print(f"  rows collapsed   : {dupes:,} duplicate drive-days, kept '{aggregate}'")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="the raw log (.csv, .tsv, .xlsx or .parquet)")
    ap.add_argument("-o", "--output", default=None,
                    help="where to write it (.csv or .xlsx); default <input>_ready.csv")
    ap.add_argument("--id-column", default=None,
                    help="name the drive-identity column instead of guessing")
    ap.add_argument("--date-column", default=None,
                    help="name the timestamp column instead of guessing")
    ap.add_argument("--aggregate", default="last", choices=["last", "max", "mean"],
                    help="how to collapse several readings in one day (default: last)")
    args = ap.parse_args(argv)

    ext = os.path.splitext(args.input)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(args.input)
    elif ext in (".xlsx", ".xlsm", ".xls"):
        try:
            df = pd.read_excel(args.input, engine="calamine")
        except ImportError:
            df = pd.read_excel(args.input)
    else:
        df = pd.read_csv(args.input, sep=None, engine="python")

    print(f"read {args.input}: {len(df):,} rows x {len(df.columns)} columns")
    out = convert(df, args.id_column, args.date_column, args.aggregate)

    dest = args.output or os.path.splitext(args.input)[0] + "_ready.csv"
    if dest.lower().endswith((".xlsx", ".xlsm")):
        out.to_excel(dest, index=False)
    else:
        out.to_csv(dest, index=False, date_format="%Y-%m-%d")

    span = out.groupby("serial_number")["date"].agg(["min", "max"])
    days = (span["max"] - span["min"]).dt.days + 1
    print(f"\nwrote {dest}: {len(out):,} rows, {span.shape[0]} drives, "
          f"{out['date'].min():%Y-%m-%d} to {out['date'].max():%Y-%m-%d}")
    print(f"  days per drive: {days.min()} to {days.max()} "
          f"({int((days >= 30).sum())} of {len(days)} reach a full 30-day window)")
    if (days < 30).any():
        print("  tick 'also score drives with a partial window' to include the rest "
              "(floor 10 days)")
    print(f"\nnow upload {dest} at http://127.0.0.1:5000")


if __name__ == "__main__":
    sys.exit(main())
