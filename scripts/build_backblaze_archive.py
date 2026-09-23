"""Fetch the whole Backblaze Drive Stats archive and reduce it to parquet shards.

    python scripts/build_backblaze_archive.py --stage download   # 45 zips, ~35 GB
    python scripts/build_backblaze_archive.py --stage shards     # zips -> parquet
    python scripts/build_backblaze_archive.py                    # both, resumable

The archive is published as three yearly zips (2013-2015) and one zip per quarter from
Q1 2016 onward. Every stage is resumable: a finished download is skipped on size, a
finished shard is skipped on a sidecar `.done` marker, so an interrupted run picks up
where it stopped rather than starting over.

Two things this does NOT do, deliberately:

* **It never extracts a zip.** The daily CSVs are read straight out of the archive with
  `ZipFile.open`, which keeps ~250 GB of extracted CSV off the disk. Only the 35 GB of
  zips and the parquet shards are written.
* **It never assumes a fixed schema.** Backblaze has added and dropped SMART columns
  repeatedly over thirteen years -- `smart_241_raw` does not exist in the 2013 files,
  `vault_id` and `pod_slot_num` only appear in recent ones. Each daily CSV's header is
  read first and the wanted-column list is intersected with what that file actually
  has; absent columns come back all-NaN so every shard shares one schema.

Unlike `load_backblaze_frame`, which filters to a single manufacturer, this keeps
**every drive model in the archive**.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import io
import os
import sys
import time
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backblaze_window_pipeline import _remote_size  # noqa: E402

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/"

# 2013-2015 ship as one zip per year; everything from 2016 on is quarterly.
YEARLY = ["2013", "2014", "2015"]
QUARTERLY = [f"Q{q}_{y}" for y in range(2016, 2027) for q in (1, 2, 3, 4)]

ARCHIVE_DIR = os.environ.get("BACKBLAZE_ARCHIVE_DIR",
                             os.path.join("data", "backblaze_archive"))
ZIP_DIR = os.path.join(ARCHIVE_DIR, "zips")
SHARD_DIR = os.path.join(ARCHIVE_DIR, "shards")

ID_COLUMNS = ["date", "serial_number", "model", "failure"]

# Raw SMART counters worth carrying forward. Deliberately wider than the nine the
# Seagate notebook settled on: which attributes a vendor populates varies by
# manufacturer and by year, and the leakage audit -- not this script -- decides which
# survive into X. Coverage is scored there.
SMART_COLUMNS = [
    "smart_1_raw",    # Read Error Rate
    "smart_3_raw",    # Spin-Up Time
    "smart_4_raw",    # Start/Stop Count
    "smart_5_raw",    # Reallocated Sectors Count
    "smart_7_raw",    # Seek Error Rate
    "smart_9_raw",    # Power-On Hours
    "smart_10_raw",   # Spin Retry Count
    "smart_12_raw",   # Power Cycle Count
    "smart_183_raw",  # SATA Downshift Error Count
    "smart_184_raw",  # End-to-End Error
    "smart_187_raw",  # Reported Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_189_raw",  # High Fly Writes
    "smart_190_raw",  # Airflow Temperature
    "smart_192_raw",  # Power-Off Retract Count
    "smart_193_raw",  # Load/Unload Cycle Count
    "smart_194_raw",  # Temperature (Celsius)
    "smart_197_raw",  # Current Pending Sector Count
    "smart_198_raw",  # Offline Uncorrectable Sector Count
    "smart_199_raw",  # UDMA CRC Error Count
    "smart_240_raw",  # Head Flying Hours
    "smart_241_raw",  # Total LBAs Written
    "smart_242_raw",  # Total LBAs Read
]

# Fleet-topology columns. Not telemetry, and mostly blocked by the audit, but they have
# to be *present* for the audit to score them -- `pod_slot_num` leaking through its own
# missingness is one of the headline findings, and it cannot be found if it was dropped
# at ingest. Absent from the older files; carried as NaN there.
TOPOLOGY_COLUMNS = ["capacity_bytes", "datacenter", "cluster_id", "vault_id",
                    "pod_id", "pod_slot_num"]

WANTED = ID_COLUMNS + SMART_COLUMNS + TOPOLOGY_COLUMNS

_STRING_TOPOLOGY = ("datacenter", "cluster_id", "vault_id", "pod_id")

_SCHEMA = pa.schema(
    [("date", pa.date32()),
     ("serial_number", pa.string()),
     ("model", pa.string()),
     ("failure", pa.int8())]
    + [(c, pa.float32()) for c in SMART_COLUMNS]
    + [("capacity_bytes", pa.float64())]
    + [(c, pa.string()) for c in _STRING_TOPOLOGY]
    + [("pod_slot_num", pa.float32())]
)


def archive_names() -> list[str]:
    """Every archive name the server might have, oldest first."""
    return YEARLY + QUARTERLY


def zip_url(name: str) -> str:
    return f"{BASE_URL}data_{name}.zip"


def zip_path(name: str) -> str:
    return os.path.join(ZIP_DIR, f"data_{name}.zip")


def shard_path(name: str) -> str:
    return os.path.join(SHARD_DIR, f"{name}.parquet")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def remote_size(name: str, retries: int = 8) -> int:
    """Size of an archive, retried.

    The host drops connections freely -- ConnectionReset, RemoteDisconnected and bare
    SSL EOFs all show up under load. A HEAD that raises must not be allowed to escape:
    it is the cheapest call in the script and the least worth failing a run over.
    """
    for attempt in range(1, retries + 1):
        try:
            return _remote_size(zip_url(name))
        except Exception:
            if attempt == retries:
                return -1
            time.sleep(min(30, 2 ** attempt))
    return -1


def probe(name: str) -> tuple[str, int]:
    return name, remote_size(name, retries=3)


def discover(verbose: bool = True) -> list[tuple[str, int]]:
    """Which archives the server actually has, and how big each one is."""
    names = archive_names()
    with cf.ThreadPoolExecutor(8) as ex:
        sizes = list(ex.map(probe, names))
    found = [(n, s) for n, s in sizes if s > 0]
    if verbose:
        total = sum(s for _, s in found)
        print(f"{len(found)} archives on the server, {total / 1e9:.1f} GB zipped",
              flush=True)
    return found


def _complete(name: str, size: int) -> bool:
    return os.path.exists(zip_path(name)) and os.path.getsize(zip_path(name)) == size


def fetch_resumable(url: str, path: str, total: int, attempts: int = 40,
                    timeout: int = 45, chunk: int = 1 << 22) -> int:
    """Fetch `url` to `path`, resuming from whatever is already there.

    Deliberately not `bw._download_with_resume`. This host drops connections every few
    minutes, and the difference that matters is the socket timeout: at 120 s a stalled
    connection burns two minutes of the transfer budget doing nothing before the retry,
    and stalls are the common case here, not the rare one. 45 s detects a dead socket
    fast enough that the retry is cheaper than the wait.

    Returns the number of bytes on disk. Never raises for a network reason -- a partial
    file is progress, and the caller loops.
    """
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if total and have == total:
            return have
        if total and have > total:            # truncated write or a changed file
            os.remove(path)
            have = 0

        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                mode = "ab"
                if have and resp.status != 206:   # Range ignored -> start over
                    have, mode = 0, "wb"
                with open(path, mode) as fh:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        fh.write(block)
                        fh.flush()
        except Exception:
            # Back off briefly, then resume. No logging per attempt: with 40 attempts
            # across 30 archives the log would be nothing but retry lines.
            time.sleep(min(15, 1.5 ** min(attempt, 8)))

    return os.path.getsize(path) if os.path.exists(path) else 0


def _fetch_one(item: tuple[str, int]) -> dict:
    """Fetch one archive, swallowing every failure into the return value.

    Nothing may raise out of here. A worker that propagates kills the whole
    `ThreadPoolExecutor.map`, which is how the first run lost 30 GB of queued work to a
    single SSL handshake error.
    """
    name, size = item
    t0 = time.time()
    try:
        got = fetch_resumable(zip_url(name), zip_path(name), size)
        return {"name": name, "got": got, "size": size, "seconds": time.time() - t0,
                "error": None}
    except Exception as exc:
        got = os.path.getsize(zip_path(name)) if os.path.exists(zip_path(name)) else 0
        return {"name": name, "got": got, "size": size, "seconds": time.time() - t0,
                "error": f"{type(exc).__name__}: {exc}"}


def download_all(found: list[tuple[str, int]], workers: int = 3,
                 rounds: int = 12) -> list[str]:
    """Fetch every archive, in rounds, until each one matches its published size.

    Partial files are kept and resumed, so a round that only advances a download by a
    few hundred megabytes is still progress. The loop is what makes the run survive a
    host that drops a connection every couple of minutes.
    """
    os.makedirs(ZIP_DIR, exist_ok=True)
    for rnd in range(1, rounds + 1):
        todo = [(n, s) for n, s in found if not _complete(n, s)]
        if not todo:
            print(f"all {len(found)} archives complete", flush=True)
            return []
        have = sum(os.path.getsize(zip_path(n)) for n, _ in todo
                   if os.path.exists(zip_path(n)))
        want = sum(s for _, s in todo)
        print(f"\n-- round {rnd}: {len(found) - len(todo)}/{len(found)} complete, "
              f"{len(todo)} to go ({have / 1e9:.1f}/{want / 1e9:.1f} GB fetched)",
              flush=True)

        # Few connections on purpose: the host resets aggressively under parallel load,
        # and a reset costs more than the concurrency buys.
        with cf.ThreadPoolExecutor(workers) as ex:
            for r in ex.map(_fetch_one, todo):
                if r["error"]:
                    print(f"  {r['name']:<10} {r['got'] / 1e9:5.2f}/"
                          f"{r['size'] / 1e9:5.2f} GB  RETRY LATER -- {r['error'][:70]}",
                          flush=True)
                else:
                    ok = "ok" if r["got"] == r["size"] else "incomplete"
                    print(f"  {r['name']:<10} {r['got'] / 1e9:5.2f} GB in "
                          f"{r['seconds'] / 60:5.1f} min  {ok}", flush=True)

    failed = [n for n, s in found if not _complete(n, s)]
    if failed:
        print(f"\nstill incomplete after {rounds} rounds: {', '.join(failed)}",
              flush=True)
    return failed


# ---------------------------------------------------------------------------
# Zip -> parquet shard
# ---------------------------------------------------------------------------

def _daily_members(zf: zipfile.ZipFile) -> list[str]:
    """Daily CSV members, skipping the AppleDouble resource forks the zips carry.

    `__MACOSX/._2019-01-01.csv` ends in .csv but is a 268-byte resource fork, and
    read_csv rejects it the moment `usecols` is in play.
    """
    return sorted(
        n for n in zf.namelist()
        if n.lower().endswith(".csv")
        and not os.path.basename(n).startswith("._")
        and "__MACOSX" not in n
    )


def _header_of(zf: zipfile.ZipFile, member: str) -> list[str]:
    """Column names of a member, read from its first line only."""
    with zf.open(member) as fh:
        first = fh.readline().decode("latin1", errors="replace")
    return [c.strip().strip('"') for c in first.rstrip("\r\n").split(",")]


def _read_member(zf: zipfile.ZipFile, member: str) -> pd.DataFrame | None:
    """One daily CSV -> a frame on the fixed shard schema, or None if it is not one."""
    header = _header_of(zf, member)
    use = [c for c in WANTED if c in header]
    if "serial_number" not in use or "date" not in use:
        return None                      # not a drive-stats daily file

    with zf.open(member) as fh:
        # Decompressed into memory rather than streamed: pandas seeks on the buffer, and
        # a daily CSV is ~50 MB, which is cheap next to holding a whole quarter at once.
        buf = io.BytesIO(fh.read())
    df = pd.read_csv(buf, encoding="latin1", low_memory=False, usecols=use)
    if df.empty:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    if "failure" in df.columns:
        df["failure"] = (pd.to_numeric(df["failure"], errors="coerce")
                         .fillna(0).astype(np.int8))
    else:
        df["failure"] = np.int8(0)
    df["serial_number"] = df["serial_number"].astype("string")
    df["model"] = (df["model"].astype("string") if "model" in df.columns
                   else pd.Series(pd.NA, index=df.index, dtype="string"))

    for c in SMART_COLUMNS:
        df[c] = (pd.to_numeric(df[c], errors="coerce").astype(np.float32)
                 if c in df.columns
                 else pd.Series(np.float32("nan"), index=df.index, dtype=np.float32))
    df["capacity_bytes"] = (
        pd.to_numeric(df["capacity_bytes"], errors="coerce").astype(np.float64)
        if "capacity_bytes" in df.columns
        else pd.Series(np.nan, index=df.index, dtype=np.float64))
    for c in _STRING_TOPOLOGY:
        df[c] = (df[c].astype("string") if c in df.columns
                 else pd.Series(pd.NA, index=df.index, dtype="string"))
    df["pod_slot_num"] = (
        pd.to_numeric(df["pod_slot_num"], errors="coerce").astype(np.float32)
        if "pod_slot_num" in df.columns
        else pd.Series(np.float32("nan"), index=df.index, dtype=np.float32))

    return df[[f.name for f in _SCHEMA]]


def build_shard(name: str, overwrite: bool = False) -> dict:
    """One archive -> one parquet shard, written a day at a time.

    A quarter is ~27 M drive-days; appending each daily CSV as its own row group keeps
    peak memory at one day rather than one quarter, which is what lets several archives
    be converted in parallel.
    """
    out = shard_path(name)
    marker = out + ".done"
    if os.path.exists(marker) and os.path.exists(out) and not overwrite:
        try:
            with open(marker, encoding="utf-8") as fh:
                rows, drives, failures = (int(v) for v in fh.read().split()[:3])
            return {"archive": name, "status": "cached", "rows": rows,
                    "drives": drives, "failures": failures}
        except Exception:
            return {"archive": name, "status": "cached"}

    zp = zip_path(name)
    if not os.path.exists(zp):
        return {"archive": name, "status": "missing zip"}

    os.makedirs(SHARD_DIR, exist_ok=True)
    tmp = out + ".partial"
    t0 = time.time()
    rows = failures = 0
    serials: set[str] = set()
    writer = None
    try:
        with zipfile.ZipFile(zp) as zf:
            for member in _daily_members(zf):
                df = _read_member(zf, member)
                if df is None or df.empty:
                    continue
                table = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, _SCHEMA, compression="zstd",
                                              compression_level=3)
                writer.write_table(table)
                rows += len(df)
                failures += int(df["failure"].sum())
                serials.update(df["serial_number"].dropna().astype(str))
                del df, table
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        return {"archive": name, "status": "no csv members"}

    os.replace(tmp, out)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(f"{rows} {len(serials)} {failures}\n")
    return {"archive": name, "status": "built", "rows": rows, "drives": len(serials),
            "failures": failures, "seconds": time.time() - t0}


def build_all(found: list[tuple[str, int]], workers: int = 6,
              overwrite: bool = False) -> pd.DataFrame:
    """Convert every *fully downloaded* archive. Partial zips are skipped, not parsed."""
    os.makedirs(SHARD_DIR, exist_ok=True)
    todo = [n for n, s in found if _complete(n, s)]
    skipped = [n for n, s in found if not _complete(n, s)]
    if skipped:
        print(f"skipping {len(skipped)} archive(s) that are not fully downloaded: "
              f"{', '.join(skipped)}", flush=True)
    print(f"converting {len(todo)} archives with {workers} workers", flush=True)
    out = []
    with cf.ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(build_shard, n, overwrite): n for n in todo}
        for fut in cf.as_completed(futs):
            res = fut.result()
            out.append(res)
            if res["status"] == "built":
                print(f"  {res['archive']:<10} {res['rows']:>12,} drive-days  "
                      f"{res['drives']:>8,} drives  {res['failures']:>6,} failures  "
                      f"{res['seconds'] / 60:5.1f} min", flush=True)
            elif res["status"] == "cached":
                print(f"  {res['archive']:<10} cached", flush=True)
            else:
                print(f"  {res['archive']:<10} {res['status']}", flush=True)
    return pd.DataFrame(out).sort_values("archive").reset_index(drop=True)


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", default="all",
                   choices=["discover", "download", "shards", "all"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--download-workers", type=int, default=3)
    p.add_argument("--rounds", type=int, default=12,
                   help="passes over the incomplete downloads; each resumes partials")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict to these archive names, e.g. --only Q1_2024 Q2_2024")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    found = discover()
    if args.only:
        keep = set(args.only)
        found = [(n, s) for n, s in found if n in keep]
    names = [n for n, _ in found]

    if args.stage == "discover":
        for n, s in found:
            print(f"  {n:<10} {s / 1e9:6.2f} GB")
        return

    if args.stage in ("download", "all"):
        download_all(found, workers=args.download_workers, rounds=args.rounds)
    if args.stage in ("shards", "all"):
        summary = build_all(found, workers=args.workers, overwrite=args.overwrite)
        if "rows" in summary:
            print(f"\n{int(summary['rows'].fillna(0).sum()):,} drive-days across "
                  f"{len(summary)} shards in {SHARD_DIR}")


if __name__ == "__main__":
    main()
