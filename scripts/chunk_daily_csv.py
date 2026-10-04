"""Group a folder of daily Backblaze CSVs into fixed-length day chunks for the webapp.

The raw dailies carry all 197 columns (~133MB each), so 30 days of them is ~4GB --
far over the page's 400MB upload cap. This keeps only what the default run reads
(the fifteen raw attributes, plus date/serial_number/model/failure) and writes one
parquet per chunk, streaming a day at a time so memory stays flat.

    .venv\\Scripts\\python.exe scripts\\chunk_daily_csv.py --src <folder>
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The fifteen the default run (backblaze_2022x_w30) reads, strongest first.
ATTRS = [f"smart_{n}_raw" for n in
         (197, 5, 198, 4, 12, 9, 240, 193, 1, 7, 3, 192, 194, 199, 10)]
KEYS = ["date", "serial_number", "model", "failure"]

# date stays the source's 'YYYY-MM-DD' string and the attributes stay float64 so
# nothing is reinterpreted on the way through; zstd does the shrinking.
SCHEMA = pa.schema([("date", pa.string()),
                    ("serial_number", pa.string()),
                    ("model", pa.string()),
                    ("failure", pa.int8())]
                   + [(a, pa.float64()) for a in ATTRS])

DAY = re.compile(r"^(\d{4}-\d{2}-\d{2})\.csv$")


def days(src: str) -> list[tuple[datetime.date, str]]:
    found = []
    for name in os.listdir(src):
        m = DAY.match(name)
        if m:
            found.append((datetime.date.fromisoformat(m.group(1)), os.path.join(src, name)))
    if not found:
        raise SystemExit(f"no YYYY-MM-DD.csv files in {src}")
    found.sort()
    for (a, _), (b, _) in zip(found, found[1:]):
        if (b - a).days != 1:
            print(f"  ! gap in the dailies: {a} -> {b}")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, help="folder of YYYY-MM-DD.csv dailies")
    ap.add_argument("--size", type=int, default=30,
                    help="days per chunk (default 30, the window the default run needs)")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "chunks"),
                    help="output folder (default data/chunks)")
    ap.add_argument("--prefix", default="chunk", help="output filename prefix")
    args = ap.parse_args()

    if args.size < 30:
        print(f"  ! --size {args.size} is under the 30-day window; no drive will score")

    all_days = days(args.src)
    os.makedirs(args.out, exist_ok=True)
    chunks = [all_days[i:i + args.size] for i in range(0, len(all_days), args.size)]
    print(f"{len(all_days)} dailies {all_days[0][0]} -> {all_days[-1][0]} "
          f"into {len(chunks)} chunk(s) of up to {args.size} days\n")

    for n, chunk in enumerate(chunks, 1):
        first, last = chunk[0][0], chunk[-1][0]
        path = os.path.join(args.out, f"{args.prefix}{n}_{first}_{last}.parquet")
        seen: Counter[str] = Counter()
        rows = failures = 0

        with pq.ParquetWriter(path, SCHEMA, compression="zstd") as w:
            for day, csv in chunk:
                df = pd.read_csv(csv, usecols=KEYS + ATTRS)
                df["date"] = df["date"].astype(str)
                df["failure"] = df["failure"].fillna(0).astype("int8")
                w.write_table(pa.Table.from_pandas(df[SCHEMA.names], schema=SCHEMA,
                                                   preserve_index=False))
                seen.update(df["serial_number"].dropna().unique().tolist())
                rows += len(df)
                failures += int(df["failure"].sum())
                print(f"  chunk {n}  {day}  {len(df):>7,} rows", end="\r")

        full = sum(1 for c in seen.values() if c >= 30)
        print(f"  chunk {n}: {os.path.basename(path)}"
              f"{' ' * 20}\n"
              f"    {len(chunk)} days {first} -> {last} | {rows:,} rows | "
              f"{len(seen):,} drives | {failures:,} failures\n"
              f"    {full:,} drives report all 30 days; the other {len(seen) - full:,} "
              f"need the partial-window box\n"
              f"    {os.path.getsize(path) / 1e6:.0f}MB"
              f"{'  ! over the 400MB upload cap' if os.path.getsize(path) > 400e6 else ''}\n")


if __name__ == "__main__":
    main()
