"""Cut an uploadable file out of the real Backblaze archive.

    python scripts/slice_real_data.py --family "TOSHIBA MG08ACA16TA"
    python scripts/slice_real_data.py --shard Q1_2026 --days 60 --healthy 500

`data/backblaze_archive/shards/*.parquet` is the real fleet, 2013 to the present, already
under the column names the model reads -- but a quarter of the whole fleet is ~32M rows,
far more than the app wants to parse. This takes the tail of one shard, keeps every drive
that failed in it plus a sample of healthy ones, and writes the fifteen attributes the
default run reads. The result drops straight onto the page.

Drives that failed stop reporting, so they are the short ones: expect the app to skip
some of them unless you tick "also score drives with a partial window".
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARDS = os.path.join(ROOT, "data", "backblaze_archive", "shards")

# The fifteen the default run (backblaze_2022x_w30) reads, strongest first.
ATTRS = [f"smart_{n}_raw" for n in
         (197, 5, 198, 4, 12, 9, 240, 193, 1, 7, 3, 192, 194, 199, 10)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", default="Q2_2026",
                    help="shard name in data/backblaze_archive/shards/ (default Q2_2026)")
    ap.add_argument("--family", default=None,
                    help="one drive model, e.g. 'TOSHIBA MG08ACA16TA'; omit for the "
                         "whole fleet")
    ap.add_argument("--days", type=int, default=45,
                    help="how many days back from the shard's last day (default 45; the "
                         "default run needs 30 to score a drive at all)")
    ap.add_argument("--healthy", type=int, default=300,
                    help="healthy drives to keep alongside every failed one, 0 for all")
    ap.add_argument("--out", default=os.path.join("data", "real_slice.parquet"),
                    help="output .parquet or .csv (default data/real_slice.parquet)")
    args = ap.parse_args()

    path = os.path.join(SHARDS, f"{args.shard}.parquet")
    if not os.path.isfile(path):
        have = sorted(f[:-8] for f in os.listdir(SHARDS) if f.endswith(".parquet"))
        raise SystemExit(f"no shard {args.shard!r}. Available: {', '.join(have)}")

    cols = ["date", "serial_number", "model", "failure"] + ATTRS
    flt = (pc.field("model") == args.family) if args.family else None
    df = ds.dataset(path, format="parquet").to_table(columns=cols, filter=flt).to_pandas()
    if df.empty:
        raise SystemExit(f"no rows for model {args.family!r} in {args.shard}")

    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= df["date"].max() - pd.Timedelta(days=args.days - 1)]

    failed = df.loc[df.failure == 1, "serial_number"].unique()
    if args.healthy:
        healthy = df.loc[~df.serial_number.isin(failed), "serial_number"].unique()
        rng = np.random.default_rng(0)
        pick = rng.choice(healthy, min(args.healthy, len(healthy)), replace=False)
        df = df[df.serial_number.isin(set(failed) | set(pick))]

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if out.lower().endswith(".csv"):
        df.to_csv(out, index=False)
    else:
        df.to_parquet(out, index=False)

    span = df.groupby("serial_number")["date"].agg(["min", "max"])
    full = int((((span["max"] - span["min"]).dt.days + 1) >= 30).sum())
    print(f"wrote {out}")
    print(f"  {len(df):,} rows | {df.serial_number.nunique()} drives | "
          f"{len(failed)} of them failed | {df.date.min().date()} -> {df.date.max().date()}")
    print(f"  {full} drives reach a full 30-day window -- the rest need the "
          f"partial-window box")


if __name__ == "__main__":
    main()
