"""Build the `demo/` folder: files to drop into the prediction app, and a worked
example of getting your own telemetry into the same shape.

    python scripts/make_demo_data.py
    python scripts/make_demo_data.py --run backblaze_window90

Every row is real telemetry lifted out of `data/toshiba_drivestats.parquet`, which is
derived from Backblaze's public drive-stats release -- nothing is synthesised and no
serial here is private. What varies between the files is the *situation*:

    01  one healthy drive, CSV                 the simplest possible input
    02  one failing drive, CSV                 the same shape, opposite answer
    03  a 60-drive fleet with ground truth      ranking, and the retrospective panel
    04  a live snapshot, no failure column      what a production pull actually looks like
    05  short and truncated windows             the "partial window" checkbox
    06  renamed headers, six attributes         aliases and the missing-column warning
    TEMPLATE.csv                                the schema, to copy

and `demo/from_smartctl/` is the real-data walkthrough: a log shaped the way a
`smartctl` cron job writes one, plus the converter that turns it into column 06's shape.

This differs from `data/examples/` on purpose. Those six files each isolate one code
path in the app and are referenced by its chips. These are bigger, messier and meant to
be opened by a person deciding whether the thing is useful.

Everything model-shaped is read from the run the app serves, so the files always carry
that model's attributes and window length.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "webapp"))

import scoring  # noqa: E402

PARQUET = os.path.join(ROOT, "data", "toshiba_drivestats.parquet")
OUT_DIR = os.path.join(ROOT, "demo")
SMART_DIR = os.path.join(OUT_DIR, "from_smartctl")

# The attribute names `smartctl -A` prints, for the attributes this model reads. The
# walkthrough renames in this direction to make the raw log, and the converter renames
# back. Keeping the table here means one definition of the mapping, used both ways.
SMARTCTL_NAMES = {
    "smart_1_raw": "Raw_Read_Error_Rate",
    "smart_3_raw": "Spin_Up_Time",
    "smart_4_raw": "Start_Stop_Count",
    "smart_5_raw": "Reallocated_Sector_Ct",
    "smart_7_raw": "Seek_Error_Rate",
    "smart_9_raw": "Power_On_Hours",
    "smart_10_raw": "Spin_Retry_Count",
    "smart_12_raw": "Power_Cycle_Count",
    "smart_192_raw": "Power-Off_Retract_Count",
    "smart_193_raw": "Load_Cycle_Count",
    "smart_194_raw": "Temperature_Celsius",
    "smart_197_raw": "Current_Pending_Sector",
    "smart_198_raw": "Offline_Uncorrectable",
    "smart_199_raw": "UDMA_CRC_Error_Count",
    "smart_240_raw": "Head_Flying_Hours",
}

MANIFEST: list[dict] = []


def score_fleet(run: scoring.Run):
    """Every drive's most recent window, scored -- the basis for picking demo drives."""
    contract, est = run.contract, run.primary
    attrs = list(contract.base_features)

    print(f"reading {os.path.basename(PARQUET)} ...")
    raw = pd.read_parquet(
        PARQUET, columns=["date", "serial_number", "model", "failure"] + attrs)
    raw["failure"] = raw["failure"].fillna(0).astype("int8")

    frame = raw[["serial_number", "date", "failure"] + attrs].rename(
        columns={"serial_number": "id", "date": "time"})
    for c in attrs:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")

    print(f"scoring with {est.label} over a {contract.window_days}-day window ...")
    store = contract.build_store(frame, contract.data_config())
    risk = est.score(store)

    fleet = pd.DataFrame({
        "serial": np.asarray(store.drive_ids, dtype=object).astype(str),
        "risk": risk,
        "days": store.lengths,
    })
    last = raw.groupby("serial_number")["date"].max()
    ever = raw.groupby("serial_number")["failure"].max()
    fleet["failed"] = fleet.serial.map(ever).fillna(0).astype(int)
    fleet["last_day"] = pd.to_datetime(fleet.serial.map(last))
    return raw, attrs, fleet.sort_values("risk", ascending=False).reset_index(drop=True)


def window_of(raw: pd.DataFrame, serial: str, n: int) -> pd.DataFrame:
    """That drive's final `n` calendar days, in order."""
    return raw[raw.serial_number == serial].sort_values("date").tail(n)


def write(df: pd.DataFrame, name: str, what: str, expect: str,
          directory: str = OUT_DIR) -> None:
    path = os.path.join(directory, name)
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for alt in ("timestamp", "time"):
        if alt in out.columns:
            out[alt] = pd.to_datetime(out[alt]).dt.strftime("%Y-%m-%d")
    if name.endswith(".csv"):
        out.to_csv(path, index=False)
    else:
        out.to_excel(path, index=False)

    kb = os.path.getsize(path) / 1024
    id_col = next((c for c in ("serial_number", "serial", "device") if c in out.columns),
                  out.columns[0])
    n_drives = int(out[id_col].nunique()) if len(out) else 0
    print(f"  {name:<32} {len(out):>6,} rows  {n_drives:>3} drives  {kb:>7.0f} KB   {what}")
    MANIFEST.append({"file": name, "rows": int(len(out)), "drives": n_drives,
                     "kb": round(kb, 1), "what": what, "expect": expect})


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=os.environ.get("RUN", "backblaze_2022x_w30"))
    ap.add_argument("--fleet-size", type=int, default=60)
    args = ap.parse_args(argv)

    runs = scoring.discover(verbose=False)
    if args.run not in runs:
        raise SystemExit(f"no run {args.run!r}; found {', '.join(runs) or 'none'}")
    run = runs[args.run]
    window = run.contract.window_days
    floor = max(10, window // 3)

    # Only ever create. `from_smartctl/` also holds convert.py, which is hand-written
    # and not regenerated here -- clearing the directory would delete it.
    os.makedirs(SMART_DIR, exist_ok=True)

    raw, attrs, fleet = score_fleet(run)
    cols = ["date", "serial_number", "model", "failure"] + attrs
    long_enough = fleet[fleet.days >= window]
    failing = long_enough[long_enough.failed == 1]
    healthy = long_enough[long_enough.failed == 0]

    print(f"\nfleet: {len(fleet):,} drives ({int(fleet.failed.sum()):,} failed). "
          f"writing to {OUT_DIR} ...\n")

    # -- 01 / 02: the two single-drive cases, as CSV so the format is obvious ---------
    quiet = healthy.iloc[-1]
    write(window_of(raw, quiet.serial, window)[cols], "01_healthy_drive.csv",
          f"one drive, {window} clean days",
          f"risk {quiet.risk:.3f} - hold")

    loud = failing.iloc[0]
    write(window_of(raw, loud.serial, window)[cols], "02_failing_drive.csv",
          f"one drive, its final {window} days",
          f"risk {loud.risk:.3f} - inspect")

    # -- 03: a fleet big enough that ranking is the point ----------------------------
    # Spread across the risk range rather than taking the top N, so the list has drives
    # the model is unsure about in the middle -- which is where a queue actually lives.
    n_fail = args.fleet_size // 3
    n_ok = args.fleet_size - n_fail
    picks = pd.concat([
        failing.iloc[np.linspace(0, len(failing) - 1, n_fail).astype(int)],
        healthy.iloc[np.linspace(0, len(healthy) - 1, n_ok).astype(int)],
    ]).drop_duplicates("serial")
    fleet_df = pd.concat([window_of(raw, s, window) for s in picks.serial])
    write(fleet_df[cols], "03_fleet_60_drives.xlsx",
          f"{len(picks)} drives, {int(picks.failed.sum())} of them failed",
          "ranked list + the retrospective precision@K panel")

    # -- 04: what a production pull looks like: no answer key, one shared 'today' -----
    # Every drive's window ends on the same day, because that is how you would query a
    # live fleet. There is no failure column at all: you do not know yet.
    asof = pd.to_datetime(raw["date"]).max().normalize()
    alive = fleet[(fleet.last_day >= asof - pd.Timedelta(days=2))
                  & (fleet.days >= window)]
    alive = alive.iloc[np.linspace(0, len(alive) - 1, min(40, len(alive))).astype(int)]
    snap = raw[raw.serial_number.isin(set(alive.serial))].copy()
    snap = snap[(pd.to_datetime(snap["date"]) > asof - pd.Timedelta(days=window))
                & (pd.to_datetime(snap["date"]) <= asof)]
    write(snap[["date", "serial_number", "model"] + attrs], "04_live_snapshot.xlsx",
          f"{alive.shape[0]} drives as of {asof:%Y-%m-%d}, no failure column",
          "a triage queue - no ground truth to check it against, which is the real case")

    # -- 05: windows that do not reach the full length -------------------------------
    # Spread across the risk range, like file 03. Taking the top of each group instead
    # would confound the demonstration: every drive would be flagged anyway, and you
    # could not tell the truncation apart from the selection.
    short = pd.concat([
        failing.iloc[np.linspace(0, len(failing) - 1, 7).astype(int)],
        healthy.iloc[np.linspace(0, len(healthy) - 1, 7).astype(int)],
    ]).drop_duplicates("serial")
    fractions = [0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.0, 1.0, 0.45, 0.55, 0.65, 0.75]
    lengths = [max(floor, int(round(f * window))) for f in fractions]
    parts = [window_of(raw, s, n) for s, n in zip(short.serial, lengths)]
    used = lengths[: len(parts)]
    n_full = sum(1 for n in used if n >= window)
    write(pd.concat(parts)[cols], "05_short_windows.xlsx",
          f"{len(parts)} drives holding {min(used)}-{max(used)} days",
          f"{n_full} scored by default, all {len(parts)} with the partial-window box")

    # -- 06: someone else's column names, and only part of the feed ------------------
    six = [c for c in ("smart_5_raw", "smart_9_raw", "smart_194_raw", "smart_197_raw",
                       "smart_198_raw", "smart_199_raw") if c in attrs]
    renamed = fleet_df[["date", "serial_number"] + six].rename(
        columns={"date": "timestamp", "serial_number": "serial"})
    write(renamed, "06_renamed_and_sparse.csv",
          f"headers timestamp/serial, only {len(six)} of {len(attrs)} attributes",
          "aliases mapped, and a warning naming every attribute it had to impute")

    # -- the schema, as something to copy --------------------------------------------
    template = pd.DataFrame({
        "date": ["2026-01-01", "2026-01-02", "2026-01-01", "2026-01-02"],
        "serial_number": ["DRIVE-A", "DRIVE-A", "DRIVE-B", "DRIVE-B"],
        "model": ["MY MODEL 1", "MY MODEL 1", "MY MODEL 1", "MY MODEL 1"],
        **{a: [0, 0, 0, 0] for a in attrs},
    })
    template.to_csv(os.path.join(OUT_DIR, "TEMPLATE.csv"), index=False)
    print(f"  {'TEMPLATE.csv':<32} {len(template):>6,} rows    2 drives  "
          f"{os.path.getsize(os.path.join(OUT_DIR, 'TEMPLATE.csv'))/1024:>7.1f} KB   "
          f"the schema, with every column the model reads")

    # -- the real-data walkthrough ---------------------------------------------------
    # A log shaped the way a nightly `smartctl -A` scrape writes one: the attributes
    # under their smartctl names, a `device` path instead of a serial column heading,
    # and an ISO timestamp. Real telemetry underneath, so converting it produces a file
    # that genuinely scores.
    walk_serials = list(picks.serial[:6])
    blocks = []
    for i, serial in enumerate(walk_serials):
        block = window_of(raw, serial, window + 5)[["date", "serial_number"] + attrs].copy()
        block.insert(1, "device", f"/dev/sd{chr(ord('a') + i)}")
        blocks.append(block)
    walk = pd.concat(blocks).rename(
        columns={"date": "timestamp", "serial_number": "serial", **SMARTCTL_NAMES})
    write(walk, "smartctl_log.csv",
          f"{len(walk_serials)} drives under smartctl attribute names",
          "not directly scoreable - run convert.py on it first", directory=SMART_DIR)

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump({"run": run.key, "window_days": window, "floor_days": floor,
                   "attributes": attrs, "smartctl_names": SMARTCTL_NAMES,
                   "files": MANIFEST}, fh, indent=2)
    print(f"\nwrote manifest.json\ndone -- upload any of these at http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
