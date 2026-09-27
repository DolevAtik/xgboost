"""Generate the example workbooks for the prediction app, into `data/examples/`.

    python scripts/make_example_workbooks.py
    python scripts/make_example_workbooks.py --run backblaze_window90

Every row is real telemetry lifted out of `data/toshiba_drivestats.parquet` -- nothing is
synthesised. What varies between the files is the *shape of the input*, so each one
exercises a different path through the app:

    01  one healthy drive, full window        the quiet case
    02  one drive on its way out              the loud case
    03  a small mixed fleet                   ranking, with blocked columns included
    04  truncated windows                     the "partial window" checkbox
    05  a handful of attributes               the missing-column warning
    06  renamed headers, no failure column    the column-alias mapping

Everything model-shaped is read from the run the app serves, via `webapp/scoring.py`:
which attributes to write out, how long a full window is, how short "too short" is, and
which drives are interesting. Hard-coding any of that is what made the previous set of
files quietly wrong when the model moved from a 90-day window over 16 attributes to a
30-day window over 15 -- the workbooks kept their old columns, so four of the new model's
inputs were median-filled on every single example.
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
OUT_DIR = os.path.join(ROOT, "data", "examples")

# Carried through to the workbooks even though the model never reads them: the app's
# transparency panel claims blocked columns are ignored, and file 03 is how you check.
BLOCKED_PASSENGERS = ["capacity_bytes", "vault_id", "pod_slot_num", "datacenter"]


def score_fleet(run: scoring.Run):
    """Every drive's most recent window, scored -- the basis for picking examples.

    Scored with the run's own primary estimator, so file 01 really is a drive the app is
    relaxed about and file 02 really is one it is not.
    """
    contract = run.contract
    attrs = list(contract.base_features)
    est = run.primary

    print(f"reading {os.path.basename(PARQUET)} ...")
    raw = pd.read_parquet(
        PARQUET,
        columns=["date", "serial_number", "model", "failure"] + attrs + BLOCKED_PASSENGERS)
    raw["failure"] = raw["failure"].fillna(0).astype("int8")

    frame = raw[["serial_number", "date", "failure"] + attrs].rename(
        columns={"serial_number": "id", "date": "time"})
    for c in attrs:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")

    print(f"building the daily calendar and scoring with {est.label} "
          f"({contract.window_days}-day window) ...")
    store = contract.build_store(frame, contract.data_config())
    risk = est.score(store)

    fleet = pd.DataFrame({
        "serial": np.asarray(store.drive_ids, dtype=object).astype(str),
        "risk": risk,
        "days": store.lengths,
    })
    ever = raw.groupby("serial_number")["failure"].max()
    fleet["failed"] = fleet.serial.map(ever).fillna(0).astype(int)
    return raw, attrs, fleet.sort_values("risk", ascending=False).reset_index(drop=True)


def last_days(raw: pd.DataFrame, serial: str, n: int) -> pd.DataFrame:
    """That drive's final `n` calendar days, in order."""
    return raw[raw.serial_number == serial].sort_values("date").tail(n)


MANIFEST: list[dict] = []


def write(df: pd.DataFrame, name: str, note: str, label: str = "",
          shows: str = "") -> None:
    path = os.path.join(OUT_DIR, name)
    df.to_excel(path, index=False)
    size_kb = os.path.getsize(path) / 1024
    id_col = "serial" if "serial" in df.columns else "serial_number"
    n_drives = int(df[id_col].nunique()) if len(df) else 0
    print(f"  {name:<34} {len(df):>6,} rows  {n_drives:>4} drives  "
          f"{size_kb:>6.0f} KB   {note}")
    MANIFEST.append({
        "file": name, "label": label or name, "shows": shows, "note": note,
        "rows": int(len(df)), "drives": n_drives, "kb": round(size_kb),
    })


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default=os.environ.get("RUN", "backblaze_2022x_w30"),
                    help="artifact prefix to build the examples against")
    args = ap.parse_args(argv)

    runs = scoring.discover(verbose=False)
    if args.run not in runs:
        raise SystemExit(f"no run {args.run!r}; found {', '.join(runs) or 'none'}")
    run = runs[args.run]
    window = run.contract.window_days
    floor = max(10, window // 3)        # the app's floor when padding is allowed

    os.makedirs(OUT_DIR, exist_ok=True)
    raw, attrs, fleet = score_fleet(run)
    cols = ["date", "serial_number", "model", "failure"] + attrs

    long_enough = fleet[fleet.days >= window]
    failing = long_enough[long_enough.failed == 1]
    healthy = long_enough[long_enough.failed == 0]

    print(f"\nfleet scored: {len(fleet):,} drives ({int(fleet.failed.sum()):,} failed) "
          f"against {run.key} / {run.primary.label}. writing to {OUT_DIR} ...\n")

    # 01 -- a healthy drive the model is relaxed about.
    quiet = healthy.iloc[-1]
    write(last_days(raw, quiet.serial, window)[cols],
          "01_single_drive_healthy.xlsx",
          f"risk {quiet.risk:.3f}, never failed",
          label="1 - healthy drive",
          shows=f"the quiet case: one drive, a clean {window} days, scored near zero")

    # 02 -- a drive the model is loud about, on its final window.
    loud = failing.iloc[0]
    write(last_days(raw, loud.serial, window)[cols],
          "02_single_drive_failing.xlsx",
          f"risk {loud.risk:.3f}, failed",
          label="2 - failing drive",
          shows=f"the loud case: the final {window} days of a drive that died, scored "
                f"near one")

    # 03 -- a mixed fleet, spread across the risk range so the ranking has work to do,
    # and carrying the blocked columns to prove they are ignored.
    picks = pd.concat([
        failing.head(4), failing.iloc[len(failing) // 2: len(failing) // 2 + 2],
        healthy.head(3), healthy.iloc[len(healthy) // 2: len(healthy) // 2 + 3],
    ]).drop_duplicates("serial")
    mixed = pd.concat([last_days(raw, s, window) for s in picks.serial])
    n_mixed, n_failed = len(picks), int(picks.failed.sum())
    write(mixed[cols + [c for c in BLOCKED_PASSENGERS if c in mixed.columns]],
          "03_small_fleet_mixed.xlsx",
          f"{n_mixed} drives, {n_failed} failed, + blocked columns",
          label="3 - mixed fleet",
          shows=f"ranking across {n_mixed} drives; the file also carries capacity_bytes, "
                f"vault_id, pod_slot_num and datacenter, none of which the model reads")

    # 04 -- windows cut short, the way a drive that dies mid-quarter really looks. The
    # lengths straddle the window so the "also score drives with a partial window"
    # checkbox has something to change: below it, a drive is skipped outright.
    short_picks = pd.concat([failing.iloc[4:9], healthy.iloc[3:6]]).drop_duplicates("serial")
    fractions = [0.40, 0.60, 0.75, 0.85, 0.95, 1.00, 1.00, 0.50]
    lengths = [max(floor, int(round(f * window))) for f in fractions]
    parts = [last_days(raw, s, n) for s, n in zip(short_picks.serial, lengths)]
    used = lengths[: len(parts)]
    n_short = sum(1 for n in used if n < window)
    n_full = len(parts) - n_short
    write(pd.concat(parts)[cols],
          "04_partial_windows.xlsx",
          f"{len(parts)} drives, {n_short} shorter than {window} days",
          label="4 - partial windows",
          shows=f"drives cut short mid-window: only {n_full} score by default, all "
                f"{len(parts)} with 'also score drives with a partial window' ticked")

    # 05 -- a file from a source that only reports a handful of attributes. Chosen from
    # the run's own list, so the warning is about genuinely absent inputs.
    preferred = ["smart_5_raw", "smart_9_raw", "smart_194_raw", "smart_197_raw",
                 "smart_198_raw"]
    few = [c for c in preferred if c in attrs] or attrs[:5]
    write(mixed[["date", "serial_number", "model", "failure"] + few],
          "05_missing_smart_columns.xlsx",
          f"only {len(few)} of {len(attrs)} attributes present",
          label="5 - missing columns",
          shows=f"the same {n_mixed} drives with {len(attrs) - len(few)} of the "
                f"{len(attrs)} attributes absent: it still scores, and warns that the "
                f"rest were filled with the training median")

    # 06 -- a file that calls its columns something else and has no ground truth, which
    # is what a genuinely unseen file looks like.
    renamed = mixed[["date", "serial_number"] + attrs].rename(
        columns={"date": "time", "serial_number": "serial"})
    write(renamed, "06_renamed_columns.xlsx",
          "headers 'time'/'serial', no failure column",
          label="6 - renamed headers",
          shows="columns called time/serial and no ground truth, which is what a "
                "genuinely unseen file looks like; aliases are mapped and the scores "
                "match example 3")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(MANIFEST, fh, indent=2)
    print("\nwrote manifest.json -- the app reads it to label the example chips")
    print("done -- upload any of these at http://127.0.0.1:5000")


if __name__ == "__main__":
    main()
