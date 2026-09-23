"""Generate six example workbooks for the prediction app, into `data/examples/`.

    python scripts/make_example_workbooks.py

Every row is real telemetry lifted out of `data/toshiba_drivestats.parquet` -- nothing
is synthesised. What varies between the files is the *shape of the input*, so that each
one exercises a different path through the app:

    01  one healthy drive, full window        the quiet case
    02  one drive on its way out              the loud case
    03  a small mixed fleet                   ranking, with blocked columns included
    04  truncated windows                     the "partial window" checkbox
    05  only 5 of the 16 attributes           the missing-column warning
    06  renamed headers, no failure column    the column-alias mapping

Drives are chosen by scoring the real fleet with the trained model first, so file 01
really is a drive the model is relaxed about and file 02 really is one it is not.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import backblaze_window_pipeline as bw  # noqa: E402

PARQUET = os.path.join(ROOT, "data", "toshiba_drivestats.parquet")
PREFIX = os.path.join(ROOT, "models", "backblaze_window90")
OUT_DIR = os.path.join(ROOT, "data", "examples")

# Carried through to the workbooks even though the model never reads them: the app's
# transparency panel claims blocked columns are ignored, and file 03 is how you check.
BLOCKED_PASSENGERS = ["capacity_bytes", "vault_id", "pod_slot_num", "datacenter"]


def score_fleet():
    """Every drive's most recent 90 days, scored -- the basis for picking examples."""
    model, config = bw.load_artifacts(PREFIX, torch.device("cpu"))
    attrs = list(config["feature_names"][: config["base_feature_count"]])

    print("reading the parquet ...")
    raw = pd.read_parquet(
        PARQUET,
        columns=["date", "serial_number", "model", "failure"] + attrs + BLOCKED_PASSENGERS)
    raw["failure"] = raw["failure"].fillna(0).astype("int8")

    frame = raw[["serial_number", "date", "failure"] + attrs].rename(
        columns={"serial_number": "id", "date": "time"})
    for c in attrs:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype("float32")

    print("building the daily calendar and scoring ...")
    cfg = bw.data_config_from_saved(config)
    store = bw.DriveSeriesStore(frame, cfg)
    store.apply_scaling(config["feature_medians"], config["feature_mean"],
                        config["feature_std"])
    risk = bw.score_latest_windows(model, store, torch.device("cpu"))

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
    d = raw[raw.serial_number == serial].sort_values("date")
    return d.tail(n)


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


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    raw, attrs, fleet = score_fleet()
    cols = ["date", "serial_number", "model", "failure"] + attrs

    long_enough = fleet[fleet.days >= 90]
    failing = long_enough[long_enough.failed == 1]
    healthy = long_enough[long_enough.failed == 0]

    print(f"\nfleet scored: {len(fleet):,} drives "
          f"({int(fleet.failed.sum()):,} failed). writing to {OUT_DIR} ...\n")

    # 01 -- a healthy drive the model is relaxed about.
    quiet = healthy.iloc[-1]
    write(last_days(raw, quiet.serial, 90)[cols],
          "01_single_drive_healthy.xlsx",
          f"risk {quiet.risk:.3f}, never failed",
          label="1 - healthy drive",
          shows="the quiet case: one drive, a clean 90 days, scored near zero")

    # 02 -- a drive the model is loud about, on its final 90 days.
    loud = failing.iloc[0]
    write(last_days(raw, loud.serial, 90)[cols],
          "02_single_drive_failing.xlsx",
          f"risk {loud.risk:.3f}, failed",
          label="2 - failing drive",
          shows="the loud case: the final 90 days of a drive that died, scored near one")

    # 03 -- a mixed fleet, spread across the risk range so the ranking has work to do,
    # and carrying the blocked columns to prove they are ignored.
    picks = pd.concat([
        failing.head(4), failing.iloc[len(failing) // 2: len(failing) // 2 + 2],
        healthy.head(3), healthy.iloc[len(healthy) // 2: len(healthy) // 2 + 3],
    ]).drop_duplicates("serial")
    mixed = pd.concat([last_days(raw, s, 90) for s in picks.serial])
    write(mixed[cols + [c for c in BLOCKED_PASSENGERS if c in mixed.columns]],
          "03_small_fleet_mixed.xlsx",
          f"{len(picks)} drives, {int(picks.failed.sum())} failed, + blocked columns",
          label="3 - mixed fleet",
          shows="ranking across 12 drives; the file also carries capacity_bytes, "
                "vault_id, pod_slot_num and datacenter, none of which the model reads")

    # 04 -- windows cut short, the way a drive that dies mid-quarter really looks.
    # Needs the "also score drives with a partial window" checkbox to score at all.
    short_picks = pd.concat([failing.iloc[4:9], healthy.iloc[3:6]]).drop_duplicates("serial")
    lengths = [41, 55, 68, 77, 84, 90, 90, 62]
    parts = [last_days(raw, s, n) for s, n in zip(short_picks.serial, lengths)]
    write(pd.concat(parts)[cols],
          "04_partial_windows.xlsx",
          f"{len(parts)} drives, {sum(1 for n in lengths[:len(parts)] if n < 90)} shorter than 90 days",
          label="4 - partial windows",
          shows="drives cut short mid-window: only 2 score by default, all 8 with "
                "'also score drives with a partial window' ticked")

    # 05 -- a file from a source that only reports a handful of attributes.
    few = ["smart_5_raw", "smart_9_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw"]
    write(mixed[["date", "serial_number", "model", "failure"] + few],
          "05_missing_smart_columns.xlsx",
          f"only {len(few)} of {len(attrs)} attributes present",
          label="5 - missing columns",
          shows="the same 12 drives with 11 of the 16 attributes absent: it still "
                "scores, warns, and the top risk falls from 0.997 to 0.752")

    # 06 -- a file that calls its columns something else and has no ground truth,
    # which is what a genuinely unseen file looks like.
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
