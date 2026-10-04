"""Append one row per drive per day to a telemetry CSV, from `smartctl`.

    python collect.py --out /var/log/smart/telemetry.csv          # all disks
    python collect.py --out telemetry.csv --device /dev/sda /dev/sdb
    python collect.py --out telemetry.csv --dry-run               # print, write nothing

Run it once a day (cron, systemd timer, Task Scheduler). It is append-only and
idempotent per day: running it twice on the same date replaces that day's row for each
drive rather than adding a second one, so a retry after a failure is safe.

Needs `smartmontools` on PATH. Reading SMART data requires privileges on most systems --
root on Linux/macOS, an elevated prompt on Windows.

The output is already in the app's shape: `date`, `serial_number`, `model` and the
fifteen `smart_<N>_raw` columns. No conversion step needed; upload it directly once you
have 30 days.

To check the parsing without any hardware:

    smartctl -A -i -j /dev/sda > sample.json
    python collect.py --out test.csv --from-json sample.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import shutil
import subprocess
import sys

# SMART id -> the column the model wants. Ids, not names, so this works across vendors
# that spell the same attribute differently.
WANTED = {
    1: "smart_1_raw", 3: "smart_3_raw", 4: "smart_4_raw", 5: "smart_5_raw",
    7: "smart_7_raw", 9: "smart_9_raw", 10: "smart_10_raw", 12: "smart_12_raw",
    192: "smart_192_raw", 193: "smart_193_raw", 194: "smart_194_raw",
    197: "smart_197_raw", 198: "smart_198_raw", 199: "smart_199_raw",
    240: "smart_240_raw",
}
COLUMNS = ["date", "serial_number", "model"] + [WANTED[k] for k in sorted(WANTED)]


def parse_smartctl(doc: dict) -> dict | None:
    """One row from a `smartctl -A -i -j` document, or None if it carries no SMART table.

    NVMe drives report through `nvme_smart_health_information_log` and have no ATA
    attribute table at all; they are skipped rather than half-filled, because this model
    was trained on spinning disks and the two are not the same measurements.
    """
    table = (doc.get("ata_smart_attributes") or {}).get("table")
    if not table:
        return None

    row = {
        "serial_number": (doc.get("serial_number") or "").strip(),
        "model": (doc.get("model_name") or doc.get("model_family") or "").strip(),
    }
    if not row["serial_number"]:
        return None

    for entry in table:
        col = WANTED.get(entry.get("id"))
        if col is None:
            continue
        # `raw.value` is the integer smartctl parsed; `raw.string` is what it printed,
        # which for temperature and power-on-hours often carries extra text such as
        # "35 (Min/Max 24/41)". The integer is the one to take.
        raw = entry.get("raw") or {}
        value = raw.get("value")
        if value is None:
            continue
        row[col] = int(value)
    return row


def scan_devices() -> list[str]:
    out = subprocess.run(["smartctl", "--scan", "-j"], capture_output=True, text=True)
    try:
        return [d["name"] for d in json.loads(out.stdout).get("devices", [])]
    except (ValueError, KeyError):
        return []


def read_device(device: str) -> dict | None:
    out = subprocess.run(["smartctl", "-A", "-i", "-j", device],
                         capture_output=True, text=True)
    # smartctl uses its exit code as a bitfield; bits 0-2 mean the command failed, while
    # higher bits are health warnings that still come with perfectly good data.
    if out.returncode & 0b111:
        print(f"  {device}: smartctl exit {out.returncode}, skipped", file=sys.stderr)
        return None
    try:
        return parse_smartctl(json.loads(out.stdout))
    except ValueError:
        print(f"  {device}: unparseable smartctl output, skipped", file=sys.stderr)
        return None


def append(path: str, rows: list[dict], today: str) -> None:
    """Write `rows` into `path`, replacing any existing rows for the same drive-day."""
    existing: list[dict] = []
    if os.path.isfile(path):
        with open(path, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
    todays = {(r["serial_number"], today) for r in rows}
    kept = [r for r in existing
            if (r.get("serial_number"), r.get("date")) not in todays]
    replaced = len(existing) - len(kept)

    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in kept + rows:
            w.writerow(r)
    shutil.move(tmp, path)          # atomic, so an interrupted run cannot truncate the log
    if replaced:
        print(f"  replaced {replaced} row(s) already recorded for {today}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV to append to")
    ap.add_argument("--device", nargs="*", default=None,
                    help="devices to read; default is everything smartctl --scan finds")
    ap.add_argument("--date", default=None, help="override the date stamp (YYYY-MM-DD)")
    ap.add_argument("--dry-run", action="store_true", help="print rows, write nothing")
    ap.add_argument("--from-json", nargs="*", default=None,
                    help="parse saved `smartctl -j` documents instead of calling smartctl")
    args = ap.parse_args(argv)

    today = args.date or dt.date.today().isoformat()

    rows = []
    if args.from_json:
        for path in args.from_json:
            with open(path, encoding="utf-8") as fh:
                row = parse_smartctl(json.load(fh))
            if row:
                rows.append(row)
            else:
                print(f"  {path}: no ATA SMART table (NVMe?) or no serial number, "
                      f"skipped", file=sys.stderr)
    else:
        if shutil.which("smartctl") is None:
            print("smartctl not found on PATH -- install smartmontools", file=sys.stderr)
            return 2
        devices = args.device or scan_devices()
        if not devices:
            print("no devices found; name them with --device", file=sys.stderr)
            return 1
        for device in devices:
            row = read_device(device)
            if row:
                rows.append(row)

    if not rows:
        print("nothing to write", file=sys.stderr)
        return 1
    for row in rows:
        row["date"] = today

    found = sorted({c for r in rows for c in r if c.startswith("smart_")},
                   key=lambda c: int(c.split("_")[1]))
    print(f"{today}: {len(rows)} drive(s), {len(found)} of {len(WANTED)} attributes")
    for row in rows:
        have = sum(1 for c in row if c.startswith("smart_"))
        print(f"  {row['serial_number']:<20} {row.get('model',''):<28} {have} attributes")
    missing = [c for c in COLUMNS[3:] if c not in found]
    if missing:
        print(f"  not reported by these drives: {', '.join(missing)}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    append(args.out, rows, today)
    with open(args.out, encoding="utf-8") as fh:
        total = sum(1 for _ in fh) - 1
    print(f"\nwrote {args.out} ({total:,} rows total)")
    print("upload it once each drive has 30 days of history")
    return 0


if __name__ == "__main__":
    sys.exit(main())
