# Demo data, and how to score your own

Files to drop into the prediction app, and a worked example of getting real telemetry
into the shape it expects.

```bash
python webapp/app.py        # then open http://127.0.0.1:5000
```

Drag any file below onto the drop zone, or click it to browse. Nothing is uploaded
anywhere — the file is parsed in the local process, scored, and dropped.

Rebuild this folder with `python scripts/make_demo_data.py`. Every row is real telemetry
from `data/toshiba_drivestats.parquet`, which derives from Backblaze's public drive-stats
release; nothing is synthesised and no serial here is private.

---

## The demo files

Scored with the default run (`backblaze_2022x_w30`, XGBoost, 30-day window, threshold
0.217). Your numbers will match these exactly.

| file | what it is | what you should see |
|---|---|---|
| `01_healthy_drive.csv` | 1 drive, 30 clean days | risk **0.000** — `hold` |
| `02_failing_drive.csv` | 1 drive, its final 30 days | risk **0.995** — `inspect` |
| `03_fleet_60_drives.xlsx` | 60 drives, 20 of which failed | 11 flagged `inspect`; **9 of the top 10** really failed |
| `04_live_snapshot.xlsx` | 40 drives as of 2026-03-31, **no** `failure` column | 1 flagged of 40 — what a healthy fleet looks like |
| `05_short_windows.xlsx` | 14 drives holding 10–30 days | **3** scored by default, **14** with the partial-window box |
| `06_renamed_and_sparse.csv` | headers `timestamp`/`serial`, only 6 of 15 attributes | scores anyway, warns, names the 9 it imputed |
| `TEMPLATE.csv` | the schema with every column, 2 dummy drives | copy this to build your own |
| `from_smartctl/` | `collect.py`, `convert.py` and a sample log | the real-data walkthrough — see below |

### Which one to open first

**`03_fleet_60_drives.xlsx`** — it is the only one where ranking is the point. Sixty
drives, twenty of which really failed, and the app has to sort them. Turn on **both**
models from the segmented control at the top: the table then shows an XGBoost and a CNN
column, their rank correlation is 0.90, and XGBoost flags three drives the CNN does not.
Because the file carries a `failure` column, a retrospective panel appears scoring the
ranking against it — 4 of the top 5, 9 of the top 10, 16 of the top 25.

Then click the top row. You get three things the table cannot show: the raw telemetry
behind the score, the **risk trajectory** (the same model re-scored with the window slid
back a day at a time, so you can see whether the drive is deteriorating or has been ill
for weeks), and the **SHAP contributions** — exactly which features pushed the score up
and by how much, in log-odds.

**`04_live_snapshot.xlsx`** is the honest one. No `failure` column, every window ending
on the same day, because that is what querying a live fleet gives you: a triage queue and
no answer key. One drive of forty is flagged. That ratio, not the demo files with six
failures in twelve drives, is what production looks like.

**`05_short_windows.xlsx`** is about a decision the app makes for you. Eleven of its
fourteen drives stopped reporting before filling a 30-day window. All fourteen are scored
anyway: the short ones are left-padded automatically, each marked with the days it
actually held (`18 of 30 d`), and seven recorded failures are visible that strict
windowing would have hidden — 4 of the top 5. Padding applies only where it is needed, and
a drive that already holds a full window scores identically either way, so it never costs
you anything on the complete drives.

Tick **"only score drives with a full window"** to see the other side: three drives
scored, eleven skipped, and every recorded failure gone — the retrospective panel has
nothing left to check and disappears entirely. That mode is the
calibrated one — padded scores are an extrapolation, not a calibrated prediction, which is
why every padded row stays labelled and counted in the input report.

---

## Putting your own data in

### The contract

Three things, and only three:

1. **A `date` column and a `serial_number` column.** The app also accepts `time`,
   `timestamp` or `day` for the first, and `serial`, `serialnumber`, `id` or `drive_id`
   for the second, and tells you on screen when it maps one.
2. **One row per drive per day**, with dates as anything pandas can parse
   (`2026-01-31`, `31/01/2026`, an ISO timestamp — it is normalised to a day).
3. **At least 30 consecutive days per drive** — or 10, with the partial-window box
   ticked. Gaps inside the span are fine: they are forward-filled and flagged to the
   model through an `observed` channel, so it can discount imputed stretches.

Everything else is optional. `.xlsx`, `.csv` and `.parquet` all work.

### The columns the model reads

Fifteen raw SMART attributes. Name them exactly like this — the app aliases the key
columns, but **not** the attributes:

| column | SMART # | what `smartctl -A` calls it | why it matters |
|---|---|---|---|
| `smart_197_raw` | 197 | `Current_Pending_Sector` | **the strongest single signal** — sectors the drive cannot read and has not yet remapped |
| `smart_5_raw` | 5 | `Reallocated_Sector_Ct` | **second strongest** — sectors already remapped |
| `smart_198_raw` | 198 | `Offline_Uncorrectable` | uncorrectable read errors found offline |
| `smart_4_raw` | 4 | `Start_Stop_Count` | spindle start/stop cycles |
| `smart_12_raw` | 12 | `Power_Cycle_Count` | power cycles |
| `smart_9_raw` | 9 | `Power_On_Hours` | age in service |
| `smart_240_raw` | 240 | `Head_Flying_Hours` | head-in-air time |
| `smart_193_raw` | 193 | `Load_Cycle_Count` | head load/unload cycles |
| `smart_1_raw` | 1 | `Raw_Read_Error_Rate` | vendor-scaled read error rate |
| `smart_7_raw` | 7 | `Seek_Error_Rate` | vendor-scaled seek error rate |
| `smart_3_raw` | 3 | `Spin_Up_Time` | ms to spin up |
| `smart_192_raw` | 192 | `Power-Off_Retract_Count` | emergency head retracts |
| `smart_194_raw` | 194 | `Temperature_Celsius` | drive temperature |
| `smart_199_raw` | 199 | `UDMA_CRC_Error_Count` | cable/interface errors |
| `smart_10_raw` | 10 | `Spin_Retry_Count` | failed spin-up retries |

**Raw, not normalised.** SMART reports each attribute twice: a vendor-normalised health
value (typically 100 counting down to a threshold) and the raw counter. The model was
trained on the raw counter. If your source has both, take the one that looks like a
count, not the one that starts at 100.

**Missing attributes are allowed.** Anything absent is filled with the training-set
median and named on screen. It costs accuracy honestly rather than silently —
`06_renamed_and_sparse.csv` supplies 6 of the 15 and the top risk falls from 0.995 to
0.833. If you can only get a few, get **197 and 5** first: XGBoost draws over half its
total gain from those two, and from their *movement* rather than their level.

**Columns you should not bother sending.** `capacity_bytes`, `vault_id`, `pod_slot_num`,
`datacenter`, `cluster_id` and anything derived from a record boundary are blocked — they
were measured to be proxies for the outcome rather than evidence about it. Sending them
is harmless (they are ignored, and the app lists them in its provenance panel) but they
will not help. A `failure` column, if you have one, is shown for reference, used for the
retrospective precision@K panel, and **never** given to the model.

### Route A — Backblaze's public drive stats

Already in exactly the right shape; the column names above *are* Backblaze's names. Grab
a quarter from <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>,
unzip, and either upload one daily CSV or concatenate a range:

```bash
python - <<'PY'
import glob, pandas as pd
keep = ["date","serial_number","model","failure"] + [
    f"smart_{n}_raw" for n in (197,5,198,4,12,9,240,193,1,7,3,192,194,199,10)]
files = sorted(glob.glob("data_Q1_2026/2026-03-*.csv"))[-40:]     # last 40 days
df = pd.concat([pd.read_csv(f, usecols=lambda c: c in keep) for f in files])
df.to_parquet("backblaze_recent.parquet", index=False)
print(len(df), "rows,", df.serial_number.nunique(), "drives")
PY
```

A full quarter of the whole fleet is ~250k drives and will not fit in a spreadsheet — use
`.parquet` or `.csv`, and expect a minute or two of parsing. To try one model family
first, add `df = df[df.model.str.contains("ST16000")]`.

### Route B — your own machines, via smartctl

Two scripts in `from_smartctl/`. Use **`collect.py`** if you are starting from scratch —
it writes the app's schema directly, so there is no conversion step. Use **`convert.py`**
if you already have a log in some other shape.

Install `smartmontools`, then collect once a day. Reading SMART data needs root on
Linux/macOS, an elevated prompt on Windows.

```bash
sudo python demo/from_smartctl/collect.py --out /var/log/smart/telemetry.csv
```

```
2026-09-27: 4 drive(s), 15 of 15 attributes
  WD-AAAA1111          WDC WD40EFRX-68N32N0         15 attributes
  ST-BBBB2222          ST16000NM001G-2KK103         13 attributes
  ...
wrote /var/log/smart/telemetry.csv (4 rows total)
upload it once each drive has 30 days of history
```

It finds disks with `smartctl --scan` (or takes `--device /dev/sda /dev/sdb`), keys on
SMART **id** rather than attribute name so it works across vendors, skips NVMe drives
(they have no ATA attribute table, and this model was trained on spinning disks), writes
atomically, and is idempotent per day — running it twice on one date replaces that day's
rows instead of duplicating them, so a cron retry is safe. `--dry-run` prints what it
would write. `--from-json sample.json` parses a saved `smartctl -A -i -j` document, which
is how to check it against your hardware before trusting the cron job.

Schedule it:

```cron
# /etc/cron.d/smart-telemetry
17 4 * * *  root  /usr/bin/python3 /path/to/demo/from_smartctl/collect.py                     --out /var/log/smart/telemetry.csv
```

**You need history.** Thirty days of it, collected a day at a time. Starting the cron job
today means the model has nothing to say for a month, and there is no shortcut: a single
`smartctl` snapshot cannot be scored, because most of what the model reads is *change
over time* — `smart_197` differenced over 7 days is its single largest input. If you have
older data in some other system, reshape that instead (Route C).

#### If you already have a log: convert.py

`convert.py` renames the timestamp and identity columns, maps smartctl attribute names
onto `smart_<N>_raw`, collapses duplicate drive-days, and prints exactly what it did.
Try it on the sample log, which is shaped the way a hand-rolled scraper usually writes
one — `timestamp`, `device`, `serial`, and attributes under their smartctl names:

```bash
$ python demo/from_smartctl/convert.py demo/from_smartctl/smartctl_log.csv       -o demo/from_smartctl/ready_to_score.csv
read demo/from_smartctl/smartctl_log.csv: 206 rows x 18 columns
  timestamp column : 'timestamp' -> date
  identity column  : 'serial' -> serial_number
  attributes mapped: 15
      Raw_Read_Error_Rate          -> smart_1_raw
      Spin_Up_Time                 -> smart_3_raw
      ...
  columns ignored  : device
wrote ready_to_score.csv: 206 rows, 6 drives, 2024-02-22 to 2025-09-18
  days per drive: 34 to 40 (6 of 6 reach a full 30-day window)
```

Upload `ready_to_score.csv` and all six drives score, every one of them `inspect` --
they were picked from the high-risk end of the fleet to make the walkthrough show
something. Note that
it keyed on `serial` rather than `device`: a serial follows the disk when it moves to
another bay, while `/dev/sdb` is just the bay. Override with `--id-column` if you prefer.

If your scraper uses names the table does not know, add them to `ATTRIBUTE_MAP` at the
top of `convert.py` — the match ignores case, spaces, hyphens and underscores.

### Route C — an existing monitoring system

If you already scrape SMART into Prometheus, InfluxDB, Zabbix or a vendor tool, you have
the data and only need to reshape it. Export one row per drive per day and pivot so each
attribute is a column:

```python
import pandas as pd
long = pd.read_csv("export.csv")          # columns: ts, serial, attribute, value
wide = (long.assign(date=pd.to_datetime(long.ts).dt.normalize())
            .pivot_table(index=["date", "serial"], columns="attribute",
                         values="value", aggfunc="last")
            .reset_index())
wide.to_csv("fleet.csv", index=False)     # then run convert.py on it
```

From `node_exporter`'s `smartmon` collector the metric is
`smartmon_attr_raw_value{name="...",smart_id="197"}` — key on `smart_id`, which saves you
the name mapping entirely.

### Before you upload — a checklist

- [ ] one row per drive per day, no duplicates
- [ ] `date` parses, and spans ≥ 30 days per drive (≥ 10 with the checkbox)
- [ ] `serial_number` is stable — a serial, not a `/dev/sdX` path that shifts on reboot
- [ ] attributes are the **raw** counters, named `smart_<N>_raw`
- [ ] counters are cumulative and monotonic; if your exporter sends per-interval deltas,
      `cumsum()` them first, because the model differences them itself
- [ ] under 400 MB (the upload cap); over ~1M rows use `.csv` or `.parquet`, not `.xlsx`

### When it goes wrong

| the app says | what happened | fix |
|---|---|---|
| `the file has no date column…` | no recognised timestamp header | rename to `date`, or use `time`/`timestamp`/`day` |
| `the file has no serial_number column…` | no recognised identity header | rename to `serial_number`, or use `serial`/`id`/`drive_id` |
| `no rows with both a date and a serial number` | dates failed to parse, or the id column is empty | check the date format and that the id is not blank |
| `no drive in this file has 30 days of telemetry — the longest run is N days` | not enough history | collect more, or tick the partial-window box (floor 10 days) |
| `unsupported file type '.xls'` | old-format Excel or something else | re-save as `.xlsx`, `.csv` or `.parquet` |
| every drive scores about the same | all 15 attributes missing or constant | check the input report — you are probably sending normalised values, or the names are wrong |
| `that upload is no longer in memory` | only the 3 most recent uploads are cached | score the file again |

### Scoring against a different model

The dropdown at the top of the page lists every run in `models/`. The 90-day runs read a
different attribute set (16, including `smart_196`, `220`, `222`, `226`, `191`), so a file
built for the default run will show 5 missing attributes against them — which is fine, and
reported. To generate demo data for a different run instead:

```bash
python scripts/make_demo_data.py --run backblaze_window90
RUN=backblaze_window90 python webapp/app.py
```
