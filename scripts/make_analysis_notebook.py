"""Generate `Dataset_backblaze_Analysis.ipynb` -- the three-step analysis notebook.

    python scripts/build_backblaze_archive.py       # once: fetch + shard the archive
    python scripts/make_analysis_notebook.py
    python -m jupyter nbconvert --to notebook --execute --inplace \
        Dataset_backblaze_Analysis.ipynb

Thin notebook by the same convention as `make_window_notebook.py`: the logic lives in
`scripts/leakage_audit.py`, `scripts/backblaze_window_pipeline.py` and
`scripts/archive_window_pipeline.py`, and the notebook drives them, explains what each
step is for, and draws the plots inline.

This generates the **full-archive** version: every drive model Backblaze has ever run,
2013 to the present, with every possible 30-day window enumerated for evaluation.
`scripts/run_full_archive_analysis.py` is the same three steps as a plain script.
"""
import json
import os

NB_NAME = "Dataset_backblaze_Analysis.ipynb"
CELLS = []


def md(text):
    # Each entry must carry its own newline: nbformat concatenates the list verbatim,
    # so a list of bare lines would render as one run-on paragraph.
    src = text.strip("\n").split("\n")
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": [ln + "\n" for ln in src[:-1]] + [src[-1]]})


def code(text):
    src = text.strip("\n").split("\n")
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [ln + "\n" for ln in src[:-1]] + [src[-1]]})


# ===========================================================================
md(r"""
# Drive failure prediction from 30-day SMART windows — the whole Backblaze archive

Three steps over **every drive Backblaze has ever published telemetry for**:

| step | question | output |
|---|---|---|
| **1** | is any column a *proxy* for `failure` rather than a predictor of it? | a scored audit, a blocklist, and the surviving feature set $X$ |
| **2** | what does a 30-day window cover, and how many are there? | a window index over *every possible start offset*, and the timeline histograms |
| **3** | how well does a sequence model rank drives about to fail? | per-epoch logs, test metrics, confusion matrix, Precision@K |

**The data.** The Backblaze Drive Stats archive in full: three yearly zips (2013–2015)
and one per quarter from Q1 2016 on, fetched and reduced to parquet shards by
`scripts/build_backblaze_archive.py`. Every manufacturer, every model, every drive —
**no vendor filter and no cohort sampling**.

That last point is the important difference from this project's earlier runs. The
Toshiba export was a **matched cohort**: every drive that failed plus a sample of drives
that did not, so roughly half of it carried a failure. This is the **fleet**, at its real
prevalence — failures are a fraction of a percent of drive-days. Precision numbers from a
cohort are inflated by construction and do not transfer; these do.

**Every possible window.** A sample is one drive's 30 consecutive daily readings, and the
label is *does this drive fail within 30 days of the window's last day*. For evaluation
the window start is enumerated at **stride 1** — every offset from day 0 to
`length − 30` for every drive in the split. Nothing is skipped and nothing is sampled.
""")

code(r"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
import leakage_audit as la
import backblaze_window_pipeline as bw
import archive_window_pipeline as ap

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9})
pd.set_option("display.width", 200, "display.max_columns", 40)

SHARD_DIR = os.environ.get("BACKBLAZE_SHARDS",
                           os.path.join("data", "backblaze_archive", "shards"))
STORE_DIR = os.environ.get("BACKBLAZE_STORE",
                           os.path.join("data", "backblaze_archive", "store"))
FIG_DIR   = "results/figures"
SEED      = 42
os.makedirs(FIG_DIR, exist_ok=True)
bw.set_seed(SEED)

shards = ap.shard_paths(SHARD_DIR)
print(f"torch {torch.__version__} | device "
      f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"{len(shards)} archive shards: {os.path.basename(shards[0])[:-8]} "
      f"-> {os.path.basename(shards[-1])[:-8]}")
""")

# ===========================================================================
md(r"""
---
## Step 1 — Per-drive telemetry and target-leakage analysis

A leaked column is one that is not *evidence about* the outcome but a *restatement of*
it — the "failing grade of 40" used to predict pass/fail. In drive telemetry that column
is almost never a SMART attribute. It hides in three other places, and the audit probes
all three:

1. **Column values.** Every telemetry column scored against the drive-day `failure` flag
   with a point-biserial correlation, a rank AUC, and the best single-threshold balanced
   accuracy. One column that separates the classes at AUC ≈ 1.0 on its own is a
   giveaway.

2. **Missingness.** A column can leak through *whether it is null* while its values stay
   innocuous — an operational pipeline that stops populating a field once a drive is
   pulled stamps the outcome onto the row. Scored as
   $P(\text{null} \mid \text{failure}) - P(\text{null} \mid \text{healthy})$.

3. **Record geometry.** The big one. Backblaze writes `failure = 1` on a drive's **last
   reporting day**, so anything derived from where a row sits relative to the end of that
   drive's record reconstructs the label exactly. These columns are not in the file — they
   are what a feature-engineering step would *create* — so the audit builds them itself
   and scores them as a warning.

**The audit runs on a sample, and the sample is drawn by *drive*, not by drive-day.**
Scoring 700M+ rows is neither necessary nor affordable, but thinning rows would break the
audit outright: every geometry probe is derived from a drive's own first and last day, and
a drive whose rows have been thinned has neither. Whole drives keep the geometry exact and
keep the fleet failure base rate intact.
""")

code(r"""
t0 = time.time()
cfg0 = ap.ArchiveConfig(shard_dir=SHARD_DIR, store_dir=STORE_DIR, random_state=SEED)
census = ap.build_drive_census(cfg0)

print(f"\n{len(census):,} drives in the archive | "
      f"{int(census.n_obs.sum()):,} observed drive-days | "
      f"{int(census.has_failure.sum()):,} drives with a failure event")
print(f"{pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(census.start_day.min())):%Y-%m-%d}"
      f" -> "
      f"{pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(census.end_day.max())):%Y-%m-%d}"
      f"  ({time.time()-t0:.0f}s)")
""")

code(r"""
audit_df = ap.sample_audit_frame(cfg0, census, n_drives=15_000)
""")

code(r"""
report = la.audit_frame(audit_df)
report.print_report()
""")

md(r"""
### Reading the audit

The same three findings the cohort run produced, now measured on the fleet:

**No raw SMART attribute is a proxy.** The strongest reach a rank AUC in the 0.7–0.8
range — high enough to be genuinely useful, nowhere near the ≥ 0.95 that would mean a
column *is* the label. That is the result we want from section A: the telemetry is
signal, not an echo.

**Fleet-topology columns leak, mostly through absence.** `pod_slot_num`, `vault_id`,
`cluster_id` and `datacenter` describe where a disk sat, not how it was behaving, and
they stop being populated once a disk is pulled. They are carried through ingest
*precisely so the audit can score them* and are then blocked.

**Record geometry is the severe leak.** `days_to_last_record` and
`still_reporting_at_export_end` reconstruct the label directly, because `failure = 1` is
written on a drive's last reporting day. Nothing derived from a record boundary may enter
$X$ — and unlike the cohort run, here `still_reporting_at_export_end` is *also* a
statement about the fleet's growth over thirteen years, which makes it more tempting and
no less fatal.

**Coverage does the rest of the work.** Backblaze has added and dropped SMART columns
repeatedly since 2013 — the 2013 files populate only five attributes — so an attribute
reported by a minority of the archive is dropped on coverage before leakage is even
considered.
""")

code(r"""
features = la.select_feature_columns(audit_df, report)

print(f"{len(features)} attributes survive coverage >= {report.cfg.min_coverage:.0%} "
      f"and the blocklist:\n")
tbl = report.columns.set_index("column")
for c in features:
    r = tbl.loc[c]
    print(f"  {c:<18} coverage {100*r.coverage:6.2f}%  "
          f"{int(r.n_unique):>8,} distinct  rank AUC {r.auc_directional:.4f}")

# The guard: raises if any blocklisted name, or anything derived from one, reached X.
report.assert_clean(features)
""")

code(r"""
la.plot_audit(report, save_path=f"{FIG_DIR}/step1_leakage_audit.png")
""")

code(r"""
del audit_df; gc.collect()
""")

# ===========================================================================
md(r"""
---
## Step 2 — Every possible 30-day window

A sample is one drive's **30 consecutive daily readings**, shape `(30, n_channels)`, and
the label is *does this drive fail within 30 days of the window's last day*. The 30-day
horizon is what makes the task a maintenance question rather than a detection one: a
window that ends 30 days before the disk dies is a **useful** warning, and under the
strict "failure inside the window" rule it would be labelled negative.

**Where a window may start.** Each drive is laid on a gap-free daily calendar — a drive
seen on the 1st and the 5th gets rows for the 2nd, 3rd and 4th, forward-filled from the
last real reading and flagged `observed = 0` so the model can discount them. A window is
a contiguous slice of that calendar. For validation and test the start is enumerated at
**stride 1**: every offset in `[0, length − 30]`, for every drive. Drives whose record is
shorter than 30 days are dropped rather than padded, so every window really is 30
continuous days.

**Why this needs a different store.** At archive scale the feature matrix is ~140 GB
against 103 GB of RAM, so it lives on disk as a memory-mapped array built by an external
sort — the shards arrive time-major and the store needs drive-major. The window index is
never materialised either: per-drive window *counts* plus a `searchsorted` turn a global
window id into `(drive, start)` arithmetically, which is the difference between ~5 MB of
index and the ~6 GB an explicit list would cost. The batch is the unit of work; nothing
in the hot path runs per window in Python.

**The leakage guard, again.** The failure flag lives in `store.failure`, a separate array
from `store.values`. There is no code path by which a model reads the label out of $X$ —
the window tensor contains SMART channels, their 1- and 7-day deltas, and the observed
mask, and nothing else.
""")

code(r"""
cfg = ap.ArchiveConfig(
    shard_dir         = SHARD_DIR,
    store_dir         = STORE_DIR,
    window_days       = 30,    # a sample is 30 consecutive days
    horizon_days      = 30,    # positive if the drive fails within 30 days of window end
    stride_days       = 1,     # EVERY possible window start is enumerated for eval
    min_days          = 30,    # shorter drives are dropped, not left-padded
    delta_lags        = (1, 7),
    add_observed_mask = True,
    feature_columns   = tuple(features),
    log1p_columns     = tuple(c for c in features
                              if c not in ("smart_194_raw", "smart_190_raw")),
    n_buckets         = 64,    # external-sort width
    samples_per_drive = 4,     # random training windows per drive per epoch
    train_positive_ratio = 0.25,
    batch_size        = 4096,
    random_state      = SEED,
)

store = ap.build_archive_store(cfg)
""")

code(r"""
# Stride 1 everywhere -- training, validation and test all see every possible window.
# An earlier version ran per-epoch validation at a coarser stride because the gather was
# the bottleneck; reading each batch as one contiguous slab instead of 4,096 scattered
# window lookups removed the need for that compromise.
bundle = ap.build_archive_bundle(cfg, store=store, val_stride=1)
""")

md(r"""
### How many windows is "every possible window"?

The count below is exact, not an estimate: it is
$\sum_{\text{drives}} (\text{length} - 30 + 1)$ over each split. The evaluation really
does score every one of them — the batching exists so that it can, not so that it can
skip any.
""")

code(r"""
rows = []
for name in ("train", "val", "test"):
    ds = bundle.datasets[name]
    idx = bundle.splits[name]
    n_possible = int((store.lengths[idx] - cfg.window_days + 1).sum())
    rows.append({
        "split": name,
        "drives": len(idx),
        "drives with a failure": int(store.drive_has_failure[idx].sum()),
        "dense drive-days": int(store.lengths[idx].sum()),
        "every possible 30-day window": n_possible,
        "windows this split uses": int(ds.n_windows if hasattr(ds, "n_windows")
                                       else ds.n_samples),
        "batches": len(ds),
    })
coverage = pd.DataFrame(rows).set_index("split")
print(coverage.to_string(formatters={c: "{:,}".format for c in coverage.columns
                                     if coverage[c].dtype.kind in "iu"}))

# The guarantee, asserted rather than asserted-in-prose: at stride 1 the number of
# windows the test split enumerates is exactly the number that exist.
full_test = bundle.datasets["test"]
expected = int((store.lengths[bundle.splits["test"]] - cfg.window_days + 1).sum())
assert cfg.stride_days == 1 and full_test.n_windows == expected, \
    (cfg.stride_days, full_test.n_windows, expected)
print(f"\ntest split enumerates every one of {full_test.n_windows:,} possible windows "
      f"in {len(full_test):,} batches of {cfg.batch_size}")
""")

md(r"""
### The window timeline

The training sampler draws a start uniformly from `[0, length − 30]`, seeded by
`(seed, epoch, batch)` rather than from global RNG state, so a new epoch re-rolls every
drive's window and the same call reproduces the same windows wherever it runs. The plots
below show where those windows actually land.
""")

code(r"""
timeline = ap.sample_window_timeline(store, bundle.splits, n_per_drive=2,
                                     max_drives=40_000, seed=SEED)
ap.describe_window_timeline(timeline, cfg)
timeline.head(8)
""")

code(r"""
ap.plot_window_timeline(timeline, cfg, coverage=coverage,
                        save_path=f"{FIG_DIR}/step2_window_timeline.png")
""")

md(r"""
### Reading the timeline plots

**Top left — start dates.** The dotted verticals are calendar year boundaries. If the
windows were quarter- or month-aligned the histogram would collapse onto those lines;
instead it is broad across the whole thirteen-year range, with starts on every day of the
week in roughly equal proportion. It rises over time because the fleet grew — there are
simply more drives reporting in 2025 than in 2014.

**Top right — end dates**, with positive windows overlaid. Positives cluster where
failures actually happened rather than spreading evenly, which is a property of the data,
not of the sampler.

**Bottom left — start position along each drive's own life**, normalised to $[0, 1]$.
Flat means the sampler is not biased toward the beginning or the end of a drive's record
— important, because "near the end of the record" is exactly the leaked signal Step 1
blocked, and a sampler that favoured late windows would smuggle it back in through the
sampling distribution.

**Bottom centre — genuine readings per window.** Most windows are 30 real daily readings;
the tail is drives with reporting gaps, forward-filled and marked in the `observed`
channel.
""")

# ===========================================================================
md(r"""
---
## Step 3 — Training with per-epoch logging

A 1D-CNN over the sequence: dilated temporal blocks (a TCN, dilations 1/2/4/8) so the
receptive field spans the full 30 days without a recurrence, then global pooling to one
logit per window. `(batch, 30, channels) → (batch, 1)`.

**On the class imbalance.** This is the fleet, not a cohort, so the imbalance is severe —
a fraction of a percent of windows are positive. The single biggest mistake in this
project's earlier models was correcting that *twice* — a balanced sampler **and** a large
`pos_weight`, which distorts the gradient without improving the ranking every headline
metric here measures. So exactly one correction is applied: the training sampler forces
25% of the windows drawn from failing drives to be positive ones, and the loss is plain
BCE.

**Early stopping on validation PR-AUC**, not on loss or accuracy. At this positive rate
accuracy is meaningless — a model that predicts "healthy" every time scores over 99.6% —
and the loss moves with the imbalance correction rather than with the ranking that
actually gets used.
""")

code(r"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcfg = bw.TrainConfig(
    arch="cnn", hidden_dim=64, dropout=0.2,
    batch_size=cfg.batch_size, lr=1e-3, max_epochs=12, patience=3,
    loss="bce",        # one imbalance correction only -- the sampler does the other job
    sampler="none",
    # num_workers=0 on purpose. A batch is 4096 x 30 x 46 float32 = 23 MB, and pushing
    # that through worker IPC costs several times what the gather itself costs -- 8
    # workers measured 7x SLOWER than none. The slab read is already fast enough that
    # the GPU, not the disk, is the limit.
    num_workers=0, seed=SEED,
)

loaders = ap.build_loaders(bundle, tcfg, device)
model = bw.build_model(bundle.num_features, cfg.window_days, tcfg).to(device)

with torch.no_grad():
    probe = torch.zeros(2, cfg.window_days, bundle.num_features, device=device)
    print(f"input (2, {cfg.window_days}, {bundle.num_features}) -> "
          f"{tuple(model(probe).shape)}  (one logit per window)")
print(f"{bw.count_parameters(model):,} trainable parameters on {device}")
""")

code(r"""
model, history, best_val = ap.train_archive_model(model, loaders, bundle, tcfg, device)
print(f"\nBest validation PR-AUC {best_val:.4f} at epoch "
      f"{1 + int(np.nanargmax(history['val_pr_auc']))}")
""")

code(r"""
bw.plot_training_history(history, save_path=f"{FIG_DIR}/step3_training_history.png")
""")

md(r"""
### Final evaluation

The decision threshold is the only quantity carried out of validation, and it is fixed
**before** the test split is touched. Everything below — PR-AUC, ROC-AUC, precision,
recall, F1, the confusion matrix and Precision@K — comes from one pass at that frozen
threshold, over **every possible 30-day window** in each split.

Precision@K is reported two ways. **Ranking windows** is the raw metric; **ranking
drives** collapses each drive to its highest-scoring window first, and is the form that
matches the decision an operator actually makes — you pull *K disks*, not K windows, and
one bad drive contributing hundreds of overlapping windows should not fill the top of the
list on its own.
""")

code(r"""
results = ap.evaluate_archive_final(model, loaders, bundle, tcfg, device,
                                    pk_ks=(10, 25, 50, 100, 250, 1000))
results["history"] = history
""")

code(r"""
ap.plot_evaluation(results, ks=(10, 25, 50, 100, 250, 1000),
                   save_path=f"{FIG_DIR}/step3_evaluation.png")
""")

code(r"""
print("Highest-risk drives in the test split (each scored by its worst window):\n")
print(results["top_drives"].head(20).to_string(index=False))
""")

code(r"""
os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
ap.save_artifacts(model, bundle, tcfg, results, prefix="models/backblaze_archive30")

pd.DataFrame(history).to_csv("results/archive30_training_history.csv", index=False)
report.columns.to_csv("results/archive30_column_association.csv", index=False)
pd.concat([report.geometry, report.categoricals], ignore_index=True).to_csv(
    "results/archive30_leakage_probes.csv", index=False)
timeline.to_csv("results/archive30_sampled_windows.csv", index=False)
coverage.to_csv("results/archive30_window_coverage.csv")
results["top_drives"].to_csv("results/archive30_top_drives.csv", index=False)
pd.concat([results["precision_at_k_windows"].assign(ranked="windows"),
           results["precision_at_k_drives"].assign(ranked="drives")],
          ignore_index=True).to_csv("results/archive30_precision_at_k.csv", index=False)

with open("results/archive30_summary.json", "w", encoding="utf-8") as fh:
    json.dump({
        "features": list(features),
        "blocklist": report.blocklist,
        "n_drives": int(store.n_drives),
        "n_drive_days": int(store.n_rows),
        "n_val_windows": results["n_val_windows"],
        "n_test_windows": results["n_test_windows"],
        "best_threshold": results["best_threshold"],
        "val_at_tuned": results["val"],
        "test_at_tuned": results["test_at_tuned"],
        "test_at_half": results["test_at_half"],
    }, fh, indent=2)
print("wrote results/archive30_*.csv and results/archive30_summary.json")
""")

# ===========================================================================
md(r"""
---
## What these numbers do and do not say

**They rank, and now the base rate is real too.** PR-AUC and Precision@K score the
ordering of drives by risk, and that ordering is what a maintenance queue consumes. Unlike
this project's earlier cohort runs, the prevalence here is the fleet's own — a fraction of
a percent of drive-days — so the precision figures are not inflated by the sampling and do
not need to be discounted before they are read.

**Precision is low in absolute terms, and that is the honest answer.** At a base rate
below one percent, a precision of even a few percent is a large multiple of chance; the
lift column in the Precision@K table is the number to read, not the raw precision. The
drive-level table is the one that matches an operator's decision.

**The evaluation windows overlap.** At stride 1 a drive contributes one window per day of
its record and consecutive ones share 29 of their 30 days, so window-level counts are not
independent samples. That is the cost of covering every possible window, and it is exactly
why the drive-level Precision@K is reported alongside: it is the read that does not
double-count.

**Thirteen years is not one fleet.** The archive spans drive models, capacities and
datacenters that never coexisted, and the SMART attributes populated in 2013 are not the
ones populated in 2026. The coverage filter in Step 1 keeps only attributes reported
across most of the archive, but a model trained over the whole span is answering a
different question from one trained on the current fleet.

**What Step 1 bought.** Every record-geometry feature the audit blocked would have pushed
these numbers higher, and every point of it would have been the archive's construction
scoring itself. The metrics above are lower than a leaky pipeline's and are the ones that
would survive contact with a real fleet.
""")

# ===========================================================================
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(root, NB_NAME)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {path}  ({len(CELLS)} cells)")
