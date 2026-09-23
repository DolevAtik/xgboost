"""Generate the 90-day-window notebook, for either dataset.

Thin notebook: all the logic lives in scripts/backblaze_window_pipeline.py, the
notebook drives it and adds the plots. Regenerate after a config change:

    python scripts/make_window_notebook.py                    # Dataset_backblaze_Window90.ipynb
    python scripts/make_window_notebook.py --source toshiba   # Dataset_Toshiba_Window90.ipynb

The two notebooks differ only in section 2 -- where the frame comes from and which SMART
attributes exist -- and in the few run-configuration numbers those imply. Windowing,
sanity checks, model, training and evaluation are the same code on the same contracts.
"""
import json
import os
import sys

SOURCE = "backblaze"
if "--source" in sys.argv:
    SOURCE = sys.argv[sys.argv.index("--source") + 1].lower()
if SOURCE not in ("backblaze", "toshiba"):
    raise SystemExit("--source must be 'backblaze' or 'toshiba'")
IS_TOSHIBA = SOURCE == "toshiba"

NB_NAME = "Dataset_%s_Window90.ipynb" % ("Toshiba" if IS_TOSHIBA else "backblaze")
DATASET_LABEL = (
    "the local Toshiba export -- 14 drive families pooled, 2024-01 to 2026-03"
    if IS_TOSHIBA else
    "Backblaze quarterly releases -- Seagate drives only"
)

# Substituted into every cell at generation time. The Toshiba export gives ~5.6k drives
# with a 685-day median series (plenty of windows per drive) and, being a cohort, about
# half of them fail -- so the batches need no resampling and one correction in the loss
# is enough. A Backblaze quarter is the opposite: many drives, few failures, short
# series.
PLACEHOLDERS = {
    "__MAX_EPOCHS__": "8" if IS_TOSHIBA else "10",
    "__SAMPLER__": '"none"' if IS_TOSHIBA else '"balanced"',
    "__STRIDE__": "30" if IS_TOSHIBA else "15",
    "__SPD__": "2" if IS_TOSHIBA else "1",
    "__COHORT_NOTE__": (
        "This export is every failed drive plus a matched sample of healthy ones, so "
        "about half\nthe drives carry a failure and the window positive rate sits orders "
        "of magnitude above\nproduction."
        if IS_TOSHIBA else
        "A single quarter of Seagate drives is not the whole population."
    ),
}

CELLS = []


def _fill(text):
    for k, v in PLACEHOLDERS.items():
        text = text.replace(k, v)
    return text


def md(text):
    text = _fill(text)
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").split("\n")})


def code(text):
    src = _fill(text).strip("\n").split("\n")
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [line + "\n" for line in src[:-1]] + [src[-1]]})


md(r"""
# Drive Failure Prediction from 90-Day SMART Windows

A refactor of `Dataset_backblaze_Comparison.ipynb` from **row-level** to **window-level**
classification.

| | row-level notebook | this notebook |
|---|---|---|
| a sample is | one drive-day | one drive's **90 consecutive days** |
| X shape | `(batch, 129)` hand-built aggregates | `(batch, 90, num_features)` raw daily sequence |
| label | will this drive fail in 10 days? | does a failure fall **inside this window**? |
| model | MLP over aggregates | dilated 1D-CNN / GRU / Transformer over the sequence |
| split | `GroupShuffleSplit` on `drive_id` | same — windows of one disk never cross splits |

The trailing-window statistics the row-level pipeline computed by hand (`_d7`, `_mean90`,
`_std90`, `_max90`) are gone: the sequence model is handed the 90 raw daily readings and
derives whatever summary it needs.

Everything is implemented in [`scripts/backblaze_window_pipeline.py`](scripts/backblaze_window_pipeline.py);
this notebook drives it and draws the results.
""" + "\n**Dataset:** %s.\n" % DATASET_LABEL)

md("## 1. Setup")

code(r"""
import os, sys, json, time, gc
sys.path.insert(0, "scripts")

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from backblaze_window_pipeline import (
    DataConfig, TrainConfig, SMART_RAW_COLUMNS, LOG1P_COLUMNS,
    load_backblaze_frame, make_synthetic_frame,
    build_window_bundle, build_loaders, run_sanity_checks, inspect_windows,
    build_model, count_parameters, train_model, evaluate_final, evaluate_loader,
    make_criterion, compute_metrics, save_artifacts, set_seed,
)

%matplotlib inline

SEED = 42
set_seed(SEED)
torch.set_num_threads(os.cpu_count() or 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- run configuration -------------------------------------------------------
# Sized for a CPU run over one quarter. On a GPU, raise HIDDEN_DIM to 64 and
# MAX_EPOCHS to 20-30, and try ARCH = "transformer".
ARCH = "cnn"          # cnn | gru | transformer
HIDDEN_DIM = 48
BATCH_SIZE = 512
MAX_EPOCHS = __MAX_EPOCHS__
PATIENCE = 3
SAMPLER = __SAMPLER__     # batch resampling; see section 4

# Categorical slots 1-3 of the validated default palette (all-pairs safe under CVD).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 10,
    "axes.titlesize": 11, "legend.frameon": False, "lines.linewidth": 2,
})

print(f"torch {torch.__version__} | device {device} | threads {torch.get_num_threads()}")
""")

if IS_TOSHIBA:
    md(r"""
## 2. Data

The Toshiba export is already local: `data/toshiba_drivestats.parquet`, 3.2M drive-days
across 14 drive families in one file. No download, and the columns this notebook needs
fit in memory whole.

**The Seagate feature list does not carry over.** `smart_187`, `smart_188` and
`smart_241` — three of the nine attributes the Backblaze notebook is built on — are not
reported at all by these families. `TOSHIBA_PIPELINE.md` derives the replacement from the
data: 11 attributes reported by all 14 families, plus 5 that the two "EY" families omit
and that get median-filled there.

This export is also **a cohort, not a fleet**: every failed drive plus a matched sample
of healthy ones, which is why roughly half the drives here carry a failure event. The
ranking metrics transfer to production; the absolute positive rate does not.
""")

    code(r"""
import pyarrow.parquet as pq

TOSHIBA_PARQUET = os.environ.get("TOSHIBA_PARQUET",
                                 os.path.join("data", "toshiba_drivestats.parquet"))
ARTIFACT_PREFIX = os.path.join("models", "toshiba_window")

CORE_ATTRS = [
    "smart_3_raw",    # Spin-Up Time
    "smart_4_raw",    # Start/Stop Count
    "smart_5_raw",    # Reallocated Sectors Count
    "smart_9_raw",    # Power-On Hours
    "smart_12_raw",   # Power Cycle Count
    "smart_192_raw",  # Power-Off Retract Count
    "smart_193_raw",  # Load/Unload Cycle Count
    "smart_194_raw",  # Temperature (Celsius)
    "smart_196_raw",  # Reallocation Event Count
    "smart_198_raw",  # Offline Uncorrectable Sector Count
    "smart_199_raw",  # UDMA CRC Error Count
]
EXTENDED_ATTRS = [
    "smart_191_raw",  # G-Sense Error Rate
    "smart_197_raw",  # Current Pending Sector Count
    "smart_220_raw",  # Disk Shift
    "smart_222_raw",  # Loaded Hours
    "smart_226_raw",  # Load-In Time
]
FEATURE_COLUMNS = CORE_ATTRS + EXTENDED_ATTRS
LOG1P_COLS = [c for c in FEATURE_COLUMNS if c != "smart_194_raw"]  # temperature is bounded

t0 = time.time()
raw_df = pq.read_table(
    TOSHIBA_PARQUET,
    columns=["date", "serial_number", "model", "failure"] + FEATURE_COLUMNS,
).to_pandas()
raw_df = raw_df.rename(columns={"serial_number": "id", "date": "time"})
raw_df["time"] = pd.to_datetime(raw_df["time"])
raw_df["failure"] = raw_df["failure"].fillna(0).astype("int8")
for c in FEATURE_COLUMNS:
    raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce").astype("float32")
raw_df["id"] = raw_df["id"].astype("category")
raw_df["model"] = raw_df["model"].astype("category")
source = f"Toshiba export ({os.path.basename(TOSHIBA_PARQUET)})"

print(f"source          : {source}  ({time.time() - t0:.0f}s)")
print(f"drive-days      : {len(raw_df):,}")
print(f"drives          : {raw_df['id'].nunique():,}")
print(f"drive models    : {raw_df['model'].nunique()}")
print(f"date range      : {raw_df['time'].min().date()} -> {raw_df['time'].max().date()}")
print(f"failure events  : {int(raw_df['failure'].sum()):,}")
print(f"drives that fail: {raw_df.loc[raw_df['failure'] == 1, 'id'].nunique():,}")
print(f"row-level positive rate: {100 * raw_df['failure'].mean():.5f}%")
print(f"attributes      : {len(FEATURE_COLUMNS)} "
      f"({len(CORE_ATTRS)} core + {len(EXTENDED_ATTRS)} extended)")
raw_df.head()
""")

    code(r"""
# How much telemetry does each drive actually have? A 90-day window needs more than 90
# days of calendar before "draw a random window" means anything at all.
span = raw_df.groupby("id", observed=True)["time"].agg(["min", "max", "count"])
span["days"] = (span["max"] - span["min"]).dt.days + 1
span["windows"] = (span["days"] - 90 + 1).clip(lower=1)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
axes[0].hist(span["days"], bins=60, color=BLUE)
axes[0].axvline(90, color=ORANGE, ls="--", label="90-day window")
axes[0].set_xlabel("calendar days of telemetry"); axes[0].set_ylabel("drives")
axes[0].set_title("Series length per drive"); axes[0].legend()
axes[1].hist(span["windows"], bins=60, color=AQUA)
axes[1].set_xlabel("distinct 90-day windows"); axes[1].set_ylabel("drives")
axes[1].set_title("Windows available per drive")
plt.tight_layout(); plt.show()

print(span[["days", "count", "windows"]].describe().round(1).to_string())
for t in (30, 90, 180, 365):
    print(f"  drives with >= {t:>3} days: {int((span['days'] >= t).sum()):,} "
          f"({100 * (span['days'] >= t).mean():.1f}%)")
""")
else:
    md(r"""
## 2. Data

Backblaze publishes one CSV per day; `load_backblaze_frame` downloads a quarter, keeps
the Seagate drives and the nine SMART attributes the row-level notebook settled on, and
caches the result as parquet. The bulk data is staged **outside** the project folder —
the extracted quarter is several GB and this project lives in a synced OneDrive
directory.

Set `DATA_DIR` to wherever you want the download to land. If the cache is missing the
notebook falls back to a synthetic frame so it still runs end to end.
""")

    code(r"""
DATA_DIR = os.environ.get("BACKBLAZE_DATA_DIR", r"C:\Users\User\AppData\Local\Temp\backblaze_data")
CACHE = os.path.join(DATA_DIR, "backblaze_seagate_h1_2023.parquet")
QUARTERS = ["Q1_2023", "Q2_2023"]
ARTIFACT_PREFIX = os.path.join("models", "backblaze_window")
FEATURE_COLUMNS = list(SMART_RAW_COLUMNS)
LOG1P_COLS = list(LOG1P_COLUMNS)

t0 = time.time()
if os.path.exists(CACHE):
    raw_df = pd.read_parquet(CACHE)
    source = f"cache {os.path.basename(CACHE)}"
else:
    try:
        raw_df = load_backblaze_frame(QUARTERS, model_prefix_filter="ST",
                                      cache=CACHE, data_dir=DATA_DIR)
        source = "Backblaze download"
    except Exception as exc:                       # offline / no disk space
        print(f"falling back to synthetic data: {type(exc).__name__}: {exc}")
        raw_df = make_synthetic_frame(n_drives=3000, seed=SEED)
        source = "SYNTHETIC (not real telemetry)"

# 20M drive ids as Python strings cost about a gigabyte; as category codes, ~80 MB.
raw_df["id"] = raw_df["id"].astype("category")
raw_df["model"] = raw_df["model"].astype("category")

print(f"source          : {source}  ({time.time() - t0:.0f}s)")
print(f"drive-days      : {len(raw_df):,}")
print(f"drives          : {raw_df['id'].nunique():,}")
print(f"drive models    : {raw_df['model'].nunique()}")
print(f"date range      : {raw_df['time'].min().date()} -> {raw_df['time'].max().date()}")
print(f"failure events  : {int(raw_df['failure'].sum()):,}")
print(f"drives that fail: {raw_df.loc[raw_df['failure'] == 1, 'id'].nunique():,}")
print(f"row-level positive rate: {100 * raw_df['failure'].mean():.5f}%")
raw_df.head()
""")

md(r"""
## 3. Windowing

`DriveSeriesStore` puts every drive on a **gap-free daily calendar**: a drive seen on the
1st and the 5th gets rows for the 2nd–4th, forward-filled from the last real reading and
flagged in an `observed` channel the model can see. Counters are `log1p`-compressed, and
7-day differences are added as extra channels — 19 channels in all. (A 1-day
difference channel is available via `delta_lags=(1, 7)`; it is left out here because a
kernel-3 convolution computes it in its first layer anyway, and 20M rows x 9 extra
float32 channels is 750 MB of RAM that this machine would rather spend elsewhere.)

The label rule is the one thing that must be exactly right:

> **y = 1** if a `failure == 1` event falls inside `[start, start + 89]`, else **0**.

`failure` itself is held in a separate array and is never part of X.
""")

code(r"""
cfg = DataConfig(
    window_days=90,
    horizon_days=0,          # strict: the failure must land INSIDE the window
    stride_days=__STRIDE__,          # val/test enumerate a window every N days
    samples_per_drive=__SPD__,     # random training windows per drive per epoch
    train_positive_ratio=0.5,
    min_days=30,             # shorter drives are dropped; 30-89 days are left-padded
    delta_lags=(7,),         # channels: levels + 7-day deltas + the observed mask
    feature_columns=tuple(FEATURE_COLUMNS),
    log1p_columns=tuple(LOG1P_COLS),
    random_state=SEED,
)

bundle = build_window_bundle(raw_df, cfg)

# The store holds its own dense copy; the 20M-row frame is not needed again.
del raw_df
gc.collect()
print(f"\nX channels ({bundle.num_features}): {bundle.feature_names}")
""")

md(r"""
## 4. Sanity checks — before a single gradient step

`run_sanity_checks` prints, in order:

1. **the leakage guard** — that `failure` is absent from X and that no channel is a copy
   of it;
2. **split integrity** — drive counts per split and the (empty) pairwise id overlaps;
3. **window counts** and the positive rate per split;
4. **representative windows** — drive id, start/end dates, tensor shape, label, how much
   of the window is real vs. forward-filled vs. padded, and the actual values across the
   90 days in both scaled and original units;
5. **the first train and test batches** as they come out of the DataLoader — shape, label
   distribution, value statistics, and a numeric slice.
""")

code(r"""
tcfg = TrainConfig(
    arch=ARCH, hidden_dim=HIDDEN_DIM, dropout=0.2,
    batch_size=BATCH_SIZE, lr=1e-3, max_epochs=MAX_EPOCHS, patience=PATIENCE,
    loss="bce_pos_weight", sampler=SAMPLER, balanced_pos_fraction=0.25,
    num_workers=0, seed=SEED,
)
loaders = build_loaders(bundle, tcfg, device)
run_sanity_checks(bundle, loaders, n_samples=3)
""")

md(r"""
## 5. Model

A dilated 1D CNN over the window. Dilations 1→16 across five residual blocks give a
receptive field of 125 days, so the classification head sees the whole window at once.
Pooling concatenates mean, max and the final timestep: the level, the worst moment, and
the drive's most recent state.

`GRUWindowClassifier` and `TransformerWindowClassifier` are drop-in alternatives with the
same `(B, 90, F) -> (B, 1)` contract.
""")

code(r"""
model = build_model(bundle.num_features, cfg.window_days, tcfg).to(device)
print(f"{tcfg.arch}: {count_parameters(model):,} trainable parameters\n")
print(model)

with torch.no_grad():
    probe = torch.zeros(4, cfg.window_days, bundle.num_features, device=device)
    print(f"\nforward: {tuple(probe.shape)} -> {tuple(model(probe).shape)}  (one logit per window)")
""")

md(r"""
## 6. Training

Early stopping on **validation PR-AUC**, not loss or accuracy: at a positive rate of a
fraction of a percent, accuracy is meaningless and the loss moves with the imbalance
correction rather than with the ranking that actually gets used.

The imbalance is corrected **once** — a balanced sampler oversamples failure-carrying
drives, and `make_criterion` therefore drops `pos_weight` back to 1. Correcting it twice
is the mistake documented in `TOSHIBA_PIPELINE.md`.
""")

code(r"""
t0 = time.time()
model, history, best_val_pr_auc = train_model(model, loaders, bundle, tcfg, device)
print(f"\nbest validation PR-AUC {best_val_pr_auc:.4f} in {(time.time() - t0) / 60:.1f} min")
""")

code(r"""
epochs = np.arange(1, len(history["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(epochs, history["train_loss"], color=BLUE, label="train")
axes[0].plot(epochs, history["val_loss"], color=ORANGE, label="validation")
axes[0].set_title("Loss")
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss"); axes[0].legend()

axes[1].plot(epochs, history["val_pr_auc"], color=BLUE, label="PR-AUC")
axes[1].plot(epochs, history["val_roc_auc"], color=ORANGE, label="ROC-AUC")
axes[1].set_title("Validation ranking quality")
axes[1].set_xlabel("epoch"); axes[1].set_ylim(0, 1.02); axes[1].legend()

best = int(np.nanargmax(history["val_pr_auc"])) + 1
axes[1].axvline(best, color=GRID, linestyle="--", linewidth=1)
axes[1].annotate(f"best epoch {best}\nPR-AUC {max(history['val_pr_auc']):.3f}",
                 xy=(best, max(history["val_pr_auc"])), xytext=(6, -28),
                 textcoords="offset points", color=INK_2, fontsize=9)
fig.suptitle("Training history", x=0.02, ha="left", color=INK, fontsize=12)
plt.tight_layout(); plt.show()
""")

md(r"""
## 7. Evaluation

The decision threshold is tuned on **validation only**, then frozen and applied once to
the test split. Test windows are enumerated deterministically every 15 days, so this
number is reproducible.
""")

code(r"""
results = evaluate_final(model, loaders, bundle, tcfg, device)
test_probs, test_targets = results["test_probs"], results["test_targets"]
""")

code(r"""
prec, rec, _ = precision_recall_curve(test_targets, test_probs)
fpr, tpr, _ = roc_curve(test_targets, test_probs)
m = results["test_at_tuned"]
base = test_targets.mean()

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

axes[0].plot(rec, prec, color=BLUE)
axes[0].axhline(base, color=GRID, linestyle="--", linewidth=1)
axes[0].annotate(f"random ranker = {base:.4f}", xy=(0.55, base), xytext=(0, 8),
                 textcoords="offset points", color=INK_2, fontsize=9)
axes[0].scatter([m["recall"]], [m["precision"]], s=60, color=ORANGE, zorder=3)
axes[0].annotate(f"tuned threshold\nP {m['precision']:.3f} / R {m['recall']:.3f}",
                 xy=(m["recall"], m["precision"]), xytext=(8, 8),
                 textcoords="offset points", color=INK_2, fontsize=9)
axes[0].set_title(f"Precision-Recall (PR-AUC {m['pr_auc']:.3f})")
axes[0].set_xlabel("recall"); axes[0].set_ylabel("precision"); axes[0].set_ylim(0, 1.02)

axes[1].plot(fpr, tpr, color=BLUE)
axes[1].plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
axes[1].set_title(f"ROC (ROC-AUC {m['roc_auc']:.3f})")
axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")

cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
axes[2].imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
axes[2].set_xticks([0, 1], ["predicted 0", "predicted 1"])
axes[2].set_yticks([0, 1], ["actual 0", "actual 1"])
axes[2].grid(False)
for i in range(2):
    for j in range(2):
        share = cm[i, j] / max(cm[i].sum(), 1)
        axes[2].text(j, i, f"{cm[i, j]:,}\n{100 * share:.1f}%", ha="center", va="center",
                     color="white" if share > 0.5 else INK, fontsize=10)
axes[2].set_title(f"Confusion @ threshold {m['threshold']:.3f}")

fig.suptitle("Test-set performance", x=0.02, ha="left", color=INK, fontsize=12)
plt.tight_layout(); plt.show()
""")

md(r"""
## 8. Precision@K — the number an operator actually acts on

Nobody inspects every window. They work a queue: rank the fleet by risk, look at the top
K. Precision@K is the share of that queue that really does contain a failure, and lift is
how many times better that is than picking windows at random.
""")

code(r"""
order = np.argsort(-test_probs)
sorted_labels = test_targets[order]
n_pos = int(test_targets.sum())
base = test_targets.mean()

rows = []
for k in [10, 25, 50, 100, 250, 500, max(n_pos, 1)]:
    k = min(k, len(sorted_labels))
    hits = int(sorted_labels[:k].sum())
    rows.append({"K": k, "hits": hits, "precision@K": hits / k,
                 "recall@K": hits / max(n_pos, 1), "lift": (hits / k) / base})
topk = pd.DataFrame(rows).drop_duplicates("K").sort_values("K").reset_index(drop=True)
display(topk.style.format({"precision@K": "{:.3f}", "recall@K": "{:.3f}", "lift": "{:.1f}x"}))

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(topk["K"].astype(str), topk["precision@K"], color=BLUE, width=0.6)
ax.axhline(base, color=GRID, linestyle="--", linewidth=1)
ax.annotate(f"random = {base:.4f}", xy=(0, base), xytext=(0, 6),
            textcoords="offset points", color=INK_2, fontsize=9)
for bar, p, lift in zip(bars, topk["precision@K"], topk["lift"]):
    ax.annotate(f"{p:.2f}\n({lift:.0f}x)",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", color=INK_2, fontsize=9)
ax.set_title("Precision@K on the test split")
ax.set_xlabel("K (windows inspected, ranked by predicted risk)")
ax.set_ylabel("precision@K"); ax.set_ylim(0, 1.1)
plt.tight_layout(); plt.show()
""")

md("## 9. Save the model and its preprocessing")

code(r"""
save_artifacts(model, bundle, tcfg, results, prefix=ARTIFACT_PREFIX)
print(json.dumps({k: v for k, v in results["test_at_tuned"].items()}, indent=2))
""")

md(r"""
## 10. What this does and does not tell you

**Reads on the numbers.** PR-AUC is the headline: with a positive rate well under 1%,
ROC-AUC flatters every model (the huge true-negative pool dominates it) and accuracy is
meaningless. Compare PR-AUC against the base rate printed beside it, not against 1.0.

**The strict label is hard by construction.** The daily-stats convention marks
`failure == 1` on a drive's *last* reported day, so under `horizon_days = 0` exactly one window per failing drive is
positive — the one ending on its final day. Every earlier window of that same drive, even
one from the week before it died, is a negative. A model that fires a week early is
*penalised*. Setting `horizon_days = 14` re-frames the task as "does this drive fail
within 14 days of the window's end", which is both easier and closer to what an operator
wants; it is a one-line change in cell 3.

**Cohort, not fleet.** __COHORT_NOTE__
Ranking metrics (PR-AUC, Precision@K) transfer; the absolute positive rate does not.

**Next steps.** Compare the three architectures on identical splits; sweep
`horizon_days`; check whether the `observed` mask channel earns its place by ablating it;
and try `stride_days = 1` on the test split to score every possible window rather than
every 15th.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT = os.environ.get(
    "TARGET",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), NB_NAME),
)
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1)
print(f"wrote {OUT}  ({len(CELLS)} cells)")
