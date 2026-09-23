"""Generate `Dataset_backblaze_Analysis_2022_XGBoost.ipynb` -- the 2022 analysis + XGBoost.

    python scripts/build_backblaze_archive.py       # once: fetch + shard the archive
    python scripts/make_analysis2022_xgb_notebook.py
    python scripts/_run_notebook.py Dataset_backblaze_Analysis_2022_XGBoost.ipynb

Everything in `Dataset_backblaze_Analysis_2022.ipynb` (made by
`make_analysis2022_notebook.py`), plus an **XGBoost** model fitted on the same flattened
30-day window table as the decision tree and compared head-to-head with the CNN on the
same test drives and the same windows.

The XGBoost section runs before the decision-tree depth ladder, which is the slowest cell
in the notebook, so its results are in hand early. Every output file carries a `2022x`
prefix so this notebook never overwrites the plain 2022 notebook's results. Both share
the cached `store_2022`, so they should not run at the same time.

The logic lives in `scripts/leakage_audit.py`, `scripts/backblaze_window_pipeline.py`
and `scripts/archive_window_pipeline.py`; the notebook drives them.
"""
import json
import os

NB_NAME = "Dataset_backblaze_Analysis_2022_XGBoost.ipynb"
CELLS = []


def md(text):
    src = text.strip("\n").split("\n")
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": [ln + "\n" for ln in src[:-1]] + [src[-1]]})


def code(text):
    src = text.strip("\n").split("\n")
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": [ln + "\n" for ln in src[:-1]] + [src[-1]]})


# ===========================================================================
md(r"""
# Drive failure from 30-day SMART windows — Backblaze, 2022 to today, CNN vs XGBoost

The same pipeline as `Dataset_backblaze_Analysis_2022.ipynb` — archives from **2022
onward** — with a second model added: **XGBoost** on a flattened summary of each 30-day
window, scored against the CNN on the same test drives and the same windows.

| § | question | what gets printed |
|---|---|---|
| **1** | is any column a proxy for `failure`? | the leakage audit and the surviving feature set $X$ |
| **2** | can the model see the label at all? | **four executed checks**, not a claim |
| **3** | how many 30-day windows cover a drive's whole life? | per-drive window and **batch counts** |
| **4** | does this drive fail in *this* 30 days? (CNN) | a verdict per window, plus per-drive histograms |
| **5** | does gradient boosting on window summaries beat the CNN? | XGBoost training, test metrics, **CNN vs XGBoost** table and curves |
| **6** | what does a tree grown to 100% accuracy look like? | the depth ladder, and what it costs |
| **Summary** | which model is best? | all three models in one table, and the winner by drive precision @100 and PR-AUC |

**Scope.** Archives `Q1_2022` through the newest quarter. A drive that was already in
service in 2021 enters with only its 2022-onward rows, which is the intended reading of
"from 2022 till today".

**Still the fleet, not a cohort.** No vendor filter and no matched sampling: failures are
a fraction of a percent of drive-days, as they are in a real datacentre.
""")

code(r"""
import gc, json, os, sys, time
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
import leakage_audit as la
import backblaze_window_pipeline as bw
import archive_window_pipeline as ap

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9})
pd.set_option("display.width", 220, "display.max_columns", 40)

SHARD_DIR  = os.environ.get("BACKBLAZE_SHARDS",
                            os.path.join("data", "backblaze_archive", "shards"))
STORE_DIR  = os.environ.get("BACKBLAZE_STORE_2022",
                            os.path.join("data", "backblaze_archive", "store_2022"))
SINCE_YEAR = int(os.environ.get("SINCE_YEAR", "2022"))
FIG_DIR    = "results/figures"
SEED       = 42
os.makedirs(FIG_DIR, exist_ok=True)
bw.set_seed(SEED)

shards = ap.shard_paths(SHARD_DIR, since_year=SINCE_YEAR)
print(f"torch {torch.__version__} | xgboost {xgb.__version__} | device "
      f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
print(f"{len(shards)} shards from {SINCE_YEAR} on: "
      f"{os.path.basename(shards[0])[:-8]} -> {os.path.basename(shards[-1])[:-8]}")
print("  " + ", ".join(os.path.basename(p)[:-8] for p in shards))
""")

# ===========================================================================
md(r"""
---
## 1 — Target-leakage audit

A leaked column is not *evidence about* the outcome but a *restatement of* it. In drive
telemetry it is almost never a SMART attribute; it hides in three places, and the audit
probes all three: the column values, the **missingness** pattern, and **record
geometry** — anything derived from where a row sits relative to the end of a drive's
record, because Backblaze writes `failure = 1` on a drive's last reporting day.

The audit samples by **drive**, never by drive-day: every geometry probe is derived from
a drive's own first and last day, and a drive whose rows have been thinned has neither.
""")

code(r"""
t0 = time.time()
cfg0 = ap.ArchiveConfig(shard_dir=SHARD_DIR, store_dir=STORE_DIR,
                        since_year=SINCE_YEAR, random_state=SEED)
census = ap.build_drive_census(cfg0)

epoch = pd.Timestamp("1970-01-01")
print(f"\n{len(census):,} drives | {int(census.n_obs.sum()):,} observed drive-days | "
      f"{int(census.has_failure.sum()):,} drives with a failure event")
print(f"{epoch + pd.Timedelta(days=int(census.start_day.min())):%Y-%m-%d} -> "
      f"{epoch + pd.Timedelta(days=int(census.end_day.max())):%Y-%m-%d}"
      f"   ({time.time()-t0:.0f}s)")
""")

code(r"""
audit_df = ap.sample_audit_frame(cfg0, census, n_drives=15_000)
report = la.audit_frame(audit_df)
report.print_report()
""")

code(r"""
features = la.select_feature_columns(audit_df, report)

print(f"{len(features)} attributes survive coverage >= {report.cfg.min_coverage:.0%} "
      f"and the blocklist:\n")
tbl = report.columns.set_index("column")
for c in features:
    r = tbl.loc[c]
    print(f"  {c:<18} coverage {100*r.coverage:6.2f}%  "
          f"{int(r.n_unique):>9,} distinct  rank AUC {r.auc_directional:.4f}")

report.assert_clean(features)          # raises if a blocked name reached X
la.plot_audit(report, save_path=f"{FIG_DIR}/2022x_step1_leakage_audit.png")
del audit_df; gc.collect()
""")

# ===========================================================================
md(r"""
---
## 2 — Building the store, then **proving** the label cannot be read

The dense matrix is built once and memory-mapped. `failure` is written to a *separate
file* from the feature matrix — it is not a column of `X` that gets dropped later, it
never enters `X` at all.

That is a claim, so the next cell tests it four ways:

1. **Names.** No channel is called `failure` or derived from it.
2. **Storage.** `values.f32` and `failure.u8` are different files and share no memory, so
   no write to one can surface in the other.
3. **Independence.** The label array is replaced with random noise and the same windows
   are gathered again. If a single byte of `X` depended on the label, `X` would move. It
   must come back bit-identical.
4. **Separability.** Every channel is scored against the window label on its own. A
   column that *was* the label would sit at AUC ≈ 1.0. The check fails the notebook if
   any channel exceeds 0.95.

Checks 1, 2 and 4 raise on failure; check 3 is the one that actually settles it, because
it is an experiment on the running code rather than an inspection of it.
""")

code(r"""
cfg = ap.ArchiveConfig(
    shard_dir         = SHARD_DIR,
    store_dir         = STORE_DIR,
    since_year        = SINCE_YEAR,
    window_days       = 30,     # a sample is 30 consecutive days
    horizon_days      = 30,     # positive if the drive fails within 30 days of window end
    stride_days       = 1,      # every possible start is enumerated for eval
    min_days          = 30,     # shorter drives are dropped, not padded
    delta_lags        = (1, 7),
    add_observed_mask = True,
    feature_columns   = tuple(features),
    log1p_columns     = tuple(c for c in features
                              if c not in ("smart_194_raw", "smart_190_raw")),
    n_buckets         = 48,
    samples_per_drive = 4,
    train_positive_ratio = 0.25,
    batch_size        = 4096,
    random_state      = SEED,
)

store = ap.build_archive_store(cfg)
""")

code(r"""
splits = ap.grouped_drive_split(store, cfg)
checks = ap.verify_no_label_leakage(store, splits["train"], n_probe=20_000, seed=SEED)
""")

code(r"""
# The strongest channels, ranked against the window label. Useful signal sits well short
# of the 0.95 line; a proxy would be pinned against 1.00.
top = checks["channel_scores"].head(12).reset_index(drop=True)
top.index += 1
print("strongest single channels vs the 30-day window label "
      f"({checks['n_probe_windows']:,} probe windows):\n")
print(top.to_string())
print(f"\nceiling for a non-leaked channel: 0.95     observed maximum: "
      f"{checks['max_channel_auc']:.4f}")
print(f"model input shape: (batch, {cfg.window_days}, {checks['n_channels']}) "
      f"-- SMART levels, 1- and 7-day deltas, observed mask. No label channel.")
""")

# ===========================================================================
md(r"""
---
## 3 — Covering every drive's whole timeline with 30-day windows

Two different questions, two different window sets, both reported below.

**Tiling** lays non-overlapping 30-day blocks end to end along each drive's record:
$\lceil \text{record days} / 30 \rceil$ windows, so **every drive-day lands in exactly
one window** — nothing counted twice, nothing skipped. This is "as many batches as it
takes to fill the whole timeline", and it is what the decision tree in §5 is fitted on.
Where a record is not a multiple of 30, the last block is pulled back to end on the
drive's final day, so it overlaps its predecessor rather than running short.

**Stride-1 enumeration** is the other extreme: every possible start offset, used for the
sequence model's validation and test passes.

Both counts are printed, per split and per drive.
""")

code(r"""
tiles = {name: ap.TilingWindows(store, splits[name], batch_size=cfg.batch_size,
                                split_name=name)
         for name in ("train", "val", "test")}

W = cfg.window_days
rows = []
for name in ("train", "val", "test"):
    idx, t = splits[name], tiles[name]
    rows.append({
        "split": name,
        "drives": len(idx),
        "drives with a failure": int(store.drive_has_failure[idx].sum()),
        "record days (dense)": int(store.lengths[idx].sum()),
        f"tiling {W}-day windows": t.n_windows,
        "tiling batches": len(t),
        "every possible window": int((store.lengths[idx] - W + 1).sum()),
    })
coverage = pd.DataFrame(rows).set_index("split")
print(coverage.to_string(formatters={c: "{:,}".format for c in coverage.columns
                                     if coverage[c].dtype.kind in "iu"}))

total_tiles = sum(t.n_windows for t in tiles.values())
total_batches = sum(len(t) for t in tiles.values())
print(f"\nTIMELINE COVERAGE")
print(f"  {store.n_drives:,} drives, {store.n_rows:,} dense drive-days")
print(f"  {total_tiles:,} non-overlapping {W}-day windows cover the entire timeline")
print(f"  at batch size {cfg.batch_size:,} that is {total_batches:,} batches in total")
for name in ("train", "val", "test"):
    print("  " + tiles[name].label_report())
""")

code(r"""
# Per-drive accounting: how many windows each drive needs to have its whole record
# covered, and how much the final window had to overlap to stay a full 30 days.
cov = tiles["test"].coverage_report()
print(f"per-drive window counts, first 20 test drives "
      f"(batch size {cfg.batch_size:,}):\n")
print(cov.head(20).to_string(index=False))

n = cov["windows_to_cover_timeline"]
print(f"\nacross all {len(cov):,} test drives:")
print(f"  windows per drive .... min {n.min()}, median {int(n.median())}, "
      f"mean {n.mean():.1f}, max {n.max()}")
print(f"  total windows ........ {int(n.sum()):,}")
print(f"  total batches ........ {len(tiles['test']):,} "
      f"of up to {cfg.batch_size:,} windows each")
print(f"  days covered ......... {int(cov.days_covered.sum()):,} "
      f"(record holds {int(cov.record_days.sum()):,}; the difference is the "
      f"overlap pulled back onto the last window of each drive)")
""")

code(r"""
bundle = ap.build_archive_bundle(cfg, store=store, val_stride=1, verbose=True)
""")

# ===========================================================================
md(r"""
---
## 4 — Training, then a verdict for every 30-day window

A 1D-CNN over the sequence: dilated temporal blocks so the receptive field spans the
full 30 days, then global pooling to one logit per window.
`(batch, 30, channels) → (batch, 1)`.

**One imbalance correction, not two.** The sampler forces 25% of the windows drawn from
failing drives to be positive ones, and the loss is plain BCE. Correcting twice — a
balanced sampler *and* a large `pos_weight` — is the mistake documented in
`TOSHIBA_PIPELINE.md`.

**Early stopping on validation PR-AUC.** At a base rate near 0.1%, accuracy is
meaningless: predicting "healthy" every time scores over 99.8%.
""")

code(r"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tcfg = bw.TrainConfig(
    arch="cnn", hidden_dim=64, dropout=0.2,
    batch_size=cfg.batch_size, lr=1e-3, max_epochs=6, patience=2,
    loss="bce", sampler="none",
    # num_workers=0 on purpose: a batch is ~23 MB and worker IPC costs several times
    # what the gather itself does -- 8 workers measured 7x slower than none.
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
bw.plot_training_history(history, save_path=f"{FIG_DIR}/2022x_step3_training_history.png")
""")

code(r"""
results = ap.evaluate_archive_final(model, loaders, bundle, tcfg, device,
                                    pk_ks=(10, 25, 50, 100, 250, 1000))
results["history"] = history
ap.plot_evaluation(results, ks=(10, 25, 50, 100, 250, 1000),
                   save_path=f"{FIG_DIR}/2022x_step3_evaluation.png")
""")

md(r"""
### The test phase, one verdict per window

The threshold is the only quantity carried out of validation, and it was fixed before the
test split was touched. Below, every tiling window of every test drive gets a single
call — **did this drive fail within 30 days of this window ending, yes or no** — scored
against the truth.

This is the tiling window set, so each test drive-day is judged exactly once.
""")

code(r"""
thr = results["best_threshold"]
pred = ap.predict_windows(model, tiles["test"], device, threshold=thr, tag="test")

tp = int(((pred.predicted_fail == 1) & (pred.actual_fail == 1)).sum())
fp = int(((pred.predicted_fail == 1) & (pred.actual_fail == 0)).sum())
fn = int(((pred.predicted_fail == 0) & (pred.actual_fail == 1)).sum())
tn = int(((pred.predicted_fail == 0) & (pred.actual_fail == 0)).sum())

print(f"\nPER-WINDOW VERDICTS -- {len(pred):,} tiling windows over "
      f"{pred.drive_index.nunique():,} test drives, threshold {thr:.4f}\n")
print(f"  predicted FAIL ....... {int(pred.predicted_fail.sum()):,}")
print(f"  actually  FAIL ....... {int(pred.actual_fail.sum()):,}")
print(f"  correct .............. {tp + tn:,} of {len(pred):,} "
      f"({100*(tp+tn)/len(pred):.4f}% accuracy)")
print(f"  TP {tp:,}   FP {fp:,}   FN {fn:,}   TN {tn:,}")
print(f"  precision {tp/max(tp+fp,1):.4f}   recall {tp/max(tp+fn,1):.4f}")
print(f"\n  note: 'accuracy' here is dominated by the {100*(1-pred.actual_fail.mean()):.3f}% "
      f"of windows that are negative -- always saying 'no fail' would score "
      f"{100*(1-pred.actual_fail.mean()):.4f}%.")
""")

code(r"""
# Every window of a few drives that really failed: the verdict as the run produced it.
failed_drives = (pred[pred.actual_fail == 1].drive_index.drop_duplicates().head(3))
for gidx in failed_drives:
    sub = pred[pred.drive_index == gidx].sort_values("window_start")
    print(f"\ndrive {sub.serial_number.iloc[0]}  --  {len(sub)} windows "
          f"covering {sub.window_start.min():%Y-%m-%d} .. {sub.window_end.max():%Y-%m-%d}")
    show = sub[["window_in_drive", "window_start", "window_end", "risk",
                "predicted_fail", "actual_fail"]].copy()
    show["verdict"] = np.where(show.predicted_fail == 1, "FAIL", "no fail")
    show["truth"] = np.where(show.actual_fail == 1, "FAIL", "no fail")
    print(show.tail(12)[["window_in_drive", "window_start", "window_end",
                         "risk", "verdict", "truth"]].to_string(index=False))
""")

code(r"""
fig, picks = ap.plot_drive_histograms(
    pred, n_drives=24, threshold=thr, seed=SEED,
    save_path=f"{FIG_DIR}/2022x_step4_drive_histograms.png")
print(f"\n{len(picks)} drives plotted, one histogram each: the distribution of that "
      f"drive's 30-day windows over predicted risk.")
""")

# ===========================================================================
md(r"""
---
## 5 — XGBoost on the same windows, head to head with the CNN

A tree model cannot take a `(30, 46)` tensor, so each **tiling** window is reduced to a
row of summary features — per channel, its last day, its 30-day mean, and its
first-to-last change: 46 channels × 3 = 138 columns. The table is built for all three
splits, from the same drive-grouped split the CNN used, so no drive crosses a boundary.

**How XGBoost is trained.**

* `tree_method="hist"` on the GPU when there is one: the 10.5M-row training table fits in
  a few minutes rather than hours.
* **Early stopping on validation PR-AUC** (`eval_metric="aucpr"`), the same criterion the
  CNN stops on, with 100 rounds of patience.
* **No imbalance weighting.** PR-AUC and Precision@K only depend on the ranking, and the
  decision threshold is tuned on validation — so `scale_pos_weight` would add a second
  correction on top of the threshold for no gain.
* **The threshold comes from validation only**, by best F1, then is frozen for test —
  exactly how the CNN's threshold was chosen.

**How the comparison is kept fair.** Both models are scored on the **same 3.1M tiling
test windows** of the same 82k test drives: the CNN's scores are the per-window verdicts
from §4, and the cell asserts that the windows and labels line up row for row. XGBoost
is not scored on the 89M stride-1 windows the CNN was also tested on in §4 — flattening
those would take ~50 GB — so the table below, not the §4 summary, is the like-for-like
read.
""")

code(r"""
t0 = time.time()
Xtr, ytr, feat_cols = ap.window_table_features(
    store, tiles["train"], aggregates=("last", "mean", "delta"))
Xva, yva, _ = ap.window_table_features(
    store, tiles["val"], aggregates=("last", "mean", "delta"))
Xte, yte, _ = ap.window_table_features(
    store, tiles["test"], aggregates=("last", "mean", "delta"))
print(f"\nbuilt in {time.time()-t0:.0f}s   train {Xtr.shape}   val {Xva.shape}   "
      f"test {Xte.shape}")
print(f"{len(feat_cols)} features = {len(store.feature_names)} channels x "
      f"3 aggregates (last, mean, delta)")
""")

code(r"""
# The CNN is finished with the GPU; hand its cached blocks back before XGBoost copies
# the training table over.
torch.cuda.empty_cache()
xgb_device = "cuda" if torch.cuda.is_available() else "cpu"

xgb_clf = xgb.XGBClassifier(
    n_estimators=3000, learning_rate=0.05, max_depth=8, min_child_weight=5,
    subsample=0.8, colsample_bytree=0.8, max_bin=256,
    tree_method="hist", device=xgb_device,
    objective="binary:logistic", eval_metric="aucpr",
    early_stopping_rounds=100, random_state=SEED,
)
t0 = time.time()
xgb_clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=100)
print(f"\nbest iteration {xgb_clf.best_iteration} of {xgb_clf.n_estimators} "
      f"(val PR-AUC {xgb_clf.best_score:.4f})  --  {time.time()-t0:.0f}s on {xgb_device}")
""")

code(r"""
# Predict on the CPU: the tables are NumPy arrays in host memory, and scoring them with a
# GPU booster makes XGBoost warn and fall back to a copy anyway.
xgb_clf.set_params(device="cpu")
p_va = xgb_clf.predict_proba(Xva)[:, 1]
p_te = xgb_clf.predict_proba(Xte)[:, 1]

# Threshold from validation only, then frozen -- the same rule the CNN followed.
thr_xgb, f1_va = ap.sweep_threshold(yva, p_va)
xgb_val = bw.compute_metrics(yva, p_va, thr_xgb)
xgb_half = bw.compute_metrics(yte, p_te, 0.5)
xgb_test = bw.compute_metrics(yte, p_te, thr_xgb)
print(f"threshold tuned on validation: {thr_xgb:.4f}  (val F1 {f1_va:.4f})\n")
for name, m in (("val   @tuned", xgb_val), ("test  @0.50", xgb_half),
                ("test  @tuned", xgb_test)):
    print("  " + bw.format_metrics(name, m))
print()

KS = (10, 25, 50, 100, 250, 1000)
g_te, _ = tiles["test"].locate(np.arange(tiles["test"].n_windows, dtype=np.int64))
xgb_pk_win = bw.print_precision_at_k(
    yte, p_te, "XGBOOST TEST PRECISION@K -- ranking windows", KS)
drive_codes, p_drv_xgb, y_drv = ap.drive_level_scores(p_te, yte, g_te)
xgb_pk_drv = bw.print_precision_at_k(
    y_drv, p_drv_xgb, "XGBOOST TEST PRECISION@K -- ranking DRIVES (each by its worst window)",
    KS)

imp_xgb = pd.DataFrame({"feature": feat_cols, "gain": xgb_clf.feature_importances_})
imp_xgb = imp_xgb.sort_values("gain", ascending=False).head(15).reset_index(drop=True)
imp_xgb.index += 1
print("\nwhat XGBoost relies on most (share of total gain):")
print(imp_xgb.to_string())
""")

code(r"""
# `pred` holds the CNN's score for exactly these tiling windows (§4). Same windows, same
# order, same labels -- asserted, not assumed.
assert len(pred) == len(yte)
assert np.array_equal(pred.drive_index.to_numpy(), g_te)
assert np.array_equal(pred.actual_fail.to_numpy(), yte.astype(int))
p_cnn = pred.risk.to_numpy()

cnn_test = bw.compute_metrics(yte, p_cnn, thr)
_, p_drv_cnn, _ = ap.drive_level_scores(p_cnn, yte, g_te)
cnn_pk_win = bw.precision_at_k_table(yte, p_cnn, KS)
cnn_pk_drv = bw.precision_at_k_table(y_drv, p_drv_cnn, KS)

comparison = pd.DataFrame([
    {"model": name, "PR-AUC": m["pr_auc"], "ROC-AUC": m["roc_auc"],
     "threshold": m["threshold"], "precision": m["precision"], "recall": m["recall"],
     "F1": m["f1"], "TP": m["tp"], "FP": m["fp"], "FN": m["fn"]}
    for name, m in (("CNN", cnn_test), ("XGBoost", xgb_test))
]).set_index("model")

print(f"CNN vs XGBOOST -- the same {len(yte):,} tiling test windows over "
      f"{len(y_drv):,} test drives")
print(f"(base rate {100*yte.mean():.3f}% of windows; each model at its own "
      f"validation-tuned threshold)\n")
print(comparison.to_string(float_format=lambda v: f"{v:.4f}"))

pk_compare = pd.DataFrame({
    "K": list(KS),
    "CNN drive hits": cnn_pk_drv.hits.to_numpy(),
    "XGBoost drive hits": xgb_pk_drv.hits.to_numpy(),
    "CNN drive precision": cnn_pk_drv.precision.to_numpy(),
    "XGBoost drive precision": xgb_pk_drv.precision.to_numpy(),
    "CNN window precision": cnn_pk_win.precision.to_numpy(),
    "XGBoost window precision": xgb_pk_win.precision.to_numpy(),
})
print(f"\nPrecision@K -- of the K drives (or windows) each model flags first, "
      f"how many really failed:\n")
print(pk_compare.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
""")

code(r"""
from sklearn.metrics import precision_recall_curve

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
for name, p, col in (("CNN", p_cnn, "#3b6ea5"), ("XGBoost", p_te, "#c1442e")):
    prec, rec, _ = precision_recall_curve(yte, p)
    ax.plot(rec, prec, color=col, lw=1.2,
            label=f"{name}  (PR-AUC {comparison.loc[name, 'PR-AUC']:.4f})")
ax.axhline(yte.mean(), color="0.5", ls="--", lw=0.8,
           label=f"random ({100*yte.mean():.2f}%)")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision-recall, test tiling windows")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(KS, pk_compare["CNN drive precision"], "o-", color="#3b6ea5", label="CNN")
ax.plot(KS, pk_compare["XGBoost drive precision"], "o-", color="#c1442e",
        label="XGBoost")
ax.set_xscale("log"); ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS])
ax.set_ylim(0, 1.02)
ax.set_xlabel("K drives flagged"); ax.set_ylabel("precision@K")
ax.set_title("Drives: of the top K flagged, the share that failed")
ax.legend(fontsize=8)
for a in axes:
    a.grid(alpha=0.25, linewidth=0.6); a.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/2022x_step5_cnn_vs_xgboost.png", dpi=130, bbox_inches="tight")
print(f"  saved {FIG_DIR}/2022x_step5_cnn_vs_xgboost.png")
plt.show()

del Xva, p_va; gc.collect()
""")

# ===========================================================================
md(r"""
---
## 6 — A decision tree grown until it is perfect on its training set

Fitted on the same flattened **tiling** table as XGBoost, so the tree sees every
training drive-day exactly once. This is the slowest section of the notebook — a single
scikit-learn tree is fitted on one CPU core, five times over — which is why it runs last.

The depth ladder below is the point of the section. A tree with no depth limit and
`min_samples_leaf=1` will reach **1.000 on the data it was fitted to** by carving a leaf
per sample; that is what "as big as needed for 100%" produces, and the test column beside
it is what it costs. Both are printed at every depth so the gap is visible rather than
asserted.

Two things to read carefully:

* **Accuracy is the wrong lens at this base rate.** Predicting "never fails" scores about
  99.9%. Balanced accuracy — the mean of per-class recall — is reported alongside, and it
  is the column that moves.
* **100% may be unreachable and that is informative.** If two windows have identical
  summary features but different labels, no tree can separate them. The cell reports how
  many such conflicts exist, which is the ceiling on training accuracy.
""")

code(r"""
# The exact ceiling: rows identical in every feature but disagreeing on the label can
# never be separated by any tree, however deep it is allowed to grow.
ceiling = ap.count_label_conflicts(Xtr, ytr)
""")

code(r"""
tree, ladder = ap.grow_tree_to_full_accuracy(
    Xtr, ytr, Xte, yte, depths=[4, 8, 16, 32], seed=SEED)
print()
print(ladder.to_string(index=False))
""")

code(r"""
final = ladder.iloc[-1]
print(f"FINAL TREE -- depth {final.actual_depth}, {final.leaves:,} leaves, "
      f"no depth limit, min_samples_leaf=1")
print(f"  training accuracy .... {final.train_accuracy:.8f}")
print(f"  test accuracy ........ {final.test_accuracy:.8f}")
print(f"  training balanced .... {final.train_balanced_acc:.4f}")
print(f"  test balanced ........ {final.test_balanced_acc:.4f}")
print()
if final.train_accuracy >= 1.0:
    print("  reached 100.000000% on the training set: one leaf per distinguishable row.")
else:
    missed = int(round((1 - final.train_accuracy) * len(ytr)))
    print(f"  it lands {missed:,} rows short of 100%, and those are the "
          f"{ceiling['unreachable_rows']:,} rows that are")
    print(f"  IDENTICAL across all {Xtr.shape[1]} features to a row carrying the "
          f"opposite label. No tree can")
    print(f"  split what it cannot tell apart, so "
          f"{100*ceiling['max_train_accuracy']:.6f}% is the ceiling -- the tree is")
    print(f"  already at it, not falling short of it.")
print(f"\n  the gap between the two ACCURACY columns is the tree memorising "
      f"{final.leaves:,} leaves of training rows;")
print(f"  the gap between the two BALANCED columns "
      f"({final.train_balanced_acc:.3f} vs {final.test_balanced_acc:.3f}) is what that "
      f"memorisation is")
print(f"  actually worth on a disk it has never seen.\n")

imp = pd.DataFrame({"feature": feat_cols, "importance": tree.feature_importances_})
imp = imp[imp.importance > 0].sort_values("importance", ascending=False).head(15)
imp.index = range(1, len(imp) + 1)
print("what the tree splits on most:")
print(imp.to_string())
""")

code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
ax = axes[0]
x = np.arange(len(ladder))
ax.plot(x, ladder.train_accuracy, "o-", color="#3b6ea5", label="train accuracy")
ax.plot(x, ladder.test_accuracy, "o-", color="#c1442e", label="test accuracy")
ax.set_xticks(x); ax.set_xticklabels(ladder.max_depth.astype(str))
ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy")
ax.set_title("Accuracy is the wrong lens here\n(both columns pinned near 1.0 by the base rate)")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(x, ladder.train_balanced_acc, "o-", color="#3b6ea5", label="train balanced acc")
ax.plot(x, ladder.test_balanced_acc, "o-", color="#c1442e", label="test balanced acc")
ax.set_xticks(x); ax.set_xticklabels(ladder.max_depth.astype(str))
ax.set_xlabel("max_depth"); ax.set_ylabel("balanced accuracy")
ax.set_title("Balanced accuracy: the gap that opens is the overfitting")
ax.legend(fontsize=8)
for a in axes:
    a.grid(alpha=0.25, linewidth=0.6); a.set_axisbelow(True)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/2022x_step6_tree_depth.png", dpi=130, bbox_inches="tight")
print(f"  saved {FIG_DIR}/2022x_step6_tree_depth.png")
plt.show()
""")

# ===========================================================================
code(r"""
os.makedirs("models", exist_ok=True); os.makedirs("results", exist_ok=True)
ap.save_artifacts(model, bundle, tcfg, results, prefix="models/backblaze_2022x_w30")
xgb_clf.save_model("models/backblaze_2022x_w30_xgb.json")
with open("models/backblaze_2022x_w30_xgb_config.json", "w", encoding="utf-8") as fh:
    json.dump({"feature_columns": feat_cols, "aggregates": ["last", "mean", "delta"],
               "window_days": cfg.window_days, "horizon_days": cfg.horizon_days,
               "threshold": thr_xgb, "best_iteration": int(xgb_clf.best_iteration),
               "params": {k: v for k, v in xgb_clf.get_params().items()
                          if isinstance(v, (int, float, str, bool)) or v is None}},
              fh, indent=2)
print("Saved models/backblaze_2022x_w30_xgb.json and its config")

comparison.to_csv("results/w2022x_cnn_vs_xgboost.csv")
pk_compare.to_csv("results/w2022x_cnn_vs_xgboost_precision_at_k.csv", index=False)
imp_xgb.to_csv("results/w2022x_xgboost_importance.csv")

pd.DataFrame(history).to_csv("results/w2022x_training_history.csv", index=False)
coverage.to_csv("results/w2022x_window_coverage.csv")
cov.to_csv("results/w2022x_per_drive_windows.csv", index=False)
ladder.to_csv("results/w2022x_tree_depth_ladder.csv", index=False)
results["top_drives"].to_csv("results/w2022x_top_drives.csv", index=False)
checks["channel_scores"].to_csv("results/w2022x_channel_label_auc.csv", index=False)
pred.head(2_000_000).to_csv("results/w2022x_window_verdicts.csv", index=False)
pd.concat([results["precision_at_k_windows"].assign(ranked="windows"),
           results["precision_at_k_drives"].assign(ranked="drives")],
          ignore_index=True).to_csv("results/w2022x_precision_at_k.csv", index=False)

with open("results/w2022x_summary.json", "w", encoding="utf-8") as fh:
    json.dump({
        "since_year": SINCE_YEAR,
        "shards": [os.path.basename(p)[:-8] for p in shards],
        "features": list(features),
        "blocklist": report.blocklist,
        "n_drives": int(store.n_drives),
        "n_drive_days": int(store.n_rows),
        "tiling_windows_total": int(total_tiles),
        "tiling_batches_total": int(total_batches),
        "leakage_checks_passed": bool(checks["x_independent_of_label"]),
        "max_channel_auc_vs_label": float(checks["max_channel_auc"]),
        "best_threshold": results["best_threshold"],
        "test_at_tuned": results["test_at_tuned"],
        "xgboost": {"best_iteration": int(xgb_clf.best_iteration),
                    "threshold": thr_xgb,
                    "val_at_tuned": xgb_val,
                    "test_at_tuned": xgb_test,
                    "test_at_half": xgb_half},
        "cnn_on_tiling_test": cnn_test,
        "tree_ceiling": {k: float(v) for k, v in ceiling.items()},
        "tree_final": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                       for k, v in ladder.iloc[-1].to_dict().items()},
    }, fh, indent=2)
print("wrote results/w2022x_*.csv and results/w2022x_summary.json")
""")

md(r"""
---
## Summary — CNN vs XGBoost vs decision tree

All three models on the **same 3.1M tiling test windows** of the same 82k test drives.
The CNN and XGBoost use the threshold each tuned on validation; the tree never saw
validation, so it is cut at 0.5.

The winner is named on two criteria, in order of importance:

1. **drive precision @100** — of the 100 drives a model flags first, the share that
   really failed. This is the decision an operator makes.
2. **PR-AUC** — the quality of the whole ranking.

When the top two sit within 5% of each other on a criterion, it is reported as a tie
rather than a win.
""")

code(r"""
p_tree = tree.predict_proba(Xte)[:, 1]
tree_test = bw.compute_metrics(yte, p_tree, 0.5)
_, p_drv_tree, _ = ap.drive_level_scores(p_tree, yte, g_te)
tree_pk_drv = bw.precision_at_k_table(y_drv, p_drv_tree, KS)

entries = [
    ("CNN", cnn_test, cnn_pk_drv),
    ("XGBoost", xgb_test, xgb_pk_drv),
    (f"Decision tree (depth {int(final.actual_depth)})", tree_test, tree_pk_drv),
]
rows = []
for name, m, pk in entries:
    pk = pk.set_index("k")
    rows.append({
        "model": name,
        "drive precision @100": pk.loc[100, "precision"],
        "drives hit @100": int(pk.loc[100, "hits"]),
        "drives hit @1000": int(pk.loc[1000, "hits"]),
        "PR-AUC": m["pr_auc"],
        "ROC-AUC": m["roc_auc"],
        "precision": m["precision"],
        "recall": m["recall"],
        "F1": m["f1"],
        "threshold": m["threshold"],
    })
summary = pd.DataFrame(rows).set_index("model")


def verdict(col):
    ranked = summary[col].sort_values(ascending=False)
    (first, a), (second, b) = list(ranked.items())[:2]
    if a <= 0 or (a - b) / a < 0.05:
        return f"{first} and {second} are tied ({a:.4f} vs {b:.4f})"
    return f"{first} ({a:.4f}, next best {second} at {b:.4f})"


rule = "=" * 88
print(rule)
print(f"SUMMARY -- {len(yte):,} test windows over {len(y_drv):,} test drives "
      f"({int(y_drv.sum()):,} failed; window base rate {100*yte.mean():.3f}%)")
print(rule)
print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
print()
print(f"  best at flagging drives (precision @100) ... {verdict('drive precision @100')}")
print(f"  best ranking overall (PR-AUC) ............. {verdict('PR-AUC')}")
print()
print(f"  for reference, not comparable to the table: the CNN on every stride-1 test "
      f"window (§4) scored")
print(f"  PR-AUC {results['test_at_tuned']['pr_auc']:.4f} over "
      f"{results['n_test_windows']:,} overlapping windows.")

summary.to_csv("results/w2022x_model_summary.csv")
print("\nwrote results/w2022x_model_summary.csv")
""")

md(r"""
---
## What these numbers do and do not say

**The label never reached the model.** §2 is an executed experiment, not a promise: the
labels were replaced with noise and the window tensors came back bit-identical. Together
with the audit's blocklist, that is the strongest statement available short of a proof —
the model's ranking comes from SMART telemetry and nothing else.

**The ranking transfers; the base rate is already real.** This is the fleet at its own
prevalence, not a matched cohort, so the precision figures need no discounting before
they are read. Read the **lift** column in Precision@K rather than raw precision, and
prefer the drive-level table — an operator pulls *K disks*, not K windows.

**Tiling and stride-1 answer different questions.** The tiling set judges every
drive-day exactly once, which is what makes the per-window verdict table and the tree
honest. The stride-1 set scores every window that exists, which is the right basis for a
ranking metric. Neither is a sample of the other.

**CNN vs XGBoost is a comparison of inputs as much as of models.** The CNN reads all
30 days of every channel in order; XGBoost reads three numbers per channel. If XGBoost
matches the CNN, most of the signal is in the level and trend of a few SMART counters
and the day-by-day shape adds little. If the CNN wins, the order of events inside the
window carries information the summaries throw away. Read the drive-level Precision@K
rows first — that is the decision an operator makes — and PR-AUC second.

**The 100% tree is a memorisation result, not a model.** It reaches perfect training
accuracy by growing a leaf per training row, and the test column next to it is the honest
read. At a 0.1% base rate even the test accuracy is near-perfect while the model is
nearly useless — which is exactly why every other number in this notebook is PR-AUC,
Precision@K or balanced accuracy rather than accuracy.

**2022 onward is not the whole archive.** Drive models, capacities and the SMART
attributes actually populated all shift over time; a model fitted on the last few years
is answering a narrower and more current question than one fitted on all thirteen.
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
