"""End-to-end run of the three-step analysis on the local Backblaze drive-stats export.

    python scripts/run_full_analysis.py                    # full run, ~35 min on CPU
    python scripts/run_full_analysis.py --epochs 3 --quick  # smoke test, ~2 min

Step 1  audit every column for target leakage, drop the proxies, keep what is left
Step 2  draw random continuous 90-day windows with unaligned starts, plot their timeline
Step 3  train a 1D-CNN over those windows with per-epoch logging, then score the test
        split at a threshold tuned on validation only

The data is `data/toshiba_drivestats.parquet` -- the Backblaze Drive Stats schema (197
columns, `serial_number` / `date` / `failure` / `smart_*`), filtered to the 14 Toshiba
families and exported as a matched cohort: every drive that failed plus a sample of
drives that did not. That cohort construction is itself a source of leakage, which is
what Step 1 spends most of its effort on.

Everything printed here comes from `scripts/leakage_audit.py` and
`scripts/backblaze_window_pipeline.py`; this file is the wiring, not the logic.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import leakage_audit as la
import backblaze_window_pipeline as bw

PARQUET = os.environ.get("DRIVESTATS_PARQUET",
                         os.path.join("data", "toshiba_drivestats.parquet"))
FIG_DIR = os.path.join("results", "figures")
RESULT_DIR = "results"
MODEL_DIR = "models"
PRECISION_AT_K = (10, 25, 50, 100)


# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--parquet", default=PARQUET)
    p.add_argument("--window-days", type=int, default=90)
    p.add_argument("--horizon-days", type=int, default=30,
                   help="a window is positive if the drive fails within this many days "
                        "of the window's last day; 0 = the failure must fall inside the "
                        "window itself")
    p.add_argument("--stride-days", type=int, default=14,
                   help="validation/test windows are enumerated every N days")
    p.add_argument("--samples-per-drive", type=int, default=6,
                   help="random training windows drawn per drive per epoch")
    p.add_argument("--imbalance", default="resample",
                   choices=["resample", "loss", "both", "none"],
                   help="resample = force a share of positive training windows with "
                        "plain BCE; loss = uniform windows with pos_weight. Correcting "
                        "twice ('both') is the mistake documented in TOSHIBA_PIPELINE.md")
    p.add_argument("--positive-ratio", type=float, default=0.25)
    p.add_argument("--arch", default="cnn", choices=["cnn", "gru", "transformer"])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--timeline-samples", type=int, default=6,
                   help="random windows drawn per drive for the Step 2 timeline plot")
    p.add_argument("--quick", action="store_true",
                   help="coarser eval stride and fewer training draws, for a smoke test")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-show", action="store_true",
                   help="save figures without calling plt.show() (scripts, CI)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------

def step1_leakage_audit(parquet: str, show: bool = True):
    """Score every column against `failure`, then hand back the clean feature set."""
    print(bw._RULE)
    print("LOADING")
    print(bw._RULE)
    t0 = time.time()
    df = pd.read_parquet(parquet)
    df["failure"] = df["failure"].fillna(0).astype("int8")
    print(f"  {parquet}")
    print(f"  {len(df):,} drive-days x {df.shape[1]} columns "
          f"({df.memory_usage(deep=False).sum() / 1e9:.1f} GB) in {time.time() - t0:.1f}s")
    print(f"  {df.serial_number.nunique():,} drives, {df.model.nunique()} families, "
          f"{df.date.min():%Y-%m-%d} -> {df.date.max():%Y-%m-%d}, "
          f"{int(df.failure.sum()):,} failure events")

    report = la.audit_frame(df)
    report.print_report()

    features = la.select_feature_columns(df, report)
    print()
    print(bw._RULE)
    print(f"FEATURE SET X AFTER DROPPING LEAKED PROXIES -- {len(features)} attributes")
    print(bw._RULE)
    tbl = report.columns.set_index("column")
    for c in features:
        r = tbl.loc[c]
        print(f"  {c:<18} coverage {100 * r.coverage:6.2f}%  "
              f"{int(r.n_unique):>6,} distinct  rank AUC {r.auc_directional:.4f}")
    report.assert_clean(features)

    la.plot_audit(report, save_path=os.path.join(FIG_DIR, "step1_leakage_audit.png"),
                  show=show)

    # Only the surviving columns go forward; the frame is renamed into the (id, time)
    # convention the window pipeline uses.
    raw = df[["serial_number", "date", "model", "failure"] + features].rename(
        columns={"serial_number": "id", "date": "time"})
    raw["time"] = pd.to_datetime(raw["time"])
    for c in features:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("float32")
    del df
    gc.collect()
    return raw, features, report


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------

def step2_windows(raw: pd.DataFrame, features: list[str], args, show: bool = True):
    """Build the window store and the three grouped splits; plot the window timeline."""
    cfg = bw.DataConfig(
        window_days=args.window_days,
        horizon_days=args.horizon_days,
        stride_days=args.stride_days,
        samples_per_drive=args.samples_per_drive,
        train_positive_ratio=(args.positive_ratio
                              if args.imbalance in ("resample", "both") else None),
        # Every window must be 90 genuinely continuous days of record, so drives shorter
        # than the window are dropped rather than left-padded.
        min_days=args.window_days,
        pad_short_drives=False,
        delta_lags=(1, 7),
        add_observed_mask=True,
        feature_columns=tuple(features),
        log1p_columns=tuple(c for c in features if c != "smart_194_raw"),
        random_state=args.seed,
    )

    print()
    print(bw._RULE)
    print(f"STEP 2 -- BUILDING {cfg.window_days}-DAY WINDOWS")
    print(bw._RULE)
    print(f"  label: positive if the drive fails within {cfg.horizon_days} days of the "
          f"window's last day")
    print(f"  eval windows enumerated every {cfg.stride_days} days; training draws "
          f"{cfg.samples_per_drive} random windows per drive per epoch")
    bundle = bw.build_window_bundle(raw, cfg)

    splits = bundle.splits
    store = bundle.store
    print()
    print("  grouped split by drive id (no drive appears in two splits):")
    for name in ("train", "val", "test"):
        idx = splits[name]
        n_fail = int(store.drive_has_failure[idx].sum())
        print(f"    {name:<5} {len(idx):>6,} drives  ({n_fail:,} with a failure event, "
              f"{100 * n_fail / max(len(idx), 1):.1f}%)")
    bw.inspect_split_integrity(bundle)

    # The Step 2 deliverable: where the random windows actually land in time.
    print()
    timelines = []
    for name in ("train", "val", "test"):
        tl = bw.sample_window_timeline(
            store, splits[name], n_per_drive=args.timeline_samples,
            seed=args.seed, split_name=name)
        timelines.append(tl)
    timeline = pd.concat(timelines, ignore_index=True)
    bw.describe_window_timeline(timeline, cfg)
    bw.plot_window_timeline(
        timeline, cfg, save_path=os.path.join(FIG_DIR, "step2_window_timeline.png"),
        show=show)
    return bundle, cfg, timeline


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------

def step3_train(bundle, cfg, args, show: bool = True):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bw.set_seed(args.seed)

    loss = {"resample": "bce", "loss": "bce_pos_weight",
            "both": "bce_pos_weight", "none": "bce"}[args.imbalance]
    tcfg = bw.TrainConfig(
        arch=args.arch, hidden_dim=args.hidden_dim, dropout=0.2,
        batch_size=args.batch_size, lr=args.lr, max_epochs=args.epochs,
        patience=args.patience, loss=loss, sampler="none", num_workers=0,
        seed=args.seed,
    )

    print()
    print(bw._RULE)
    print("STEP 3 -- TRAINING")
    print(bw._RULE)
    print(f"  device ............... {device}")
    print(f"  architecture ......... {tcfg.arch}")
    print(f"  input shape .......... (batch, {cfg.window_days}, {bundle.num_features}) "
          f"-- {cfg.window_days} days x {bundle.num_features} channels")
    print(f"  channels ............. {len(cfg.feature_columns)} SMART attributes, "
          f"their 1- and 7-day deltas, and an observed/forward-filled mask")
    print(f"  imbalance strategy ... {args.imbalance}  (loss={tcfg.loss}"
          + (f", positive_ratio={cfg.train_positive_ratio}"
             if cfg.train_positive_ratio else "") + ")")
    if args.imbalance == "both":
        print("  WARNING: correcting the imbalance twice -- see TOSHIBA_PIPELINE.md")

    loaders = bw.build_loaders(bundle, tcfg, device)
    model = bw.build_model(bundle.num_features, cfg.window_days, tcfg).to(device)
    with torch.no_grad():
        probe = torch.zeros(2, cfg.window_days, bundle.num_features, device=device)
        out_shape = tuple(model(probe).shape)
    print(f"  parameters ........... {bw.count_parameters(model):,}")
    print(f"  forward check ........ (2, {cfg.window_days}, {bundle.num_features}) -> "
          f"{out_shape}  (one logit per window)")
    print()

    model, history, best_val = bw.train_model(model, loaders, bundle, tcfg, device)
    print(f"\nBest validation PR-AUC: {best_val:.4f} "
          f"(epoch {1 + int(np.nanargmax(history['val_pr_auc']))})")
    bw.plot_training_history(
        history, save_path=os.path.join(FIG_DIR, "step3_training_history.png"), show=show)

    print()
    results = bw.evaluate_final(model, loaders, bundle, tcfg, device, pk_ks=PRECISION_AT_K)
    results["history"] = history
    bw.plot_evaluation(results, ks=PRECISION_AT_K,
                       save_path=os.path.join(FIG_DIR, "step3_evaluation.png"), show=show)
    return model, tcfg, results, history


# ---------------------------------------------------------------------------

def save_outputs(model, bundle, tcfg, results, history, timeline, report, features):
    prefix = os.path.join(MODEL_DIR, "backblaze_window90")
    bw.save_artifacts(model, bundle, tcfg, results, prefix=prefix)

    pd.DataFrame(history).to_csv(
        os.path.join(RESULT_DIR, "window90_training_history.csv"), index=False)
    report.columns.to_csv(
        os.path.join(RESULT_DIR, "window90_column_association.csv"), index=False)
    pd.concat([report.geometry, report.categoricals], ignore_index=True).to_csv(
        os.path.join(RESULT_DIR, "window90_leakage_probes.csv"), index=False)
    timeline.to_csv(
        os.path.join(RESULT_DIR, "window90_sampled_windows.csv"), index=False)
    if results.get("precision_at_k_windows") is not None:
        pk = results["precision_at_k_windows"].assign(ranked="windows")
        if results.get("precision_at_k_drives") is not None:
            pk = pd.concat([pk, results["precision_at_k_drives"].assign(ranked="drives")],
                           ignore_index=True)
        pk.to_csv(os.path.join(RESULT_DIR, "window90_precision_at_k.csv"), index=False)

    summary = {
        "features": features,
        "blocklist": report.blocklist,
        "best_threshold": results["best_threshold"],
        "val_at_tuned": {k: v for k, v in results["val"].items()},
        "test_at_tuned": {k: v for k, v in results["test_at_tuned"].items()},
        "test_at_half": {k: v for k, v in results["test_at_half"].items()},
    }
    with open(os.path.join(RESULT_DIR, "window90_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved metric tables and {os.path.join(RESULT_DIR, 'window90_summary.json')}")


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.stride_days = max(args.stride_days, 30)
        args.samples_per_drive = min(args.samples_per_drive, 3)
        args.timeline_samples = min(args.timeline_samples, 2)
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    show = not args.no_show

    t0 = time.time()
    raw, features, report = step1_leakage_audit(args.parquet, show=show)
    bundle, cfg, timeline = step2_windows(raw, features, args, show=show)
    del raw
    gc.collect()
    model, tcfg, results, history = step3_train(bundle, cfg, args, show=show)
    if not args.no_save:
        save_outputs(model, bundle, tcfg, results, history, timeline, report, features)
    print(f"\nTotal wall time: {time.time() - t0:.0f}s")
    return results


if __name__ == "__main__":
    main()
