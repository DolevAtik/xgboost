"""End-to-end run of the three-step analysis on the **whole Backblaze archive**.

    python scripts/build_backblaze_archive.py            # once: fetch + shard, ~35 GB
    python scripts/run_full_archive_analysis.py          # full run
    python scripts/run_full_archive_analysis.py --quick  # smoke test on 2 shards

Step 1  audit every column for target leakage, drop the proxies, keep what is left
Step 2  build the memory-mapped window store and enumerate **every possible 30-day
        window** for validation and test
Step 3  train a 1D-CNN over those windows with per-epoch logging, then score the test
        split at a threshold tuned on validation only

This is `Dataset_backblaze_Analysis.ipynb` as a plain script -- same modules, same
configuration, no plots shown. `scripts/run_full_analysis.py` is the older single-vendor
cohort run over `data/toshiba_drivestats.parquet`; this one has no vendor filter and no
cohort sampling, so the failure prevalence is the fleet's own.
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
import archive_window_pipeline as ap

FIG_DIR = os.path.join("results", "figures")
RESULT_DIR = "results"
MODEL_DIR = "models"
PRECISION_AT_K = (10, 25, 50, 100, 250, 1000)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--shard-dir",
                   default=os.environ.get("BACKBLAZE_SHARDS",
                                          os.path.join("data", "backblaze_archive", "shards")))
    p.add_argument("--store-dir",
                   default=os.environ.get("BACKBLAZE_STORE",
                                          os.path.join("data", "backblaze_archive", "store")))
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--horizon-days", type=int, default=30)
    p.add_argument("--stride-days", type=int, default=1,
                   help="1 = every possible window start is evaluated")
    p.add_argument("--val-stride", type=int, default=7,
                   help="stride for the per-epoch early-stopping signal only; the "
                        "reported evaluation always uses --stride-days")
    p.add_argument("--samples-per-drive", type=int, default=4)
    p.add_argument("--positive-ratio", type=float, default=0.25)
    p.add_argument("--audit-drives", type=int, default=15_000)
    p.add_argument("--n-buckets", type=int, default=64)
    p.add_argument("--arch", default="cnn", choices=["cnn", "gru", "transformer"])
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--quick", action="store_true",
                   help="coarser eval stride, fewer epochs, smaller audit sample")
    p.add_argument("--no-save", action="store_true")
    return p.parse_args(argv)


def step1(args):
    """Score every column against `failure`, then hand back the clean feature set."""
    print(bw._RULE); print("STEP 1 -- TARGET LEAKAGE AUDIT"); print(bw._RULE)
    cfg0 = ap.ArchiveConfig(shard_dir=args.shard_dir, store_dir=args.store_dir,
                            random_state=args.seed)
    census = ap.build_drive_census(cfg0)
    epoch = pd.Timestamp("1970-01-01")
    print(f"\n  {len(census):,} drives | {int(census.n_obs.sum()):,} observed drive-days "
          f"| {int(census.has_failure.sum()):,} with a failure event")
    print(f"  {epoch + pd.Timedelta(days=int(census.start_day.min())):%Y-%m-%d} -> "
          f"{epoch + pd.Timedelta(days=int(census.end_day.max())):%Y-%m-%d}")

    # Sampled by drive, not by drive-day: the geometry probes are derived from each
    # drive's own first and last day, and thinned rows would have neither.
    audit_df = ap.sample_audit_frame(cfg0, census, n_drives=args.audit_drives)
    report = la.audit_frame(audit_df)
    report.print_report()

    features = la.select_feature_columns(audit_df, report)
    print()
    print(bw._RULE)
    print(f"FEATURE SET X AFTER DROPPING LEAKED PROXIES -- {len(features)} attributes")
    print(bw._RULE)
    tbl = report.columns.set_index("column")
    for c in features:
        r = tbl.loc[c]
        print(f"  {c:<18} coverage {100 * r.coverage:6.2f}%  "
              f"{int(r.n_unique):>9,} distinct  rank AUC {r.auc_directional:.4f}")
    report.assert_clean(features)
    la.plot_audit(report, save_path=os.path.join(FIG_DIR, "step1_leakage_audit.png"),
                  show=False)

    del audit_df
    gc.collect()
    return census, features, report


def step2(args, features):
    """Build the memory-mapped store and enumerate every possible window."""
    cfg = ap.ArchiveConfig(
        shard_dir=args.shard_dir, store_dir=args.store_dir,
        window_days=args.window_days, horizon_days=args.horizon_days,
        stride_days=args.stride_days, min_days=args.window_days,
        delta_lags=(1, 7), add_observed_mask=True,
        feature_columns=tuple(features),
        log1p_columns=tuple(c for c in features
                            if c not in ("smart_194_raw", "smart_190_raw")),
        n_buckets=args.n_buckets,
        samples_per_drive=args.samples_per_drive,
        train_positive_ratio=args.positive_ratio,
        batch_size=args.batch_size, random_state=args.seed,
    )
    print()
    print(bw._RULE)
    print(f"STEP 2 -- EVERY POSSIBLE {cfg.window_days}-DAY WINDOW")
    print(bw._RULE)
    print(f"  label: positive if the drive fails within {cfg.horizon_days} days of the "
          f"window's last day")
    print(f"  eval windows enumerated at stride {cfg.stride_days}; training draws "
          f"{cfg.samples_per_drive} random windows per drive per epoch")

    store = ap.build_archive_store(cfg)
    bundle = ap.build_archive_bundle(cfg, store=store, val_stride=args.val_stride)

    rows = []
    for name in ("train", "val", "test"):
        ds = bundle.datasets[name]
        idx = bundle.splits[name]
        rows.append({
            "split": name,
            "drives": len(idx),
            "drives with a failure": int(store.drive_has_failure[idx].sum()),
            "dense drive-days": int(store.lengths[idx].sum()),
            "every possible window": int(
                (store.lengths[idx] - cfg.window_days + 1).sum()),
            "windows this split uses": int(
                ds.n_windows if hasattr(ds, "n_windows") else ds.n_samples),
            "batches": len(ds),
        })
    coverage = pd.DataFrame(rows).set_index("split")
    print()
    print(coverage.to_string())

    test_ds = bundle.datasets["test"]
    expected = int((store.lengths[bundle.splits["test"]] - cfg.window_days + 1).sum())
    if cfg.stride_days == 1:
        # The guarantee only means anything at stride 1; --quick deliberately coarsens it.
        assert test_ds.n_windows == expected, (test_ds.n_windows, expected)
        print(f"\n  test split enumerates every one of {test_ds.n_windows:,} possible "
              f"windows in {len(test_ds):,} batches of {cfg.batch_size}")
    else:
        print(f"\n  test split enumerates {test_ds.n_windows:,} of {expected:,} "
              f"possible windows (stride {cfg.stride_days}) in {len(test_ds):,} batches")

    timeline = ap.sample_window_timeline(store, bundle.splits, n_per_drive=2,
                                         max_drives=40_000, seed=args.seed)
    print()
    ap.describe_window_timeline(timeline, cfg)
    ap.plot_window_timeline(
        timeline, cfg, coverage=coverage,
        save_path=os.path.join(FIG_DIR, "step2_window_timeline.png"), show=False)
    return cfg, bundle, timeline, coverage


def step3(args, cfg, bundle):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    bw.set_seed(args.seed)
    tcfg = bw.TrainConfig(
        arch=args.arch, hidden_dim=args.hidden_dim, dropout=0.2,
        batch_size=cfg.batch_size, lr=args.lr, max_epochs=args.epochs,
        patience=args.patience, loss="bce", sampler="none", num_workers=0,
        seed=args.seed,
    )
    print()
    print(bw._RULE); print("STEP 3 -- TRAINING"); print(bw._RULE)
    print(f"  device ............... {device}")
    print(f"  input shape .......... (batch, {cfg.window_days}, {bundle.num_features})")
    print(f"  channels ............. {len(cfg.feature_columns)} SMART attributes, their "
          f"1- and 7-day deltas, and an observed/forward-filled mask")
    print(f"  imbalance strategy ... sampler only (positive_ratio="
          f"{cfg.train_positive_ratio}, loss=bce)")

    loaders = ap.build_loaders(bundle, tcfg, device)
    model = bw.build_model(bundle.num_features, cfg.window_days, tcfg).to(device)
    print(f"  parameters ........... {bw.count_parameters(model):,}")
    print()

    model, history, best_val = ap.train_archive_model(model, loaders, bundle, tcfg, device)
    print(f"\nBest validation PR-AUC: {best_val:.4f} "
          f"(epoch {1 + int(np.nanargmax(history['val_pr_auc']))})")
    bw.plot_training_history(
        history, save_path=os.path.join(FIG_DIR, "step3_training_history.png"), show=False)

    print()
    results = ap.evaluate_archive_final(model, loaders, bundle, tcfg, device,
                                        pk_ks=PRECISION_AT_K)
    results["history"] = history
    ap.plot_evaluation(results, ks=PRECISION_AT_K,
                       save_path=os.path.join(FIG_DIR, "step3_evaluation.png"), show=False)
    return model, tcfg, results, history


def save_outputs(model, bundle, tcfg, results, history, timeline, coverage,
                 report, features):
    ap.save_artifacts(model, bundle, tcfg, results,
                      prefix=os.path.join(MODEL_DIR, "backblaze_archive30"))
    pd.DataFrame(history).to_csv(
        os.path.join(RESULT_DIR, "archive30_training_history.csv"), index=False)
    report.columns.to_csv(
        os.path.join(RESULT_DIR, "archive30_column_association.csv"), index=False)
    pd.concat([report.geometry, report.categoricals], ignore_index=True).to_csv(
        os.path.join(RESULT_DIR, "archive30_leakage_probes.csv"), index=False)
    timeline.to_csv(os.path.join(RESULT_DIR, "archive30_sampled_windows.csv"), index=False)
    coverage.to_csv(os.path.join(RESULT_DIR, "archive30_window_coverage.csv"))
    results["top_drives"].to_csv(
        os.path.join(RESULT_DIR, "archive30_top_drives.csv"), index=False)
    pd.concat([results["precision_at_k_windows"].assign(ranked="windows"),
               results["precision_at_k_drives"].assign(ranked="drives")],
              ignore_index=True).to_csv(
        os.path.join(RESULT_DIR, "archive30_precision_at_k.csv"), index=False)

    summary = {
        "features": list(features),
        "blocklist": report.blocklist,
        "n_drives": int(bundle.store.n_drives),
        "n_drive_days": int(bundle.store.n_rows),
        "n_val_windows": results["n_val_windows"],
        "n_test_windows": results["n_test_windows"],
        "best_threshold": results["best_threshold"],
        "val_at_tuned": results["val"],
        "test_at_tuned": results["test_at_tuned"],
        "test_at_half": results["test_at_half"],
    }
    with open(os.path.join(RESULT_DIR, "archive30_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved metric tables and "
          f"{os.path.join(RESULT_DIR, 'archive30_summary.json')}")


def main(argv=None):
    args = parse_args(argv)
    if args.quick:
        args.stride_days = max(args.stride_days, 15)
        args.val_stride = max(args.val_stride, 30)
        args.epochs = min(args.epochs, 3)
        args.audit_drives = min(args.audit_drives, 4_000)
        args.samples_per_drive = min(args.samples_per_drive, 2)
    for d in (FIG_DIR, RESULT_DIR, MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    census, features, report = step1(args)
    cfg, bundle, timeline, coverage = step2(args, features)
    model, tcfg, results, history = step3(args, cfg, bundle)
    if not args.no_save:
        save_outputs(model, bundle, tcfg, results, history, timeline, coverage,
                     report, features)
    print(f"\nTotal wall time: {(time.time() - t0) / 60:.1f} min")
    return results


if __name__ == "__main__":
    main()
