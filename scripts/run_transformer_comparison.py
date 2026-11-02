"""Train the transformer variations, then score every model on the same windows.

    python scripts/run_transformer_comparison.py                # train + compare + Excel
    python scripts/run_transformer_comparison.py --skip-train   # compare + Excel only
    python scripts/run_transformer_comparison.py --tree         # also the depth-8 tree
    python scripts/run_transformer_comparison.py --full-test    # + winner on stride-1 test

The same steps as section 5b and the summary of
`Dataset_backblaze_Analysis_2022_XGBoost.ipynb`, as a plain script, so they can run
without re-running the CNN, its sweep and XGBoost. It needs what that notebook already
left on disk: the cached `store_2022`, the CNN checkpoints in `models/experiments/`
(`baseline.pt` is the CNN selected in §4) and XGBoost (`models/backblaze_2022x_w30_xgb.json`).

1. **Train** each transformer variation (`model_comparison.transformer_variations`)
   under the CNN's exact protocol -- same training stream, monitors, early stopping,
   overfitting rule and seed -- and log it to `results/experiments_results.xlsx` next
   to the CNN runs. Variations already in the log are skipped, so it resumes.
2. **Compare** on the 3.09M tiling test windows: every CNN variation,
   every transformer, XGBoost (and with `--tree`, a depth-8 decision tree). Each model's
   threshold is tuned on the tiling validation windows.
3. **Write** `results/w2022x_tiling_comparison.csv`,
   `results/w2022x_tiling_precision_at_k.csv` and `results/model_comparison.xlsx`.
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
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive_window_pipeline as ap
import backblaze_window_pipeline as bw
import model_comparison as mc
import overfit_experiments as ox

SEED = 42
MONITOR_STRIDE = 30


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--store-dir", default=os.path.join("data", "backblaze_archive",
                                                       "store_2022"))
    p.add_argument("--log", default=os.path.join("results", "experiments_results.xlsx"))
    p.add_argument("--out", default=os.path.join("results", "model_comparison.xlsx"))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--only", nargs="*", help="train only these variation names")
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--full-test", action="store_true",
                   help="score the transformer winner on every stride-1 test window")
    p.add_argument("--tree", action="store_true",
                   help="fit the depth-8 decision tree and add it to the comparison")
    args = p.parse_args(argv)

    t_all = time.time()
    c = json.load(open(os.path.join(args.store_dir, "config.json"), encoding="utf-8"))
    c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}
    cfg = ap.ArchiveConfig(**c)
    store = ap.build_archive_store(cfg, verbose=False)
    bundle = ap.build_archive_bundle(cfg, store=store, val_stride=MONITOR_STRIDE,
                                     verbose=False)
    print(store.summary())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tiles = {n: ap.TilingWindows(store, bundle.splits[n], batch_size=cfg.batch_size,
                                 split_name=n) for n in ("val", "test")}

    # -- 1. train ---------------------------------------------------------------
    log = ox.ExperimentLog(args.log, resume=True)
    if not args.skip_train:
        # Monitors, rule and early stopping exactly as the notebook builds them for the CNN.
        val_monitor = ap.EnumeratedWindows(store, bundle.splits["val"],
                                           batch_size=cfg.batch_size,
                                           stride=MONITOR_STRIDE, split_name="val")
        probe = ox.make_train_probe(store, bundle.splits["train"], like=val_monitor,
                                    seed=SEED)
        test_monitor = ap.EnumeratedWindows(store, bundle.splits["test"],
                                            batch_size=cfg.batch_size,
                                            stride=MONITOR_STRIDE, split_name="test")
        rule = ox.OverfitRule(pr_auc_gap_pct=15.0, roc_auc_gap_pct=3.0,
                              bal_acc_gap_pct=10.0, loss_gap_pct=25.0, min_consecutive=2,
                              stop_on_overfit=True)
        base = bw.TrainConfig(batch_size=cfg.batch_size, max_epochs=args.max_epochs,
                              patience=3, loss="bce", sampler="none", num_workers=0,
                              seed=SEED, **mc.TRANSFORMER_BASE)
        variations = mc.transformer_variations()
        if args.only:
            variations = [v for v in variations if v["name"] in args.only]
        ox.run_experiments(
            variations, base, bundle, val_monitor, probe, test_monitor, device, log,
            test_full_ds=bundle.datasets["test"] if args.full_test else None,
            select_by="val_pr_auc", rule=rule,
            es_kwargs={"monitor": "val_pr_auc", "mode": "max", "min_delta": 1e-3,
                       "restore_best_weights": True},
            train_positive_ratio=cfg.train_positive_ratio, seed=SEED,
            fig_dir="results/figures", model_type="transformer")
        del val_monitor, probe, test_monitor
        log = ox.ExperimentLog(args.log, resume=True)

    # -- 2. compare on the tiling windows ----------------------------------------------
    print(f"\n{bw._RULE}\nHEAD TO HEAD -- {tiles['test'].n_windows:,} tiling test windows, "
          f"thresholds from {tiles['val'].n_windows:,} tiling validation windows\n{bw._RULE}")
    rows = []
    sel = "models/backblaze_2022x_w30"
    cfgs = log.frames["Configs"]
    for _, cr in cfgs.iterrows():
        fam, ck = cr.get("model_type"), cr.get("checkpoint")
        if fam not in ("cnn", "transformer") or not isinstance(ck, str):
            continue
        if not os.path.exists(ck):
            print(f"  {cr['variation']}: checkpoint {ck} missing -- skipped")
            continue
        model, tcfg = mc.load_network(ck, bundle.num_features, cfg.window_days, device)
        note = f"{tcfg.arch}, {bw.count_parameters(model):,} parameters"
        if cr["variation"] == "baseline":
            # The notebook's §4 CNN: the same weights as models/backblaze_2022x_w30_cnn.pth.
            note += "; the CNN selected in notebook §4 (the search winner)"
        rows.append(mc.score_network(model, cr["variation"], fam, tiles, device, note=note))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    xgb_path = f"{sel}_xgb.json"
    Xtr = ytr = None
    if os.path.exists(xgb_path):
        t0 = time.time()
        Xva, yva, _ = ap.window_table_features(store, tiles["val"], verbose=False)
        Xte, yte, _ = ap.window_table_features(store, tiles["test"], verbose=False)
        clf = xgb.XGBClassifier()
        clf.load_model(xgb_path)
        clf.set_params(device="cpu")
        p_va, p_te = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
        codes, _ = tiles["test"].locate(np.arange(tiles["test"].n_windows, dtype=np.int64))
        rows.append(mc.score_probs("XGBoost", "xgboost", p_va, yva, p_te, yte, codes,
                                   note=f"{clf.get_booster().num_boosted_rounds()} rounds, "
                                        f"max_depth 8, 138 window-summary features"))
        print(f"  {'XGBoost':<28} {'xgboost':<12} val PR-AUC {rows[-1]['val_pr_auc']:.4f}  "
              f"test PR-AUC {rows[-1]['test_pr_auc']:.4f}  drive P@100 "
              f"{rows[-1]['drive_P@100']:.2f}  ({time.time() - t0:.0f}s)")
        if args.tree:
            from sklearn.tree import DecisionTreeClassifier
            train_tiles = ap.TilingWindows(store, bundle.splits["train"],
                                           batch_size=cfg.batch_size, split_name="train")
            Xtr, ytr, _ = ap.window_table_features(store, train_tiles, verbose=True)
            t0 = time.time()
            tree = DecisionTreeClassifier(max_depth=8, random_state=SEED).fit(Xtr, ytr)
            rows.append(mc.score_probs(
                "Decision tree (depth 8)", "tree", None, None,
                tree.predict_proba(Xte)[:, 1], yte, codes, threshold=0.5,
                note="one tree, max_depth 8, cut at 0.5 (never saw validation)"))
            print(f"  depth-8 tree fitted in {time.time() - t0:.0f}s, test PR-AUC "
                  f"{rows[-1]['test_pr_auc']:.4f}")
            del Xtr, ytr, train_tiles
        del Xva, Xte, p_va, p_te
        gc.collect()

    table, pk = mc.comparison_table(rows)
    os.makedirs("results", exist_ok=True)
    table.to_csv("results/w2022x_tiling_comparison.csv", index=False)
    pk.to_csv("results/w2022x_tiling_precision_at_k.csv", index=False)
    show = ["rank", "model", "family", "family_winner", "drive_P@100", "drive_P@1000",
            "test_pr_auc", "test_roc_auc", "val_pr_auc", "f1"]
    print()
    print(table[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    win = mc.family_winners(table)
    print(f"\n  best at flagging drives (precision @100) ... "
          f"{mc.verdict(win, 'drive_P@100')}")
    print(f"  best ranking overall (PR-AUC) ............. {mc.verdict(win, 'test_pr_auc')}")

    # -- 3. the workbook ------------------------------------------------------------
    mc.write_comparison_workbook(table, pk, args.out, exp_log_path=args.log)
    print(f"\nwrote results/w2022x_tiling_comparison.csv, "
          f"results/w2022x_tiling_precision_at_k.csv and {args.out} "
          f"-- total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
