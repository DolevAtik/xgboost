"""Round 2: every network trained on XGBoost's data, then added to the comparison.

    python scripts/run_tiling_experiments.py                  # train, score, Excel
    python scripts/run_tiling_experiments.py --skip-train     # score + Excel only
    python scripts/run_tiling_experiments.py --only tf_channel_tiling

The first round trained every CNN and transformer on `RandomWindows` -- 4 draws per drive
per epoch, 1.1M windows -- while XGBoost was fitted on all 10.5M tiling windows. These runs
(`model_comparison.TILING_RUNS`) train on that same full set (`ShuffledTilingWindows`):

* `tf_channel_tiling` -- one token per SMART channel (`ChannelTokenTransformer`)
* `tf_day_tiling`     -- the best round-1 transformer (day tokens, width 64)
* `cnn_tiling`        -- the best round-1 CNN (`lr_3e-03`), the control

Everything else is the round-1 protocol: the same stride-30 validation monitor and train
probe, early stopping on validation PR-AUC (patience 3), the overfitting rule, seed 42,
logged to `results/experiments_results.xlsx`. Each new checkpoint is then scored on the
tiling validation and test windows, merged into `results/w2022x_tiling_comparison.csv`,
and `results/model_comparison.xlsx` is rebuilt.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import archive_window_pipeline as ap
import backblaze_window_pipeline as bw
import model_comparison as mc
import overfit_experiments as ox

SEED = 42


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--store-dir", default=os.path.join("data", "backblaze_archive",
                                                       "store_2022"))
    p.add_argument("--log", default=os.path.join("results", "experiments_results.xlsx"))
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--only", nargs="*")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args(argv)

    t_all = time.time()
    c = json.load(open(os.path.join(args.store_dir, "config.json"), encoding="utf-8"))
    c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}
    cfg = ap.ArchiveConfig(**c)
    store = ap.build_archive_store(cfg, verbose=False)
    bundle = ap.build_archive_bundle(cfg, store=store, val_stride=30, verbose=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs = {fam: [v for v in vs if not args.only or v["name"] in args.only]
            for fam, vs in mc.TILING_RUNS.items()}

    log = ox.ExperimentLog(args.log, resume=True)
    if not args.skip_train:
        val_monitor = ap.EnumeratedWindows(store, bundle.splits["val"],
                                           batch_size=cfg.batch_size, stride=30,
                                           split_name="val")
        probe = ox.make_train_probe(store, bundle.splits["train"], like=val_monitor,
                                    seed=SEED)
        test_monitor = ap.EnumeratedWindows(store, bundle.splits["test"],
                                            batch_size=cfg.batch_size, stride=30,
                                            split_name="test")
        rule = ox.OverfitRule(pr_auc_gap_pct=15.0, roc_auc_gap_pct=3.0,
                              bal_acc_gap_pct=10.0, loss_gap_pct=25.0, min_consecutive=2,
                              stop_on_overfit=True)
        base = bw.TrainConfig(batch_size=cfg.batch_size, max_epochs=args.max_epochs,
                              patience=3, loss="bce", sampler="none", num_workers=0,
                              seed=SEED)
        for fam, variations in runs.items():
            if variations:
                ox.run_variations(
                    variations, base, bundle, val_monitor, probe, test_monitor, device,
                    log, rule=rule,
                    es_kwargs={"monitor": "val_pr_auc", "mode": "max", "min_delta": 1e-3,
                               "restore_best_weights": True},
                    train_positive_ratio=cfg.train_positive_ratio, seed=SEED,
                    fig_dir="results/figures", model_type=fam, train_stream="tiling")
        del val_monitor, probe, test_monitor
        log = ox.ExperimentLog(args.log, resume=True)

    tiles = {n: ap.TilingWindows(store, bundle.splits[n], batch_size=cfg.batch_size,
                                 split_name=n) for n in ("val", "test")}
    print(f"\n{bw._RULE}\nSCORING ON THE TILING WINDOWS\n{bw._RULE}")
    rows = []
    for fam, variations in runs.items():
        for v in variations:
            if not log.has(v["name"]):
                print(f"  {v['name']}: not in the log -- skipped")
                continue
            ck = log.config_of(v["name"])["checkpoint"]
            model, tcfg = mc.load_network(ck, bundle.num_features, cfg.window_days, device)
            rows.append(mc.score_network(
                model, v["name"], fam, tiles, device,
                note=f"{tcfg.arch}, {bw.count_parameters(model):,} parameters; trained on "
                     f"all tiling windows ({tcfg.optimizer}, {tcfg.schedule})"))
            del model
            torch.cuda.empty_cache()
    if rows:
        table, _ = mc.merge_into_csv(rows)
        print(table[["rank", "model", "family", "family_winner", "drive_P@100",
                     "test_pr_auc", "val_pr_auc", "precision", "recall", "f1"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    mc.rebuild_from_csv(exp_log_path=args.log)
    print(f"total {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
