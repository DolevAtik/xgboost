"""Hyper-parameter / architecture search for the 30-day CNN on the 2022+ store.

    python scripts/run_cnn_search.py --trials 24 --epochs 3
    python scripts/run_cnn_search.py --mode grid --trials 40

Gives the model a list of options (`ap.CNN_SEARCH_SPACE`) and runs one short training
run per configuration, ranking them by validation PR-AUC on a fixed set of validation
windows. The number of CNN layers is one of the options: each `dilations` entry is one
temporal block, so the search adds and removes layers as well as resizing them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import archive_window_pipeline as ap


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--store-dir", default=os.path.join("data", "backblaze_archive",
                                                       "store_2022"))
    p.add_argument("--mode", default="random", choices=["random", "grid"])
    p.add_argument("--trials", type=int, default=24)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--val-stride", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=os.path.join("results", "w2022x_cnn_search.csv"))
    args = p.parse_args(argv)

    c = json.load(open(os.path.join(args.store_dir, "config.json"), encoding="utf-8"))
    c = {k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()}
    cfg = ap.ArchiveConfig(**c)
    store = ap.build_archive_store(cfg, verbose=False)
    splits = ap.grouped_drive_split(store, cfg)
    print(store.summary())

    t0 = time.time()
    table = ap.cnn_search(cfg, store, splits, mode=args.mode, n_trials=args.trials,
                          epochs=args.epochs, val_stride=args.val_stride,
                          seed=args.seed, out_csv=args.out)
    print()
    print(table.to_string())
    best = table.iloc[0]
    print(f"\nBEST: {int(best.layers)} layers (dilations {best.dilations}), "
          f"kernel {int(best.kernel_size)}, hidden {int(best.hidden_dim)}, "
          f"dropout {best.dropout}, lr {best.lr:g}, "
          f"weight_decay {best.weight_decay:g}, "
          f"positive_ratio {best.train_positive_ratio}")
    print(f"     val PR-AUC {best.val_pr_auc:.4f} "
          f"({best.parameters:,} parameters)")
    print(f"total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
