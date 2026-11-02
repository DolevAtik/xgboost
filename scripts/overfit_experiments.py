"""Overfitting diagnostics, early stopping, and logged architecture variations.

Drives the archive CNN of `archive_window_pipeline.py` (and, for comparison, XGBoost)
through one training protocol, so every variation is measured the same way:

    probe  = make_train_probe(store, splits["train"], like=val_monitor)
    run    = train_with_diagnostics(model, train_ds, val_monitor, probe, tcfg, device)
    test   = evaluate_on_test(run.model, tiles["test"], run.val_threshold, device)
    log.add(name, config, run.history, run.diagnostics, test)      # -> .xlsx

**Why a train probe.** The training stream is resampled -- `RandomWindows` forces a share
of draws onto failure regions, so it runs at ~1.5% positives against validation's ~0.2%
-- and its running loss is measured with dropout on. Neither number can be compared with
validation: the prevalence difference alone makes the training loss ~3x *higher* than the
validation loss. The probe is a fixed set of windows from *training* drives, enumerated
exactly like the validation monitor (same stride, same failing-drive share, natural
prevalence) and scored in eval mode. The difference between probe and validation is then
the generalisation gap and nothing else.

**The overfitting rule** (`OverfitRule`). Per epoch, gaps are signed so that positive
means "train looks better":

    pr_auc_gap_pct  = 100 * (train PR-AUC  - val PR-AUC)  / val PR-AUC
    roc_auc_gap_pct = 100 * (train ROC-AUC - val ROC-AUC) / val ROC-AUC
    bal_acc_gap_pct = 100 * (train bal.acc - val bal.acc) / val bal.acc
    loss_gap_pct    = 100 * (val loss      - train loss)  / train loss

An epoch is *flagged* when any gap exceeds its threshold. The *divergence epoch* is found
separately, from the trend rather than a threshold: the first epoch from which, for
`min_consecutive` epochs in a row, the train metric keeps improving while the validation
metric stays below its best and the gap widens.
"""
from __future__ import annotations

import copy
import dataclasses
import os
import re
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import archive_window_pipeline as ap
import backblaze_window_pipeline as bw

_RULE = bw._RULE

# Metrics compared between the train probe and validation, and which way is "better".
GAP_METRICS = {"pr_auc": True, "roc_auc": True, "bal_acc": True, "loss": False}
# Everything recorded per split, per epoch: the gap metrics plus the threshold-dependent
# ones, all at the validation-tuned threshold.
REPORT_METRICS = (*GAP_METRICS, "accuracy", "precision", "recall", "f1")


# ---------------------------------------------------------------------------
# 1. A train-side measurement that is comparable to validation
# ---------------------------------------------------------------------------

def make_train_probe(store, train_drives: np.ndarray, like: ap.EnumeratedWindows,
                     n_drives: int | None = None, seed: int = 42,
                     split_name: str = "trn*") -> ap.EnumeratedWindows:
    """Fixed natural-prevalence windows from training drives, enumerated like `like`.

    Drives are drawn so the probe carries the same share of failing drives as `like`
    (the validation monitor), and the same number of drives unless `n_drives` says
    otherwise. Same stride, same batch size: the two passes differ only in which drives
    the model was trained on.
    """
    rng = np.random.default_rng(seed)
    train_drives = np.asarray(train_drives, dtype=np.int64)
    n_drives = int(n_drives or len(like.drives))
    fail_share = float(store.drive_has_failure[like.drives].mean())

    is_fail = store.drive_has_failure[train_drives].astype(bool)
    pool_f, pool_h = train_drives[is_fail], train_drives[~is_fail]
    n_f = min(len(pool_f), int(round(n_drives * fail_share)))
    n_h = min(len(pool_h), n_drives - n_f)
    pick = np.concatenate([rng.choice(pool_f, n_f, replace=False),
                           rng.choice(pool_h, n_h, replace=False)])
    return ap.EnumeratedWindows(store, np.sort(pick), batch_size=like.batch_size,
                                stride=like.stride, split_name=split_name)


def balanced_accuracy(m: dict) -> float:
    """Mean of per-class recall, from the confusion counts of `bw.compute_metrics`."""
    tpr = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else float("nan")
    tnr = m["tn"] / (m["tn"] + m["fp"]) if (m["tn"] + m["fp"]) else float("nan")
    return 0.5 * (tpr + tnr)


def _loader(dataset, device) -> DataLoader:
    return DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0,
                      pin_memory=(device.type == "cuda"))


def score_split(model, loader, criterion, device, threshold: float | None = None,
                tag: str = "eval") -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Loss, PR-AUC, ROC-AUC and balanced accuracy on one split, in eval mode.

    With `threshold=None` the best-F1 cut on *this* split is used and returned -- only
    ever call it that way on validation. Returns (summary, probs, targets, drive codes).
    """
    loss, _, probs, targets, codes = ap.evaluate_archive_loader(
        model, loader, criterion, device, tag=tag)
    if threshold is None:
        threshold, _ = ap.sweep_threshold(targets, probs)
    m = bw.compute_metrics(targets, probs, threshold)
    m["loss"] = float(loss)
    m["bal_acc"] = balanced_accuracy(m)
    m["accuracy"] = (m["tp"] + m["tn"]) / m["n"] if m["n"] else float("nan")
    return m, probs, targets, codes


# ---------------------------------------------------------------------------
# 2. Early stopping and the overfitting rule
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop after `patience` epochs without a `min_delta` gain; keep the best weights.

    The best state is copied to host memory, so holding it costs no GPU memory and
    `restore()` puts it back on whatever device the model lives on.
    """

    def __init__(self, monitor: str = "val_pr_auc", mode: str = "max", patience: int = 3,
                 min_delta: float = 1e-3, restore_best_weights: bool = True,
                 restore_best: bool | None = None):
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        self.monitor, self.mode = monitor, mode
        self.patience, self.min_delta = int(patience), float(min_delta)
        # `restore_best` is the older spelling of the same switch.
        self.restore_best = restore_best_weights if restore_best is None else restore_best
        self.best = -np.inf if mode == "max" else np.inf
        self.best_epoch, self.bad_epochs, self.best_state = 0, 0, None

    def improved(self, value: float) -> bool:
        if not np.isfinite(value):
            return False
        if self.mode == "max":
            return value > self.best + self.min_delta
        return value < self.best - self.min_delta

    def step(self, value: float, model, epoch: int) -> bool:
        """Record one epoch. True means stop."""
        if self.improved(value):
            self.best, self.best_epoch, self.bad_epochs = float(value), int(epoch), 0
            self.best_state = {k: v.detach().to("cpu", copy=True)
                               for k, v in model.state_dict().items()}
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model) -> None:
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


@dataclass
class OverfitRule:
    """Thresholds, in percent, above which a train/validation gap counts as overfitting.

    Defaults are set above the epoch-to-epoch noise of this problem: validation PR-AUC
    near 0.15 moves by ~5% relative between neighbouring epochs on its own, so a PR-AUC
    gap has to clear 15% before it means anything. ROC-AUC sits near 0.88 and is far less
    noisy, so a 3% gap there is already substantial. Balanced accuracy also moves with
    the threshold, which makes it the noisiest of the four. `None` disables a metric.

    They assume thousands of positive windows per split; with a few hundred, every
    gap is mostly sampling noise and `train_with_diagnostics` says so.
    """

    pr_auc_gap_pct: float | None = 15.0
    roc_auc_gap_pct: float | None = 3.0
    bal_acc_gap_pct: float | None = 10.0
    loss_gap_pct: float | None = 25.0
    min_consecutive: int = 2            # epochs a trend must persist to count
    stop_on_overfit: bool = True        # stop once flagged this long with no new best

    def thresholds(self) -> dict[str, float]:
        return {m: getattr(self, f"{m}_gap_pct") for m in GAP_METRICS
                if getattr(self, f"{m}_gap_pct") is not None}


def gap_pct(train: float, val: float, higher_is_better: bool) -> float:
    """Signed relative gap, positive when train looks better than validation."""
    if not (np.isfinite(train) and np.isfinite(val)):
        return float("nan")
    if higher_is_better:
        return 100.0 * (train - val) / val if val else float("nan")
    return 100.0 * (val - train) / train if train else float("nan")


def gap_columns(history: pd.DataFrame, rule: OverfitRule) -> pd.DataFrame:
    """Add `<metric>_gap_pct`, `overfit_flag` and `overfit_reasons` to an epoch table."""
    h = history.copy()
    for m, hib in GAP_METRICS.items():
        if f"train_{m}" in h and f"val_{m}" in h:
            h[f"{m}_gap_pct"] = [gap_pct(t, v, hib)
                                 for t, v in zip(h[f"train_{m}"], h[f"val_{m}"])]
    limits = rule.thresholds()
    reasons = []
    for _, r in h.iterrows():
        hit = [f"{m} {r[f'{m}_gap_pct']:.1f}%>{lim:g}%" for m, lim in limits.items()
               if f"{m}_gap_pct" in r and np.isfinite(r[f"{m}_gap_pct"])
               and r[f"{m}_gap_pct"] > lim]
        reasons.append("; ".join(hit))
    h["overfit_flag"] = [bool(s) for s in reasons]
    h["overfit_reasons"] = reasons
    return h


def find_divergence(train, val, higher_is_better: bool = True,
                    min_consecutive: int = 2, epochs=None) -> int | None:
    """First epoch of a sustained train-up / val-down split, or None.

    Only epochs after validation's best one are candidates: if validation later sets a
    new best, whatever came before it was noise, not divergence. From there an epoch
    counts as diverging when, against the epoch before it, the train metric did not get
    worse and the gap grew. The run must last `min_consecutive` epochs; the returned
    epoch is where it started.
    """
    s = 1.0 if higher_is_better else -1.0
    t = s * np.asarray(train, dtype=float)
    v = s * np.asarray(val, dtype=float)
    epochs = np.arange(1, len(t) + 1) if epochs is None else np.asarray(epochs)
    if not np.isfinite(v).any():
        return None
    gap = t - v
    run = 0
    for e in range(int(np.nanargmax(v)) + 1, len(t)):
        diverging = (t[e] >= t[e - 1]) and (gap[e] > gap[e - 1])
        run = run + 1 if diverging else 0
        if run >= min_consecutive:
            return int(epochs[e - min_consecutive + 1])
    return None


def overfit_status(diag: dict) -> str:
    """One verdict from the diagnostics, keeping "a gap broke its limit" apart from
    "the trend detector saw train pull ahead while every gap stayed inside its limit"."""
    def has(v):
        return v is not None and not (isinstance(v, float) and np.isnan(v))
    if diag.get("flagged_at_best"):
        return "overfit at best epoch (gap over limit)"
    if has(diag.get("n_flagged_epochs")) and diag["n_flagged_epochs"] > 0:
        return "overfit after best epoch (gap over limit; best weights restored)"
    if has(diag.get("divergence_epoch")):
        return "early divergence after best epoch (gaps within limits)"
    return "no overfitting detected"


def diagnose(history: pd.DataFrame, best_epoch: int, rule: OverfitRule,
             stop_reason: str = "") -> dict:
    """Divergence point, first flagged epoch, and the gaps at the best and last epoch."""
    h = history.set_index("epoch")
    ep = h.index.to_numpy()
    div = {m: find_divergence(h[f"train_{m}"], h[f"val_{m}"], hib,
                              rule.min_consecutive, ep)
           for m, hib in GAP_METRICS.items()
           if f"train_{m}" in h and h[f"train_{m}"].notna().any()}
    # Only the metrics the rule watches decide the headline divergence epoch.
    found = {m: e for m, e in div.items() if e is not None and m in rule.thresholds()}
    first = min(found, key=found.get) if found else None
    flagged = h.index[h["overfit_flag"]].tolist()
    last = int(ep[-1])

    out = {
        "epochs_run": len(h),
        "best_epoch": int(best_epoch),
        "stopped_epoch": last,
        "stop_reason": stop_reason,
        "divergence_epoch": found[first] if first else None,
        "divergence_metric": first,
        **{f"divergence_epoch_{m}": e for m, e in div.items()},
        "first_flagged_epoch": int(flagged[0]) if flagged else None,
        "n_flagged_epochs": len(flagged),
        "flagged_at_best": bool(h.loc[best_epoch, "overfit_flag"]),
    }
    out["status"] = overfit_status(out)
    for when, e in (("best", best_epoch), ("stop", last)):
        for m in GAP_METRICS:
            col = f"{m}_gap_pct"
            if col in h:
                out[f"{col}_at_{when}"] = float(h.loc[e, col])
        out[f"flagged_at_{when}"] = bool(h.loc[e, "overfit_flag"])
        out[f"reasons_at_{when}"] = h.loc[e, "overfit_reasons"]
    return out


# ---------------------------------------------------------------------------
# 3. The training loop
# ---------------------------------------------------------------------------

@dataclass
class TrainRun:
    model: torch.nn.Module
    history: pd.DataFrame                # one row per epoch, train-probe vs val columns
    diagnostics: dict                    # divergence, flags, gaps at best/stop
    best_epoch: int
    val_threshold: float                 # best-F1 cut on validation at the best epoch
    seconds: float

    @property
    def history_dict(self) -> dict:
        """`history` as the dict of lists `bw.plot_training_history` expects."""
        return {c: self.history[c].tolist() for c in self.history.columns}


def train_with_diagnostics(model, train_ds, val_ds, probe_ds, tcfg, device,
                           early_stopping: EarlyStopping | None = None,
                           rule: OverfitRule | None = None, lr_patience: int = 1,
                           min_positives: int = 500, test_ds=None,
                           verbose: bool = True) -> TrainRun:
    """Train one model; after every epoch score the train probe and validation alike.

    The optimiser, loss and LR schedule are those of `ap.train_archive_model`. What is
    new: a train-probe pass in eval mode each epoch, the gap columns, the overfitting
    rule as an additional stopping condition, and best-weight restoration in memory.

    `test_ds` is **watch-only**: when given, it is scored every epoch at the validation
    threshold and recorded as `test_*` columns so the train-vs-test curve can be drawn
    (`plot_train_test_curve`). Nothing reads those columns while training -- stopping,
    the best epoch and the threshold all come from validation alone.
    """
    early_stopping = early_stopping or EarlyStopping(patience=tcfg.patience)
    rule = rule or OverfitRule()
    monitor = early_stopping.monitor
    criterion = bw.make_criterion(tcfg, 1.0, device, verbose=False)
    opt_cls = (torch.optim.AdamW if getattr(tcfg, "optimizer", "adam") == "adamw"
               else torch.optim.Adam)
    optimizer = opt_cls(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    train_loader = _loader(train_ds, device)
    grad_clip = float(getattr(tcfg, "grad_clip", 0.0) or 0.0)
    step_scheduler, scheduler = None, None
    if getattr(tcfg, "schedule", "plateau") == "warmup_cosine":
        # Per batch: linear warm-up, then cosine down to 5% of the peak by max_epochs.
        total = max(1, tcfg.max_epochs * len(train_loader))
        warm = max(1, int(tcfg.warmup_epochs * len(train_loader)))

        def lr_factor(step):
            if step < warm:
                return (step + 1) / warm
            t = min(1.0, (step - warm) / max(1, total - warm))
            return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * t))
        step_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=early_stopping.mode, factor=0.5, patience=lr_patience)
    val_loader, probe_loader = _loader(val_ds, device), _loader(probe_ds, device)
    test_loader = _loader(test_ds, device) if test_ds is not None else None

    if verbose:
        print(_RULE)
        print(f"TRAINING -- {bw.count_parameters(model):,} parameters, up to "
              f"{tcfg.max_epochs} epochs; early stop on {monitor} ({early_stopping.mode}), "
              f"patience {early_stopping.patience}, min_delta {early_stopping.min_delta:g}")
        print(f"  overfit rule: gap > {rule.thresholds()} (percent)"
              f"{', stops the run' if rule.stop_on_overfit else ''}")
        n_pos = {"train probe": int(probe_ds.labels.sum()), "val": int(val_ds.labels.sum())}
        print(f"  train probe {probe_ds.n_windows:,} windows / {len(probe_ds.drives):,} "
              f"drives / {n_pos['train probe']:,} positive  vs  val {val_ds.n_windows:,} "
              f"windows / {len(val_ds.drives):,} drives / {n_pos['val']:,} positive")
        thin = [k for k, v in n_pos.items() if v < min_positives]
        if thin:
            print(f"  WARNING: {' and '.join(thin)} hold fewer than {min_positives} "
                  f"positive windows -- the gaps below are mostly sampling noise")
        print(_RULE)
        print(f"{'ep':>3} | {'loss tr/val':>17} | {'PR-AUC tr/val':>15} | "
              f"{'ROC tr/val':>13} | {'balacc tr/val':>13} | {'PR gap':>7} | "
              f"{'sec':>5} |")

    rows, stop_reason, t_start = [], "max_epochs", time.time()
    flagged_run = 0
    for epoch in range(1, tcfg.max_epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        model.train()
        running, n_seen = 0.0, 0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if step_scheduler is not None:
                step_scheduler.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        # Validation picks the threshold; the train probe is judged at that same cut, so
        # balanced accuracy differs only through the model, never through the threshold.
        vm, _, _, _ = score_split(model, val_loader, criterion, device, tag="val")
        tm, _, _, _ = score_split(model, probe_loader, criterion, device,
                                  threshold=vm["threshold"], tag="probe")
        lr_now = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "train_loss_stream": running / max(n_seen, 1)}
        for m in REPORT_METRICS:
            row[f"train_{m}"], row[f"val_{m}"] = tm[m], vm[m]
        if test_loader is not None:
            sm, _, _, _ = score_split(model, test_loader, criterion, device,
                                      threshold=vm["threshold"], tag="test")
            for m in REPORT_METRICS:
                row[f"test_{m}"] = sm[m]
        row.update({"val_f1": vm["f1"], "val_threshold": vm["threshold"],
                    "train_base_rate": tm["base_rate"], "val_base_rate": vm["base_rate"],
                    "lr": lr_now, "epoch_seconds": time.time() - t0})
        rows.append(row)

        hist = gap_columns(pd.DataFrame(rows), rule)
        last = hist.iloc[-1]
        value = float(last[monitor])
        is_best = early_stopping.improved(value)
        stop = early_stopping.step(value, model, epoch)
        if scheduler is not None:
            scheduler.step(value if np.isfinite(value) else
                           (0.0 if early_stopping.mode == "max" else np.inf))
        flagged_run = flagged_run + 1 if (last["overfit_flag"] and not is_best) else 0

        if verbose:
            print(f"{epoch:>3} | {last.train_loss:>8.5f}/{last.val_loss:<8.5f} | "
                  f"{last.train_pr_auc:>7.4f}/{last.val_pr_auc:<7.4f} | "
                  f"{last.train_roc_auc:>6.4f}/{last.val_roc_auc:<6.4f} | "
                  f"{last.train_bal_acc:>6.4f}/{last.val_bal_acc:<6.4f} | "
                  f"{last.pr_auc_gap_pct:>+6.1f}% | {last.epoch_seconds:>5.0f} |"
                  + (f" test PR {last.test_pr_auc:.4f} |" if "test_pr_auc" in last else "")
                  + (" best" if is_best else "")
                  + (f" OVERFIT: {last.overfit_reasons}" if last.overfit_flag else ""),
                  flush=True)

        if stop:
            stop_reason = f"patience ({early_stopping.patience} epochs without a gain)"
            break
        if rule.stop_on_overfit and flagged_run >= rule.min_consecutive:
            stop_reason = (f"overfit rule ({flagged_run} flagged epochs since the "
                           f"best one)")
            break

    early_stopping.restore(model)
    best_epoch = early_stopping.best_epoch or int(hist["epoch"].iloc[-1])
    diag = diagnose(hist, best_epoch, rule, stop_reason)
    # The stopping settings this run used, so the log can say what stopped it and why.
    diag.update({
        "es_monitor": monitor, "es_patience": early_stopping.patience,
        "es_min_delta": early_stopping.min_delta,
        "es_restore_best_weights": bool(early_stopping.restore_best),
        "max_epochs": int(tcfg.max_epochs), "lr_patience": int(lr_patience),
        "rule_min_consecutive": rule.min_consecutive,
        "rule_stop_on_overfit": bool(rule.stop_on_overfit),
        **{f"rule_{m}_gap_pct": v for m, v in rule.thresholds().items()},
    })
    val_thr = float(hist.set_index("epoch").loc[best_epoch, "val_threshold"])
    if verbose:
        print(f"stopped at epoch {diag['stopped_epoch']}: {stop_reason}")
        print(f"best epoch {best_epoch} ({monitor} {early_stopping.best:.4f}) -- "
              f"weights restored; {diag['status']}")
        if diag["divergence_epoch"]:
            print(f"train and validation diverge from epoch {diag['divergence_epoch']}")
    return TrainRun(model=model, history=hist, diagnostics=diag, best_epoch=best_epoch,
                    val_threshold=val_thr, seconds=time.time() - t_start)


# ---------------------------------------------------------------------------
# 4. Test evaluation and plotting
# ---------------------------------------------------------------------------

def evaluate_on_test(model, test_ds, threshold: float, device,
                     ks=(10, 25, 50, 100, 250, 1000), probs: np.ndarray | None = None,
                     targets: np.ndarray | None = None,
                     codes: np.ndarray | None = None) -> dict:
    """Test PR-AUC, ROC-AUC, balanced accuracy and Precision@K at a frozen threshold.

    Pass `probs`/`targets`/`codes` to score predictions made elsewhere (XGBoost)
    instead of running `model` over `test_ds`.
    """
    loss = float("nan")
    if probs is None:
        crit = torch.nn.BCEWithLogitsLoss()
        loss, _, probs, targets, codes = ap.evaluate_archive_loader(
            model, _loader(test_ds, device), crit, device, tag="test")
    m = bw.compute_metrics(targets, probs, threshold)
    m["loss"] = float(loss)
    m["bal_acc"] = balanced_accuracy(m)
    m["accuracy"] = (m["tp"] + m["tn"]) / m["n"] if m["n"] else float("nan")
    pk_win = bw.precision_at_k_table(targets, probs, ks).assign(ranked="windows")
    tables = [pk_win]
    if codes is not None:
        _, p_drv, y_drv = ap.drive_level_scores(probs, targets, codes)
        tables.append(bw.precision_at_k_table(y_drv, p_drv, ks).assign(ranked="drives"))
    m["precision_at_k"] = pd.concat(tables, ignore_index=True)
    return m


def save_checkpoint(model, path: str, tcfg, train_positive_ratio: float) -> str:
    """The restored best weights plus the config needed to rebuild the network."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "tcfg": dataclasses.asdict(tcfg),
                "train_positive_ratio": float(train_positive_ratio)}, path)
    return path


def load_checkpoint(path: str, num_features: int, window_days: int, device):
    """Rebuild the model a checkpoint describes and load its weights, ready for eval."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    tcfg = bw.TrainConfig(**{**ck["tcfg"], "dilations": tuple(ck["tcfg"]["dilations"])})
    model = bw.build_model(num_features, window_days, tcfg)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), tcfg


def plot_divergence(history: pd.DataFrame, diagnostics: dict, rule: OverfitRule,
                    title: str = "", xlabel: str = "epoch", save_path: str | None = None,
                    show: bool = True):
    """Train-probe vs validation loss and PR-AUC, and the PR-AUC gap against its limit."""
    import matplotlib.pyplot as plt
    ep = history["epoch"]
    mk = "o-" if len(ep) <= 60 else "-"           # boosting rounds: lines, not dots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for ax, m, lab in ((axes[0], "loss", "loss"), (axes[1], "pr_auc", "PR-AUC")):
        ax.plot(ep, history[f"train_{m}"], mk, color="#3b6ea5", label="train probe")
        ax.plot(ep, history[f"val_{m}"], mk, color="#c1442e", label="validation")
        ax.set_title(f"{lab}: train probe vs validation"); ax.set_xlabel(xlabel)
    if "train_loss_stream" in history:
        axes[0].plot(ep, history["train_loss_stream"], ":", color="0.55",
                     label="training stream (resampled, dropout on)")
    ax = axes[2]
    ax.plot(ep, history["pr_auc_gap_pct"], mk, color="#7a4fa0", label="PR-AUC gap")
    if "loss_gap_pct" in history:
        ax.plot(ep, history["loss_gap_pct"], "--", color="#7a9b57", label="loss gap")
    if rule.pr_auc_gap_pct is not None:
        ax.axhline(rule.pr_auc_gap_pct, color="#7a4fa0", ls=":", lw=1,
                   label=f"PR-AUC limit {rule.pr_auc_gap_pct:g}%")
    if rule.loss_gap_pct is not None:
        ax.axhline(rule.loss_gap_pct, color="#7a9b57", ls=":", lw=1,
                   label=f"loss limit {rule.loss_gap_pct:g}%")
    ax.set_title("Gap, % (positive = train looks better)"); ax.set_xlabel(xlabel)
    for a in axes:
        a.axvline(diagnostics["best_epoch"], color="0.4", ls="--", lw=1,
                  label=f"best {xlabel} {diagnostics['best_epoch']}")
        if diagnostics.get("divergence_epoch"):
            a.axvline(diagnostics["divergence_epoch"], color="#d08a1e", ls="-.", lw=1,
                      label=f"divergence {diagnostics['divergence_epoch']} "
                            f"({diagnostics.get('divergence_metric')})")
        a.grid(alpha=0.25, linewidth=0.6); a.set_axisbelow(True)
        a.legend(fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_train_test_curve(history: pd.DataFrame, chosen: int, title: str = "",
                          x: str = "epoch", xlabel: str = "epoch", log_x: bool = False,
                          save_path: str | None = None, show: bool = True):
    """Train vs test, the depth-ladder chart's layout: accuracy | balanced accuracy | PR-AUC.

    Needs `train_*` and `test_*` columns for `accuracy`, `bal_acc` and `pr_auc` (and draws
    `val_*` dashed when present) -- what `train_with_diagnostics(..., test_ds=...)`
    records. `chosen` is the epoch (or round) the model was kept at.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    for ax, m, lab in zip(axes, ("accuracy", "bal_acc", "pr_auc"),
                          ("accuracy", "balanced accuracy", "PR-AUC")):
        ax.plot(history[x], history[f"train_{m}"], "o-", color="#3b6ea5",
                label=f"train {lab}")
        ax.plot(history[x], history[f"test_{m}"], "o-", color="#c1442e",
                label=f"test {lab}")
        if f"val_{m}" in history:
            ax.plot(history[x], history[f"val_{m}"], "--", color="0.6", lw=1,
                    label=f"validation {lab}")
        ax.axvline(chosen, color="0.35", ls=":", lw=1.2,
                   label=f"early stopping picks {chosen}")
        if log_x:
            ax.set_xscale("log")
        elif x == "epoch":
            ax.set_xticks(list(history[x]))
        ax.set_xlabel(xlabel); ax.set_ylabel(lab)
        ax.grid(alpha=0.25, linewidth=0.6); ax.set_axisbelow(True)
        ax.legend(fontsize=8)
    axes[0].set_title("Accuracy is the wrong lens here\n(both pinned near 1.0 by the base rate)")
    axes[1].set_title("Balanced accuracy\n(at the validation-tuned threshold)")
    axes[2].set_title("PR-AUC: the ranking the model is chosen on\n(threshold-free)")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_variation_gaps(log: "ExperimentLog", rule: OverfitRule, model_type: str = "cnn",
                        save_path: str | None = None, show: bool = True):
    """Every logged variation on one pair of axes: val PR-AUC, and the PR-AUC gap."""
    import matplotlib.pyplot as plt
    cfgs = log.frames["Configs"]
    names = cfgs.loc[cfgs["model_type"] == model_type, "variation"].tolist()
    ep = log.frames["Epoch History"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(names):
        h = ep[ep["variation"] == name]
        best = int(log.frames["Diagnostics"].set_index("variation").loc[name, "best_epoch"])
        for ax, col in ((axes[0], "val_pr_auc"), (axes[1], "pr_auc_gap_pct")):
            ax.plot(h["epoch"], h[col], "o-", ms=3, lw=1.1, color=cmap(i % 10), label=name)
            b = h[h["epoch"] == best]
            ax.plot(b["epoch"], b[col], "*", ms=11, color=cmap(i % 10))
    if rule.pr_auc_gap_pct is not None:
        axes[1].axhline(rule.pr_auc_gap_pct, color="0.4", ls=":", lw=1,
                        label=f"overfit limit {rule.pr_auc_gap_pct:g}%")
    axes[0].set_title("Validation PR-AUC by epoch (star = restored best epoch)")
    axes[1].set_title("Train-probe vs validation PR-AUC gap, %")
    for a in axes:
        a.set_xlabel("epoch"); a.grid(alpha=0.25, linewidth=0.6); a.set_axisbelow(True)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 5. Variations
# ---------------------------------------------------------------------------

def eval_protocol(val_ds, test_ds) -> str:
    """'train/val enumerated stride 30, test tiling' -- the label for the log."""
    def one(ds):
        if isinstance(ds, ap.EnumeratedWindows):
            return f"enumerated stride {ds.stride}"
        return "tiling" if isinstance(ds, ap.TilingWindows) else type(ds).__name__
    return f"train/val {one(val_ds)}, test {one(test_ds)}"


def default_variations(base: dict) -> list[dict]:
    """One knob at a time around `base`, so each row of the log isolates one change.

    `base` holds `dilations`, `kernel_size`, `hidden_dim`, `dropout` and friends -- the
    search winner. Variations that would equal the base are dropped.
    """
    d = tuple(base["dilations"])
    h = int(base["hidden_dim"])
    cands = [
        ("baseline", {}),
        ("shallow", {"dilations": d[:max(2, len(d) - 2)]}),
        ("deeper", {"dilations": d + (2 * d[-1],)}),
        ("no_dilation", {"dilations": (1,) * len(d)}),
        ("kernel_5" if base["kernel_size"] != 5 else "kernel_3",
         {"kernel_size": 5 if base["kernel_size"] != 5 else 3}),
        ("narrow", {"hidden_dim": max(16, h // 2)}),
        ("wide", {"hidden_dim": 2 * h}),
        ("dropout_0.3" if base["dropout"] < 0.3 else "dropout_0.1",
         {"dropout": 0.3 if base["dropout"] < 0.3 else 0.1}),
        ("weight_decay_1e-4" if base["weight_decay"] < 1e-4 else "weight_decay_1e-5",
         {"weight_decay": 1e-4 if base["weight_decay"] < 1e-4 else 1e-5}),
    ]
    if "lr" in base:                    # a third either side, the usual step for Adam
        lr = float(base["lr"])
        cands += [(f"lr_{lr * 3:.0e}", {"lr": lr * 3}), (f"lr_{lr / 3:.0e}", {"lr": lr / 3})]
    out, seen = [], set()
    for name, over in cands:
        cfg = {**base, **over}
        sig = tuple(sorted((k, str(v)) for k, v in cfg.items()))
        if sig in seen:
            continue
        seen.add(sig)
        out.append({"name": name, **over})
    return out


def variation_name(cfg: dict) -> str:
    """A readable, unique id for a grid point: 'L5_d1-2-4-8-16_k3_h128_do0.1_lr1e-03'."""
    parts = []
    if "dilations" in cfg:
        d = tuple(cfg["dilations"])
        parts += [f"L{len(d)}", "d" + "-".join(str(x) for x in d)]
    for key, tag, fmt in (("kernel_size", "k", "{}"), ("hidden_dim", "h", "{}"),
                          ("dropout", "do", "{:g}"), ("lr", "lr", "{:.0e}"),
                          ("weight_decay", "wd", "{:.0e}"),
                          ("train_positive_ratio", "pos", "{:g}")):
        if key in cfg:
            parts.append(tag + fmt.format(cfg[key]))
    return "_".join(parts)


def grid_variations(space: dict, mode: str = "grid", n_trials: int | None = None,
                    seed: int = 42) -> list[dict]:
    """Every combination of `space` (`mode="grid"`), or `n_trials` random draws from it.

    `space` maps TrainConfig fields (and `train_positive_ratio`) to the values to try,
    e.g. `{"dilations": [(1, 2, 4), (1, 2, 4, 8, 16)], "hidden_dim": [64, 128],
    "lr": [3e-4, 1e-3]}`. A grid multiplies fast -- that example is 8 runs, adding
    `dropout: [0.1, 0.3]` makes it 16 -- so `n_trials` caps it with a shuffled subset.
    """
    trials = ap.iter_search_configs(space, mode=mode, n_trials=n_trials or 0, seed=seed)
    if mode == "random" and n_trials:
        trials = trials[:n_trials]
    return [{"name": variation_name(t), **t} for t in trials]


def describe_config(tcfg, train_positive_ratio: float, model=None) -> dict:
    """The architecture/optimiser columns written to the log."""
    dil = tuple(tcfg.dilations)
    if tcfg.arch == "cnn":
        shape = {"layers": len(dil), "dilations": "-".join(str(x) for x in dil),
                 "kernel_size": int(tcfg.kernel_size),
                 "receptive_field_days": ap._receptive_field(dil, int(tcfg.kernel_size))}
    else:
        # Attention sees every day from the first layer: no dilations, no kernel, and
        # the receptive field is the window itself.
        shape = {"layers": int(getattr(tcfg, "n_layers", 0)), "dilations": None,
                 "kernel_size": None, "n_heads": int(getattr(tcfg, "n_heads", 0)),
                 "receptive_field_days": None}
    out = {
        "arch": tcfg.arch,
        **shape,
        "hidden_dim": int(tcfg.hidden_dim),
        "dropout": float(tcfg.dropout),
        "lr": float(tcfg.lr),
        "weight_decay": float(tcfg.weight_decay),
        "batch_size": int(tcfg.batch_size),
        "optimizer": getattr(tcfg, "optimizer", "adam"),
        "schedule": getattr(tcfg, "schedule", "plateau"),
        "max_epochs": int(tcfg.max_epochs),
        "patience": int(tcfg.patience),
        "train_positive_ratio": float(train_positive_ratio),
    }
    if model is not None:
        out["parameters"] = bw.count_parameters(model)
    return out


def run_variations(variations: list[dict], base_tcfg, bundle, val_ds, probe_ds, test_ds,
                   device, log: "ExperimentLog", rule: OverfitRule | None = None,
                   es_kwargs: dict | None = None, train_positive_ratio: float | None = None,
                   ks=(10, 25, 50, 100, 250, 1000), seed: int = 42,
                   fig_dir: str | None = None, skip_logged: bool = True,
                   ckpt_dir: str = "models/experiments",
                   model_type: str = "cnn", train_stream: str = "random") -> pd.DataFrame:
    """Train, diagnose, test and log every variation under one protocol.

    Each entry of `variations` is `{"name": ..., <TrainConfig field>: value, ...}`;
    `train_positive_ratio` may be set per variation too. `model_type` labels the rows in
    the log ("cnn", "transformer"), and the winner is chosen within one model type.
    `train_stream` (also settable per variation) is "random" -- `RandomWindows`, 4 draws
    per drive with forced positives -- or "tiling", every training window once per epoch
    at natural prevalence (`ShuffledTilingWindows`, XGBoost's training set).
    Every run uses the same seed,
    the same validation monitor, the same train probe and the same test windows. The
    workbook is rewritten after each variation, so an interrupted sweep loses at most
    the run in progress, and with `skip_logged` a rerun resumes where it stopped.

    Each run's best weights (already restored by early stopping) are saved to
    `<ckpt_dir>/<name>.pt`; the train, validation and test numbers in the log are the
    ones those weights score.
    """
    rule = rule or OverfitRule()
    es_kwargs = es_kwargs or {}
    base_ratio = (train_positive_ratio if train_positive_ratio is not None
                  else bundle.cfg.train_positive_ratio)
    for i, var in enumerate(variations, 1):
        var = dict(var)
        name = var.pop("name")
        if skip_logged and log.has(name):
            print(f"[{i}/{len(variations)}] {name}: already in {log.path}, skipped")
            continue
        ratio = float(var.pop("train_positive_ratio", base_ratio))
        stream = var.pop("train_stream", train_stream)
        if "dilations" in var:
            var["dilations"] = tuple(int(x) for x in var["dilations"])
        tcfg = dataclasses.replace(base_tcfg, **var)

        print(f"\n[{i}/{len(variations)}] {name}: {var or 'search winner, unchanged'}"
              f" -- train stream: {stream}")
        bw.set_seed(seed)
        if stream == "tiling":
            train_ds = ap.ShuffledTilingWindows(
                bundle.store, bundle.splits["train"], batch_size=bundle.cfg.batch_size,
                seed=seed, split_name="train")
        else:
            train_ds = ap.RandomWindows(
                bundle.store, bundle.splits["train"],
                samples_per_drive=bundle.cfg.samples_per_drive, positive_ratio=ratio,
                batch_size=bundle.cfg.batch_size, seed=seed, split_name="train")
        model = bw.build_model(bundle.num_features, bundle.cfg.window_days, tcfg).to(device)
        es = EarlyStopping(**{"patience": tcfg.patience, **es_kwargs})
        run = train_with_diagnostics(model, train_ds, val_ds, probe_ds, tcfg, device,
                                     early_stopping=es, rule=rule)
        test = evaluate_on_test(run.model, test_ds, run.val_threshold, device, ks=ks)
        config = describe_config(tcfg, ratio, run.model)
        config["train_stream"] = ("tiling: all train windows, natural prevalence"
                                  if stream == "tiling" else
                                  f"random: {bundle.cfg.samples_per_drive} per drive, "
                                  f"{ratio:g} forced positive")
        config["checkpoint"] = save_checkpoint(
            run.model, os.path.join(ckpt_dir, f"{_safe(name)}.pt"), tcfg, ratio)
        log.add(name, config, run.history, run.diagnostics, test, model_type=model_type,
                seconds=run.seconds, eval_protocol=eval_protocol(val_ds, test_ds))
        log.save()
        if fig_dir:
            plot_divergence(run.history, run.diagnostics, rule, title=name, show=False,
                            save_path=os.path.join(fig_dir, f"divergence_{name}.png"))
        del model, run, train_ds
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return log.summary()


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def evaluate_winner(log: "ExperimentLog", test_full_ds, num_features: int,
                    window_days: int, device, select_by: str = "val_pr_auc",
                    ks=(10, 25, 50, 100, 250, 1000), model_type: str = "cnn") -> dict:
    """Reload the winner's checkpoint and score it once on the full test set.

    The winner is chosen by `select_by` on validation among `model_type` runs (see
    `ExperimentLog.best_model`); the threshold is the one its validation run tuned. The
    result lands in the `Full Test` sheet (and, for the CNN, the `Best Model` sheet)
    next to the stride-30 numbers every variation got.
    """
    row = log.best_model(select_by, model_type=model_type)
    ckpt = log.config_of(row["variation"]).get("checkpoint")
    if not isinstance(ckpt, str) or not os.path.exists(ckpt):
        print(f"winner {row['variation']} has no checkpoint on disk -- skipped the "
              f"full-test pass (re-run that variation to create one)")
        return {}
    model, _ = load_checkpoint(ckpt, num_features, window_days, device)
    print(_RULE)
    print(f"FULL TEST PASS -- winner '{row['variation']}' (best {select_by} "
          f"{row[select_by]:.4f}), {test_full_ds.n_windows:,} windows, "
          f"threshold {row['threshold']:.4f} from its validation run")
    print(_RULE)
    m = evaluate_on_test(model, test_full_ds, float(row["threshold"]), device, ks=ks)
    m["protocol"] = (f"enumerated stride {test_full_ds.stride}"
                     if isinstance(test_full_ds, ap.EnumeratedWindows) else
                     type(test_full_ds).__name__)
    log.add_full_test(row["variation"], m)
    log.save()
    print("  " + bw.format_metrics("full test", m))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return m


def run_experiments(variations: list[dict], base_tcfg, bundle, val_ds, probe_ds, test_ds,
                    device, log: "ExperimentLog", test_full_ds=None,
                    select_by: str = "val_pr_auc", **run_kwargs) -> pd.Series:
    """The whole pipeline in one call: every variation, the winner, its full test.

    1. `run_variations`: train each with early stopping and the overfitting rule, save
       its best checkpoint, score train/val/test, log to Excel after every run.
    2. Pick the winner on validation (`select_by`).
    3. `evaluate_winner`: reload its checkpoint, score the full test set once.

    Interrupt it and call it again: finished variations are skipped, and the full-test
    pass is redone for whichever variation wins once everything is in.
    """
    run_variations(variations, base_tcfg, bundle, val_ds, probe_ds, test_ds, device, log,
                   **run_kwargs)
    model_type = run_kwargs.get("model_type", "cnn")
    if test_full_ds is not None:
        evaluate_winner(log, test_full_ds, bundle.num_features, bundle.cfg.window_days,
                        device, select_by=select_by, model_type=model_type,
                        ks=run_kwargs.get("ks", (10, 25, 50, 100, 250, 1000)))
    log.save()
    best = log.best_model(select_by, model_type=model_type)
    print(f"\nBEST {model_type.upper()}: {best['variation']} -- val PR-AUC {best['val_pr_auc']:.4f}, "
          f"test PR-AUC {best['test_pr_auc']:.4f} (stride-30 test), {best['status']}")
    return best


# ---------------------------------------------------------------------------
# 6. XGBoost on the same footing
# ---------------------------------------------------------------------------

def xgb_history(clf, rule: OverfitRule, y_probe=None, y_val=None,
                train_key: str = "validation_0", val_key: str = "validation_1") -> pd.DataFrame:
    """Per-round train-probe vs validation curves from a fitted `XGBClassifier`.

    Fit with `eval_set=[(X_probe, y_probe), (X_val, y_val)]` -- validation last, so it
    is the set early stopping watches -- and `eval_metric=["logloss", "auc", "aucpr"]`.
    A boosting round plays the role of an epoch in the log. Pass the label arrays to
    record each set's base rate alongside.
    """
    ev = clf.evals_result()
    names = {"logloss": "loss", "auc": "roc_auc", "aucpr": "pr_auc"}
    n = len(next(iter(ev[val_key].values())))
    h = pd.DataFrame({"epoch": np.arange(1, n + 1)})
    for xm, m in names.items():
        if xm in ev[val_key]:
            h[f"train_{m}"] = ev[train_key][xm]
            h[f"val_{m}"] = ev[val_key][xm]
    for col, y in (("train_base_rate", y_probe), ("val_base_rate", y_val)):
        if y is not None:
            h[col] = float(np.mean(y))
    return gap_columns(h, rule)


# ---------------------------------------------------------------------------
# 7. The Excel log
# ---------------------------------------------------------------------------

class ExperimentLog:
    """Every variation's config, epoch curves, diagnostics and scores in one .xlsx.

    Sheets, in order:

    * `Best Model` -- the single winner (chosen on validation), its configuration, how
      its training went, and every metric on train / validation / test, plus the full
      test pass when `evaluate_winner` has run.
    * `Summary Table` -- the leaderboard: one row per variation, hyperparameters, best /
      stopped / divergence epoch, Overfitting Triggered (Yes/No), and each metric for
      train, val and test side by side. Ranked by validation PR-AUC.
    * `Stopping Rules` -- the three stopping conditions with their settings, the
      overfitting limits, and which condition stopped each run.
    * `Epoch History` -- every epoch of every run, long format.
    * `Configs`, `Diagnostics`, `Test_PrecisionAtK`, `Full Test`.
    * `C01_<name>`, ... -- one per variation: its epochs with line charts of train vs
      validation loss, PR-AUC, ROC-AUC and the gap.
    """

    SHEETS = ("Summary Table", "Configs", "Epoch History", "Diagnostics",
              "Test_PrecisionAtK", "Full Test")
    # Workbooks written before the sheets were renamed still resume.
    LEGACY = {"Summary Table": "Summary", "Epoch History": "Epochs"}
    METRICS = (("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("loss", "Loss"),
               ("bal_acc", "Balanced accuracy"), ("accuracy", "Accuracy"),
               ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"))

    def __init__(self, path: str = "results/experiments_results.xlsx", resume: bool = True):
        self.path = path
        self.frames = {s: pd.DataFrame() for s in self.SHEETS}
        if resume and os.path.exists(path):
            got = pd.read_excel(path, sheet_name=None)
            for s in self.SHEETS:
                src = s if s in got else self.LEGACY.get(s)
                if src in got:
                    self.frames[s] = got[src].drop(columns="rank_by_val", errors="ignore")
            print(f"resumed {path}: {self.names()}")

    def names(self) -> list[str]:
        cfgs = self.frames["Configs"]
        return cfgs["variation"].tolist() if "variation" in cfgs else []

    def has(self, name: str) -> bool:
        return name in self.names()

    def config_of(self, name: str) -> dict:
        c = self.frames["Configs"]
        return c[c["variation"] == name].iloc[0].dropna().to_dict()

    def _put(self, sheet: str, name: str, df: pd.DataFrame) -> None:
        old = self.frames[sheet]
        if "variation" in old:
            old = old[old["variation"] != name]
        self.frames[sheet] = pd.concat([old, df], ignore_index=True)

    def add(self, name: str, config: dict, history: pd.DataFrame, diagnostics: dict,
            test: dict, model_type: str = "cnn", seconds: float | None = None,
            eval_protocol: str = "") -> None:
        """Record one run; re-adding a name replaces its rows.

        Train and validation numbers are read at the best epoch -- the weights early
        stopping restored -- and test was scored with those same weights.
        `eval_protocol` names the window sets the metrics came from: PR-AUC moves with
        the base rate, so rows are only comparable when it matches.
        """
        h = history.set_index("epoch")
        best = h.loc[diagnostics["best_epoch"]]

        def at_best(col):
            return float(best[col]) if col in best and pd.notna(best[col]) else np.nan

        pk = test.get("precision_at_k", pd.DataFrame())
        pk_at = {}
        for _, r in pk.iterrows():
            if int(r["k"]) in (100, 1000):
                pk_at[f"test_P@{int(r['k'])}_{r['ranked']}"] = float(r["precision"])

        n_flag = diagnostics.get("n_flagged_epochs") or 0
        stop_reason = str(diagnostics.get("stop_reason", ""))
        summary = {
            "variation": name, "model_type": model_type, "eval_protocol": eval_protocol,
            **{k: config.get(k) for k in ("layers", "dilations", "kernel_size",
                                          "hidden_dim", "dropout", "lr", "weight_decay",
                                          "parameters")},
            "best_epoch": diagnostics["best_epoch"],
            "stopped_epoch": diagnostics["stopped_epoch"],
            "divergence_epoch": diagnostics.get("divergence_epoch"),
            "overfitting_triggered": "Yes" if n_flag > 0 else "No",
            "stopped_by_overfit_rule": "Yes" if stop_reason.startswith("overfit") else "No",
            "status": diagnostics["status"],
            "stop_reason": stop_reason,
        }
        for m, _ in self.METRICS:
            summary[f"train_{m}"] = at_best(f"train_{m}")
            summary[f"val_{m}"] = at_best(f"val_{m}")
            summary[f"test_{m}"] = test.get(m, np.nan)
            if m in GAP_METRICS:
                summary[f"{m}_gap_pct"] = at_best(f"{m}_gap_pct")
        summary.update({
            "train_base_rate": at_best("train_base_rate"),
            "val_base_rate": at_best("val_base_rate"),
            "test_base_rate": test.get("base_rate"),
            **pk_at,
            "threshold": test.get("threshold"),
            "test_windows": test.get("n"), "test_positives": test.get("n_pos"),
            "train_minutes": None if seconds is None else round(seconds / 60, 1),
            "logged_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        })

        self._put("Summary Table", name, pd.DataFrame([summary]))
        self._put("Configs", name, pd.DataFrame([{"variation": name,
                                                  "model_type": model_type, **config}]))
        self._put("Epoch History", name, history.assign(variation=name)
                  [["variation"] + list(history.columns)])
        self._put("Diagnostics", name, pd.DataFrame([{"variation": name, **diagnostics}]))
        if len(pk):
            self._put("Test_PrecisionAtK", name, pk.assign(variation=name)
                      [["variation", "ranked"] + [c for c in pk.columns if c != "ranked"]])

    def add_full_test(self, name: str, test: dict) -> None:
        """The winner's one pass over the full test set (`evaluate_winner`)."""
        row = {"variation": name, "protocol": test.get("protocol", ""),
               **{m: test.get(m) for m, _ in self.METRICS},
               "threshold": test.get("threshold"), "windows": test.get("n"),
               "positives": test.get("n_pos"), "base_rate": test.get("base_rate")}
        for _, r in test.get("precision_at_k", pd.DataFrame()).iterrows():
            row[f"P@{int(r['k'])}_{r['ranked']}"] = float(r["precision"])
        row["evaluated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
        self._put("Full Test", name, pd.DataFrame([row]))

    def summary(self) -> pd.DataFrame:
        """All variations, ranked by validation PR-AUC -- never by test.

        Ranks restart within each `eval_protocol`: PR-AUC moves with the base rate, so a
        tiling-window score and a stride-30 score are not on one scale.
        """
        s = self.frames["Summary Table"]
        if not len(s):
            return s
        s = s.sort_values(["eval_protocol", "val_pr_auc"], ascending=[True, False])
        s = s.reset_index(drop=True)
        s.insert(0, "rank_by_val", s.groupby("eval_protocol").cumcount() + 1)
        for c in ("best_epoch", "stopped_epoch", "divergence_epoch", "layers",
                  "kernel_size", "hidden_dim", "parameters"):
            if c in s:
                s[c] = pd.to_numeric(s[c], errors="coerce").round().astype("Int64")
        return s

    def best_model(self, select_by: str = "val_pr_auc",
                   model_type: str = "cnn") -> pd.Series:
        """The winner: highest `select_by` among `model_type` runs.

        Only runs scored under the same protocol as the most common one compete, so a
        tiling-window row never outranks a stride-30 row on base rate alone. Choose on
        validation: choosing on test makes the reported test score a selected maximum,
        biased upward by however lucky the winner got on that one test split.
        """
        s = self.frames["Summary Table"]
        s = s[s["model_type"] == model_type]
        if not len(s):
            raise ValueError(f"no {model_type} runs logged yet")
        s = s[s["eval_protocol"] == s["eval_protocol"].mode().iloc[0]]
        return s.loc[s[select_by].astype(float).idxmax()]

    # -- writing -------------------------------------------------------------

    @staticmethod
    def _sheet_name(i: int, name: str) -> str:
        return f"C{i:02d}_" + re.sub(r"[\[\]:*?/\\]", "_", name)[:26]

    def save(self, path: str | None = None, select_by: str = "val_pr_auc") -> str:
        """Write every sheet, the Best Model page, the per-variation charts.

        Written to a temporary file and swapped in, so a crash mid-write never leaves
        a truncated workbook. If the target is open in Excel (locked on Windows), the
        log goes to a timestamped copy next to it instead of being lost.
        """
        path = path or self.path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp.xlsx"
        with pd.ExcelWriter(tmp, engine="openpyxl") as xw:
            ws_best = xw.book.create_sheet("Best Model")
            self.summary().to_excel(xw, sheet_name="Summary Table", index=False)
            ws_stop = xw.book.create_sheet("Stopping Rules")
            for s in self.SHEETS[1:]:
                if s == "Full Test" and not len(self.frames[s]):
                    continue
                self.frames[s].to_excel(xw, sheet_name=s, index=False)
            epochs = self.frames["Epoch History"]
            for i, name in enumerate(self.names(), 1):
                sub = (epochs[epochs["variation"] == name].drop(columns="variation")
                       .dropna(axis=1, how="all"))
                sn = self._sheet_name(i, name)
                sub.to_excel(xw, sheet_name=sn, index=False)
                _add_curve_charts(xw.sheets[sn], sub, title=name)
            for ws in xw.book.worksheets:
                if ws.title not in ("Best Model", "Stopping Rules"):
                    _format_sheet(ws)
            _write_best_sheet(ws_best, self, select_by)
            write_stopping_sheet(ws_stop, self)
        try:
            os.replace(tmp, path)
        except PermissionError:
            alt = path.replace(".xlsx", f"_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
            os.replace(tmp, alt)
            print(f"  {path} is locked (open in Excel?) -- wrote {alt} instead")
            return alt
        print(f"  wrote {path} ({len(self.names())} runs)")
        return path


# What the notebook configures (`ES`, `RULE`, `tcfg`); used for runs logged before each
# run recorded its own settings.
STOPPING_DEFAULTS = {
    "es_monitor": "val_pr_auc", "es_patience": 3, "es_min_delta": 1e-3,
    "es_restore_best_weights": True, "max_epochs": 20, "lr_patience": 1,
    "rule_min_consecutive": 2, "rule_stop_on_overfit": True,
    "rule_pr_auc_gap_pct": 15.0, "rule_roc_auc_gap_pct": 3.0,
    "rule_bal_acc_gap_pct": 10.0, "rule_loss_gap_pct": 25.0,
}


def stop_condition(reason) -> str:
    """The stop reason as one of the named conditions on the Stopping Rules sheet."""
    r = str(reason or "")
    if r.startswith("patience"):
        return "1 - No improvement (patience)"
    if r.startswith("overfit"):
        return "2 - Overfitting rule"
    if r.startswith("max_epochs"):
        return "3 - Epoch limit"
    if r.startswith("early_stopping_rounds"):
        return "XGBoost - no improvement for 100 rounds"
    if r.startswith("n_estimators"):
        return "XGBoost - round limit"
    return r or "unknown"


def write_stopping_sheet(ws, log: "ExperimentLog") -> None:
    """The stopping rules with their settings, then which one stopped each run."""
    from openpyxl.styles import Alignment, Font, PatternFill
    bold, title = Font(bold=True), Font(bold=True, size=14)
    band = PatternFill("solid", start_color="DCE6F1")
    hit = PatternFill("solid", start_color="F8D7D3")
    widths = {"A": 26, "B": 62, "C": 16, "D": 13, "E": 13, "F": 13, "G": 38, "H": 15,
              "I": 15, "J": 44}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    wrap = Alignment(wrap_text=True, vertical="top")

    diag = log.frames["Diagnostics"]
    cfgs = log.frames["Configs"]
    cnn = (set(cfgs.loc[cfgs["model_type"] == "cnn", "variation"])
           if {"variation", "model_type"} <= set(cfgs.columns) else set())
    # Settings: what the CNN runs recorded, else their config, else the notebook defaults.
    st, source = dict(STOPPING_DEFAULTS), "notebook defaults"
    rec = diag[diag["variation"].isin(cnn)] if "variation" in diag else diag
    if "es_patience" in rec and rec["es_patience"].notna().any():
        row = rec[rec["es_patience"].notna()].iloc[0]
        st.update({k: row[k] for k in STOPPING_DEFAULTS if k in row and pd.notna(row[k])})
        source = "recorded by the runs"
    elif "patience" in cfgs and len(cnn):
        c = cfgs[cfgs["variation"].isin(cnn)].iloc[0]
        st["es_patience"], st["max_epochs"] = int(c["patience"]), int(c["max_epochs"])
        source = "patience and max_epochs from Configs; the rest as the notebook sets them"

    r = 1

    def section(label):
        nonlocal r
        r += 1
        for j in range(1, 11):
            ws.cell(r, j).fill = band
        ws.cell(r, 1, label).font = bold
        r += 1

    def line(*vals, font=None, fill=None):
        nonlocal r
        for j, v in enumerate(vals, 1):
            c = ws.cell(r, j, v)
            c.alignment = wrap
            if font:
                c.font = font
            if fill:
                c.fill = fill
        r += 1

    ws.cell(r, 1, "STOPPING RULES -- when training stops, and what stopped each run").font = title
    r += 1
    ws.cell(r, 1, f"Settings: {source}.")

    section("CNN and transformer: training stops at whichever comes first "
            "(checked after every epoch)")
    line("Condition", "Rule", "Setting", font=bold)
    line("1 - No improvement",
         f"Validation PR-AUC has not improved by more than {st['es_min_delta']:g} for "
         f"{int(st['es_patience'])} epoch{'s' if int(st['es_patience']) != 1 else ''} "
         f"in a row.",
         f"patience {int(st['es_patience'])}")
    line("2 - Overfitting rule",
         f"{int(st['rule_min_consecutive'])} epochs in a row are flagged as overfitting "
         f"(limits below) and neither set a new best. A flagged epoch that is still a new "
         f"best does not count: if validation is still improving, training goes on.",
         "on" if st["rule_stop_on_overfit"] else "off (report only)")
    line("3 - Epoch limit", f"Epoch {int(st['max_epochs'])} is reached.",
         f"max_epochs {int(st['max_epochs'])}")
    line("After stopping",
         "The weights from the best epoch (highest validation PR-AUC) are restored. Every "
         "reported number and the saved checkpoint come from that epoch, not the last one.",
         "restored" if st["es_restore_best_weights"] else "not restored")
    line("Learning rate",
         f"Halved after {int(st['lr_patience'])} epoch(s) without a gain. Does not stop "
         f"training; gives the model a chance to improve before patience runs out.",
         "factor 0.5")

    section("An epoch is flagged as overfitting when train beats validation by more than:")
    line("Metric", "Gap", "Limit", font=bold)
    for m, label, how in (("pr_auc", "PR-AUC", "(train - val) / val"),
                          ("roc_auc", "ROC-AUC", "(train - val) / val"),
                          ("bal_acc", "Balanced accuracy", "(train - val) / val"),
                          ("loss", "Loss", "(val loss - train loss) / train loss")):
        v = st.get(f"rule_{m}_gap_pct")
        line(label, how, "off" if v is None or pd.isna(v) else f"{v:g}%")

    section("XGBoost stops differently")
    line("No improvement",
         "Stops when validation PR-AUC has not improved for 100 boosting rounds and keeps "
         "the best round (early_stopping_rounds=100, at most 3,000 rounds). The overfitting "
         "rule only reports on XGBoost; it never stops it.", "100 rounds")

    section("What stopped each run")
    heads = ("Run", "Stopped by", "Model", "Best epoch", "Stopped at", "Epochs after best",
             "Verdict", "Flagged epochs", "Divergence from", "Stop reason (as logged)")
    line(*heads, font=bold)
    summ = log.frames["Summary Table"].set_index("variation") \
        if "variation" in log.frames["Summary Table"] else pd.DataFrame()
    counts = {}
    for _, d in diag.iterrows():
        name = d["variation"]
        cond = stop_condition(d.get("stop_reason"))
        counts[cond] = counts.get(cond, 0) + 1
        mt = summ.loc[name, "model_type"] if name in summ.index else ""
        best, stop = int(d["best_epoch"]), int(d["stopped_epoch"])
        div = d.get("divergence_epoch")
        nflag = d.get("n_flagged_epochs")
        line(name, cond, mt, best, stop, stop - best,
             summ.loc[name, "status"] if name in summ.index else d.get("status"),
             int(nflag) if pd.notna(nflag) else 0,
             int(div) if pd.notna(div) else "none", d.get("stop_reason"),
             fill=hit if cond.startswith("2") else None)

    section("Totals")
    line("Stopped by", "Runs", font=bold)
    for cond, n in sorted(counts.items()):
        line(cond, n)
    ws.freeze_panes = "A3"


def _write_best_sheet(ws, log: "ExperimentLog", select_by: str) -> None:
    """The winner on one page: config, training story, metrics by split."""
    from openpyxl.styles import Alignment, Font, PatternFill
    bold, title = Font(bold=True), Font(bold=True, size=14)
    band = PatternFill("solid", start_color="DCE6F1")
    win = PatternFill("solid", start_color="E2EFDA")
    ws.column_dimensions["A"].width = 30
    for col in "BCDE":
        ws.column_dimensions[col].width = 20
    try:
        best = log.best_model(select_by)
    except (ValueError, KeyError):
        ws["A1"] = "No CNN runs logged yet."
        return
    name = best["variation"]
    cfg = log.config_of(name)
    full = log.frames["Full Test"]
    full = full[full["variation"] == name].iloc[0] if "variation" in full and (
        full["variation"] == name).any() else None

    r = 1
    ws.cell(r, 1, f"BEST MODEL: {name}").font = title
    r += 1
    ws.cell(r, 1, f"Chosen by highest {select_by.replace('_', ' ')} among "
                  f"{(log.frames['Summary Table']['model_type'] == 'cnn').sum()} CNN runs "
                  f"({best['eval_protocol']}). Test was never used to choose.")
    r += 2

    def section(label):
        nonlocal r
        c = ws.cell(r, 1, label)
        c.font, c.fill = bold, band
        for j in range(2, 6):
            ws.cell(r, j).fill = band
        r += 1

    section("Configuration")
    for k in ("layers", "dilations", "kernel_size", "receptive_field_days", "hidden_dim",
              "dropout", "lr", "weight_decay", "batch_size", "train_positive_ratio",
              "parameters", "max_epochs", "patience", "checkpoint"):
        if k in cfg:
            ws.cell(r, 1, k)
            ws.cell(r, 2, cfg[k])
            r += 1
    r += 1

    section("Training")
    for label, key in (("Best epoch (weights kept)", "best_epoch"),
                       ("Stopped at epoch", "stopped_epoch"),
                       ("Stop reason", "stop_reason"),
                       ("Divergence started at epoch", "divergence_epoch"),
                       ("Overfitting triggered", "overfitting_triggered"),
                       ("Verdict", "status"),
                       ("Training minutes", "train_minutes")):
        v = best.get(key)
        ws.cell(r, 1, label)
        ws.cell(r, 2, "none" if (v is None or (isinstance(v, float) and np.isnan(v)))
                else (int(v) if isinstance(v, float) and v.is_integer() else v))
        r += 1
    r += 1

    section("Metrics")
    heads = ["Metric", "Train", "Validation", "Test"] + (
        [f"Test (full: {full['protocol']})"] if full is not None else [])
    for j, hd in enumerate(heads, 1):
        c = ws.cell(r, j, hd)
        c.font = bold
        c.alignment = Alignment(wrap_text=True)
    r += 1
    for m, label in log.METRICS:
        ws.cell(r, 1, label)
        vals = [best.get(f"train_{m}"), best.get(f"val_{m}"), best.get(f"test_{m}")]
        if full is not None:
            vals.append(full.get(m))
        for j, v in enumerate(vals, 2):
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                c = ws.cell(r, j, float(v))
                c.number_format = "0.0000"
        r += 1
    for label, key in (("Drive precision @100", "P@100_drives"),
                       ("Drive precision @1000", "P@1000_drives")):
        ws.cell(r, 1, label)
        v = best.get(f"test_{key}")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            ws.cell(r, 4, float(v)).number_format = "0.0000"
        if full is not None and pd.notna(full.get(key)):
            ws.cell(r, 5, float(full[key])).number_format = "0.0000"
        r += 1
    ws.cell(r, 1, "Threshold (tuned on validation)")
    ws.cell(r, 2, float(best["threshold"])).number_format = "0.0000"
    r += 2

    section("Leaderboard (top 5 by validation)")
    s = log.summary()
    s = s[(s["model_type"] == "cnn") & (s["eval_protocol"] == best["eval_protocol"])]
    for j, hd in enumerate(("Variation", "Val PR-AUC", "Test PR-AUC", "Overfitting"), 1):
        ws.cell(r, j, hd).font = bold
    r += 1
    for _, row in s.head(5).iterrows():
        for j, v in enumerate((row["variation"], row["val_pr_auc"], row["test_pr_auc"],
                               row["overfitting_triggered"]), 1):
            c = ws.cell(r, j, float(v) if isinstance(v, (float, np.floating)) else v)
            if isinstance(v, (float, np.floating)):
                c.number_format = "0.0000"
            if row["variation"] == name:
                c.fill = win
        r += 1

    others = log.frames["Summary Table"]
    others = others[others["model_type"] != "cnn"]
    if len(others):
        r += 1
        section("Other model families (comparable to the CNN only where the windows match)")
        for _, row in others.iterrows():
            ws.cell(r, 1, f"{row['variation']} ({row['model_type']})")
            ws.cell(r, 2, f"val PR-AUC {row['val_pr_auc']:.4f}, "
                          f"test PR-AUC {row['test_pr_auc']:.4f}")
            ws.cell(r, 4, row["eval_protocol"])
            ws.cell(r, 5, "same windows" if row["eval_protocol"] == best["eval_protocol"]
                    else "other windows")
            r += 1
        ws.cell(r, 1, "For the head-to-head on identical windows see "
                      "Final_Model_Summary / CNN_vs_XGBoost.")


def _format_sheet(ws) -> None:
    """Bold frozen header, filters, readable widths, overfit rows tinted."""
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
    if ws.max_row < 1:
        return
    for c in ws[1]:
        c.font = Font(bold=True)
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    header = [c.value for c in ws[1]]
    for j, h in enumerate(header, 1):
        col = get_column_letter(j)
        width = max([len(str(h or ""))] + [len(f"{c.value:.4f}" if isinstance(c.value, float)
                                               else str(c.value or ""))
                                           for c in ws[col][1:200]])
        ws.column_dimensions[col].width = min(max(8, width + 2), 48)
        for c in ws[col][1:]:
            if isinstance(c.value, float):
                c.number_format = "0.0" if str(h).endswith("_pct") else "0.0000"
    flag_cols = [j for j, h in enumerate(header, 1)
                 if h in ("status", "overfit_flag", "overfitting_triggered")]
    red = PatternFill("solid", start_color="F8D7D3")
    for row in ws.iter_rows(min_row=2):
        for j in flag_cols:
            v = row[j - 1].value
            if v is True or v == "Yes" or (isinstance(v, str)
                                            and v.startswith("overfit at")):
                for c in row:
                    c.fill = red


def _add_curve_charts(ws, sub: pd.DataFrame, title: str) -> None:
    """Native Excel line charts: loss, PR-AUC and the gap, train vs validation."""
    from openpyxl.chart import LineChart, Reference
    cols = list(sub.columns)
    n = len(sub)
    if n < 2:
        return
    x = Reference(ws, min_col=cols.index("epoch") + 1, min_row=2, max_row=n + 1)
    anchor_col = len(cols) + 2
    from openpyxl.utils import get_column_letter
    for k, (label, series) in enumerate((
            ("Loss", ["train_loss", "val_loss"]),
            ("PR-AUC", ["train_pr_auc", "val_pr_auc"]),
            ("ROC-AUC", ["train_roc_auc", "val_roc_auc"]),
            ("Gap % (train better > 0)", ["pr_auc_gap_pct", "loss_gap_pct"]))):
        present = [s for s in series if s in cols and sub[s].notna().any()]
        if not present:
            continue
        ch = LineChart()
        ch.title = f"{title}: {label}"
        ch.x_axis.title = "epoch"
        ch.height, ch.width = 7, 13
        for s in present:
            j = cols.index(s) + 1
            ch.add_data(Reference(ws, min_col=j, min_row=1, max_row=n + 1),
                        titles_from_data=True)
        ch.set_categories(x)
        ws.add_chart(ch, f"{get_column_letter(anchor_col)}{1 + 15 * k}")
