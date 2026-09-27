"""The trained artifacts, loaded and made scoreable.

`app.py` owns the HTTP surface. This module owns everything between a parsed frame of
telemetry and a number: which runs exist on disk, what preprocessing contract each one
was trained under, and how each estimator turns a drive's most recent window into
`P(fails within the horizon)`.

Two things live here rather than in the app because they are properties of the *model*,
not of the web page:

*Two config dialects.* The archive pipeline writes `feature_columns` + `channel_names`;
the older cohort pipeline writes `feature_names` + `base_feature_count`. Both describe
the same object -- base SMART attributes, their lagged differences, an observed mask --
so both are read into one `Contract` and nothing downstream has to know which run it is
looking at.

*Two estimator kinds over identical input.* The 2022x run trained a CNN on the (30, 46)
window and XGBoost on a flattened summary of that same window. They share the store, the
calendar and the scaler; they differ only in the last step. `Estimator.score` is that
last step, so the app can run either -- or both -- over a single parse of the file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch

import backblaze_window_pipeline as bw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")

METRIC_KEYS = ("pr_auc", "roc_auc", "precision", "recall", "f1",
               "tp", "fp", "fn", "n", "n_pos", "base_rate", "threshold")


# ---------------------------------------------------------------------------
# the preprocessing contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Contract:
    """What the training run promised its input would look like.

    Reproducing this exactly at serving time is the whole point of saving it: the same
    base attributes in the same order, the same log1p set, the same lags, and above all
    the *same scaler statistics*. Refitting a scaler on whatever file the user dropped in
    would make one drive's score depend on which other drives were uploaded alongside it.
    """

    prefix: str
    window_days: int
    horizon_days: int
    base_features: tuple[str, ...]
    channel_names: tuple[str, ...]
    log1p_columns: tuple[str, ...]
    delta_lags: tuple[int, ...]
    add_observed_mask: bool
    medians: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    arch: str
    hidden_dim: int
    dropout: float
    cnn_threshold: float
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def n_channels(self) -> int:
        return len(self.channel_names)

    def data_config(self, pad_short: bool = False,
                    min_days: int | None = None) -> bw.DataConfig:
        return bw.DataConfig(
            window_days=self.window_days,
            horizon_days=self.horizon_days,
            stride_days=1,
            min_days=min_days or self.window_days,
            pad_short_drives=pad_short,
            delta_lags=tuple(self.delta_lags),
            add_observed_mask=self.add_observed_mask,
            feature_columns=tuple(self.base_features),
            log1p_columns=tuple(self.log1p_columns),
        )

    def build_store(self, frame: pd.DataFrame, cfg: bw.DataConfig) -> bw.DriveSeriesStore:
        """A dense daily calendar for `frame`, scaled with the statistics from training."""
        store = bw.DriveSeriesStore(frame, cfg)
        if list(store.feature_names) != list(self.channel_names):
            raise ValueError(
                f"the store built {len(store.feature_names)} channels but the saved "
                f"contract names {len(self.channel_names)} -- the feature set changed")
        store.apply_scaling(self.medians, self.mean, self.std)
        return store


def read_contract(prefix: str) -> Contract:
    """Read `<prefix>_preprocessing_config.json` in either dialect."""
    with open(f"{prefix}_preprocessing_config.json", encoding="utf-8") as fh:
        c = json.load(fh)

    if "channel_names" in c:                     # archive pipeline
        base = list(c["feature_columns"])
        channels = list(c["channel_names"])
    else:                                        # cohort pipeline
        channels = list(c["feature_names"])
        base = channels[: c["base_feature_count"]]

    return Contract(
        prefix=prefix,
        window_days=int(c["window_days"]),
        horizon_days=int(c.get("horizon_days", 0)),
        base_features=tuple(base),
        channel_names=tuple(channels),
        log1p_columns=tuple(c["log1p_columns"]),
        delta_lags=tuple(c["delta_lags"]),
        add_observed_mask=bool(c["add_observed_mask"]),
        medians=np.asarray(c["feature_medians"], dtype=np.float32),
        mean=np.asarray(c["feature_mean"], dtype=np.float32),
        std=np.asarray(c["feature_std"], dtype=np.float32),
        arch=c["arch"],
        hidden_dim=int(c["hidden_dim"]),
        dropout=float(c.get("dropout", 0.2)),
        cnn_threshold=float(c["best_threshold"]),
        raw=c,
    )


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------

def latest_windows(store: bw.DriveSeriesStore,
                   drives: Sequence[int] | None = None) -> np.ndarray:
    """(n, window_days, channels) -- each drive's most recent window.

    "As of today": one window per drive, ending on its last day of telemetry. That is the
    question an operator actually asks of a fleet.
    """
    idx = range(store.n_drives) if drives is None else drives
    return np.stack([store.get_window(g, store.max_start(g))[0] for g in idx])


def trailing_windows(store: bw.DriveSeriesStore, g: int,
                     n: int) -> tuple[np.ndarray, list[str]]:
    """The last `n` windows of drive `g`, oldest first, with the date each one ends on.

    Sliding the window back a day at a time turns one score into a trajectory, which is
    what distinguishes a drive that is deteriorating from one that is merely unwell.
    Computed on demand for a single drive: doing it for a whole fleet up front would
    multiply the work by `n` for a panel most drives never have opened.
    """
    hi = store.max_start(g)
    starts = list(range(max(0, hi - n + 1), hi + 1))
    windows, ends = [], []
    for s in starts:
        w, _, meta = store.get_window(g, s)
        windows.append(w)
        ends.append(meta["end_date"])
    return np.stack(windows), ends


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------

class Estimator:
    """One trained decision rule over a window.

    Subclasses differ only in `score_windows`; everything else -- batching, the "as of
    today" convention, the threshold -- is shared, which is what makes the head-to-head
    on the page an honest comparison rather than two pipelines that happen to agree.
    """

    kind = "?"
    label = "?"
    blurb = ""

    def __init__(self, contract: Contract, threshold: float):
        self.contract = contract
        self.threshold = float(threshold)

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def score(self, store: bw.DriveSeriesStore, batch: int = 256) -> np.ndarray:
        """P(failure) for every drive's most recent window, in drive-index order."""
        out = np.zeros(store.n_drives, dtype=np.float32)
        for a in range(0, store.n_drives, batch):
            b = min(a + batch, store.n_drives)
            out[a:b] = self.score_windows(latest_windows(store, range(a, b)))
        return out

    def explain(self, windows: np.ndarray) -> list[dict] | None:
        """Per-feature contributions for a single window, or None if not available."""
        return None


class CnnEstimator(Estimator):
    kind = "cnn"
    label = "CNN"
    blurb = ("a dilated 1-D convolutional net reading the window as a sequence, so it "
             "sees the order of the days and not only their summary")

    def __init__(self, contract: Contract, device: torch.device | None = None):
        super().__init__(contract, contract.cnn_threshold)
        self.device = device or torch.device("cpu")
        tcfg = bw.TrainConfig(arch=contract.arch, hidden_dim=contract.hidden_dim,
                              dropout=contract.dropout)
        model = bw.build_model(contract.n_channels, contract.window_days, tcfg)
        state = torch.load(f"{contract.prefix}_{contract.arch}.pth",
                           map_location=self.device)
        model.load_state_dict(state)
        self.model = model.to(self.device).eval()
        self.n_params = bw.count_parameters(model)

    @torch.no_grad()
    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        xb = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        probs = torch.sigmoid(self.model(xb.to(self.device)))
        return probs.cpu().numpy().ravel().astype(np.float32)


class XgbEstimator(Estimator):
    """Gradient boosting over a flattened summary of the same window.

    A tree cannot take a (30, 46) tensor, so each channel is reduced to its last day, its
    mean and its first-to-last change -- the shape of the month rather than the month
    itself. `window_table_features` in the archive pipeline builds exactly these columns
    in exactly this order during training; this reproduces it a batch at a time.
    """

    kind = "xgb"
    label = "XGBoost"
    blurb = ("gradient-boosted trees over each channel's last value, its mean and its "
             "first-to-last change across the window")

    def __init__(self, contract: Contract):
        import xgboost as xgb

        with open(f"{contract.prefix}_xgb_config.json", encoding="utf-8") as fh:
            xc = json.load(fh)
        super().__init__(contract, xc["threshold"])
        self.xgb = xgb
        self.feature_columns = list(xc["feature_columns"])
        self.aggregates = list(xc["aggregates"])
        self.best_iteration = int(xc.get("best_iteration", 0))
        self.clf = xgb.XGBClassifier()
        self.clf.load_model(f"{contract.prefix}_xgb.json")
        self.booster = self.clf.get_booster()
        self.n_trees = self.booster.num_boosted_rounds()
        expected = contract.n_channels * len(self.aggregates)
        if len(self.feature_columns) != expected:
            raise ValueError(
                f"the xgb config lists {len(self.feature_columns)} features but "
                f"{contract.n_channels} channels x {len(self.aggregates)} aggregates "
                f"is {expected}")

    def table(self, windows: np.ndarray) -> np.ndarray:
        """(n, channels * aggregates), in the training column order."""
        parts = []
        for agg in self.aggregates:
            if agg == "last":
                parts.append(windows[:, -1, :])
            elif agg == "mean":
                parts.append(windows.mean(axis=1))
            elif agg == "min":
                parts.append(windows.min(axis=1))
            elif agg == "max":
                parts.append(windows.max(axis=1))
            elif agg == "delta":
                parts.append(windows[:, -1, :] - windows[:, 0, :])
            else:
                raise ValueError(f"unknown aggregate {agg!r}")
        X = np.concatenate(parts, axis=1).astype(np.float32)
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def score_windows(self, windows: np.ndarray) -> np.ndarray:
        # predict_proba honours the early-stopping iteration recorded in the saved model,
        # so this scores with the trees that were actually selected, not every tree fitted.
        return self.clf.predict_proba(self.table(windows))[:, 1].astype(np.float32)

    def explain(self, windows: np.ndarray, top: int = 12) -> list[dict]:
        """Signed SHAP contributions in log-odds for one window, largest magnitude first.

        Exact for trees, and they sum to the logit of the score -- so this is an account
        of *this* prediction, not a global importance ranking shown next to it.
        """
        X = self.table(windows[:1])
        contrib = self.booster.predict(self.xgb.DMatrix(X), pred_contribs=True)[0]
        order = np.argsort(-np.abs(contrib[:-1]))[:top]
        return [{"feature": self.feature_columns[int(j)],
                 "contribution": float(contrib[int(j)]),
                 "value": float(X[0, int(j)])} for j in order
                if abs(float(contrib[int(j)])) > 1e-6]


# ---------------------------------------------------------------------------
# runs: an artifact prefix plus everything known about how it was trained
# ---------------------------------------------------------------------------

# Editorial, not derivable from the files: which run is the current one, what each was
# trained on, and how to say so in one line on the page.
CATALOGUE: list[dict] = [
    {
        "prefix": "backblaze_2022x_w30",
        "label": "Backblaze 2022 -> today",
        "tag": "current",
        "scope": "the whole fleet, every manufacturer, at its real failure rate",
        "note": ("18 quarterly archives from Q1 2022 on: 413k drives, 459M drive-days. "
                 "Validation and test enumerate every possible window at stride 1, so "
                 "the numbers are measured at the fleet's true prevalence of 0.24% "
                 "rather than on a balanced cohort."),
        "summary": "w2022x_summary.json",
        "default": "xgb",
    },
    {
        "prefix": "backblaze_2022_w30",
        "label": "Backblaze 2022 -> today, narrow CNN",
        "tag": "superseded",
        "scope": "the same shards, the 64-unit CNN from before the capacity search",
        "note": "the first pass over the 2022+ shards, kept for comparison.",
        "summary": "w2022_summary.json",
        "default": "cnn",
    },
    {
        "prefix": "backblaze_archive30",
        "label": "Backblaze full archive, 2013 -> today",
        "tag": "archive",
        "scope": "every published quarter since 2013",
        "note": ("508k drives across thirteen years. Backblaze has added and dropped "
                 "SMART columns repeatedly over that span, so fewer attributes survive "
                 "the coverage filter."),
        "summary": "archive30_summary.json",
        "default": "cnn",
    },
    {
        "prefix": "backblaze_window90",
        "label": "Backblaze cohort, 90-day window",
        "tag": "cohort",
        "scope": "a matched cohort: every failure plus a sample of healthy drives",
        "note": ("the original run. Precision here is inflated by construction -- the "
                 "cohort is roughly half failures -- so only the ranking transfers."),
        "summary": None,
        "default": "cnn",
    },
    {
        "prefix": "toshiba_window",
        "label": "Toshiba export, 90-day window",
        "tag": "cohort",
        "scope": "14 Toshiba families from the local export",
        "note": "single-vendor, and a cohort rather than a fleet. See TOSHIBA_PIPELINE.md.",
        "summary": None,
        "default": "cnn",
    },
]

BLOCKED_COLUMNS = {
    "days_to_last_record":
        "AUC 0.99 -- failure=1 is written on a drive's last reporting day, so any count "
        "of days to that boundary reconstructs the label",
    "days_from_record_end_to_export_end":
        "AUC 0.96 -- the same boundary, measured from the other end",
    "still_reporting_at_export_end":
        "AUC 0.97 per drive -- a disk still reporting on the archive's final day has not "
        "failed, so the probe restates the label",
    "record_length_days":
        "balanced accuracy 0.91 -- an artefact of how the cohort was sampled",
    "pod_slot_num":
        "leaks through absence: null on 53.6% of failure days against 2.2% of healthy ones",
    "vault_id": "8.2x lift on one level -- fleet topology, not drive behaviour",
    "cluster_id": "fleet topology -- where the disk sat, not how it behaved",
    "datacenter": "fleet topology -- where the disk sat, not how it behaved",
    "capacity_bytes": "5.5x lift -- a proxy for drive family and generation",
    "serial_number": "identity: a tree will memorise the outcome drive by drive",
    "date": "calendar position encodes how and when the cohort was assembled",
    "failure": "the label itself",
    "model": "drive family, which stands in for the manufacturer's failure rate",
}
# Attributes dropped for coverage rather than leakage: reported by too small a slice of
# the archive to be trusted, so the leakage audit never gets to them.
SPARSE_ATTRIBUTES = ("smart_184_raw", "smart_187_raw", "smart_188_raw", "smart_189_raw",
                     "smart_190_raw", "smart_241_raw", "smart_242_raw")


@dataclass
class Run:
    key: str
    meta: dict
    contract: Contract
    estimators: dict[str, Estimator]
    default: str
    summary: dict = field(default_factory=dict, repr=False)
    tables: dict = field(default_factory=dict, repr=False)

    @property
    def primary(self) -> Estimator:
        return self.estimators[self.default]

    def metrics_for(self, kind: str) -> dict:
        """Held-out test metrics for one estimator, from whichever file recorded them."""
        s = self.summary
        block = None
        if kind == "xgb" and isinstance(s.get("xgboost"), dict):
            block = s["xgboost"].get("test_at_tuned")
        elif kind == "cnn" and isinstance(s.get("cnn_on_tiling_test"), dict):
            block = s["cnn_on_tiling_test"]
        if block is None:
            block = s.get("test_at_tuned") or self.contract.raw.get("test_metrics") or {}
        return {k: block[k] for k in METRIC_KEYS if k in block}

    def describe(self) -> dict:
        """Everything the page needs to introduce this run, as plain JSON."""
        c = self.contract
        blank = np.zeros((1, c.window_days, c.n_channels), dtype=np.float32)
        ests = []
        for kind, est in self.estimators.items():
            ests.append({
                "kind": kind,
                "label": est.label,
                "blurb": est.blurb,
                "threshold": est.threshold,
                "metrics": self.metrics_for(kind),
                "is_default": kind == self.default,
                "explains": est.explain(blank) is not None,
                "size": (f"{getattr(est, 'n_params', 0):,} parameters" if kind == "cnn"
                         else f"{getattr(est, 'best_iteration', 0) + 1:,} of "
                              f"{getattr(est, 'n_trees', 0):,} trees"),
            })
        ests.sort(key=lambda e: (not e["is_default"], e["kind"]))
        blocklist = self.summary.get("blocklist", [])
        return {
            "key": self.key,
            "label": self.meta["label"],
            "tag": self.meta["tag"],
            "scope": self.meta["scope"],
            "note": self.meta["note"],
            "window_days": c.window_days,
            "horizon_days": c.horizon_days,
            "attributes": list(c.base_features),
            "n_channels": c.n_channels,
            "delta_lags": list(c.delta_lags),
            "default": self.default,
            "estimators": ests,
            "n_drives": self.summary.get("n_drives"),
            "n_drive_days": self.summary.get("n_drive_days"),
            "shards": self.summary.get("shards"),
            "blocklist": [{"column": col,
                           "why": BLOCKED_COLUMNS.get(
                               col, "dropped by the leakage audit before training")}
                          for col in blocklist if col not in SPARSE_ATTRIBUTES],
            "sparse_dropped": [col for col in blocklist if col in SPARSE_ATTRIBUTES],
            "max_channel_auc": self.summary.get("max_channel_auc_vs_label"),
            "leakage_checks_passed": self.summary.get("leakage_checks_passed"),
            "tree_baseline": self.summary.get("tree_final"),
            "tables": self.tables,
        }


def _read_csv(name: str, limit: int | None = None) -> list[dict]:
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    if limit:
        df = df.head(limit)
    return df.replace({np.nan: None}).to_dict("records")


def _tables_for(prefix_name: str) -> dict:
    """The result tables the page shows beside a run, where that run produced them."""
    if prefix_name != "backblaze_2022x_w30":
        return {}
    return {
        "precision_at_k": _read_csv("w2022x_cnn_vs_xgboost_precision_at_k.csv"),
        "importance": _read_csv("w2022x_xgboost_importance.csv", 12),
        "channel_auc": _read_csv("w2022x_channel_label_auc.csv", 10),
        "comparison": _read_csv("w2022x_model_summary.csv"),
    }


def discover(models_dir: str = MODELS_DIR, verbose: bool = True) -> dict[str, Run]:
    """Every run in `models_dir` whose config and at least one estimator are present.

    Missing artifacts are skipped rather than fatal: the app is useful with one run, and
    a half-synced `models/` should not stop it booting.
    """
    runs: dict[str, Run] = {}
    for entry in CATALOGUE:
        prefix = os.path.join(models_dir, entry["prefix"])
        if not os.path.isfile(f"{prefix}_preprocessing_config.json"):
            continue
        try:
            contract = read_contract(prefix)
        except (OSError, ValueError, KeyError) as e:
            if verbose:
                print(f"  skipped {entry['prefix']}: {e}")
            continue

        estimators: dict[str, Estimator] = {}
        if os.path.isfile(f"{prefix}_{contract.arch}.pth"):
            try:
                estimators["cnn"] = CnnEstimator(contract)
            except Exception as e:                        # noqa: BLE001
                if verbose:
                    print(f"  skipped {entry['prefix']} cnn: {type(e).__name__}: {e}")
        if os.path.isfile(f"{prefix}_xgb.json"):
            try:
                estimators["xgb"] = XgbEstimator(contract)
            except Exception as e:                        # noqa: BLE001
                if verbose:
                    print(f"  skipped {entry['prefix']} xgb: {type(e).__name__}: {e}")
        if not estimators:
            continue

        summary: dict = {}
        if entry["summary"]:
            path = os.path.join(RESULTS_DIR, entry["summary"])
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as fh:
                    summary = json.load(fh)

        default = entry["default"] if entry["default"] in estimators else next(iter(estimators))
        runs[entry["prefix"]] = Run(key=entry["prefix"], meta=entry, contract=contract,
                                    estimators=estimators, default=default,
                                    summary=summary, tables=_tables_for(entry["prefix"]))
    return runs
