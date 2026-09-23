"""90-day window sequence classification for Backblaze drive telemetry.

Refactor of the row-level pipeline in `Dataset_backblaze_Comparison.ipynb`. Instead of
scoring one drive-day at a time from hand-built trailing-window aggregates, a *sample*
here is a block of `WINDOW_DAYS` consecutive daily readings for one drive:

    X : (WINDOW_DAYS, num_features)   the daily sequence, `failure` NEVER included
    y : {0, 1}                        1 <=> a failure event falls inside the window

Pipeline stages
---------------
1.  load          -- Backblaze quarterly CSVs (or a synthetic frame for smoke tests)
2.  DriveSeriesStore
                  -- one dense daily calendar per drive: calendar gaps are
                     forward-filled and flagged, deltas are precomputed, the failure
                     flag is split off into a separate array that X can never see
3.  grouped split -- by drive id, so no drive's windows appear in two splits
4.  scaling       -- medians + StandardScaler fit on TRAIN drive-days only
5.  Dataset       -- `random` mode (a fresh random window per drive per epoch) for
                     training, `enumerate` mode (fixed stride, reproducible) for
                     val/test
6.  inspection    -- print real samples and real batches before a single gradient step
7.  model         -- 1D-CNN (dilated / TCN-style), GRU, or Transformer encoder;
                     all map (B, WINDOW_DAYS, F) -> (B, 1) logit
8.  train / eval  -- PR-AUC, ROC-AUC, precision, recall, F1, with pos_weight,
                     focal loss, or a balanced sampler for the class imbalance

Usage
-----
    # end-to-end smoke test on generated data, no download required
    python scripts/backblaze_window_pipeline.py --synthetic --epochs 3

    # inspection only: build the windows, print samples/batches, stop before training
    python scripts/backblaze_window_pipeline.py --synthetic --inspect-only

    # the real thing (downloads ~1.5 GB per quarter on first run, caches to parquet)
    python scripts/backblaze_window_pipeline.py --quarters Q1_2023 Q2_2023 \
        --arch cnn --epochs 20

From a notebook:

    from scripts.backblaze_window_pipeline import *
    cfg = DataConfig(window_days=90)
    raw = load_backblaze_frame(["Q1_2023"], cache="data/backblaze_seagate.parquet")
    bundle = build_window_bundle(raw, cfg)
    inspect_windows(bundle.datasets["train"], n_samples=3)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42

ID_COLUMNS = ["date", "serial_number", "model", "failure"]

# The same nine attributes the row-level notebook settled on (Section 3).
SMART_RAW_COLUMNS = [
    "smart_5_raw",    # Reallocated Sectors Count
    "smart_9_raw",    # Power-On Hours
    "smart_187_raw",  # Reported Uncorrectable Errors
    "smart_188_raw",  # Command Timeout
    "smart_194_raw",  # Temperature (Celsius)
    "smart_197_raw",  # Current Pending Sector Count
    "smart_198_raw",  # Offline Uncorrectable Sector Count
    "smart_199_raw",  # UDMA CRC Error Count
    "smart_241_raw",  # Total LBAs Written
]

# Counters spanning many orders of magnitude; compressed with log1p before scaling so a
# drive with 40,000 reallocated sectors does not dominate the whole channel. Temperature
# (smart_194) is a bounded physical reading and is left alone.
LOG1P_COLUMNS = [c for c in SMART_RAW_COLUMNS if c != "smart_194_raw"]

# Extracted Backblaze quarters run to several GB. They default to `data/`, but this
# project lives in a synced OneDrive folder, so point BACKBLAZE_DATA_DIR at somewhere
# unsynced before pulling real quarters (Dataset_backblaze_Window90.ipynb does this).
BULK_DATA_DIR = os.environ.get("BACKBLAZE_DATA_DIR", "data")

QUARTER_URL_TEMPLATE = (
    "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_{quarter}.zip"
)


@dataclass
class DataConfig:
    """Everything that decides what a window is and which windows exist."""

    window_days: int = 90
    # A window is positive if a failure falls in [start, end + horizon_days].
    # 0 == the strict definition: the failure event must lie inside the window itself.
    # >0 turns the task into "does this drive fail within `horizon_days` of the window
    # ending", which is the row-level notebook's FAILURE_HORIZON_DAYS in window form.
    horizon_days: int = 0

    # Eval windows are enumerated every `stride_days`; 1 = every possible window.
    stride_days: int = 15
    # Training draws this many random windows per drive per epoch.
    samples_per_drive: int = 1
    # Fraction of TRAINING samples forced to be positive windows where the drive has
    # one. None = pure uniform random window choice (positives then become very rare;
    # see the note in DriveWindowDataset).
    train_positive_ratio: float | None = None

    # Drives with fewer than this many calendar days of telemetry are dropped outright.
    min_days: int = 30
    # Drives with min_days <= L < window_days are kept and left-padded to window_days.
    pad_short_drives: bool = True

    # Per-attribute differences appended as extra channels (computed on the dense daily
    # calendar, so lag k really means "k days ago").
    delta_lags: tuple[int, ...] = (1, 7)
    # Extra channel that is 1 on a genuinely observed day and 0 on a forward-filled
    # calendar gap or a padded step. Lets the model discount imputed history.
    add_observed_mask: bool = True

    feature_columns: tuple[str, ...] = tuple(SMART_RAW_COLUMNS)
    log1p_columns: tuple[str, ...] = tuple(LOG1P_COLUMNS)

    test_size: float = 0.20
    val_size: float = 0.15  # of the remaining train+val pool
    # Split failing and healthy drives with separate GroupShuffleSplits so every split
    # gets its share of the (rare) failures. Still a strict grouped split: a drive id
    # lands in exactly one partition either way.
    stratify_split_by_failure: bool = True
    random_state: int = RANDOM_STATE


@dataclass
class TrainConfig:
    arch: str = "cnn"                # cnn | gru | transformer
    hidden_dim: int = 64
    dropout: float = 0.2
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 20
    patience: int = 5
    # bce_pos_weight | bce | focal   -- how the class imbalance is handled in the loss
    loss: str = "bce_pos_weight"
    pos_weight_cap: float = 50.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    # none | balanced -- `balanced` oversamples drives that can yield a positive window
    sampler: str = "none"
    balanced_pos_fraction: float = 0.3
    num_workers: int = 0
    seed: int = RANDOM_STATE


# ---------------------------------------------------------------------------
# 1. Loading
# ---------------------------------------------------------------------------

def _find_daily_csvs(extract_folder: str) -> list[str]:
    """All daily CSV paths under an extracted quarter.

    Skips the `__MACOSX/._*.csv` AppleDouble stubs the archives carry: they end in
    ".csv" but are 268-byte resource forks, and read_csv would reject them once
    `usecols` is in play.
    """
    return sorted(
        os.path.join(root, f)
        for root, _, files in os.walk(extract_folder)
        for f in files
        if f.endswith(".csv") and not f.startswith("._")
    )


def _remote_size(url: str, timeout: int = 60) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.headers.get("Content-Length", 0))


def _download_with_resume(url: str, path: str, retries: int = 6,
                          chunk: int = 1 << 20) -> str:
    """Fetch `url` to `path`, resuming a partial file and retrying on a dropped link.

    A quarter is the better part of a gigabyte and the host does drop connections
    mid-transfer; restarting from byte zero each time can fail indefinitely. Resumes
    with a Range request, and falls back to a clean restart if the server ignores it.
    """
    total = _remote_size(url)
    for attempt in range(1, retries + 1):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if total and have == total:
            return path
        if total and have > total:          # truncated write or a changed file
            os.remove(path)
            have = 0
        if have:
            print(f"  resuming at {have / 1e9:.2f} / {total / 1e9:.2f} GB")

        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                mode = "ab"
                if have and resp.status != 206:   # Range ignored -> start over
                    have, mode = 0, "wb"
                with open(path, mode) as fh:
                    while True:
                        block = resp.read(chunk)
                        if not block:
                            break
                        fh.write(block)
        except Exception as exc:                  # reset, timeout, partial read
            wait = min(30, 2 ** attempt)
            got = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"  interrupted at {got / 1e9:.2f} GB ({type(exc).__name__}); "
                  f"retry {attempt}/{retries} in {wait}s")
            time.sleep(wait)

    got = os.path.getsize(path) if os.path.exists(path) else 0
    if total and got != total:
        raise RuntimeError(f"download incomplete: {got:,}/{total:,} bytes from {url}")
    return path


def load_backblaze_frame(
    quarters: Sequence[str],
    model_prefix_filter: str | None = "ST",
    cache: str | None = None,
    data_dir: str = BULK_DATA_DIR,
    feature_columns: Sequence[str] = SMART_RAW_COLUMNS,
) -> pd.DataFrame:
    """Download/extract Backblaze quarters and return the long telemetry frame.

    Columns: id, time, model, failure, + `feature_columns`. One row per drive-day, as
    published; calendar gaps are not filled here (DriveSeriesStore does that).
    """
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    if cache and os.path.exists(cache):
        print(f"Loading cached frame: {cache}")
        df = pd.read_parquet(cache)
        print(f"  {len(df):,} rows, {df['id'].nunique():,} drives")
        return df

    use_columns = ID_COLUMNS + list(feature_columns)
    chunks = []
    for quarter in quarters:
        url = QUARTER_URL_TEMPLATE.format(quarter=quarter)
        os.makedirs(data_dir, exist_ok=True)
        zip_path = os.path.join(data_dir, f"data_{quarter}.zip")
        extract_folder = os.path.join(data_dir, f"data_{quarter}")

        if not os.path.exists(extract_folder):
            print(f"Downloading {url} ...")
            _download_with_resume(url, zip_path)
        if not os.path.exists(extract_folder):
            print(f"Extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(extract_folder)

        csv_files = _find_daily_csvs(extract_folder)
        print(f"  {quarter}: {len(csv_files)} daily CSVs")
        for fp in csv_files:
            chunk = pd.read_csv(fp, encoding="latin1", low_memory=False, usecols=use_columns)
            chunk["date"] = pd.to_datetime(chunk["date"])
            if model_prefix_filter:
                chunk = chunk[chunk["model"].astype(str).str.startswith(model_prefix_filter)]
            # float32 per chunk rather than float64 at the end: it halves the peak of
            # the concat below, and the store is float32 anyway. The widest attribute
            # (smart_241_raw, LBAs written, ~1e13) keeps 7 significant digits, which
            # survives log1p + standardisation with room to spare.
            for col in feature_columns:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype(np.float32)
            chunk["failure"] = chunk["failure"].fillna(0).astype(np.int8)
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df = df.rename(columns={"serial_number": "id", "date": "time"})
    df = df.sort_values(["id", "time"]).reset_index(drop=True)
    if cache:
        df.to_parquet(cache, index=False)
        print(f"Cached to {cache}")
    return df


def make_synthetic_frame(
    n_drives: int = 900,
    failure_rate: float = 0.06,
    min_days: int = 60,
    max_days: int = 400,
    gap_prob: float = 0.02,
    feature_columns: Sequence[str] = SMART_RAW_COLUMNS,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """A stand-in telemetry frame with the shape of the real thing.

    Same column names, same "failure == 1 on the drive's last observed day" convention,
    the same rare-positive imbalance, missing calendar days, and a degradation ramp on
    the error-counter attributes over the final ~45 days of a failing drive. Enough to
    exercise every branch of the pipeline without a multi-GB download.
    """
    rng = np.random.default_rng(seed)
    origin = pd.Timestamp("2023-01-01")
    frames = []
    for d in range(n_drives):
        length = int(rng.integers(min_days, max_days))
        start = origin + pd.Timedelta(days=int(rng.integers(0, 120)))
        fails = bool(rng.random() < failure_rate)

        t = np.arange(length)
        days_left = length - 1 - t
        ramp = np.where(days_left < 45, (45 - days_left) / 45.0, 0.0) if fails else np.zeros(length)

        realloc = float(rng.integers(0, 4)) + np.cumsum(rng.poisson(0.02 + 6.0 * ramp))
        pending = np.cumsum(rng.poisson(0.005 + 3.0 * ramp)).astype(float)
        poh = float(rng.integers(1_000, 40_000)) + 24.0 * (t + 1)
        lba = float(rng.integers(1e6, 5e7)) + np.cumsum(rng.normal(4e5, 8e4, length))

        values = {
            "smart_5_raw": realloc.astype(float),
            "smart_9_raw": poh,
            "smart_187_raw": rng.poisson(0.01 + 4.0 * ramp).astype(float),
            "smart_188_raw": rng.poisson(0.01 + 2.0 * ramp).astype(float),
            "smart_194_raw": rng.normal(30, 3) + rng.normal(0, 1.5, length) + 4.0 * ramp,
            "smart_197_raw": pending,
            "smart_198_raw": pending * rng.uniform(0.3, 0.8, length),
            "smart_199_raw": rng.poisson(0.02, length).astype(float),
            "smart_241_raw": lba,
        }

        keep = rng.random(length) > gap_prob
        keep[0] = keep[-1] = True  # first/last day always present
        frame = pd.DataFrame(
            {
                "id": f"SYN{d:06d}",
                "time": start + pd.to_timedelta(t, unit="D"),
                "model": "ST_SYNTH",
                "failure": np.where(fails & (t == length - 1), 1, 0).astype(np.int8),
                **{c: values.get(c, np.zeros(length)) for c in feature_columns},
            }
        )
        frames.append(frame.loc[keep])
    return pd.concat(frames, ignore_index=True).sort_values(["id", "time"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. DriveSeriesStore -- dense daily calendar per drive
# ---------------------------------------------------------------------------

_EPOCH = pd.Timestamp("1970-01-01")


class DriveSeriesStore:
    """Every drive's telemetry on a gap-free daily calendar, in flat arrays.

    One (N, F) float32 matrix holds all drives back to back; `offsets` says where each
    drive starts. Window extraction is then a contiguous slice -- no per-drive DataFrame
    lookups at __getitem__ time.

    The failure flag lives in `self.failure`, a separate array that is never part of
    `self.values`. That separation is the leakage guard: there is no code path by which
    a model can read the label out of X.

    Missing-day handling
    --------------------
    A drive observed on 2023-01-01 and 2023-01-05 gets rows for 01-02..01-04 too. Those
    rows are forward-filled from the last real reading (SMART attributes are counters
    and slow-moving gauges, so "unchanged since the last report" is the right prior) and
    marked `observed = 0`. The `observed` channel is handed to the model as a feature so
    it can discount imputed stretches.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        cfg: DataConfig,
        id_col: str = "id",
        time_col: str = "time",
        failure_col: str = "failure",
    ):
        self.cfg = cfg
        feature_cols = list(cfg.feature_columns)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature columns absent from the frame: {missing}")

        df = df[[id_col, time_col, failure_col] + feature_cols]
        # A duplicated (drive, day) would corrupt the scatter below. Sorting failure
        # ascending and keeping the last occurrence retains the failure row if one of
        # the duplicates carries the event. Dropping duplicates preserves order, so the
        # frame is still in (id, time) order afterwards -- no second sort, which on a
        # 20M-row frame would mean another full copy.
        df = df.sort_values([id_col, time_col, failure_col], kind="stable")
        df = df.drop_duplicates([id_col, time_col], keep="last").reset_index(drop=True)

        day = ((pd.to_datetime(df[time_col]) - _EPOCH).dt.days).to_numpy(np.int64)
        codes, drive_ids = pd.factorize(df[id_col], sort=True)
        n_drives = len(drive_ids)

        day_ser = pd.Series(day)
        start_days = day_ser.groupby(codes).min().to_numpy(np.int64)
        end_days = day_ser.groupby(codes).max().to_numpy(np.int64)
        lengths = (end_days - start_days + 1).astype(np.int64)

        # Drop drives too short to be worth a window before materialising anything.
        keep_drive = lengths >= cfg.min_days
        if not cfg.pad_short_drives:
            keep_drive &= lengths >= cfg.window_days
        if not keep_drive.all():
            keep_row = keep_drive[codes]
            df = df.loc[keep_row].reset_index(drop=True)
            day = day[keep_row]
            codes, drive_ids = pd.factorize(df[id_col], sort=True)
            n_drives = len(drive_ids)
            day_ser = pd.Series(day)
            start_days = day_ser.groupby(codes).min().to_numpy(np.int64)
            end_days = day_ser.groupby(codes).max().to_numpy(np.int64)
            lengths = (end_days - start_days + 1).astype(np.int64)

        offsets = np.zeros(n_drives + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        n_rows = int(offsets[-1])

        # Scatter the observed rows onto the dense calendar.
        n_base = len(feature_cols)
        n_channels = n_base * (1 + len(cfg.delta_lags)) + (1 if cfg.add_observed_mask else 0)
        # One allocation for the whole feature matrix, filled block by block. Building
        # the blocks separately and concatenating them at the end would briefly hold two
        # copies of a matrix that runs to gigabytes on a full quarter of Seagate data.
        values = np.empty((n_rows, n_channels), dtype=np.float32)

        dest = offsets[codes] + (day - start_days[codes])
        levels = values[:, :n_base]          # a view: writes land in `values`
        levels[:] = np.nan
        # Column at a time: `df[feature_cols].to_numpy()` would materialise the whole
        # observed block as one temporary array before the scatter.
        for j, col in enumerate(feature_cols):
            levels[dest, j] = df[col].to_numpy(dtype=np.float32)

        observed = np.zeros(n_rows, dtype=bool)
        observed[dest] = True
        failure = np.zeros(n_rows, dtype=np.uint8)
        failure[dest] = df[failure_col].fillna(0).to_numpy(dtype=np.uint8)
        del df, dest, codes        # nothing below reads the frame again

        row_drive_start = np.repeat(offsets[:-1], lengths)
        row_pos = np.arange(n_rows, dtype=np.int64) - row_drive_start

        # log1p the heavy-tailed counters before anything differences them.
        log_idx = [i for i, c in enumerate(feature_cols) if c in set(cfg.log1p_columns)]
        if log_idx:
            block = levels[:, log_idx]
            levels[:, log_idx] = np.log1p(np.clip(block, 0.0, None))

        self._ffill_within_drive(levels, row_drive_start)

        feature_names = list(feature_cols)
        for k, lag in enumerate(cfg.delta_lags, start=1):
            block = values[:, n_base * k : n_base * (k + 1)]
            if lag < n_rows:
                # Written straight into the destination view -- no temporary difference
                # array the size of the whole level block.
                np.subtract(levels[lag:], levels[:-lag], out=block[lag:])
            block[row_pos < lag] = np.nan  # no genuine `lag`-days-ago reading yet
            feature_names += [f"{c}_d{lag}" for c in feature_cols]
        if cfg.add_observed_mask:
            values[:, -1] = observed
            feature_names.append("observed")

        self.values = values
        self.feature_names = feature_names
        self.observed = observed
        self.failure = failure
        self.offsets = offsets
        self.lengths = lengths
        self.start_days = start_days
        self.drive_ids = np.asarray(drive_ids, dtype=object)
        self.n_drives = n_drives
        self.base_feature_count = len(feature_cols)
        # `observed` is a 0/1 indicator; scaling it would destroy that reading.
        self.scaled_columns = np.array(
            [name != "observed" for name in feature_names], dtype=bool
        )
        self.log_columns = np.array(
            [name in set(cfg.log1p_columns) for name in feature_names], dtype=bool
        )
        self.feature_medians: np.ndarray | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self._scaled = False

        assert failure_col not in self.feature_names, (
            "the failure column leaked into the feature matrix"
        )

        # Per-drive failure bookkeeping, used for the split, the sampler and labelling.
        self.failure_positions: list[np.ndarray] = []
        for g in range(n_drives):
            lo, hi = self.offsets[g], self.offsets[g + 1]
            self.failure_positions.append(np.flatnonzero(self.failure[lo:hi]))
        self.drive_has_failure = np.array(
            [len(p) > 0 for p in self.failure_positions], dtype=bool
        )

    # -- construction helpers -------------------------------------------------

    @staticmethod
    def _ffill_within_drive(mat: np.ndarray, row_drive_start: np.ndarray) -> np.ndarray:
        """Forward-fill NaNs down each column without crossing a drive boundary.

        `np.maximum.accumulate` over the index of the last valid row gives a global
        forward fill; clamping that index to the row's own drive start keeps a drive
        from inheriting the previous drive's readings. A drive's first dense row is
        always an observed day (the calendar starts at its first reading), so the clamp
        only bites for a column that was already NaN on that first day -- those stay
        NaN and are median-imputed with the rest.
        """
        n_rows = mat.shape[0]
        row_idx = np.arange(n_rows, dtype=np.int64)
        for j in range(mat.shape[1]):
            col = mat[:, j]
            valid = ~np.isnan(col)
            if valid.all():
                continue
            src = np.where(valid, row_idx, -1)
            np.maximum.accumulate(src, out=src)
            np.maximum(src, row_drive_start, out=src)
            mat[:, j] = col[src]
        return mat

    # -- scaling --------------------------------------------------------------

    def fit_scaling(
        self, train_drives: np.ndarray, max_fit_rows: int = 2_000_000
    ) -> None:
        """Median-impute + standardise, with both statistics fit on TRAIN drives only.

        Fit on genuinely observed training drive-days: forward-filled rows repeat values
        and would over-weight drives with patchy reporting. Statistics are estimated
        from at most `max_fit_rows` of them -- a median over two million drive-days is
        already exact to more decimals than anything downstream cares about, and the
        full Seagate matrix does not want to be copied.
        """
        rows = self.drive_rows(train_drives)
        fit_rows = rows[self.observed[rows]]
        if fit_rows.size == 0:
            raise ValueError("no observed training rows to fit the scaler on")
        if fit_rows.size > max_fit_rows:
            rng = np.random.default_rng(self.cfg.random_state)
            fit_rows = np.sort(rng.choice(fit_rows, size=max_fit_rows, replace=False))
        block = self.values[fit_rows]

        with np.errstate(invalid="ignore"):
            medians = np.nanmedian(block, axis=0)
        medians = np.nan_to_num(medians, nan=0.0).astype(np.float32)

        # Column-at-a-time so the NaN mask never materialises for the whole matrix.
        for j in range(self.values.shape[1]):
            col = self.values[:, j]
            nan_j = np.isnan(col)
            if nan_j.any():
                col[nan_j] = medians[j]
                self.values[:, j] = col

        block = self.values[fit_rows]
        mean = block.mean(axis=0).astype(np.float32)
        std = block.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        mean[~self.scaled_columns] = 0.0
        std[~self.scaled_columns] = 1.0

        self.values = (self.values - mean) / std
        self.feature_medians, self.feature_mean, self.feature_std = medians, mean, std
        self._scaled = True

    def apply_scaling(
        self, medians: np.ndarray, mean: np.ndarray, std: np.ndarray
    ) -> None:
        """Apply statistics fit on a previous run instead of fitting new ones.

        This is the inference path. Re-fitting the scaler on whatever data happens to be
        in front of the model at serving time is a silent train/serve skew: the same
        drive would score differently depending on which other drives were loaded
        alongside it. The training run's medians, means and standard deviations are
        saved to the preprocessing config, and this replays them exactly.
        """
        medians = np.asarray(medians, dtype=np.float32)
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32).copy()
        n_cols = self.values.shape[1]
        if not (len(medians) == len(mean) == len(std) == n_cols):
            raise ValueError(
                f"saved scaler has {len(mean)} columns but this store has {n_cols}; "
                f"the feature set changed since the model was trained"
            )
        std[std < 1e-6] = 1.0

        for j in range(n_cols):
            col = self.values[:, j]
            nan_j = np.isnan(col)
            if nan_j.any():
                col[nan_j] = medians[j]
                self.values[:, j] = col

        self.values = (self.values - mean) / std
        self.feature_medians, self.feature_mean, self.feature_std = medians, mean, std
        self._scaled = True

    def to_original_units(self, window: np.ndarray) -> np.ndarray:
        """Undo scaling (and log1p on level channels) for a (T, F) window, for display."""
        if not self._scaled:
            return window
        out = window * self.feature_std + self.feature_mean
        lvl = np.zeros(len(self.feature_names), dtype=bool)
        lvl[: self.base_feature_count] = True
        undo = lvl & self.log_columns
        out[:, undo] = np.expm1(out[:, undo])
        return out

    # -- indexing -------------------------------------------------------------

    def drive_rows(self, drives: np.ndarray) -> np.ndarray:
        """Row indices of every dense day belonging to `drives`."""
        drives = np.asarray(drives, dtype=np.int64)
        return np.concatenate(
            [np.arange(self.offsets[g], self.offsets[g + 1]) for g in drives]
        )

    def max_start(self, g: int) -> int:
        return int(max(0, self.lengths[g] - self.cfg.window_days))

    def n_windows(self, g: int) -> int:
        return self.max_start(g) + 1

    def window_label(self, g: int, start: int) -> int:
        """1 if a failure event falls in [start, start + window_days - 1 + horizon]."""
        length = int(self.lengths[g])
        end = min(start + self.cfg.window_days - 1 + self.cfg.horizon_days, length - 1)
        pos = self.failure_positions[g]
        if pos.size == 0:
            return 0
        return int(np.any((pos >= start) & (pos <= end)))

    def positive_starts(self, g: int) -> np.ndarray:
        """Every window start for drive `g` whose window is labelled 1."""
        pos = self.failure_positions[g]
        if pos.size == 0:
            return np.empty(0, dtype=np.int64)
        hi = self.max_start(g)
        span = self.cfg.window_days - 1 + self.cfg.horizon_days
        starts = set()
        for f in pos:
            lo_s = max(0, int(f) - span)
            hi_s = min(hi, int(f))
            if lo_s <= hi_s:
                starts.update(range(lo_s, hi_s + 1))
        return np.array(sorted(starts), dtype=np.int64)

    def window_dates(self, g: int, start: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Calendar dates of the first and last day *represented* by the window.

        For a drive shorter than the window the returned span is the drive's own span;
        the missing head of the window is padding, not calendar time.
        """
        length = int(self.lengths[g])
        first = _EPOCH + pd.Timedelta(days=int(self.start_days[g] + start))
        last_pos = min(start + self.cfg.window_days - 1, length - 1)
        last = _EPOCH + pd.Timedelta(days=int(self.start_days[g] + last_pos))
        return first, last

    def get_window(self, g: int, start: int) -> tuple[np.ndarray, int, dict]:
        """(window, label, meta) for drive `g` starting at within-drive day `start`.

        Short drives are right-aligned into the window: the padding sits at the front so
        that the last row of X is always the drive's most recent day, which is what the
        head of the model sees as "now".
        """
        cfg = self.cfg
        lo, hi = int(self.offsets[g]), int(self.offsets[g + 1])
        length = hi - lo
        take = min(cfg.window_days, length - start)
        chunk = self.values[lo + start : lo + start + take]

        if take == cfg.window_days:
            window = chunk.astype(np.float32, copy=True)
            n_pad = 0
        else:
            window = np.zeros((cfg.window_days, self.values.shape[1]), dtype=np.float32)
            window[cfg.window_days - take :] = chunk
            n_pad = cfg.window_days - take

        label = self.window_label(g, start)
        first, last = self.window_dates(g, start)
        n_observed = int(self.observed[lo + start : lo + start + take].sum())
        meta = {
            "drive_id": str(self.drive_ids[g]),
            "drive_index": int(g),
            "start": int(start),
            "start_date": first.strftime("%Y-%m-%d"),
            "end_date": last.strftime("%Y-%m-%d"),
            "label": int(label),
            "n_real_days": n_observed,
            "n_filled_days": int(take - n_observed),
            "n_padded_days": int(n_pad),
            "failure_day_in_window": (
                int(self.failure_positions[g][0] - start)
                if label and self.failure_positions[g].size else -1
            ),
        }
        return window, label, meta

    def summary(self) -> str:
        total_days = int(self.lengths.sum())
        real = int(self.observed.sum())
        return (
            f"DriveSeriesStore: {self.n_drives:,} drives, {total_days:,} dense drive-days "
            f"({real:,} observed, {total_days - real:,} forward-filled gaps), "
            f"{len(self.feature_names)} features, "
            f"{int(self.drive_has_failure.sum()):,} drives with a failure event"
        )


# ---------------------------------------------------------------------------
# 3. Grouped split by drive id
# ---------------------------------------------------------------------------

def grouped_drive_split(store: DriveSeriesStore, cfg: DataConfig) -> dict[str, np.ndarray]:
    """Split *drive indices* into train/val/test with GroupShuffleSplit.

    Every window of a drive is derived from that drive's own rows, so splitting at the
    drive level is what keeps windows of one disk out of two partitions. With
    `stratify_split_by_failure` the failing and healthy pools are split separately --
    still grouped (a drive is in exactly one split), but each split is guaranteed its
    share of the rare positive drives.
    """
    all_idx = np.arange(store.n_drives)

    def _split(idx: np.ndarray, test_size: float, seed: int):
        if len(idx) < 2 or test_size <= 0:
            return idx, np.empty(0, dtype=np.int64)
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        a, b = next(gss.split(idx, groups=store.drive_ids[idx]))
        return idx[a], idx[b]

    if cfg.stratify_split_by_failure:
        pools = [all_idx[store.drive_has_failure], all_idx[~store.drive_has_failure]]
    else:
        pools = [all_idx]

    train, val, test = [], [], []
    for pool in pools:
        pool_tv, pool_test = _split(pool, cfg.test_size, cfg.random_state)
        pool_train, pool_val = _split(pool_tv, cfg.val_size, cfg.random_state)
        train.append(pool_train)
        val.append(pool_val)
        test.append(pool_test)

    splits = {
        "train": np.sort(np.concatenate(train)),
        "val": np.sort(np.concatenate(val)),
        "test": np.sort(np.concatenate(test)),
    }

    ids = {k: set(store.drive_ids[v]) for k, v in splits.items()}
    assert not (ids["train"] & ids["val"]), "drive id in both train and val"
    assert not (ids["train"] & ids["test"]), "drive id in both train and test"
    assert not (ids["val"] & ids["test"]), "drive id in both val and test"
    return splits


# ---------------------------------------------------------------------------
# 4. Dataset / DataLoader
# ---------------------------------------------------------------------------

class DriveWindowDataset(Dataset):
    """Fixed-length windows of one drive's telemetry, labelled by failure-in-window.

    Two modes:

    ``random``     one sample per drive per epoch (or `samples_per_drive`), the window
                   start drawn at random inside that drive's history. Used for
                   training: over many epochs the model sees a drive's whole life
                   without ever being handed the same window twice in a row.
    ``enumerate``  every `stride` days a window, plus always the drive's most recent
                   window. Deterministic, so validation and test scores are comparable
                   epoch to epoch and run to run.

    Randomness in ``random`` mode is derived from (seed, epoch, index) rather than from
    global NumPy state, which makes __getitem__ pure: the same index yields the same
    window whether it is fetched by the training loop, by an inspection helper, or by
    DataLoader worker #3.

    On positive rarity
    ------------------
    Backblaze marks `failure == 1` on a drive's *last* reported day, so under the strict
    in-window definition (`horizon_days = 0`) exactly one window per failing drive is
    positive -- the one ending on its final day. A uniformly drawn window from a
    300-day failing drive is therefore positive with probability ~1/211. Two knobs
    counteract that: `positive_ratio` (pick that drive's positive window with a given
    probability) and, at the loader level, `make_balanced_sampler` (draw failing drives
    more often). Setting `DataConfig.horizon_days > 0` instead re-frames the task as
    "does this drive fail within N days of the window's end", which widens the positive
    region rather than resampling it.
    """

    def __init__(
        self,
        store: DriveSeriesStore,
        drives: np.ndarray,
        mode: str = "enumerate",
        split_name: str = "",
        samples_per_drive: int = 1,
        positive_ratio: float | None = None,
        stride: int | None = None,
        seed: int = RANDOM_STATE,
        return_meta: bool = True,
    ):
        if mode not in ("random", "enumerate"):
            raise ValueError(f"unknown mode {mode!r}")
        self.store = store
        self.drives = np.asarray(drives, dtype=np.int64)
        self.mode = mode
        self.split_name = split_name or mode
        self.samples_per_drive = max(1, samples_per_drive)
        self.positive_ratio = positive_ratio
        self.stride = max(1, stride or store.cfg.stride_days)
        self.seed = seed
        self.return_meta = return_meta
        self.epoch = 0
        self.window_days = store.cfg.window_days
        self.num_features = len(store.feature_names)
        self.feature_names = list(store.feature_names)

        # Per-drive window bookkeeping. positive_starts() only does real work for the
        # handful of drives that carry a failure event.
        self._pos_starts: dict[int, np.ndarray] = {}
        self._neg_starts: dict[int, np.ndarray] = {}
        for g in self.drives:
            if store.drive_has_failure[g]:
                pos = store.positive_starts(int(g))
                self._pos_starts[int(g)] = pos
                alls = np.arange(store.n_windows(int(g)), dtype=np.int64)
                self._neg_starts[int(g)] = np.setdiff1d(alls, pos, assume_unique=True)

        if mode == "enumerate":
            pairs, labels = [], []
            for g in self.drives:
                g = int(g)
                hi = store.max_start(g)
                starts = list(range(0, hi + 1, self.stride))
                if starts[-1] != hi:
                    starts.append(hi)  # never skip the window ending on the last day
                for s in starts:
                    pairs.append((g, s))
                    labels.append(store.window_label(g, s))
            self.index = np.array(pairs, dtype=np.int64).reshape(-1, 2)
            self.labels = np.array(labels, dtype=np.int64)
        else:
            self.index = None
            self.labels = None

    # -- sizing ---------------------------------------------------------------

    def __len__(self) -> int:
        if self.mode == "enumerate":
            return len(self.index)
        return len(self.drives) * self.samples_per_drive

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the random windows. Call once per epoch before iterating."""
        self.epoch = int(epoch)

    # -- window choice --------------------------------------------------------

    def locate(self, idx: int) -> tuple[int, int]:
        """(drive index, window start) for dataset index `idx`."""
        if self.mode == "enumerate":
            g, s = self.index[idx]
            return int(g), int(s)

        g = int(self.drives[idx // self.samples_per_drive])
        rng = np.random.default_rng([self.seed, self.epoch, int(idx)])
        pos = self._pos_starts.get(g)
        if self.positive_ratio is not None and pos is not None and pos.size:
            if rng.random() < self.positive_ratio:
                return g, int(pos[rng.integers(0, pos.size)])
            neg = self._neg_starts[g]
            if neg.size:
                return g, int(neg[rng.integers(0, neg.size)])
        return g, int(rng.integers(0, self.store.max_start(g) + 1))

    def label_of(self, idx: int) -> int:
        """The label for a dataset index without materialising the window."""
        if self.mode == "enumerate":
            return int(self.labels[idx])
        g, s = self.locate(idx)
        return self.store.window_label(g, s)

    def describe(self, idx: int) -> dict:
        """Metadata for a sample without going through tensor conversion."""
        g, s = self.locate(idx)
        _, _, meta = self.store.get_window(g, s)
        meta["dataset_index"] = int(idx)
        meta["split"] = self.split_name
        return meta

    def raw_window(self, idx: int) -> tuple[np.ndarray, np.ndarray, dict]:
        """(scaled window, window in original units, meta) -- for inspection."""
        g, s = self.locate(idx)
        window, _, meta = self.store.get_window(g, s)
        meta["dataset_index"] = int(idx)
        meta["split"] = self.split_name
        return window, self.store.to_original_units(window), meta

    # -- the actual sample ----------------------------------------------------

    def __getitem__(self, idx: int):
        g, s = self.locate(idx)
        window, label, meta = self.store.get_window(g, s)
        x = torch.from_numpy(window)                      # (window_days, num_features)
        y = torch.tensor([float(label)], dtype=torch.float32)
        if not self.return_meta:
            return x, y
        meta["dataset_index"] = int(idx)
        meta["split"] = self.split_name
        return x, y, meta

    # -- class balance --------------------------------------------------------

    @property
    def drive_can_be_positive(self) -> np.ndarray:
        """Per dataset index: can this index ever produce a positive window?"""
        if self.mode == "enumerate":
            return self.labels.astype(bool)
        flags = np.array(
            [bool(self._pos_starts.get(int(g), np.empty(0)).size) for g in self.drives],
            dtype=bool,
        )
        return np.repeat(flags, self.samples_per_drive)

    def expected_positive_rate(self) -> float:
        """Analytic positive prevalence -- exact for `enumerate`, in expectation for
        `random` (averaging the per-drive probability of drawing a positive window)."""
        if self.mode == "enumerate":
            return float(self.labels.mean()) if len(self.labels) else 0.0
        probs = []
        for g in self.drives:
            g = int(g)
            pos = self._pos_starts.get(g)
            if pos is None or pos.size == 0:
                probs.append(0.0)
            elif self.positive_ratio is not None:
                # A drive whose every window is positive (one shorter than the window
                # length, say) is drawn as a positive whatever the ratio says.
                has_neg = self._neg_starts[g].size > 0
                probs.append(float(self.positive_ratio) if has_neg else 1.0)
            else:
                probs.append(pos.size / self.store.n_windows(g))
        return float(np.mean(probs)) if probs else 0.0

    def label_report(self) -> str:
        rate = self.expected_positive_rate()
        n = len(self)
        drives_pos = int(
            sum(1 for g in self.drives if self._pos_starts.get(int(g), np.empty(0)).size)
        )
        kind = "exact" if self.mode == "enumerate" else "expected"
        return (
            f"{self.split_name:<5} | {self.mode:<9} | {n:>8,} windows | "
            f"{len(self.drives):>6,} drives ({drives_pos:,} with a positive window) | "
            f"positive rate {100 * rate:6.3f}% ({kind})"
        )


def window_collate(batch):
    """Stack (x, y) and keep the per-sample meta dicts as a plain list."""
    if len(batch[0]) == 2:
        xs, ys = zip(*batch)
        return torch.stack(xs), torch.stack(ys)
    xs, ys, metas = zip(*batch)
    return torch.stack(xs), torch.stack(ys), list(metas)


def make_balanced_sampler(
    dataset: DriveWindowDataset, pos_fraction: float = 0.3, seed: int = RANDOM_STATE
) -> WeightedRandomSampler:
    """Oversample indices that can yield a positive window.

    Draws roughly `pos_fraction` of each batch from failure-carrying drives. In `random`
    mode combine with `positive_ratio` -- the realised positive share is about
    `pos_fraction * positive_ratio`, since a failing drive still only produces its
    positive window `positive_ratio` of the time.

    Do NOT stack this with a large `pos_weight`: correcting the imbalance twice was the
    single biggest thing holding this project's earlier models back (see
    TOSHIBA_PIPELINE.md). Pick the sampler or the loss weight, then measure on
    validation.
    """
    can_pos = dataset.drive_can_be_positive
    n_pos, n_neg = int(can_pos.sum()), int((~can_pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("balanced sampler needs both failing and healthy drives")
    weights = np.where(can_pos, pos_fraction / n_pos, (1 - pos_fraction) / n_neg)
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )


@dataclass
class WindowBundle:
    """Everything the training code needs, built once from the raw frame."""

    store: DriveSeriesStore
    cfg: DataConfig
    splits: dict[str, np.ndarray]
    datasets: dict[str, DriveWindowDataset]
    pos_weight: float

    @property
    def feature_names(self) -> list[str]:
        return list(self.store.feature_names)

    @property
    def num_features(self) -> int:
        return len(self.store.feature_names)


def build_window_bundle(
    df: pd.DataFrame, cfg: DataConfig, verbose: bool = True
) -> WindowBundle:
    """Raw long frame -> scaled store -> grouped split -> three window datasets."""
    t0 = time.time()
    store = DriveSeriesStore(df, cfg)
    if verbose:
        print(store.summary())

    splits = grouped_drive_split(store, cfg)
    store.fit_scaling(splits["train"])

    datasets = {
        "train": DriveWindowDataset(
            store, splits["train"], mode="random", split_name="train",
            samples_per_drive=cfg.samples_per_drive,
            positive_ratio=cfg.train_positive_ratio, seed=cfg.random_state,
        ),
        "val": DriveWindowDataset(
            store, splits["val"], mode="enumerate", split_name="val",
            stride=cfg.stride_days, seed=cfg.random_state,
        ),
        "test": DriveWindowDataset(
            store, splits["test"], mode="enumerate", split_name="test",
            stride=cfg.stride_days, seed=cfg.random_state,
        ),
    }

    rate = datasets["train"].expected_positive_rate()
    pos_weight = (1.0 - rate) / rate if rate > 0 else 1.0

    if verbose:
        print(f"\nWindow datasets (window_days={cfg.window_days}, "
              f"horizon_days={cfg.horizon_days}, stride_days={cfg.stride_days}):")
        for name in ("train", "val", "test"):
            print("  " + datasets[name].label_report())
        print(f"\nGrouped split integrity: no drive id is shared between splits "
              f"(asserted in grouped_drive_split).")
        print(f"Raw pos_weight from the training positive rate: {pos_weight:,.1f}")
        print(f"Bundle built in {time.time() - t0:.1f}s")

    return WindowBundle(store=store, cfg=cfg, splits=splits, datasets=datasets,
                        pos_weight=pos_weight)


def build_loaders(
    bundle: WindowBundle, tcfg: TrainConfig, device: torch.device
) -> dict[str, DataLoader]:
    """DataLoaders for the three splits, with the chosen imbalance strategy applied."""
    pin = device.type == "cuda"
    train_ds = bundle.datasets["train"]

    if tcfg.sampler == "balanced":
        sampler = make_balanced_sampler(train_ds, tcfg.balanced_pos_fraction, tcfg.seed)
        shuffle = False
    else:
        sampler, shuffle = None, True

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=tcfg.batch_size, sampler=sampler, shuffle=shuffle,
            num_workers=tcfg.num_workers, pin_memory=pin, collate_fn=window_collate,
            drop_last=False,
        )
    }
    for name in ("val", "test"):
        loaders[name] = DataLoader(
            bundle.datasets[name], batch_size=tcfg.batch_size, shuffle=False,
            num_workers=tcfg.num_workers, pin_memory=pin, collate_fn=window_collate,
        )
    return loaders


# ---------------------------------------------------------------------------
# 5. Data inspection / debug printing
# ---------------------------------------------------------------------------

_RULE = "=" * 88
_THIN = "-" * 88


def _day_labels(meta: dict, window_days: int) -> list[str]:
    """Row labels for a window: PAD for padded steps, the calendar date otherwise."""
    n_pad = meta["n_padded_days"]
    first = pd.Timestamp(meta["start_date"])
    labels = []
    for i in range(window_days):
        if i < n_pad:
            labels.append(f"t{i:02d} PAD")
        else:
            labels.append(f"t{i:02d} {(first + pd.Timedelta(days=i - n_pad)).date()}")
    return labels


def _window_table(
    window: np.ndarray,
    feature_names: Sequence[str],
    meta: dict,
    columns: Sequence[str],
    n_head: int = 3,
    n_tail: int = 3,
    blank_pad: bool = False,
) -> pd.DataFrame:
    """A head/tail slice of one window as a printable DataFrame.

    `blank_pad` is for the original-units view: padded steps are zeros in model space,
    and inverting the scaler would print them as the training mean, which reads like a
    real reading. They are shown as `--` instead.
    """
    window_days = window.shape[0]
    idx = [feature_names.index(c) for c in columns]
    labels = _day_labels(meta, window_days)
    frame = pd.DataFrame(window[:, idx], columns=list(columns), index=labels)
    if blank_pad and meta["n_padded_days"]:
        frame = frame.astype(object)
        frame.iloc[: meta["n_padded_days"], :] = "--"
    if window_days <= n_head + n_tail:
        return frame
    head, tail = frame.iloc[:n_head], frame.iloc[-n_tail:]
    gap = pd.DataFrame(
        [["..."] * len(columns)],
        columns=list(columns),
        index=[f"... {window_days - n_head - n_tail} rows ..."],
    )
    return pd.concat([head.astype(object), gap, tail.astype(object)])


def assert_no_label_leakage(bundle: WindowBundle, verbose: bool = True) -> None:
    """Prove that the failure flag is not reachable from X.

    Two checks: the name is absent from the feature list, and no feature column is a
    copy of the failure array (which would catch an accidental duplicate under another
    name).
    """
    store = bundle.store
    names = store.feature_names
    offenders = [n for n in names if "failure" in n.lower() or n.lower() == "target"]
    assert not offenders, f"label-like columns present in X: {offenders}"

    fail = store.failure.astype(np.float32)
    if fail.any():
        for j, name in enumerate(names):
            col = store.values[:, j]
            # Only a 0/1-valued column could be a copy of the flag; skip the rest
            # without touching every row of a multi-million-row matrix twice.
            if col.min() < -1e-6 or col.max() > 1 + 1e-6:
                continue
            if np.allclose(col, fail, atol=1e-6):
                raise AssertionError(f"feature {name!r} is identical to the failure flag")

    if verbose:
        print(_RULE)
        print("LEAKAGE GUARD")
        print(_RULE)
        print(f"  'failure' present in X feature names ......... "
              f"{any(n == 'failure' for n in names)}")
        print(f"  label-like column names in X ................. {offenders or 'none'}")
        print(f"  any X column identical to the failure flag ... False")
        print(f"  the failure flag lives in store.failure, shape {store.failure.shape}, "
              f"used only by window_label()")
        print(f"\n  X carries {len(names)} channels:")
        base = store.base_feature_count
        print(f"    levels ({base}): {names[:base]}")
        rest = names[base:]
        for lag in bundle.cfg.delta_lags:
            block = [n for n in rest if n.endswith(f"_d{lag}")]
            print(f"    deltas d{lag} ({len(block)}): {block}")
        if "observed" in names:
            print("    mask (1): ['observed']  -- 1 = real reading, 0 = filled gap / pad")


def inspect_split_integrity(bundle: WindowBundle) -> None:
    """Print the grouped-split evidence: drive counts and empty pairwise overlaps."""
    store = bundle.store
    print(_RULE)
    print("SPLIT INTEGRITY (grouped by drive_id)")
    print(_RULE)
    ids = {k: set(store.drive_ids[v]) for k, v in bundle.splits.items()}
    for name, drives in bundle.splits.items():
        n_fail = int(store.drive_has_failure[drives].sum())
        days = int(store.lengths[drives].sum())
        print(f"  {name:<5}: {len(drives):>7,} drives  {n_fail:>5,} with a failure  "
              f"{days:>10,} drive-days")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        print(f"  overlap {a:<5} n {b:<5}: {len(ids[a] & ids[b])} drive ids")
    print("  -> windows from one disk can never appear in two splits")


def inspect_windows(
    dataset: DriveWindowDataset,
    n_samples: int = 3,
    indices: Sequence[int] | None = None,
    columns: Sequence[str] | None = None,
    n_head: int = 3,
    n_tail: int = 3,
    show_original_units: bool = True,
) -> None:
    """Print representative windows straight out of the Dataset.

    Shows, per sample: drive id, window start/end dates, tensor shape, the assigned
    binary label, how much of the window is real vs. imputed vs. padded, and a slice of
    the actual feature values across the 90 days -- in scaled model units and, if asked,
    back in original units (sectors, hours, degrees).
    """
    store = dataset.store
    names = dataset.feature_names
    if columns is None:
        base = names[: store.base_feature_count][:6]
        columns = base + (["observed"] if "observed" in names else [])

    if indices is None:
        # Show a mix: prefer at least one positive window if the split has any.
        rng = np.random.default_rng(dataset.seed)
        pool = rng.choice(len(dataset), size=min(len(dataset), 4000), replace=False)
        labels = [dataset.label_of(int(i)) for i in pool]
        pos = [i for i, y in zip(pool, labels) if y == 1]
        neg = [i for i, y in zip(pool, labels) if y == 0]
        n_pos = min(len(pos), max(1, n_samples // 2)) if pos else 0
        indices = [int(i) for i in pos[:n_pos]] + [int(i) for i in neg[: n_samples - n_pos]]
        indices = indices[:n_samples]

    print(_RULE)
    print(f"DATASET INSPECTION -- split '{dataset.split_name}' (mode={dataset.mode})")
    print(_RULE)
    print(f"  windows in split ..... {len(dataset):,}")
    print(f"  window length ........ {dataset.window_days} days")
    print(f"  features per day ..... {dataset.num_features}")
    print(f"  sample X shape ....... ({dataset.window_days}, {dataset.num_features})")
    print(f"  label rule ........... y=1 if failure in [start, end"
          f"{f' + {store.cfg.horizon_days}d' if store.cfg.horizon_days else ''}]")

    for k, idx in enumerate(indices, start=1):
        window, original, meta = dataset.raw_window(int(idx))
        x, y = dataset[int(idx)][0], dataset[int(idx)][1]
        print(_THIN)
        print(f"  sample {k}/{len(indices)}  (dataset index {meta['dataset_index']})")
        print(f"    drive_id ......... {meta['drive_id']}")
        print(f"    window ........... {meta['start_date']} -> {meta['end_date']}  "
              f"(start day {meta['start']} of this drive's history)")
        print(f"    X tensor ......... {tuple(x.shape)}  dtype={x.dtype}")
        print(f"    y ................ {int(y.item())}  "
              f"({'FAILURE inside the window' if int(y.item()) else 'no failure'})")
        if meta["failure_day_in_window"] >= 0:
            print(f"    failure lands on . t{meta['failure_day_in_window']:02d} "
                  f"of the {dataset.window_days}-day window")
        print(f"    composition ...... {meta['n_real_days']} real readings, "
              f"{meta['n_filled_days']} forward-filled calendar gaps, "
              f"{meta['n_padded_days']} padded steps")
        print(f"\n    X (scaled, model input) -- {len(columns)} of "
              f"{dataset.num_features} channels:")
        table = _window_table(window, names, meta, columns, n_head, n_tail)
        print(table.to_string(float_format=lambda v: f"{v:9.3f}"))
        if show_original_units:
            print(f"\n    same rows in original units "
                  f"(log1p undone; 'failure' is nowhere in this table):")
            table = _window_table(original, names, meta, columns, n_head, n_tail,
                                  blank_pad=True)
            print(table.to_string(float_format=lambda v: f"{v:12.2f}"))
    print(_THIN)


def inspect_batch(batch, tag: str, n_show: int = 2, n_days: int = 4, n_feats: int = 6) -> None:
    """Print one batch as it comes out of the DataLoader."""
    if len(batch) == 3:
        xb, yb, metas = batch
    else:
        xb, yb = batch
        metas = None

    y = yb.detach().cpu().numpy().ravel()
    n_pos = int(y.sum())
    x = xb.detach().cpu()

    print(_RULE)
    print(f"BATCH INSPECTION -- {tag}")
    print(_RULE)
    print(f"  X batch .............. {tuple(xb.shape)}  dtype={xb.dtype}  "
          f"(batch, window_days, num_features)")
    print(f"  y batch .............. {tuple(yb.shape)}  dtype={yb.dtype}")
    print(f"  label distribution ... 0 -> {len(y) - n_pos:,} ({100 * (1 - y.mean()):.2f}%)   "
          f"1 -> {n_pos:,} ({100 * y.mean():.2f}%)")
    print(f"  X stats .............. min {x.min():.3f}  max {x.max():.3f}  "
          f"mean {x.mean():.3f}  std {x.std():.3f}")
    print(f"  non-finite values .... {int((~torch.isfinite(x)).sum())}")
    if metas:
        for m in metas[:n_show]:
            print(f"    sample: drive {m['drive_id']}  {m['start_date']} -> {m['end_date']}"
                  f"  y={m['label']}  real/filled/pad="
                  f"{m['n_real_days']}/{m['n_filled_days']}/{m['n_padded_days']}")
    print(f"  X[0, :{n_days}, :{n_feats}] =")
    slice_ = x[0, :n_days, :n_feats].numpy()
    for row_i, row in enumerate(slice_):
        print("    t{:02d}  ".format(row_i) + "  ".join(f"{v:8.3f}" for v in row))
    print(_THIN)


def run_sanity_checks(
    bundle: WindowBundle, loaders: dict[str, DataLoader], n_samples: int = 3
) -> None:
    """Everything item 4 of the brief asks for, in one call, before training starts."""
    assert_no_label_leakage(bundle)
    inspect_split_integrity(bundle)
    print(_RULE)
    print("WINDOW COUNTS")
    print(_RULE)
    for name in ("train", "val", "test"):
        print("  " + bundle.datasets[name].label_report())
    inspect_windows(bundle.datasets["train"], n_samples=n_samples)
    inspect_windows(bundle.datasets["test"], n_samples=max(1, n_samples // 2),
                    show_original_units=False)
    for split in ("train", "test"):
        batch = next(iter(loaders[split]))
        inspect_batch(batch, tag=f"first batch of the '{split}' DataLoader")


# ---------------------------------------------------------------------------
# 6. Sequence models -- (B, window_days, num_features) -> (B, 1) logit
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Two dilated convolutions with a residual connection, length-preserving."""

    def __init__(self, in_ch: int, out_ch: int, dilation: int, kernel_size: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):                      # x: (B, C, T)
        h = self.drop(self.act(self.bn1(self.conv1(x))))
        h = self.drop(self.act(self.bn2(self.conv2(h))))
        return self.act(h + self.down(x))


class CNN1DWindowClassifier(nn.Module):
    """Dilated 1D CNN (TCN-style) over the 90-day window.

    Dilations 1-16 across five blocks give a receptive field of 125 days, so the head
    sees the entire window at once. Pooling concatenates mean, max and the final
    timestep: mean carries the window's level, max the worst moment, last the drive's
    most recent state -- which is the one the operator actually acts on.
    """

    def __init__(self, num_features: int, window_days: int = 90, hidden_dim: int = 64,
                 dropout: float = 0.2, dilations: Sequence[int] = (1, 2, 4, 8, 16)):
        super().__init__()
        blocks, in_ch = [], num_features
        for d in dilations:
            blocks.append(TemporalBlock(in_ch, hidden_dim, dilation=d, dropout=dropout))
            in_ch = hidden_dim
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):                      # x: (B, T, F)
        h = self.blocks(x.transpose(1, 2))     # (B, C, T)
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2), h[:, :, -1]], dim=1)
        return self.head(pooled)               # (B, 1)


class GRUWindowClassifier(nn.Module):
    """Bidirectional GRU, pooled over time plus the final hidden state."""

    def __init__(self, num_features: int, window_days: int = 90, hidden_dim: int = 64,
                 dropout: float = 0.2, num_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=num_features, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(4 * hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):                      # x: (B, T, F)
        out, _ = self.rnn(x)                   # (B, T, 2H)
        pooled = torch.cat([out.mean(dim=1), out[:, -1, :]], dim=1)
        return self.head(pooled)


class TransformerWindowClassifier(nn.Module):
    """Transformer encoder with learned positional embeddings over the 90 days."""

    def __init__(self, num_features: int, window_days: int = 90, hidden_dim: int = 64,
                 dropout: float = 0.2, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.proj = nn.Linear(num_features, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, window_days, hidden_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        # enable_nested_tensor has no effect with norm_first=True and only emits a
        # warning; every window is the same length, so there is nothing to nest anyway.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):                      # x: (B, T, F)
        h = self.norm(self.encoder(self.proj(x) + self.pos))
        pooled = torch.cat([h.mean(dim=1), h[:, -1, :]], dim=1)
        return self.head(pooled)


ARCHITECTURES = {
    "cnn": CNN1DWindowClassifier,
    "gru": GRUWindowClassifier,
    "transformer": TransformerWindowClassifier,
}


def build_model(num_features: int, window_days: int, tcfg: TrainConfig) -> nn.Module:
    if tcfg.arch not in ARCHITECTURES:
        raise ValueError(f"arch must be one of {sorted(ARCHITECTURES)}, got {tcfg.arch!r}")
    return ARCHITECTURES[tcfg.arch](
        num_features=num_features, window_days=window_days,
        hidden_dim=tcfg.hidden_dim, dropout=tcfg.dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 7. Losses for extreme imbalance
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal loss for binary classification with logits.

    Down-weights easy (already well-classified) examples via (1-pt)**gamma so training
    focuses on the hard, rare positives -- an alternative to class-weighted resampling.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - pt).pow(self.gamma) * bce).mean()


def make_criterion(tcfg: TrainConfig, pos_weight: float, device: torch.device,
                   verbose: bool = True) -> nn.Module:
    """Pick the loss, and refuse to correct the imbalance twice.

    A balanced sampler already rebalances the batches; stacking a large `pos_weight` on
    top of it inflates the positive gradient a second time. That exact double correction
    is what capped the earlier models in this project (TOSHIBA_PIPELINE.md), so when the
    sampler is on, the pos_weight is dropped back to 1 and the choice is announced.
    """
    if tcfg.loss == "focal":
        if verbose:
            print(f"Loss: FocalLoss(alpha={tcfg.focal_alpha}, gamma={tcfg.focal_gamma})")
        return FocalLoss(tcfg.focal_alpha, tcfg.focal_gamma)

    if tcfg.loss == "bce" or tcfg.sampler == "balanced":
        if verbose:
            reason = ("balanced sampler already rebalances the batches"
                      if tcfg.sampler == "balanced" else "requested")
            print(f"Loss: BCEWithLogitsLoss(pos_weight=1.0)  [{reason}]")
        return nn.BCEWithLogitsLoss()

    w = float(min(pos_weight, tcfg.pos_weight_cap))
    if verbose:
        capped = " (capped)" if pos_weight > tcfg.pos_weight_cap else ""
        print(f"Loss: BCEWithLogitsLoss(pos_weight={w:.1f}{capped}; "
              f"raw class ratio {pos_weight:,.1f})")
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w], device=device))


# ---------------------------------------------------------------------------
# 8. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """PR-AUC, ROC-AUC and the threshold-dependent trio, plus the confusion counts.

    Both AUCs come back as NaN when the split happens to contain a single class -- that
    is a property of the split, not a score of 0, and averaging a 0 into a report would
    misrepresent it.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    probs = np.asarray(probs).ravel()
    preds = (probs >= threshold).astype(int)
    both_classes = 0 < y_true.sum() < len(y_true)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    tn, fp, fn, tp = (
        confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        if len(y_true) else (0, 0, 0, 0)
    )
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probs)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probs)) if both_classes else float("nan"),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
    }


def precision_at_k(y_true: np.ndarray, probs: np.ndarray, k: int) -> dict:
    """Precision among the K highest-scoring items -- the metric an operator lives by.

    A maintenance team can pull a fixed number of disks a week. What matters is not the
    score at some threshold but how many of the top K flagged drives were really about
    to fail, so this is reported alongside a `lift` over the base rate and the `best`
    attainable value (min(K, n_pos) / K), which caps precision when there are fewer than
    K positives in the split.
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    probs = np.asarray(probs).ravel()
    n, n_pos = len(y_true), int(y_true.sum())
    k_eff = int(min(k, n))
    if k_eff == 0:
        return {"k": k, "k_effective": 0, "hits": 0, "precision": float("nan"),
                "recall": float("nan"), "best_possible": float("nan"),
                "lift": float("nan")}
    # argpartition, not a full sort: only the identity of the top K matters, and on a
    # test split of a few hundred thousand windows the sort is the expensive part.
    top = np.argpartition(-probs, k_eff - 1)[:k_eff]
    hits = int(y_true[top].sum())
    base = n_pos / n if n else float("nan")
    prec = hits / k_eff
    return {
        "k": k,
        "k_effective": k_eff,
        "hits": hits,
        "precision": prec,
        "recall": hits / n_pos if n_pos else float("nan"),
        "best_possible": min(k_eff, n_pos) / k_eff,
        "lift": prec / base if base else float("nan"),
    }


def precision_at_k_table(
    y_true: np.ndarray, probs: np.ndarray, ks: Sequence[int] = (10, 25, 50, 100)
) -> pd.DataFrame:
    return pd.DataFrame([precision_at_k(y_true, probs, k) for k in ks])


def drive_level_scores(
    probs: np.ndarray, targets: np.ndarray, drive_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse many windows per drive to one score and one label per drive.

    Precision@K over *windows* double-counts: a drive contributing 50 windows can fill
    half the top-100 on its own. Ranking drives by their highest-scoring window is the
    form of the metric that matches the decision -- K disks get pulled, not K windows.
    """
    ids = np.asarray(drive_ids, dtype=object)
    order = np.argsort(ids, kind="stable")
    ids_s, probs_s, targ_s = ids[order], np.asarray(probs)[order], np.asarray(targets)[order]
    uniq, starts = np.unique(ids_s, return_index=True)
    max_prob = np.maximum.reduceat(probs_s, starts)
    any_pos = np.maximum.reduceat(targ_s, starts)
    return uniq, max_prob, any_pos.astype(int)


def print_precision_at_k(
    y_true: np.ndarray, probs: np.ndarray, title: str,
    ks: Sequence[int] = (10, 25, 50, 100),
) -> pd.DataFrame:
    tbl = precision_at_k_table(y_true, probs, ks)
    n_pos = int(np.asarray(y_true).sum())
    print(_RULE)
    print(title)
    print(_RULE)
    print(f"  {len(y_true):,} items ranked, {n_pos:,} positive "
          f"(base rate {100 * n_pos / max(len(y_true), 1):.3f}%)")
    print(f"  {'K':>6} | {'hits':>6} | {'precision':>10} | {'best':>7} | "
          f"{'recall@K':>9} | {'lift':>8}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*10}-+-{'-'*7}-+-{'-'*9}-+-{'-'*8}")
    for _, r in tbl.iterrows():
        print(f"  {int(r['k']):>6} | {int(r['hits']):>6} | {r['precision']:>10.4f} | "
              f"{r['best_possible']:>7.4f} | {r['recall']:>9.4f} | {r['lift']:>7.1f}x")
    return tbl


def tune_threshold(y_true: np.ndarray, probs: np.ndarray, grid: int = 200) -> tuple[float, float]:
    """Threshold that maximises F1. Fit on VALIDATION only, then frozen for test."""
    y_true = np.asarray(y_true).ravel().astype(int)
    probs = np.asarray(probs).ravel()
    if y_true.sum() == 0:
        return 0.5, float("nan")
    candidates = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, grid)))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        preds = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1


def format_metrics(name: str, m: dict) -> str:
    return (
        f"{name:<22} PR-AUC {m['pr_auc']:.4f} | ROC-AUC {m['roc_auc']:.4f} | "
        f"P {m['precision']:.4f} | R {m['recall']:.4f} | F1 {m['f1']:.4f} | "
        f"thr {m['threshold']:.4f} | TP {m['tp']} FP {m['fp']} FN {m['fn']} TN {m['tn']}"
    )


def print_metrics_block(title: str, m: dict) -> None:
    print(_RULE)
    print(title)
    print(_RULE)
    print(f"  windows evaluated .... {m['n']:,}  ({m['n_pos']:,} positive, "
          f"base rate {100 * m['base_rate']:.3f}%)")
    print(f"  PR-AUC ............... {m['pr_auc']:.4f}   "
          f"(a random ranker scores {m['base_rate']:.4f})")
    print(f"  ROC-AUC .............. {m['roc_auc']:.4f}")
    print(f"  threshold ............ {m['threshold']:.4f}")
    print(f"  precision ............ {m['precision']:.4f}")
    print(f"  recall ............... {m['recall']:.4f}")
    print(f"  F1 ................... {m['f1']:.4f}")
    print(f"  confusion ............ TP {m['tp']}  FP {m['fp']}  FN {m['fn']}  TN {m['tn']}")


def print_confusion(m: dict, title: str = "CONFUSION MATRIX") -> None:
    """The 2x2 laid out, with the two error rates an operator actually trades off."""
    tp, fp, fn, tn = m["tp"], m["fp"], m["fn"], m["tn"]
    print(_RULE)
    print(f"{title}   (threshold {m['threshold']:.4f})")
    print(_RULE)
    print(f"                    {'predicted 0':>14} {'predicted 1':>14}")
    print(f"    {'actual 0':<10} {tn:>14,} {fp:>14,}   <- {fp:,} false alarms")
    print(f"    {'actual 1':<10} {fn:>14,} {tp:>14,}   <- {tp:,} caught, {fn:,} missed")
    print()
    print(f"    TP {tp:,}   FP {fp:,}   TN {tn:,}   FN {fn:,}")
    print(f"    precision  TP/(TP+FP) = {m['precision']:.4f}   "
          f"({fp:,} wasted call-outs per {tp:,} real catches)")
    print(f"    recall     TP/(TP+FN) = {m['recall']:.4f}   "
          f"({fn:,} failures went unflagged)")
    print(f"    false-alarm rate FP/(FP+TN) = "
          f"{fp / max(fp + tn, 1):.4f}")


# ---------------------------------------------------------------------------
# 9. Training / validation / evaluation
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unpack(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    device: torch.device, threshold: float = 0.5,
                    debug_first_batch: bool = False, tag: str = "eval"):
    """Run a split end to end; returns (loss, metrics, probs, targets, drive_ids).

    `drive_ids` is parallel to `probs` and comes from the per-sample meta dicts, so
    downstream metrics can collapse windows to drives. It is empty when the loader was
    built with `return_meta=False`.
    """
    model.eval()
    total_loss, n_seen = 0.0, 0
    all_probs, all_targets, all_ids = [], [], []
    for i, batch in enumerate(loader):
        xb, yb, metas = _unpack(batch)
        if debug_first_batch and i == 0:
            inspect_batch((xb, yb, metas) if metas else (xb, yb),
                          tag=f"{tag} -- first batch (inference)")
        if metas:
            all_ids.extend(m["drive_id"] for m in metas)
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n_seen += xb.size(0)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(yb.cpu().numpy())

    probs = np.concatenate(all_probs).ravel()
    targets = np.concatenate(all_targets).ravel()
    metrics = compute_metrics(targets, probs, threshold)
    return total_loss / max(n_seen, 1), metrics, probs, targets, np.asarray(all_ids, dtype=object)


def train_model(model: nn.Module, loaders: dict[str, DataLoader], bundle: WindowBundle,
                tcfg: TrainConfig, device: torch.device, verbose: bool = True):
    """Train with early stopping on validation PR-AUC.

    PR-AUC rather than loss or accuracy: with a positive rate in the fractions of a
    percent, accuracy is meaningless and the loss moves with the imbalance correction
    rather than with the ranking that actually gets used.
    """
    criterion = make_criterion(tcfg, bundle.pos_weight, device, verbose=verbose)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr,
                                 weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    train_ds: DriveWindowDataset = bundle.datasets["train"]

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_pr_auc": [],
               "val_roc_auc": [], "val_f1": [], "lr": [], "epoch_seconds": []}
    best_pr_auc, best_state, epochs_no_improve = -1.0, None, 0

    # Printed lazily, just before the first epoch line: the first-batch inspections fire
    # partway through epoch 1 and would otherwise land in the middle of the table.
    header_printed = False

    def _print_header():
        print(_RULE)
        print(f"TRAINING -- {tcfg.arch.upper()}, {count_parameters(model):,} parameters, "
              f"up to {tcfg.max_epochs} epochs, early stop after {tcfg.patience} "
              f"epochs without a val PR-AUC gain")
        print(_RULE)
        print(f"{'epoch':>5} | {'train loss':>10} | {'val loss':>9} | {'val PR-AUC':>10} | "
              f"{'val ROC-AUC':>11} | {'val F1@.5':>9} | {'lr':>8} | {'sec':>6}")
        print(f"{'-'*5}-+-{'-'*10}-+-{'-'*9}-+-{'-'*10}-+-{'-'*11}-+-{'-'*9}-+-"
              f"{'-'*8}-+-{'-'*6}")

    for epoch in range(1, tcfg.max_epochs + 1):
        # Re-roll every drive's random window: a new epoch is a new view of the fleet.
        train_ds.set_epoch(epoch)
        model.train()
        t0, running, n_seen, seen_pos = time.time(), 0.0, 0, 0

        for i, batch in enumerate(loaders["train"]):
            xb, yb, metas = _unpack(batch)
            if verbose and epoch == 1 and i == 0:
                inspect_batch((xb, yb, metas) if metas else (xb, yb),
                              tag="training step 0 (epoch 1)")
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            seen_pos += int(yb.sum().item())

        train_loss = running / max(n_seen, 1)
        val_loss, val_m, _, _, _ = evaluate_loader(
            model, loaders["val"], criterion, device, threshold=0.5,
            debug_first_batch=(verbose and epoch == 1), tag="validation",
        )
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["pr_auc"] if np.isfinite(val_m["pr_auc"]) else 0.0)
        elapsed = time.time() - t0

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pr_auc"].append(val_m["pr_auc"])
        history["val_roc_auc"].append(val_m["roc_auc"])
        history["val_f1"].append(val_m["f1"])
        history["lr"].append(lr_now)
        history["epoch_seconds"].append(elapsed)
        history.setdefault("train_pos_seen", []).append(seen_pos)
        history.setdefault("train_n_seen", []).append(n_seen)

        score = val_m["pr_auc"] if np.isfinite(val_m["pr_auc"]) else -1.0
        improved = score > best_pr_auc

        if verbose:
            if not header_printed:
                _print_header()
                header_printed = True
            mark = "  <- best" if improved else ""
            print(f"{epoch:>5} | {train_loss:>10.5f} | {val_loss:>9.5f} | "
                  f"{val_m['pr_auc']:>10.4f} | {val_m['roc_auc']:>11.4f} | "
                  f"{val_m['f1']:>9.4f} | {lr_now:>8.2e} | {elapsed:>6.1f}{mark}",
                  flush=True)

        if improved:
            best_pr_auc = score
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= tcfg.patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} "
                          f"(best val PR-AUC {best_pr_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_pr_auc


def evaluate_final(model: nn.Module, loaders: dict[str, DataLoader], bundle: WindowBundle,
                   tcfg: TrainConfig, device: torch.device,
                   pk_ks: Sequence[int] = (10, 25, 50, 100)) -> dict:
    """Tune the decision threshold on validation, then score the test split once.

    The threshold is the only quantity carried from validation into the test report, and
    it is fit before the test split is touched. Everything printed for test -- PR-AUC,
    ROC-AUC, precision, recall, F1, the confusion matrix and the Precision@K table --
    comes from a single pass at that frozen threshold.
    """
    criterion = make_criterion(tcfg, bundle.pos_weight, device, verbose=False)

    _, _, val_probs, val_targets, val_ids = evaluate_loader(
        model, loaders["val"], criterion, device)
    best_t, best_val_f1 = tune_threshold(val_targets, val_probs)
    print(_RULE)
    print("THRESHOLD TUNING (validation only -- the test split is not consulted)")
    print(_RULE)
    print(f"  best F1 threshold .... {best_t:.4f}  (val F1 {best_val_f1:.4f})")

    _, test_m_05, test_probs, test_targets, test_ids = evaluate_loader(
        model, loaders["test"], criterion, device, threshold=0.5,
        debug_first_batch=True, tag="test",
    )
    test_m_tuned = compute_metrics(test_targets, test_probs, best_t)

    print_metrics_block("TEST @ threshold 0.5", test_m_05)
    print_metrics_block(f"TEST @ tuned threshold {best_t:.4f}", test_m_tuned)
    print_confusion(test_m_tuned, f"TEST CONFUSION MATRIX @ {best_t:.4f}")

    pk_window = print_precision_at_k(
        test_targets, test_probs, "TEST PRECISION@K -- ranking windows", ks=pk_ks)
    pk_drive = None
    if len(test_ids):
        _, drive_probs, drive_targets = drive_level_scores(test_probs, test_targets, test_ids)
        pk_drive = print_precision_at_k(
            drive_targets, drive_probs,
            "TEST PRECISION@K -- ranking DRIVES (each drive scored by its worst window)",
            ks=pk_ks,
        )

    val_m = compute_metrics(val_targets, val_probs, best_t)
    print(_RULE)
    print("SUMMARY")
    print(_RULE)
    print("  " + format_metrics("val   @tuned", val_m))
    print("  " + format_metrics("test  @0.50", test_m_05))
    print("  " + format_metrics("test  @tuned", test_m_tuned))
    return {
        "best_threshold": best_t,
        "val": val_m,
        "val_probs": val_probs,
        "val_targets": val_targets,
        "test_at_half": test_m_05,
        "test_at_tuned": test_m_tuned,
        "test_probs": test_probs,
        "test_targets": test_targets,
        "test_drive_ids": test_ids,
        "precision_at_k_windows": pk_window,
        "precision_at_k_drives": pk_drive,
    }


def save_artifacts(model: nn.Module, bundle: WindowBundle, tcfg: TrainConfig,
                   results: dict, prefix: str = "backblaze_window") -> None:
    """Weights + everything needed to rebuild the exact preprocessing at inference."""
    store = bundle.store
    torch.save(model.state_dict(), f"{prefix}_{tcfg.arch}.pth")
    config = {
        "arch": tcfg.arch,
        "window_days": bundle.cfg.window_days,
        "horizon_days": bundle.cfg.horizon_days,
        "stride_days": bundle.cfg.stride_days,
        "feature_names": store.feature_names,
        "base_feature_count": store.base_feature_count,
        "delta_lags": list(bundle.cfg.delta_lags),
        "log1p_columns": list(bundle.cfg.log1p_columns),
        "add_observed_mask": bundle.cfg.add_observed_mask,
        "feature_medians": np.asarray(store.feature_medians).tolist(),
        "feature_mean": np.asarray(store.feature_mean).tolist(),
        "feature_std": np.asarray(store.feature_std).tolist(),
        "hidden_dim": tcfg.hidden_dim,
        "dropout": tcfg.dropout,
        "loss": tcfg.loss,
        "sampler": tcfg.sampler,
        "best_threshold": results["best_threshold"],
        "test_metrics": {k: v for k, v in results["test_at_tuned"].items()},
    }
    with open(f"{prefix}_preprocessing_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"\nSaved {prefix}_{tcfg.arch}.pth and {prefix}_preprocessing_config.json")


def load_artifacts(prefix: str = "backblaze_window90",
                   device: torch.device | None = None) -> tuple[nn.Module, dict]:
    """Rebuild a trained model and its preprocessing contract from disk.

    The config is the contract: it pins the feature list, the window length, the delta
    lags and the exact scaler statistics, so a serving process reproduces the training
    preprocessing rather than guessing at it.
    """
    device = device or torch.device("cpu")
    with open(f"{prefix}_preprocessing_config.json", encoding="utf-8") as fh:
        config = json.load(fh)

    tcfg = TrainConfig(arch=config["arch"], hidden_dim=config["hidden_dim"],
                       dropout=config.get("dropout", 0.2))
    model = build_model(len(config["feature_names"]), config["window_days"], tcfg)
    state = torch.load(f"{prefix}_{config['arch']}.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, config


def data_config_from_saved(config: dict, **overrides) -> DataConfig:
    """The DataConfig that reproduces the training-time windowing, from a saved config."""
    base = config["feature_names"][: config["base_feature_count"]]
    cfg = DataConfig(
        window_days=config["window_days"],
        horizon_days=config.get("horizon_days", 0),
        stride_days=config.get("stride_days", 15),
        min_days=config["window_days"],
        pad_short_drives=False,
        delta_lags=tuple(config["delta_lags"]),
        add_observed_mask=config["add_observed_mask"],
        feature_columns=tuple(base),
        log1p_columns=tuple(config["log1p_columns"]),
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@torch.no_grad()
def score_latest_windows(
    model: nn.Module, store: DriveSeriesStore, device: torch.device | None = None,
    batch_size: int = 512,
) -> np.ndarray:
    """P(failure) for every drive's most recent window -- the "as of today" question.

    One window per drive, ending on its last day of telemetry. This is what an operator
    asks: given everything reported so far, which disks are most likely to go next?
    """
    device = device or torch.device("cpu")
    model.eval()
    n = store.n_drives
    out = np.zeros(n, dtype=np.float32)
    buf, idx = [], []
    for g in range(n):
        window, _, _ = store.get_window(g, store.max_start(g))
        buf.append(window)
        idx.append(g)
        if len(buf) == batch_size or g == n - 1:
            xb = torch.from_numpy(np.stack(buf)).to(device)
            out[idx] = torch.sigmoid(model(xb)).cpu().numpy().ravel()
            buf, idx = [], []
    return out


# ---------------------------------------------------------------------------
# 9b. Window timeline sampling and plots
# ---------------------------------------------------------------------------

def _plt():
    """Imported lazily so the module stays usable in a headless / plotless environment."""
    import matplotlib
    if matplotlib.get_backend().lower() == "agg" or "inline" in matplotlib.get_backend():
        pass
    import matplotlib.pyplot as plt
    return plt


def iter_random_windows(
    store: DriveSeriesStore,
    drives: np.ndarray,
    n_per_drive: int = 1,
    seed: int = RANDOM_STATE,
    epoch: int = 0,
):
    """Yield (drive index, window start) for random windows with unaligned starts.

    The start is drawn uniformly from ``[0, length - window_days]`` on the drive's own
    dense daily calendar, so a window may begin on any day of that drive's history -- a
    Tuesday in the middle of February as readily as the first of a quarter. Nothing here
    snaps to a calendar boundary; the only constraint is that the whole window fits
    inside the drive's record, which is what makes it 90 *continuous* days.

    Reproducible without touching global RNG state: the draw for one (drive, replicate)
    is seeded by (seed, epoch, drive, replicate), the same rule `DriveWindowDataset`
    uses, so this generator and the training loader agree on what they are looking at.
    """
    for g in np.asarray(drives, dtype=np.int64):
        g = int(g)
        hi = store.max_start(g)
        for r in range(n_per_drive):
            rng = np.random.default_rng([seed, epoch, g, r])
            yield g, int(rng.integers(0, hi + 1))


def sample_window_timeline(
    store: DriveSeriesStore,
    drives: np.ndarray,
    n_per_drive: int = 1,
    seed: int = RANDOM_STATE,
    epoch: int = 0,
    split_name: str = "",
) -> pd.DataFrame:
    """A table of randomly drawn windows with their calendar span and composition.

    One row per window: where it starts in calendar time, where it starts as a fraction
    of the drive's own life, how many of its days are genuine readings rather than
    forward-filled gaps or left padding, and its label.
    """
    rows = []
    for g, start in iter_random_windows(store, drives, n_per_drive, seed, epoch):
        _, _, meta = store.get_window(g, start)
        length = int(store.lengths[g])
        max_start = store.max_start(g)
        last_pos = min(start + store.cfg.window_days - 1, length - 1)
        rows.append({
            "split": split_name,
            "drive_id": meta["drive_id"],
            "drive_index": g,
            "start_date": pd.Timestamp(meta["start_date"]),
            "end_date": pd.Timestamp(meta["end_date"]),
            "start_offset_days": start,
            "start_fraction_of_life": start / max_start if max_start else 0.0,
            "drive_record_days": length,
            "n_real_days": meta["n_real_days"],
            "n_filled_days": meta["n_filled_days"],
            "n_padded_days": meta["n_padded_days"],
            "label": meta["label"],
            "days_to_record_end": length - 1 - last_pos,
        })
    return pd.DataFrame(rows)


def describe_window_timeline(tl: pd.DataFrame, cfg: DataConfig) -> None:
    """Print what the sampled windows cover, and check they are not quarter-aligned."""
    print(_RULE)
    print(f"STEP 2 -- RANDOM {cfg.window_days}-DAY CONTINUOUS WINDOW SAMPLING")
    print(_RULE)
    print(f"  windows drawn ........ {len(tl):,} over {tl.drive_id.nunique():,} drives")
    print(f"  positive windows ..... {int(tl.label.sum()):,} "
          f"({100 * tl.label.mean():.3f}%)")
    print(f"  start dates .......... {tl.start_date.min():%Y-%m-%d} .. "
          f"{tl.start_date.max():%Y-%m-%d}  "
          f"({tl.start_date.dt.normalize().nunique():,} distinct start days)")
    print(f"  end dates ............ {tl.end_date.min():%Y-%m-%d} .. "
          f"{tl.end_date.max():%Y-%m-%d}")
    span = (tl.end_date - tl.start_date).dt.days + 1
    pad_note = ("all exactly {0}".format(cfg.window_days) if span.nunique() == 1
                else f"{cfg.window_days} except where a short drive was left-padded")
    print(f"  calendar span ........ {span.min()}..{span.max()} days "
          f"(mean {span.mean():.1f}) -- {pad_note}")
    print(f"  genuine readings ..... mean {tl.n_real_days.mean():.1f}/{cfg.window_days}"
          f" days per window ({tl.n_filled_days.mean():.1f} forward-filled, "
          f"{tl.n_padded_days.mean():.1f} padded)")

    # The alignment check the brief asks for: were windows snapped to calendar quarters,
    # essentially every start would land on a quarter boundary. The baseline is the share
    # of quarter-start days among the calendar days a start could have fallen on -- over
    # a two-year span that is nine days, not four.
    lo, hi = tl.start_date.min(), tl.start_date.max()
    all_days = pd.date_range(lo, hi, freq="D")
    quarter_days = pd.date_range(lo, hi, freq="QS")
    on_quarter = tl.start_date.isin(quarter_days).mean()
    expected = len(quarter_days) / max(len(all_days), 1)
    print(f"  quarter-aligned ...... {100 * on_quarter:.2f}% of starts land on one of "
          f"the {len(quarter_days)} quarter boundaries in range")
    print(f"                         (uniform expectation {100 * expected:.2f}% -- "
          f"starts are NOT snapped to quarters)")
    print(f"  distinct start days .. {tl.start_date.dt.normalize().nunique():,} of "
          f"{len(all_days):,} calendar days in range")
    weekdays = (tl.start_date.dt.day_name().str[:3]
                .value_counts(normalize=True).sort_index())
    print("  weekday spread ....... " + ", ".join(
        f"{d} {100 * v:.1f}%" for d, v in weekdays.items()))


def plot_window_timeline(
    tl: pd.DataFrame, cfg: DataConfig, save_path: str | None = None, show: bool = True
):
    """Six panels describing where the sampled windows sit in time.

    Top row is the calendar view -- when windows start, when they end, and how many are
    live on any given day. Bottom row is the per-drive view: where a window sits along
    its own drive's life, how much of it is genuine telemetry, and how long the drives
    it was drawn from actually ran.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 8.5))
    pos = tl[tl.label == 1]
    BLUE, RED, GREEN = "#3b6ea5", "#c1442e", "#7a9b57"

    ax = axes[0, 0]
    ax.hist(tl.start_date, bins=60, color=BLUE, edgecolor="white", linewidth=0.4)
    for q in pd.date_range(tl.start_date.min(), tl.start_date.max(), freq="QS"):
        ax.axvline(q, color="#999999", linestyle=":", linewidth=0.9, zorder=0)
    ax.set_title("Window START dates\n(dotted = calendar quarter boundaries)", fontsize=10)
    ax.set_xlabel("calendar date")
    ax.set_ylabel("windows")
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    ax = axes[0, 1]
    ax.hist(tl.end_date, bins=60, color=GREEN, edgecolor="white", linewidth=0.4,
            label="all windows")
    if len(pos):
        ax.hist(pos.end_date, bins=60, color=RED, edgecolor="white", linewidth=0.4,
                label="positive")
    ax.set_title("Window END dates", fontsize=10)
    ax.set_xlabel("calendar date")
    ax.set_ylabel("windows")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(fontsize=8)

    # Coverage: how many sampled windows overlap each calendar day.
    ax = axes[0, 2]
    days = pd.date_range(tl.start_date.min(), tl.end_date.max(), freq="D")
    opened = tl.start_date.value_counts().reindex(days, fill_value=0).to_numpy()
    closed = ((tl.end_date + pd.Timedelta(days=1)).value_counts()
              .reindex(days, fill_value=0).to_numpy())
    ax.fill_between(days, np.cumsum(opened - closed), color=BLUE, alpha=0.75)
    ax.set_title("Fleet timeline coverage\n(windows overlapping each calendar day)",
                 fontsize=10)
    ax.set_xlabel("calendar date")
    ax.set_ylabel("windows live")
    ax.tick_params(axis="x", rotation=30, labelsize=8)

    ax = axes[1, 0]
    ax.hist(tl.start_fraction_of_life, bins=40, range=(0, 1), color=BLUE,
            edgecolor="white", linewidth=0.4)
    ax.set_title("Start position along the drive's own life\n"
                 "(0 = earliest valid start, 1 = latest)", fontsize=10)
    ax.set_xlabel("fraction of that drive's available starts")
    ax.set_ylabel("windows")

    ax = axes[1, 1]
    ax.hist(tl.n_real_days, bins=np.arange(0, cfg.window_days + 2) - 0.5, color=GREEN,
            edgecolor="white", linewidth=0.4)
    ax.axvline(cfg.window_days, color=RED, linestyle="--", linewidth=1.2,
               label=f"{cfg.window_days} = fully observed")
    ax.set_title("Genuine daily readings per window\n"
                 "(remainder forward-filled or padded)", fontsize=10)
    ax.set_xlabel("observed days")
    ax.set_ylabel("windows")
    ax.set_yscale("log")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.hist(tl.drive_record_days, bins=50, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(cfg.window_days, color=RED, linestyle="--", linewidth=1.2,
               label=f"{cfg.window_days}-day minimum")
    ax.set_title("Record length of the drives sampled\n"
                 "(longer record = more distinct windows)", fontsize=10)
    ax.set_xlabel("days of telemetry")
    ax.set_ylabel("windows")
    ax.legend(fontsize=8)

    for a in axes.ravel():
        a.grid(alpha=0.25, linewidth=0.6)
        a.set_axisbelow(True)
    fig.suptitle(
        f"{cfg.window_days}-day continuous windows, random unaligned starts -- "
        f"{len(tl):,} windows over {tl.drive_id.nunique():,} drives", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_training_history(history: dict, save_path: str | None = None, show: bool = True):
    """Loss curves and the two validation ranking metrics, epoch by epoch."""
    plt = _plt()
    ep = history.get("epoch") or list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(ep, history["train_loss"], "o-", color="#3b6ea5", label="train")
    ax.plot(ep, history["val_loss"], "o-", color="#c1442e", label="validation")
    ax.set_title("Loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(ep, history["val_pr_auc"], "o-", color="#7a4fa0")
    if len(ep):
        best = int(np.nanargmax(history["val_pr_auc"]))
        ax.axvline(ep[best], color="#999999", linestyle="--", linewidth=1,
                   label=f"best epoch {ep[best]} ({history['val_pr_auc'][best]:.4f})")
        ax.legend(fontsize=8)
    ax.set_title("Validation PR-AUC (early-stopping criterion)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("PR-AUC")

    ax = axes[2]
    ax.plot(ep, history["val_roc_auc"], "o-", color="#7a9b57")
    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1, label="random ranker")
    ax.set_title("Validation ROC-AUC")
    ax.set_xlabel("epoch")
    ax.set_ylabel("ROC-AUC")
    ax.legend(fontsize=8)

    for a in axes:
        a.grid(alpha=0.25, linewidth=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_evaluation(results: dict, ks: Sequence[int] = (10, 25, 50, 100),
                    save_path: str | None = None, show: bool = True):
    """PR curve, ROC curve, score separation, and the Precision@K table as a curve."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    plt = _plt()
    y, p = results["test_targets"], results["test_probs"]
    m = results["test_at_tuned"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    prec, rec, _ = precision_recall_curve(y, p)
    ax = axes[0]
    ax.plot(rec, prec, color="#3b6ea5", linewidth=1.6)
    ax.axhline(m["base_rate"], color="#999999", linestyle=":", linewidth=1,
               label=f"base rate {m['base_rate']:.4f}")
    ax.plot(m["recall"], m["precision"], "o", color="#c1442e",
            label=f"tuned threshold {m['threshold']:.3f}")
    ax.set_title(f"Precision-Recall  (AP = {m['pr_auc']:.4f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend(fontsize=8)

    fpr, tpr, _ = roc_curve(y, p)
    ax = axes[1]
    ax.plot(fpr, tpr, color="#7a4fa0", linewidth=1.6)
    ax.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1)
    ax.set_title(f"ROC  (AUC = {m['roc_auc']:.4f})")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")

    ax = axes[2]
    bins = np.linspace(0, 1, 51)
    ax.hist(p[y == 0], bins=bins, color="#7a9b57", alpha=0.8, density=True,
            label="healthy window")
    ax.hist(p[y == 1], bins=bins, color="#c1442e", alpha=0.7, density=True,
            label="failure window")
    ax.axvline(m["threshold"], color="#333333", linestyle="--", linewidth=1.2,
               label="tuned threshold")
    ax.set_yscale("log")
    ax.set_title("Predicted probability by true class")
    ax.set_xlabel("P(failure in window)")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[3]
    for tag, tbl, colour in (("windows", results.get("precision_at_k_windows"), "#3b6ea5"),
                             ("drives", results.get("precision_at_k_drives"), "#c1442e")):
        if tbl is None or not len(tbl):
            continue
        ax.plot(tbl["k"], tbl["precision"], "o-", color=colour, label=f"P@K, {tag}")
        ax.plot(tbl["k"], tbl["best_possible"], ":", color=colour, alpha=0.6,
                label=f"ceiling, {tag}")
    ax.set_xscale("log")
    ax.set_xticks(list(ks))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision@K")
    ax.set_xlabel("K (highest-scoring items)")
    ax.set_ylabel("precision")
    ax.legend(fontsize=8)

    for a in axes:
        a.grid(alpha=0.25, linewidth=0.6)
        a.set_axisbelow(True)
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
# 10. CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = p.add_argument_group("data source")
    src.add_argument("--synthetic", action="store_true",
                     help="generate a stand-in frame instead of downloading quarters")
    src.add_argument("--n-drives", type=int, default=900, help="synthetic drives")
    src.add_argument("--quarters", nargs="+", default=["Q1_2023", "Q2_2023"])
    src.add_argument("--cache",
                     default=os.path.join(BULK_DATA_DIR, "backblaze_window_frame.parquet"))
    src.add_argument("--model-prefix", default="ST", help="'' to keep all manufacturers")

    win = p.add_argument_group("windowing")
    win.add_argument("--window-days", type=int, default=90)
    win.add_argument("--horizon-days", type=int, default=0)
    win.add_argument("--stride-days", type=int, default=15)
    win.add_argument("--samples-per-drive", type=int, default=1)
    win.add_argument("--positive-ratio", type=float, default=None)
    win.add_argument("--min-days", type=int, default=30)

    trn = p.add_argument_group("training")
    trn.add_argument("--arch", default="cnn", choices=sorted(ARCHITECTURES))
    trn.add_argument("--hidden-dim", type=int, default=64)
    trn.add_argument("--dropout", type=float, default=0.2)
    trn.add_argument("--batch-size", type=int, default=256)
    trn.add_argument("--lr", type=float, default=1e-3)
    trn.add_argument("--epochs", type=int, default=20)
    trn.add_argument("--patience", type=int, default=5)
    trn.add_argument("--loss", default="bce_pos_weight",
                     choices=["bce_pos_weight", "bce", "focal"])
    trn.add_argument("--sampler", default="none", choices=["none", "balanced"])
    trn.add_argument("--pos-fraction", type=float, default=0.3)
    trn.add_argument("--num-workers", type=int, default=0)
    trn.add_argument("--seed", type=int, default=RANDOM_STATE)
    trn.add_argument("--device", default=None, help="cuda | cpu (default: auto)")

    p.add_argument("--inspect-only", action="store_true",
                   help="build the windows, run every sanity check, then stop")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--out-prefix", default="backblaze_window",
                   help="prefix for the saved weights and preprocessing config")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict | None:
    args = parse_args(argv)
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    positive_ratio = args.positive_ratio
    if args.sampler == "balanced" and positive_ratio is None:
        positive_ratio = 0.5
        print("note: --sampler balanced with no --positive-ratio; using 0.5 so failing "
              "drives still contribute their healthy windows.")

    cfg = DataConfig(
        window_days=args.window_days,
        horizon_days=args.horizon_days,
        stride_days=args.stride_days,
        samples_per_drive=args.samples_per_drive,
        train_positive_ratio=positive_ratio,
        min_days=args.min_days,
        random_state=args.seed,
    )
    tcfg = TrainConfig(
        arch=args.arch, hidden_dim=args.hidden_dim, dropout=args.dropout,
        batch_size=args.batch_size, lr=args.lr, max_epochs=args.epochs,
        patience=args.patience, loss=args.loss, sampler=args.sampler,
        balanced_pos_fraction=args.pos_fraction, num_workers=args.num_workers,
        seed=args.seed,
    )

    print(_RULE)
    print("BACKBLAZE 90-DAY WINDOW SEQUENCE CLASSIFICATION")
    print(_RULE)
    print(f"  device {device} | arch {tcfg.arch} | window {cfg.window_days}d | "
          f"horizon {cfg.horizon_days}d | loss {tcfg.loss} | sampler {tcfg.sampler}")

    if args.synthetic:
        print(f"\nGenerating a synthetic frame ({args.n_drives} drives) ...")
        raw = make_synthetic_frame(n_drives=args.n_drives, seed=args.seed)
    else:
        raw = load_backblaze_frame(
            args.quarters, model_prefix_filter=args.model_prefix or None, cache=args.cache
        )
    print(f"Raw frame: {len(raw):,} drive-days, {raw['id'].nunique():,} drives, "
          f"{int(raw['failure'].sum()):,} failure events, "
          f"{raw['time'].min().date()} -> {raw['time'].max().date()}")

    bundle = build_window_bundle(raw, cfg)
    loaders = build_loaders(bundle, tcfg, device)
    run_sanity_checks(bundle, loaders)

    if args.inspect_only:
        print("\n--inspect-only: stopping before training.")
        return None

    model = build_model(bundle.num_features, cfg.window_days, tcfg).to(device)
    print(_RULE)
    print(f"MODEL: {tcfg.arch}  ({count_parameters(model):,} trainable parameters)")
    print(_RULE)
    print(model)
    with torch.no_grad():
        probe = torch.zeros(2, cfg.window_days, bundle.num_features, device=device)
        print(f"  forward check: {tuple(probe.shape)} -> "
              f"{tuple(model(probe).shape)} (one logit per window)")

    print(_RULE)
    print("TRAINING")
    print(_RULE)
    model, history, best_val = train_model(model, loaders, bundle, tcfg, device)
    print(f"\nBest validation PR-AUC: {best_val:.4f}")

    results = evaluate_final(model, loaders, bundle, tcfg, device)
    results["history"] = history
    if not args.no_save:
        save_artifacts(model, bundle, tcfg, results, prefix=args.out_prefix)
    return results


if __name__ == "__main__":
    main()
