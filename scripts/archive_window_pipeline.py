"""Window pipeline for the *whole* Backblaze archive, not a single-vendor cohort.

`backblaze_window_pipeline` holds the entire telemetry matrix in RAM and enumerates
windows through Python lists. That is the right shape for a 3 M drive-day Toshiba
cohort and the wrong shape for 730 M drive-days across every model Backblaze has ever
run. Four things break at that size, and this module replaces exactly those four:

| what breaks | at archive scale | what this does instead |
|---|---|---|
| the feature matrix | ~143 GB, against 103 GB of RAM | a **memory-mapped** matrix on disk, built by an external sort |
| the window index | a Python list of 730 M tuples, ~53 GB | per-drive window **offsets**; `(drive, start)` is arithmetic |
| per-window labelling | 730 M calls to `window_label` | failures are rare: **mark positive ranges**, leave the rest zero |
| `__getitem__` per window | 730 M Python round-trips per pass | the dataset item **is a batch**, gathered with one fancy index |

Everything else -- the models, the loss, the metrics, the training loop, the plots --
is imported from `backblaze_window_pipeline` unchanged.

Build order
-----------
1. `build_drive_census`   one pass over the shards for per-drive first/last day and
                          whether the drive ever failed. Decides the dense layout.
2. `bucketize_shards`     one pass that rewrites every row into one of `n_buckets`
                          fixed-width temp files, keyed by a contiguous range of drive
                          codes. This is the sort: shards arrive time-major, the store
                          needs drive-major, and scattering 730 M rows straight into a
                          143 GB memmap would be one random write per row.
3. `densify_buckets`      each bucket is small enough to sort in RAM, so its rows go
                          down as one sequential write, then get forward-filled,
                          log1p'd and differenced in place.

Steps 1-3 are resumable: each writes a marker and is skipped if it already ran.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import backblaze_window_pipeline as bw  # noqa: E402

_EPOCH = pd.Timestamp("1970-01-01")
_RULE = bw._RULE
_THIN = bw._THIN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ArchiveConfig:
    """Where the archive lives and how the dense store is laid out."""

    shard_dir: str = os.path.join("data", "backblaze_archive", "shards")
    store_dir: str = os.path.join("data", "backblaze_archive", "store")
    # Restrict the run to archives from this year on. None = the whole archive.
    since_year: int | None = None

    # A window is `window_days` consecutive days; it is positive if the drive fails
    # within `horizon_days` of the window's last day.
    window_days: int = 30
    horizon_days: int = 30
    # 1 = every possible window start is enumerated for val/test.
    stride_days: int = 1

    # Drives shorter than a window are dropped rather than padded, so every window is
    # `window_days` genuinely continuous days.
    min_days: int = 30

    delta_lags: tuple[int, ...] = (1, 7)
    add_observed_mask: bool = True

    feature_columns: tuple[str, ...] = ()
    log1p_columns: tuple[str, ...] = ()

    # External-sort width. Each bucket holds a contiguous range of drive codes and must
    # fit in RAM once: 730 M rows / 64 ~ 11 M rows, ~1.5 GB with 23 attributes.
    n_buckets: int = 64

    samples_per_drive: int = 4
    train_positive_ratio: float | None = 0.25
    batch_size: int = 1024

    test_size: float = 0.20
    val_size: float = 0.15
    random_state: int = 42

    def __post_init__(self):
        if self.min_days < self.window_days:
            self.min_days = self.window_days

    @property
    def n_base(self) -> int:
        return len(self.feature_columns)

    @property
    def n_channels(self) -> int:
        return (self.n_base * (1 + len(self.delta_lags))
                + (1 if self.add_observed_mask else 0))

    def channel_names(self) -> list[str]:
        names = list(self.feature_columns)
        for lag in self.delta_lags:
            names += [f"{c}_d{lag}" for c in self.feature_columns]
        if self.add_observed_mask:
            names.append("observed")
        return names


def shard_paths(shard_dir: str, since_year: int | None = None) -> list[str]:
    """Every built shard, oldest archive first, optionally from `since_year` on.

    `since_year` is what scopes a run to a slice of the archive. It filters on the
    archive the shard came from, not on row dates, so a drive first seen in 2021 enters
    a 2022-onward run with only its 2022-onward rows -- which is the intended reading of
    "the data from 2022 till today".
    """
    if not os.path.isdir(shard_dir):
        raise FileNotFoundError(
            f"{shard_dir} does not exist -- run scripts/build_backblaze_archive.py first")
    names = [f for f in os.listdir(shard_dir)
             if f.endswith(".parquet") and os.path.exists(os.path.join(shard_dir, f + ".done"))]

    def key(f: str):
        stem = f[:-len(".parquet")]
        if stem.startswith("Q"):
            q, y = stem.split("_")
            return (int(y), int(q[1]))
        return (int(stem), 0)

    keep = sorted(names, key=key)
    if since_year is not None:
        keep = [f for f in keep if key(f)[0] >= int(since_year)]
        if not keep:
            raise FileNotFoundError(
                f"no shards in {shard_dir} from {since_year} onward")
    return [os.path.join(shard_dir, f) for f in keep]


def _marker(path: str) -> str:
    return path + ".done"


def _mark(path: str, payload: str = "ok") -> None:
    with open(_marker(path), "w", encoding="utf-8") as fh:
        fh.write(payload)


def _is_done(path: str) -> bool:
    return os.path.exists(path) and os.path.exists(_marker(path))


# ---------------------------------------------------------------------------
# 1. Drive census
# ---------------------------------------------------------------------------

def build_drive_census(cfg: ArchiveConfig, verbose: bool = True) -> pd.DataFrame:
    """Per-drive first day, last day, observation count and ever-failed flag.

    One pass over the shards reading four columns. This is what decides the dense
    layout, so it has to see every row -- but it never holds more than one shard's
    worth of them, and the result is ~600 k rows.
    """
    out = os.path.join(cfg.store_dir, "census.parquet")
    if _is_done(out):
        census = pd.read_parquet(out)
        if verbose:
            print(f"  census cached: {len(census):,} drives")
        return census

    os.makedirs(cfg.store_dir, exist_ok=True)
    paths = shard_paths(cfg.shard_dir, cfg.since_year)
    parts = []
    for i, p in enumerate(paths, 1):
        t0 = time.time()
        df = pq.read_table(p, columns=["serial_number", "date", "failure"]).to_pandas()
        day = (pd.to_datetime(df["date"]) - _EPOCH).dt.days.to_numpy(np.int32)
        g = pd.DataFrame({"serial_number": df["serial_number"],
                          "day": day,
                          "failure": df["failure"].to_numpy(np.int8)})
        agg = g.groupby("serial_number", sort=False).agg(
            start_day=("day", "min"), end_day=("day", "max"),
            n_obs=("day", "size"), n_fail=("failure", "sum"))
        parts.append(agg.reset_index())
        if verbose:
            print(f"  [{i:>2}/{len(paths)}] {os.path.basename(p):<16} "
                  f"{len(df):>12,} rows -> {len(agg):>8,} drives  "
                  f"({time.time() - t0:5.1f}s)", flush=True)
        del df, g, agg

    # A drive spans shards, so the per-shard aggregates are combined once at the end.
    census = pd.concat(parts, ignore_index=True)
    census = census.groupby("serial_number", sort=True).agg(
        start_day=("start_day", "min"), end_day=("end_day", "max"),
        n_obs=("n_obs", "sum"), n_fail=("n_fail", "sum")).reset_index()

    census["length"] = (census["end_day"] - census["start_day"] + 1).astype(np.int32)
    census["has_failure"] = (census["n_fail"] > 0)
    census.to_parquet(out, index=False)
    _mark(out, f"{len(census)}")
    if verbose:
        print(f"  census: {len(census):,} drives, "
              f"{int(census.has_failure.sum()):,} with a failure event")
    return census


def layout_from_census(census: pd.DataFrame, cfg: ArchiveConfig,
                       verbose: bool = True) -> dict:
    """Drop drives too short for a window, then assign codes and dense row offsets."""
    keep = census[census["length"] >= cfg.min_days].reset_index(drop=True)
    keep = keep.sort_values("serial_number", kind="stable").reset_index(drop=True)

    lengths = keep["length"].to_numpy(np.int64)
    offsets = np.zeros(len(keep) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])

    # Buckets are contiguous code ranges, so each bucket's rows land in one contiguous
    # slab of the memmap and go down as a sequential write.
    bounds = np.linspace(0, len(keep), cfg.n_buckets + 1).astype(np.int64)
    bounds = np.unique(bounds)

    if verbose:
        dropped = len(census) - len(keep)
        print(f"  {len(keep):,} drives kept, {dropped:,} dropped "
              f"(shorter than {cfg.min_days} days)")
        print(f"  {int(offsets[-1]):,} dense drive-days "
              f"({int(keep.n_obs.sum()):,} observed, "
              f"{int(offsets[-1] - keep.n_obs.sum()):,} forward-filled)")
        print(f"  matrix {int(offsets[-1]):,} x {cfg.n_channels} float32 = "
              f"{offsets[-1] * cfg.n_channels * 4 / 1e9:.1f} GB memory-mapped")
    return {"drives": keep, "lengths": lengths, "offsets": offsets,
            "bucket_bounds": bounds, "n_rows": int(offsets[-1])}


# ---------------------------------------------------------------------------
# 2. External sort: shards (time-major) -> buckets (drive-major)
# ---------------------------------------------------------------------------

def _rec_dtype(n_base: int) -> np.dtype:
    return np.dtype([("dest", np.int64), ("fail", np.uint8),
                     ("vals", np.float32, (n_base,))])


def bucketize_shards(cfg: ArchiveConfig, layout: dict, verbose: bool = True) -> str:
    """Rewrite every row into the temp bucket file its drive belongs to.

    The destination row is computed here, so the densify pass never needs the census
    again -- it sorts on `dest` and writes.
    """
    bucket_dir = os.path.join(cfg.store_dir, "buckets")
    done = os.path.join(bucket_dir, "_bucketize")
    if _is_done(done):
        if verbose:
            print("  buckets cached")
        return bucket_dir

    os.makedirs(bucket_dir, exist_ok=True)
    drives = layout["drives"]
    offsets, bounds = layout["offsets"], layout["bucket_bounds"]
    n_bucket = len(bounds) - 1
    feats = list(cfg.feature_columns)
    rec = _rec_dtype(len(feats))

    code_of = pd.Series(np.arange(len(drives), dtype=np.int64),
                        index=pd.Index(drives["serial_number"]))
    start_of = drives["start_day"].to_numpy(np.int64)

    handles = [open(os.path.join(bucket_dir, f"b{b:03d}.bin"), "wb")
               for b in range(n_bucket)]
    paths = shard_paths(cfg.shard_dir, cfg.since_year)
    written = 0
    try:
        for i, p in enumerate(paths, 1):
            t0 = time.time()
            tbl = pq.read_table(p, columns=["serial_number", "date", "failure"] + feats)
            df = tbl.to_pandas()
            del tbl

            code = code_of.reindex(df["serial_number"]).to_numpy()
            ok = ~pd.isna(code)                       # drives dropped as too short
            if not ok.all():
                df = df.loc[ok].reset_index(drop=True)
                code = code[ok]
            code = code.astype(np.int64)
            day = (pd.to_datetime(df["date"]) - _EPOCH).dt.days.to_numpy(np.int64)

            arr = np.empty(len(df), dtype=rec)
            arr["dest"] = offsets[code] + (day - start_of[code])
            arr["fail"] = df["failure"].to_numpy(np.uint8)
            block = np.empty((len(df), len(feats)), dtype=np.float32)
            for j, c in enumerate(feats):
                block[:, j] = df[c].to_numpy(np.float32)
            arr["vals"] = block
            del df, block

            bucket = np.searchsorted(bounds, code, side="right") - 1
            np.clip(bucket, 0, n_bucket - 1, out=bucket)
            order = np.argsort(bucket, kind="stable")
            arr = arr[order]
            edges = np.searchsorted(bucket[order], np.arange(n_bucket + 1))
            for b in range(n_bucket):
                lo, hi = edges[b], edges[b + 1]
                if hi > lo:
                    handles[b].write(arr[lo:hi].tobytes())
            written += len(arr)
            if verbose:
                print(f"  [{i:>2}/{len(paths)}] {os.path.basename(p):<16} "
                      f"{len(arr):>12,} rows bucketed ({time.time() - t0:5.1f}s)",
                      flush=True)
            del arr, code, day, bucket, order
    finally:
        for h in handles:
            h.close()

    _mark(done, str(written))
    with open(done, "w", encoding="utf-8") as fh:
        fh.write(str(written))
    if verbose:
        print(f"  {written:,} rows across {n_bucket} buckets "
              f"({sum(os.path.getsize(os.path.join(bucket_dir, f)) for f in os.listdir(bucket_dir) if f.endswith('.bin')) / 1e9:.1f} GB temp)")
    return bucket_dir


# ---------------------------------------------------------------------------
# 3. Buckets -> the dense memory-mapped matrix
# ---------------------------------------------------------------------------

def _ffill_block(mat: np.ndarray, row_drive_start: np.ndarray) -> None:
    """Forward-fill NaNs down each column without crossing a drive boundary."""
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


def densify_buckets(cfg: ArchiveConfig, layout: dict, verbose: bool = True) -> dict:
    """Turn each bucket's rows into the dense, forward-filled, differenced matrix.

    Each bucket is handled start to finish in RAM and written once, so the 143 GB
    memmap is filled by `n_buckets` sequential writes rather than 730 M random ones.
    """
    values_path = os.path.join(cfg.store_dir, "values.f32")
    failure_path = os.path.join(cfg.store_dir, "failure.u8")
    done = os.path.join(cfg.store_dir, "_densify")
    n_rows, n_ch = layout["n_rows"], cfg.n_channels

    if _is_done(done):
        if verbose:
            print("  dense matrix cached")
        return {"values_path": values_path, "failure_path": failure_path,
                "n_rows": n_rows, "n_channels": n_ch}

    bucket_dir = os.path.join(cfg.store_dir, "buckets")
    offsets, bounds = layout["offsets"], layout["bucket_bounds"]
    lengths = layout["lengths"]
    n_bucket = len(bounds) - 1
    n_base = cfg.n_base
    rec = _rec_dtype(n_base)
    log_idx = [i for i, c in enumerate(cfg.feature_columns)
               if c in set(cfg.log1p_columns)]

    values = np.memmap(values_path, dtype=np.float32, mode="w+", shape=(n_rows, n_ch))
    failure = np.memmap(failure_path, dtype=np.uint8, mode="w+", shape=(n_rows,))

    for b in range(n_bucket):
        t0 = time.time()
        lo_code, hi_code = int(bounds[b]), int(bounds[b + 1])
        if hi_code <= lo_code:
            continue
        row_lo, row_hi = int(offsets[lo_code]), int(offsets[hi_code])
        n = row_hi - row_lo

        path = os.path.join(bucket_dir, f"b{b:03d}.bin")
        arr = np.fromfile(path, dtype=rec)

        # The bucket's slab of the matrix, built in RAM then written in one go.
        block = np.empty((n, n_ch), dtype=np.float32)
        levels = block[:, :n_base]
        levels[:] = np.nan
        observed = np.zeros(n, dtype=bool)
        fail_local = np.zeros(n, dtype=np.uint8)

        dest = arr["dest"] - row_lo
        # A duplicated (drive, day) would have two rows writing the same cell. Sorting
        # by (dest, fail) and letting the later write win keeps the failure row.
        order = np.lexsort((arr["fail"], dest))
        dest = dest[order]
        levels[dest] = arr["vals"][order]
        observed[dest] = True
        fail_local[dest] = arr["fail"][order]
        del arr, order

        blk_lengths = lengths[lo_code:hi_code]
        blk_starts = (offsets[lo_code:hi_code] - row_lo).astype(np.int64)
        row_drive_start = np.repeat(blk_starts, blk_lengths)
        row_pos = np.arange(n, dtype=np.int64) - row_drive_start

        if log_idx:
            sub = levels[:, log_idx]
            levels[:, log_idx] = np.log1p(np.clip(sub, 0.0, None))
        _ffill_block(levels, row_drive_start)

        for k, lag in enumerate(cfg.delta_lags, start=1):
            dblk = block[:, n_base * k: n_base * (k + 1)]
            if lag < n:
                np.subtract(levels[lag:], levels[:-lag], out=dblk[lag:])
            dblk[row_pos < lag] = np.nan   # no genuine `lag`-days-ago reading yet
        if cfg.add_observed_mask:
            block[:, -1] = observed

        values[row_lo:row_hi] = block
        failure[row_lo:row_hi] = fail_local
        if verbose:
            print(f"  bucket {b + 1:>3}/{n_bucket}  drives {lo_code:>7,}-{hi_code:>7,}  "
                  f"{n:>11,} rows  ({time.time() - t0:5.1f}s)", flush=True)
        del block, levels, observed, fail_local, row_drive_start, row_pos

    values.flush(); failure.flush()
    del values, failure
    _mark(done, str(n_rows))
    with open(done, "w", encoding="utf-8") as fh:
        fh.write(str(n_rows))

    shutil.rmtree(bucket_dir, ignore_errors=True)   # ~70 GB of temp, no longer needed
    return {"values_path": values_path, "failure_path": failure_path,
            "n_rows": n_rows, "n_channels": n_ch}


# ---------------------------------------------------------------------------
# 4. The store
# ---------------------------------------------------------------------------

class ArchiveSeriesStore:
    """Every drive's telemetry on a gap-free daily calendar, memory-mapped.

    Same contract as `DriveSeriesStore` -- `values`, `offsets`, `lengths`, `get_window`
    -- but `values` is an on-disk `np.memmap` rather than an in-RAM array, and the
    failure flag lives in `failure`, a separate array that is never a channel of
    `values`. That separation is the leakage guard: no code path lets a model read the
    label out of X.

    The memmaps are opened lazily so the object survives being pickled into a DataLoader
    worker, which on Windows means being rebuilt in a fresh process.
    """

    def __init__(self, cfg: ArchiveConfig, layout: dict, paths: dict):
        self.cfg = cfg
        self.store_dir = cfg.store_dir
        self._values_path = paths["values_path"]
        self._failure_path = paths["failure_path"]
        self.n_rows = int(paths["n_rows"])
        self.n_channels = int(paths["n_channels"])

        drives = layout["drives"]
        self.drive_ids = drives["serial_number"].to_numpy(dtype=object)
        self.lengths = layout["lengths"].astype(np.int64)
        self.offsets = layout["offsets"].astype(np.int64)
        self.start_days = drives["start_day"].to_numpy(np.int64)
        self.models = drives["model"].to_numpy(dtype=object) if "model" in drives else None
        self.n_drives = len(drives)
        self.drive_has_failure = drives["has_failure"].to_numpy(bool)
        self.feature_names = cfg.channel_names()
        self.base_feature_count = cfg.n_base

        self.scaled_columns = np.array([n != "observed" for n in self.feature_names])
        self.log_columns = np.array(
            [n in set(cfg.log1p_columns) for n in self.feature_names])
        self.feature_medians = self.feature_mean = self.feature_std = None

        self._values = None
        self._failure = None
        self._build_failure_index()
        self._load_scaler()

    # -- lazy memmaps ---------------------------------------------------------

    @property
    def values(self) -> np.memmap:
        if self._values is None:
            self._values = np.memmap(self._values_path, dtype=np.float32, mode="r+",
                                     shape=(self.n_rows, self.n_channels))
        return self._values

    @property
    def failure(self) -> np.memmap:
        if self._failure is None:
            self._failure = np.memmap(self._failure_path, dtype=np.uint8, mode="r",
                                      shape=(self.n_rows,))
        return self._failure

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_values"] = None      # a memmap does not survive pickling; reopen it
        state["_failure"] = None
        return state

    # -- failure bookkeeping --------------------------------------------------

    def _build_failure_index(self) -> None:
        """Failure positions per drive, in CSR form.

        Built from the ~50 k rows where `failure == 1` rather than by slicing the
        failure array once per drive: at 600 k drives that loop is the expensive part
        and the answer is almost everywhere empty.
        """
        cache = os.path.join(self.store_dir, "failure_index.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            self.fail_offsets, self.fail_pos = z["fail_offsets"], z["fail_pos"]
            return
        rows = np.flatnonzero(np.asarray(self.failure))
        g = np.searchsorted(self.offsets, rows, side="right") - 1
        pos = rows - self.offsets[g]
        order = np.lexsort((pos, g))
        g, pos = g[order], pos[order]
        counts = np.bincount(g, minlength=self.n_drives)
        fail_offsets = np.zeros(self.n_drives + 1, dtype=np.int64)
        np.cumsum(counts, out=fail_offsets[1:])
        self.fail_offsets = fail_offsets
        self.fail_pos = pos.astype(np.int32)
        np.savez(cache, fail_offsets=fail_offsets, fail_pos=self.fail_pos)

    def failures_of(self, g: int) -> np.ndarray:
        lo, hi = self.fail_offsets[g], self.fail_offsets[g + 1]
        return self.fail_pos[lo:hi]

    # -- window geometry ------------------------------------------------------

    @property
    def span(self) -> int:
        """Days after a window's start in which a failure still counts as positive."""
        return self.cfg.window_days - 1 + self.cfg.horizon_days

    def max_start(self, g: int) -> int:
        return int(self.lengths[g] - self.cfg.window_days)

    def n_windows(self, g: int) -> int:
        return self.max_start(g) + 1

    def window_label(self, g: int, start: int) -> int:
        f = self.failures_of(g)
        if f.size == 0:
            return 0
        return int(np.any((f >= start) & (f <= start + self.span)))

    def positive_starts(self, g: int) -> np.ndarray:
        f = self.failures_of(g)
        if f.size == 0:
            return np.empty(0, dtype=np.int64)
        hi = self.max_start(g)
        parts = [np.arange(max(0, int(x) - self.span), min(hi, int(x)) + 1) for x in f]
        parts = [p for p in parts if p.size]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, np.int64)

    def window_dates(self, g: int, start: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        first = _EPOCH + pd.Timedelta(days=int(self.start_days[g] + start))
        last = first + pd.Timedelta(days=self.cfg.window_days - 1)
        return first, last

    # -- gathering ------------------------------------------------------------

    def gather(self, drives: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """(B, window_days, channels) for B (drive, start) pairs, in one fancy index.

        This is the hot path: every window of a pass goes through here, so it never
        touches Python per window. Consecutive stride-1 windows of one drive overlap by
        29 of their 30 days, so in enumerate order the pages are already resident and
        the memmap read is effectively a sequential scan.
        """
        row0 = self.offsets[drives] + starts
        rows = row0[:, None] + np.arange(self.cfg.window_days, dtype=np.int64)[None, :]
        flat = np.asarray(self.values[rows.ravel()])
        return flat.reshape(len(drives), self.cfg.window_days, self.n_channels)

    def gather_run(self, drive: int, start0: int, n: int, stride: int = 1) -> np.ndarray:
        """`n` evenly-spaced windows of one drive, from a single contiguous read.

        The fancy-index path in `gather` reads `n * window_days` rows for a batch whose
        windows overlap almost completely. A run of `n` starts spaced `stride` apart
        spans only `(n - 1) * stride + window_days` distinct rows, so the slab is read
        once and the windows become a strided view over it. At `window_days = 30` and
        stride 1 that is a ~30x cut in memory traffic, which is the difference between
        an evaluation pass that is disk-bound and one that is GPU-bound.
        """
        W, C = self.cfg.window_days, self.n_channels
        need = (n - 1) * stride + W
        lo = int(self.offsets[drive]) + int(start0)
        slab = np.asarray(self.values[lo: lo + need])
        if slab.shape[0] < need:                 # defensive: never seen for valid starts
            return self.gather(
                np.full(n, drive, dtype=np.int64),
                start0 + np.arange(n, dtype=np.int64) * stride)
        s0, s1 = slab.strides
        view = np.lib.stride_tricks.as_strided(slab, shape=(n, W, C),
                                               strides=(s0 * stride, s0, s1))
        return np.ascontiguousarray(view)

    def labels_for(self, drives: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """Labels for B (drive, start) pairs. Loops only over drives that ever failed."""
        out = np.zeros(len(drives), dtype=np.float32)
        span = self.span
        for i in np.flatnonzero(self.drive_has_failure[drives]):
            f = self.failures_of(int(drives[i]))
            s = int(starts[i])
            if f.size and np.any((f >= s) & (f <= s + span)):
                out[i] = 1.0
        return out

    def get_window(self, g: int, start: int):
        """(window, label, meta) -- the single-window path, used for inspection."""
        window = self.gather(np.array([g]), np.array([start]))[0]
        label = self.window_label(g, start)
        first, last = self.window_dates(g, start)
        lo = int(self.offsets[g] + start)
        obs = (int(np.asarray(self.values[lo:lo + self.cfg.window_days, -1]).sum())
               if self.cfg.add_observed_mask else self.cfg.window_days)
        meta = {
            "drive_id": str(self.drive_ids[g]), "drive_index": int(g),
            "start": int(start), "start_date": first.strftime("%Y-%m-%d"),
            "end_date": last.strftime("%Y-%m-%d"), "label": int(label),
            "n_real_days": obs,
            "n_filled_days": int(self.cfg.window_days - obs),
            "n_padded_days": 0,
        }
        return window, label, meta

    # -- scaling --------------------------------------------------------------

    def _load_scaler(self) -> None:
        path = os.path.join(self.store_dir, "scaler.npz")
        self._scaled = os.path.exists(path)
        if self._scaled:
            z = np.load(path)
            self.feature_medians, self.feature_mean, self.feature_std = (
                z["medians"], z["mean"], z["std"])

    def fit_scaling(self, train_drives: np.ndarray, max_fit_rows: int = 5_000_000,
                    verbose: bool = True) -> None:
        """Median-impute and standardise the matrix in place, on TRAIN statistics only.

        The memmap is rewritten rather than copied -- a copy would be another 143 GB --
        so the transformation is recorded in `scaler.npz` and skipped if it already ran.
        Statistics come from genuinely observed training drive-days: forward-filled rows
        repeat values and would over-weight drives that report patchily.
        """
        if self._scaled:
            if verbose:
                print("  scaler cached -- matrix already standardised in place")
            return

        rng = np.random.default_rng(self.cfg.random_state)
        # Sample drives, then a run of rows inside each: materialising every train row
        # index would be a multi-GB array before a single statistic was computed.
        pick = (train_drives if len(train_drives) <= 40_000
                else rng.choice(train_drives, 40_000, replace=False))
        per = max(1, max_fit_rows // len(pick))
        chunks = []
        for g in pick:
            lo, hi = int(self.offsets[g]), int(self.offsets[g + 1])
            take = min(per, hi - lo)
            s = int(rng.integers(lo, hi - take + 1))
            chunks.append(np.asarray(self.values[s:s + take]))
        block = np.concatenate(chunks)
        del chunks
        if self.cfg.add_observed_mask:
            obs = block[:, -1] > 0.5
            if obs.sum() > 1000:
                block = block[obs]

        with np.errstate(invalid="ignore"):
            medians = np.nan_to_num(np.nanmedian(block, axis=0), nan=0.0).astype(np.float32)
        block = np.where(np.isnan(block), medians, block)
        mean = block.mean(axis=0).astype(np.float32)
        std = block.std(axis=0).astype(np.float32)
        std[std < 1e-6] = 1.0
        mean[~self.scaled_columns] = 0.0
        std[~self.scaled_columns] = 1.0
        del block

        if verbose:
            print(f"  scaler fit on {len(pick):,} train drives; rewriting "
                  f"{self.n_rows * self.n_channels * 4 / 1e9:.0f} GB in place")
        self._apply_inplace(medians, mean, std, verbose=verbose)
        np.savez(os.path.join(self.store_dir, "scaler.npz"),
                 medians=medians, mean=mean, std=std)
        self.feature_medians, self.feature_mean, self.feature_std = medians, mean, std
        self._scaled = True

    def _apply_inplace(self, medians, mean, std, chunk_rows: int = 4_000_000,
                       verbose: bool = True) -> None:
        values = self.values
        n = self.n_rows
        for a in range(0, n, chunk_rows):
            b = min(a + chunk_rows, n)
            blk = np.asarray(values[a:b], dtype=np.float32)
            np.copyto(blk, np.broadcast_to(medians, blk.shape), where=np.isnan(blk))
            blk -= mean
            blk /= std
            values[a:b] = blk
            if verbose and (a // chunk_rows) % 20 == 0:
                print(f"    scaled {b:,}/{n:,} rows", flush=True)
        values.flush()

    def to_original_units(self, window: np.ndarray) -> np.ndarray:
        if not self._scaled:
            return window
        out = window * self.feature_std + self.feature_mean
        lvl = np.zeros(len(self.feature_names), dtype=bool)
        lvl[: self.base_feature_count] = True
        undo = lvl & self.log_columns
        out[:, undo] = np.expm1(out[:, undo])
        return out

    def summary(self) -> str:
        return (f"ArchiveSeriesStore: {self.n_drives:,} drives, {self.n_rows:,} dense "
                f"drive-days, {self.n_channels} channels "
                f"({self.n_rows * self.n_channels * 4 / 1e9:.0f} GB memory-mapped), "
                f"{int(self.drive_has_failure.sum()):,} drives with a failure event")


# ---------------------------------------------------------------------------
# 5. Window datasets -- the item IS a batch
# ---------------------------------------------------------------------------

class EnumeratedWindows(Dataset):
    """Every window of every drive in the split, at `stride`, as ready-made batches.

    The index is not materialised. Per drive it stores how many windows that drive has
    and the running total; a global window id maps back to `(drive, start)` with one
    `searchsorted`. At stride 1 over the whole archive that is the difference between
    ~6 GB of index and ~5 MB.

    Labels *are* materialised -- one int8 per window -- because the metrics need
    `y_true` anyway. They are produced by marking the positive range around each of the
    (rare) failure events and leaving everything else zero, not by asking each window.
    """

    def __init__(self, store: ArchiveSeriesStore, drives: np.ndarray,
                 batch_size: int = 1024, stride: int = 1, split_name: str = "eval"):
        self.store = store
        self.drives = np.asarray(drives, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.stride = max(1, int(stride))
        self.split_name = split_name

        max_start = store.lengths[self.drives] - store.cfg.window_days
        self.n_win = (max_start // self.stride) + 1
        self.win_offsets = np.zeros(len(self.drives) + 1, dtype=np.int64)
        np.cumsum(self.n_win, out=self.win_offsets[1:])
        self.n_windows = int(self.win_offsets[-1])
        self.labels = self._mark_labels()

    def _mark_labels(self) -> np.ndarray:
        """int8 label per window, by painting the positive range around each failure."""
        labels = np.zeros(self.n_windows, dtype=np.int8)
        span = self.store.span
        for i in np.flatnonzero(self.store.drive_has_failure[self.drives]):
            g = int(self.drives[i])
            hi = self.store.max_start(g)
            base = int(self.win_offsets[i])
            for f in self.store.failures_of(g):
                lo_s = max(0, int(f) - span)
                hi_s = min(hi, int(f))
                if lo_s > hi_s:
                    continue
                # Window k of this drive starts at k * stride.
                k_lo = (lo_s + self.stride - 1) // self.stride
                k_hi = hi_s // self.stride
                if k_hi >= k_lo:
                    labels[base + k_lo: base + k_hi + 1] = 1
        return labels

    def locate(self, win_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global window ids -> (drive index, within-drive start day)."""
        i = np.searchsorted(self.win_offsets, win_ids, side="right") - 1
        starts = (win_ids - self.win_offsets[i]) * self.stride
        return self.drives[i], starts

    def __len__(self) -> int:
        return (self.n_windows + self.batch_size - 1) // self.batch_size

    def __getitem__(self, b: int):
        lo = b * self.batch_size
        hi = min(lo + self.batch_size, self.n_windows)
        win_ids = np.arange(lo, hi, dtype=np.int64)
        g, s = self.locate(win_ids)

        # Windows arrive in (drive, start) order, so the batch is a handful of runs of
        # evenly-spaced starts -- usually one or two. Each run is one slab read.
        cuts = np.flatnonzero(np.diff(g)) + 1
        parts = [self.store.gather_run(int(g[a]), int(s[a]), int(b_ - a), self.stride)
                 for a, b_ in zip(np.r_[0, cuts], np.r_[cuts, len(g)])]
        x = torch.from_numpy(parts[0] if len(parts) == 1
                             else np.concatenate(parts, axis=0))

        y = torch.from_numpy(self.labels[lo:hi].astype(np.float32)).unsqueeze(1)
        return x, y, torch.from_numpy(g.astype(np.int32))

    def label_report(self) -> str:
        pos = int(self.labels.sum())
        return (f"{self.split_name:<5} | enumerate stride {self.stride:<2} | "
                f"{self.n_windows:>12,} windows | {len(self.drives):>7,} drives | "
                f"positive rate {100 * pos / max(self.n_windows, 1):7.4f}% "
                f"({pos:,} positive windows)")


class RandomWindows(Dataset):
    """`samples_per_drive` random windows per drive per epoch, as ready-made batches.

    The start is drawn uniformly from every valid offset, so nothing snaps to a quarter,
    a month or a week, and a new epoch re-rolls every drive. `positive_ratio` forces a
    share of the draws from failing drives to land in that drive's positive region --
    the single imbalance correction this pipeline applies.
    """

    def __init__(self, store: ArchiveSeriesStore, drives: np.ndarray,
                 samples_per_drive: int = 4, positive_ratio: float | None = 0.25,
                 batch_size: int = 1024, seed: int = 42, split_name: str = "train"):
        self.store = store
        self.drives = np.asarray(drives, dtype=np.int64)
        self.samples_per_drive = max(1, int(samples_per_drive))
        self.positive_ratio = positive_ratio
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.split_name = split_name
        self.epoch = 0
        self.n_samples = len(self.drives) * self.samples_per_drive
        self._order = np.tile(self.drives, self.samples_per_drive)

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the windows and reshuffle which drives share a batch."""
        self.epoch = int(epoch)
        rng = np.random.default_rng([self.seed, self.epoch, 0])
        self._order = np.tile(self.drives, self.samples_per_drive)
        rng.shuffle(self._order)

    def __len__(self) -> int:
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __getitem__(self, b: int):
        lo = b * self.batch_size
        hi = min(lo + self.batch_size, self.n_samples)
        g = self._order[lo:hi]
        # Seeded by (seed, epoch, batch) rather than global RNG state, so the same call
        # yields the same windows in the main process or in a worker.
        rng = np.random.default_rng([self.seed, self.epoch, int(b)])
        max_start = self.store.lengths[g] - self.store.cfg.window_days
        s = (rng.random(len(g)) * (max_start + 1)).astype(np.int64)
        np.minimum(s, max_start, out=s)

        if self.positive_ratio:
            span = self.store.span
            force = rng.random(len(g)) < self.positive_ratio
            for i in np.flatnonzero(force & self.store.drive_has_failure[g]):
                f = self.store.failures_of(int(g[i]))
                if not f.size:
                    continue
                pick = int(f[rng.integers(0, f.size)])
                lo_s = max(0, pick - span)
                hi_s = min(int(max_start[i]), pick)
                if lo_s <= hi_s:
                    s[i] = int(rng.integers(lo_s, hi_s + 1))

        # Read in drive order. The draw above picks 4096 windows from 4096 drives
        # scattered across a 138 GB memmap; fetching them in shuffled order is 4096
        # random seeks per batch, while fetching them in ascending row order is a
        # forward scan the page cache can read ahead. A batch is order-invariant for
        # the loss, and y is derived after the sort, so nothing can fall out of step.
        order = np.lexsort((s, g))
        g, s = g[order], s[order]

        x = torch.from_numpy(self.store.gather(g, s))
        y = torch.from_numpy(self.store.labels_for(g, s)).unsqueeze(1)
        return x, y, torch.from_numpy(g.astype(np.int32))

    def expected_positive_rate(self, sample_batches: int = 40) -> float:
        """Measured on a few drawn batches -- there is no closed form once the
        positive-region draw and the uniform draw are mixed."""
        n = min(sample_batches, len(self))
        if n == 0:
            return 0.0
        tot = pos = 0
        for b in range(n):
            lo = b * self.batch_size
            hi = min(lo + self.batch_size, self.n_samples)
            g = self._order[lo:hi]
            rng = np.random.default_rng([self.seed, self.epoch, int(b)])
            max_start = self.store.lengths[g] - self.store.cfg.window_days
            s = (rng.random(len(g)) * (max_start + 1)).astype(np.int64)
            np.minimum(s, max_start, out=s)
            if self.positive_ratio:
                span = self.store.span
                force = rng.random(len(g)) < self.positive_ratio
                for i in np.flatnonzero(force & self.store.drive_has_failure[g]):
                    f = self.store.failures_of(int(g[i]))
                    if not f.size:
                        continue
                    pick = int(f[rng.integers(0, f.size)])
                    lo_s, hi_s = max(0, pick - span), min(int(max_start[i]), pick)
                    if lo_s <= hi_s:
                        s[i] = int(rng.integers(lo_s, hi_s + 1))
            y = self.store.labels_for(g, s)
            tot += len(y)
            pos += int(y.sum())
        return pos / max(tot, 1)

    def label_report(self) -> str:
        rate = self.expected_positive_rate()
        n_fail = int(self.store.drive_has_failure[self.drives].sum())
        return (f"{self.split_name:<5} | random spd {self.samples_per_drive:<3} | "
                f"{self.n_samples:>12,} windows | {len(self.drives):>7,} drives | "
                f"positive rate {100 * rate:7.4f}% (measured, {n_fail:,} failing drives)")


# ---------------------------------------------------------------------------
# 6. Splits and the bundle
# ---------------------------------------------------------------------------

def grouped_drive_split(store: ArchiveSeriesStore,
                        cfg: ArchiveConfig) -> dict[str, np.ndarray]:
    """Split *drive indices* into train/val/test, stratified on ever-failed.

    Every window of a drive comes from that drive's own rows, so splitting at the drive
    level is what keeps windows of one disk out of two partitions. Failing and healthy
    drives are split separately so each partition gets its share of the rare positives;
    it is still a strict grouped split -- a drive lands in exactly one of the three.
    """
    rng = np.random.default_rng(cfg.random_state)
    splits = {"train": [], "val": [], "test": []}
    for flag in (True, False):
        pool = np.flatnonzero(store.drive_has_failure == flag)
        rng.shuffle(pool)
        n = len(pool)
        n_test = int(round(n * cfg.test_size))
        n_val = int(round((n - n_test) * cfg.val_size))
        splits["test"].append(pool[:n_test])
        splits["val"].append(pool[n_test:n_test + n_val])
        splits["train"].append(pool[n_test + n_val:])
    out = {k: np.sort(np.concatenate(v)) for k, v in splits.items()}

    total = sum(len(v) for v in out.values())
    assert len(np.unique(np.concatenate(list(out.values())))) == total, \
        "a drive id appears in more than one split"
    return out


@dataclass
class ArchiveBundle:
    """Everything the training code needs, built once from the archive."""

    store: ArchiveSeriesStore
    cfg: ArchiveConfig
    splits: dict[str, np.ndarray]
    datasets: dict
    pos_weight: float

    @property
    def feature_names(self) -> list[str]:
        return list(self.store.feature_names)

    @property
    def num_features(self) -> int:
        return self.store.n_channels


def build_archive_store(cfg: ArchiveConfig, verbose: bool = True) -> ArchiveSeriesStore:
    """Census -> layout -> external sort -> dense memmap -> store. Resumable throughout."""
    os.makedirs(cfg.store_dir, exist_ok=True)
    with open(os.path.join(cfg.store_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in asdict(cfg).items()}, fh, indent=2)

    if verbose:
        print(_RULE); print("STEP A -- DRIVE CENSUS"); print(_RULE)
    census = build_drive_census(cfg, verbose=verbose)

    if verbose:
        print(); print(_RULE); print("STEP B -- DENSE LAYOUT"); print(_RULE)
    layout = layout_from_census(census, cfg, verbose=verbose)

    if verbose:
        print(); print(_RULE)
        print(f"STEP C -- EXTERNAL SORT INTO {cfg.n_buckets} BUCKETS"); print(_RULE)
    bucketize_shards(cfg, layout, verbose=verbose)

    if verbose:
        print(); print(_RULE); print("STEP D -- DENSIFY"); print(_RULE)
    paths = densify_buckets(cfg, layout, verbose=verbose)

    store = ArchiveSeriesStore(cfg, layout, paths)
    if verbose:
        print(); print(store.summary())
    return store


def build_archive_bundle(cfg: ArchiveConfig, store: ArchiveSeriesStore | None = None,
                         val_stride: int | None = None,
                         verbose: bool = True) -> ArchiveBundle:
    """Store -> grouped split -> scaler -> the three window datasets.

    `val_stride` exists because per-epoch validation is an early-stopping signal, not a
    reported number: enumerating every stride-1 validation window fifteen times over
    costs more than the entire training run. The *reported* evaluation -- threshold
    tuning and the test pass -- always runs at `cfg.stride_days`.
    """
    store = store or build_archive_store(cfg, verbose=verbose)
    splits = grouped_drive_split(store, cfg)
    if verbose:
        print()
        print("grouped split by drive id -- no drive appears in two splits:")
        for name in ("train", "val", "test"):
            idx = splits[name]
            nf = int(store.drive_has_failure[idx].sum())
            print(f"  {name:<5} {len(idx):>8,} drives  ({nf:,} carry a failure, "
                  f"{100 * nf / max(len(idx), 1):.2f}%)")

    store.fit_scaling(splits["train"], verbose=verbose)

    datasets = {
        "train": RandomWindows(
            store, splits["train"], samples_per_drive=cfg.samples_per_drive,
            positive_ratio=cfg.train_positive_ratio, batch_size=cfg.batch_size,
            seed=cfg.random_state, split_name="train"),
        "val": EnumeratedWindows(
            store, splits["val"], batch_size=cfg.batch_size,
            stride=val_stride or cfg.stride_days, split_name="val"),
        "test": EnumeratedWindows(
            store, splits["test"], batch_size=cfg.batch_size,
            stride=cfg.stride_days, split_name="test"),
    }
    datasets["val_full"] = (
        datasets["val"] if (val_stride or cfg.stride_days) == cfg.stride_days
        else EnumeratedWindows(store, splits["val"], batch_size=cfg.batch_size,
                               stride=cfg.stride_days, split_name="val"))

    rate = datasets["train"].expected_positive_rate()
    pos_weight = (1.0 - rate) / rate if rate > 0 else 1.0
    if verbose:
        print(f"\nWindow datasets (window_days={cfg.window_days}, "
              f"horizon_days={cfg.horizon_days}, stride_days={cfg.stride_days}):")
        for name in ("train", "val", "test"):
            print("  " + datasets[name].label_report())
        if datasets["val_full"] is not datasets["val"]:
            print("  " + datasets["val_full"].label_report()
                  .replace("val  ", "val*"))
            print("    * the stride-1 validation pass used for threshold tuning")
        print(f"\nRaw pos_weight from the training positive rate: {pos_weight:,.1f}")
    return ArchiveBundle(store=store, cfg=cfg, splits=splits, datasets=datasets,
                         pos_weight=pos_weight)


def build_loaders(bundle: ArchiveBundle, tcfg: bw.TrainConfig,
                  device: torch.device) -> dict[str, DataLoader]:
    """DataLoaders over batch-shaped datasets: `batch_size=None` means one item, one batch.

    `num_workers > 0` is what keeps the GPU fed. The model is small enough to finish a
    batch in well under the time it takes to pull the next one off a 138 GB memmap, so
    with a single process the GPU sits idle and the run is disk-bound. The store holds
    its memmaps lazily and drops them in `__getstate__`, so each worker reopens its own
    rather than trying to pickle 138 GB.
    """
    pin = device.type == "cuda"
    nw = int(tcfg.num_workers)
    extra = {}
    if nw > 0:
        extra = {"persistent_workers": True, "prefetch_factor": 4}

    loaders = {}
    for name in ("train", "val", "test", "val_full"):
        if name not in bundle.datasets:
            continue
        loaders[name] = DataLoader(bundle.datasets[name], batch_size=None,
                                   shuffle=False, num_workers=nw,
                                   pin_memory=pin, **extra)
    return loaders


# ---------------------------------------------------------------------------
# 7. Evaluation at archive scale
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_archive_loader(model, loader, criterion, device, threshold: float = 0.5,
                            tag: str = "eval", verbose: bool = False):
    """Run a split; returns (loss, metrics, probs, targets, drive_codes).

    Drive identity comes back as an **int32 code**, not a serial string. At 146 M test
    windows a list of Python strings would be ~15 GB; the codes are 584 MB and index
    straight into `store.drive_ids` for the handful that get reported.
    """
    model.eval()
    total_loss, n_seen = 0.0, 0
    n = len(loader)
    probs_out, targ_out, code_out = [], [], []
    t0 = time.time()
    for i, (xb, yb, gb) in enumerate(loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n_seen += xb.size(0)
        probs_out.append(torch.sigmoid(logits).squeeze(1).to(torch.float32).cpu().numpy())
        targ_out.append(yb.squeeze(1).cpu().numpy().astype(np.int8))
        code_out.append(gb.numpy())
        if verbose and n and i % max(1, n // 20) == 0:
            done = i + 1
            rate = done / max(time.time() - t0, 1e-9)
            print(f"    {tag}: {done:,}/{n:,} batches "
                  f"({100 * done / n:5.1f}%)  {rate * xb.size(0):,.0f} windows/s",
                  flush=True)

    probs = np.concatenate(probs_out); del probs_out
    targets = np.concatenate(targ_out); del targ_out
    codes = np.concatenate(code_out); del code_out
    metrics = bw.compute_metrics(targets, probs, threshold)
    return total_loss / max(n_seen, 1), metrics, probs, targets, codes


def sweep_threshold(targets: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Best-F1 threshold over every achievable cut, in one sort.

    `bw.tune_threshold` walks a 200-point grid, which is 200 passes over the array. At
    100 M+ windows one sort plus two cumulative sums is both faster and exact: every
    distinct probability is considered, not just the grid points.
    """
    order = np.argsort(probs, kind="stable")[::-1]
    y = targets[order].astype(np.int64)
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1, dtype=np.int64)
    n_pos = int(tp[-1])
    if n_pos == 0:
        return 0.5, float("nan")
    precision = tp / k
    recall = tp / n_pos
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    best = int(np.argmax(f1))
    return float(probs[order[best]]), float(f1[best])


def drive_level_scores(probs: np.ndarray, targets: np.ndarray,
                       codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse windows to drives: each drive keeps its highest-scoring window.

    The form that matches the decision an operator actually makes -- you pull K disks,
    not K windows, and one bad drive contributing thousands of overlapping windows
    should not fill the top of the list on its own.
    """
    df = pd.DataFrame({"code": codes, "p": probs, "y": targets})
    g = df.groupby("code", sort=True).agg(p=("p", "max"), y=("y", "max"))
    return g.index.to_numpy(), g["p"].to_numpy(), g["y"].to_numpy()


def evaluate_archive_final(model, loaders, bundle: ArchiveBundle, tcfg, device,
                           pk_ks=(10, 25, 50, 100)) -> dict:
    """Tune the threshold on validation, then score the test split once.

    The threshold is the only quantity carried out of validation, and it is fixed
    before the test split is touched.
    """
    criterion = bw.make_criterion(tcfg, bundle.pos_weight, device, verbose=False)
    val_loader = loaders.get("val_full", loaders["val"])

    print(_RULE)
    print("FINAL VALIDATION PASS -- every window at stride "
          f"{bundle.datasets['val_full'].stride}")
    print(_RULE)
    _, _, val_probs, val_targets, _ = evaluate_archive_loader(
        model, val_loader, criterion, device, tag="val", verbose=True)
    best_t, best_val_f1 = sweep_threshold(val_targets, val_probs)
    print(f"  best F1 threshold .... {best_t:.6f}  (val F1 {best_val_f1:.4f}) "
          f"over {len(val_probs):,} windows")

    print()
    print(_RULE)
    print(f"TEST PASS -- every window at stride {bundle.datasets['test'].stride}")
    print(_RULE)
    _, test_m_05, test_probs, test_targets, test_codes = evaluate_archive_loader(
        model, loaders["test"], criterion, device, threshold=0.5,
        tag="test", verbose=True)
    test_m_tuned = bw.compute_metrics(test_targets, test_probs, best_t)

    bw.print_metrics_block("TEST @ threshold 0.5", test_m_05)
    bw.print_metrics_block(f"TEST @ tuned threshold {best_t:.6f}", test_m_tuned)
    bw.print_confusion(test_m_tuned, f"TEST CONFUSION MATRIX @ {best_t:.6f}")

    pk_window = bw.print_precision_at_k(
        test_targets, test_probs, "TEST PRECISION@K -- ranking windows", ks=pk_ks)
    codes_u, drive_probs, drive_targets = drive_level_scores(
        test_probs, test_targets, test_codes)
    pk_drive = bw.print_precision_at_k(
        drive_targets, drive_probs,
        "TEST PRECISION@K -- ranking DRIVES (each drive scored by its worst window)",
        ks=pk_ks)

    val_m = bw.compute_metrics(val_targets, val_probs, best_t)
    print(_RULE); print("SUMMARY"); print(_RULE)
    print("  " + bw.format_metrics("val   @tuned", val_m))
    print("  " + bw.format_metrics("test  @0.50", test_m_05))
    print("  " + bw.format_metrics("test  @tuned", test_m_tuned))

    # The top-ranked drives, resolved back to serials for the report.
    top = np.argsort(drive_probs)[::-1][:100]
    top_table = pd.DataFrame({
        "rank": np.arange(1, len(top) + 1),
        "serial_number": bundle.store.drive_ids[codes_u[top]],
        "risk": drive_probs[top],
        "actually_failed": drive_targets[top].astype(int),
    })
    return {
        "best_threshold": best_t,
        "val": val_m,
        "test_at_half": test_m_05,
        "test_at_tuned": test_m_tuned,
        "test_probs": test_probs,
        "test_targets": test_targets,
        "n_val_windows": int(len(val_probs)),
        "n_test_windows": int(len(test_probs)),
        "n_test_drives": int(len(drive_probs)),
        "precision_at_k_windows": pk_window,
        "precision_at_k_drives": pk_drive,
        "drive_probs": drive_probs,
        "drive_targets": drive_targets,
        "top_drives": top_table,
    }


# ---------------------------------------------------------------------------
# 8. Training
# ---------------------------------------------------------------------------

def train_archive_model(model, loaders, bundle: ArchiveBundle, tcfg, device,
                        verbose: bool = True):
    """Train with early stopping on validation PR-AUC.

    Mirrors `bw.train_model`; it is repeated rather than reused only because the
    archive loaders yield int drive codes where the cohort loaders yield meta dicts.
    """
    import copy
    criterion = bw.make_criterion(tcfg, bundle.pos_weight, device, verbose=verbose)
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg.lr,
                                 weight_decay=tcfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.5, patience=2)
    train_ds = bundle.datasets["train"]
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_pr_auc": [],
               "val_roc_auc": [], "val_f1": [], "lr": [], "epoch_seconds": [],
               "train_pos_seen": [], "train_n_seen": []}
    best_pr_auc, best_state, no_improve = -1.0, None, 0

    if verbose:
        print(_RULE)
        print(f"TRAINING -- {tcfg.arch.upper()}, {bw.count_parameters(model):,} "
              f"parameters, up to {tcfg.max_epochs} epochs, early stop after "
              f"{tcfg.patience} without a val PR-AUC gain")
        print(f"  {len(train_ds):,} training batches/epoch, "
              f"{len(bundle.datasets['val']):,} validation batches/epoch")
        print(_RULE)
        print(f"{'epoch':>5} | {'train loss':>10} | {'val loss':>9} | {'val PR-AUC':>10} | "
              f"{'val ROC-AUC':>11} | {'val F1@.5':>9} | {'lr':>8} | {'sec':>7}")
        print("-" * 5 + "-+-" + "-" * 10 + "-+-" + "-" * 9 + "-+-" + "-" * 10 + "-+-"
              + "-" * 11 + "-+-" + "-" * 9 + "-+-" + "-" * 8 + "-+-" + "-" * 7)

    for epoch in range(1, tcfg.max_epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        t0, running, n_seen, seen_pos = time.time(), 0.0, 0, 0
        n_batches = len(loaders["train"])
        for i, (xb, yb, _) in enumerate(loaders["train"]):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen += xb.size(0)
            seen_pos += int(yb.sum().item())
            if verbose and n_batches and i and i % max(1, n_batches // 5) == 0:
                print(f"      epoch {epoch} training {100 * i / n_batches:5.1f}%  "
                      f"loss {running / max(n_seen, 1):.5f}", flush=True)

        train_loss = running / max(n_seen, 1)
        val_loss, val_m, _, _, _ = evaluate_archive_loader(
            model, loaders["val"], criterion, device, threshold=0.5, tag="val")
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step(val_m["pr_auc"] if np.isfinite(val_m["pr_auc"]) else 0.0)
        elapsed = time.time() - t0

        for k, v in (("epoch", epoch), ("train_loss", train_loss),
                     ("val_loss", val_loss), ("val_pr_auc", val_m["pr_auc"]),
                     ("val_roc_auc", val_m["roc_auc"]), ("val_f1", val_m["f1"]),
                     ("lr", lr_now), ("epoch_seconds", elapsed),
                     ("train_pos_seen", seen_pos), ("train_n_seen", n_seen)):
            history[k].append(v)

        score = val_m["pr_auc"] if np.isfinite(val_m["pr_auc"]) else -1.0
        improved = score > best_pr_auc
        if verbose:
            print(f"{epoch:>5} | {train_loss:>10.5f} | {val_loss:>9.5f} | "
                  f"{val_m['pr_auc']:>10.4f} | {val_m['roc_auc']:>11.4f} | "
                  f"{val_m['f1']:>9.4f} | {lr_now:>8.2e} | {elapsed:>7.1f}"
                  + ("  <- best" if improved else ""), flush=True)

        if improved:
            best_pr_auc, no_improve = score, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, os.path.join(bundle.cfg.store_dir, "best_model.pth"))
        else:
            no_improve += 1
            if no_improve >= tcfg.patience:
                if verbose:
                    print(f"early stopping at epoch {epoch} "
                          f"(best val PR-AUC {best_pr_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_pr_auc


# ---------------------------------------------------------------------------
# 9. A frame for the leakage audit
# ---------------------------------------------------------------------------

def sample_audit_frame(cfg: ArchiveConfig, census: pd.DataFrame,
                       n_drives: int = 15_000, verbose: bool = True) -> pd.DataFrame:
    """A frame for `leakage_audit.audit_frame`, sampled by **drive**, not by drive-day.

    Sampling drive-days would break the audit outright: `build_record_geometry` derives
    every probe from each drive's own first and last day, and a drive whose rows have
    been thinned has neither. Whole drives keep the geometry exact, keep the fleet
    failure base rate intact, and keep `still_reporting_at_export_end` meaningful -- the
    probe that scores 0.97 on the cohort and is the headline finding of Step 1.
    """
    rng = np.random.default_rng(cfg.random_state)
    pick = census["serial_number"].to_numpy()
    if len(pick) > n_drives:
        pick = rng.choice(pick, n_drives, replace=False)
    wanted = set(pick.tolist())

    paths = shard_paths(cfg.shard_dir, cfg.since_year)
    parts = []
    for i, p in enumerate(paths, 1):
        df = pq.read_table(p).to_pandas()
        df = df[df["serial_number"].isin(wanted)]
        if len(df):
            parts.append(df)
        if verbose:
            print(f"  [{i:>2}/{len(paths)}] {os.path.basename(p):<16} "
                  f"{len(df):>9,} rows kept", flush=True)
    frame = pd.concat(parts, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["failure"] = frame["failure"].fillna(0).astype("int8")
    frame = frame.sort_values(["serial_number", "date"]).reset_index(drop=True)
    if verbose:
        print(f"  audit frame: {len(frame):,} drive-days over "
              f"{frame.serial_number.nunique():,} drives, "
              f"{int(frame.failure.sum()):,} failure events "
              f"(base rate {100 * frame.failure.mean():.5f}%)")
    return frame


def save_artifacts(model, bundle: ArchiveBundle, tcfg, results: dict,
                   prefix: str = "models/backblaze_archive30") -> None:
    """Weights plus everything needed to rebuild the exact preprocessing at inference."""
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)
    store, cfg = bundle.store, bundle.cfg
    torch.save(model.state_dict(), f"{prefix}_{tcfg.arch}.pth")
    config = {
        "window_days": cfg.window_days,
        "horizon_days": cfg.horizon_days,
        "stride_days": cfg.stride_days,
        "feature_columns": list(cfg.feature_columns),
        "log1p_columns": list(cfg.log1p_columns),
        "delta_lags": list(cfg.delta_lags),
        "add_observed_mask": cfg.add_observed_mask,
        "channel_names": store.feature_names,
        "num_features": store.n_channels,
        "arch": tcfg.arch,
        "hidden_dim": tcfg.hidden_dim,
        "dropout": tcfg.dropout,
        "best_threshold": results["best_threshold"],
        "feature_medians": np.asarray(store.feature_medians).tolist(),
        "feature_mean": np.asarray(store.feature_mean).tolist(),
        "feature_std": np.asarray(store.feature_std).tolist(),
        "n_drives": int(store.n_drives),
        "n_drive_days": int(store.n_rows),
        "n_val_windows": results.get("n_val_windows"),
        "n_test_windows": results.get("n_test_windows"),
    }
    with open(f"{prefix}_preprocessing_config.json", "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    print(f"Saved {prefix}_{tcfg.arch}.pth and {prefix}_preprocessing_config.json")


# ---------------------------------------------------------------------------
# 10. The Step 2 deliverable: where the windows actually land
# ---------------------------------------------------------------------------

def sample_window_timeline(store: ArchiveSeriesStore, splits: dict,
                           n_per_drive: int = 2, max_drives: int = 40_000,
                           seed: int = 42) -> pd.DataFrame:
    """A table of randomly drawn windows with their calendar span and composition.

    One row per window: where it starts in calendar time, where it starts as a fraction
    of the drive's own life, how many of its days are genuine readings rather than
    forward-filled gaps, and its label.

    Drawn over a capped sample of drives per split -- the point of the table is the
    *shape* of the sampling distribution, and 600 k drives would make the histograms no
    more informative and the plot a great deal slower.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for name in ("train", "val", "test"):
        idx = splits[name]
        if len(idx) > max_drives:
            idx = np.sort(rng.choice(idx, max_drives, replace=False))
        g = np.repeat(idx, n_per_drive)
        max_start = store.lengths[g] - store.cfg.window_days
        s = (rng.random(len(g)) * (max_start + 1)).astype(np.int64)
        np.minimum(s, max_start, out=s)

        # The observed mask is the last channel; read it alone rather than whole windows.
        n_real = np.empty(len(g), dtype=np.int32)
        step = 20_000
        for a in range(0, len(g), step):
            b = min(a + step, len(g))
            row0 = store.offsets[g[a:b]] + s[a:b]
            rows = row0[:, None] + np.arange(store.cfg.window_days)[None, :]
            n_real[a:b] = np.asarray(
                store.values[rows.ravel(), -1]).reshape(b - a, -1).sum(axis=1) \
                if store.cfg.add_observed_mask else store.cfg.window_days

        start_date = (pd.Timestamp("1970-01-01")
                      + pd.to_timedelta(store.start_days[g] + s, unit="D"))
        frames.append(pd.DataFrame({
            "split": name,
            "drive_index": g,
            "start_date": start_date,
            "end_date": start_date + pd.Timedelta(days=store.cfg.window_days - 1),
            "start_offset_days": s,
            "start_fraction_of_life": np.where(max_start > 0, s / np.maximum(max_start, 1), 0.0),
            "drive_record_days": store.lengths[g],
            "n_real_days": n_real,
            "n_filled_days": store.cfg.window_days - n_real,
            "n_padded_days": 0,
            "label": store.labels_for(g, s).astype(np.int8),
        }))
    return pd.concat(frames, ignore_index=True)


def describe_window_timeline(tl: pd.DataFrame, cfg: ArchiveConfig) -> None:
    """Print what the sampled windows cover, and check they are not calendar-aligned."""
    print(_RULE)
    print(f"STEP 2 -- RANDOM {cfg.window_days}-DAY CONTINUOUS WINDOW SAMPLING")
    print(_RULE)
    print(f"  windows drawn ........ {len(tl):,} over "
          f"{tl.drive_index.nunique():,} drives")
    print(f"  positive windows ..... {int(tl.label.sum()):,} "
          f"({100 * tl.label.mean():.4f}%)")
    print(f"  start dates .......... {tl.start_date.min():%Y-%m-%d} .. "
          f"{tl.start_date.max():%Y-%m-%d}  "
          f"({tl.start_date.dt.normalize().nunique():,} distinct start days)")
    print(f"  end dates ............ {tl.end_date.min():%Y-%m-%d} .. "
          f"{tl.end_date.max():%Y-%m-%d}")
    span = (tl.end_date - tl.start_date).dt.days + 1
    print(f"  calendar span ........ {span.min()}..{span.max()} days "
          f"-- every window is exactly {cfg.window_days} continuous days")
    print(f"  genuine readings ..... mean {tl.n_real_days.mean():.1f}/{cfg.window_days}"
          f" days per window ({tl.n_filled_days.mean():.1f} forward-filled)")

    # The alignment check: were windows snapped to calendar quarters, essentially every
    # start would land on a quarter boundary. The baseline is the share of quarter-start
    # days among the calendar days a start could have fallen on.
    lo, hi = tl.start_date.min(), tl.start_date.max()
    all_days = pd.date_range(lo, hi, freq="D")
    quarter_days = pd.date_range(lo, hi, freq="QS")
    month_days = pd.date_range(lo, hi, freq="MS")
    on_quarter = tl.start_date.isin(quarter_days).mean()
    on_month = tl.start_date.isin(month_days).mean()
    print(f"  quarter-aligned ...... {100 * on_quarter:.3f}% of starts land on one of "
          f"the {len(quarter_days)} quarter boundaries "
          f"(uniform expectation {100 * len(quarter_days) / max(len(all_days), 1):.3f}%)")
    print(f"  month-aligned ........ {100 * on_month:.3f}% "
          f"(uniform expectation {100 * len(month_days) / max(len(all_days), 1):.3f}%)")
    print(f"                         starts are NOT snapped to a calendar grid")
    print(f"  distinct start days .. {tl.start_date.dt.normalize().nunique():,} of "
          f"{len(all_days):,} calendar days in range")
    weekdays = (tl.start_date.dt.day_name().str[:3]
                .value_counts(normalize=True).sort_index())
    print("  weekday spread ....... " + ", ".join(
        f"{d} {100 * v:.1f}%" for d, v in weekdays.items()))


def plot_window_timeline(tl: pd.DataFrame, cfg: ArchiveConfig,
                         coverage: pd.DataFrame | None = None,
                         save_path: str | None = None, show: bool = True):
    """Six panels: where windows start and end, how they sit in a drive's life, and
    how many windows each split actually enumerates."""
    plt = bw._plt()
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    # -- top left: start dates
    ax = axes[0, 0]
    ax.hist(tl.start_date, bins=160, color="#3b6ea5", alpha=0.85)
    for y in pd.date_range(tl.start_date.min(), tl.start_date.max(), freq="YS"):
        ax.axvline(y, color="#999999", linestyle=":", linewidth=0.8)
    ax.set_title(f"Window start dates ({len(tl):,} sampled windows)\n"
                 "dotted = calendar year boundaries")
    ax.set_xlabel("start date")
    ax.set_ylabel("windows")
    ax.tick_params(axis="x", rotation=30)

    # -- top centre: end dates, positives overlaid
    ax = axes[0, 1]
    ax.hist(tl.end_date, bins=160, color="#7a9b57", alpha=0.85, label="all windows")
    pos = tl[tl.label == 1]
    if len(pos):
        ax.hist(pos.end_date, bins=160, color="#c1442e", alpha=0.9,
                label=f"positive ({len(pos):,})")
    ax.set_title("Window end dates")
    ax.set_xlabel("end date")
    ax.set_ylabel("windows")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=8)

    # -- top right: how many windows each split enumerates
    ax = axes[0, 2]
    if coverage is not None and "every possible 30-day window" in coverage.columns:
        col = "every possible 30-day window"
    elif coverage is not None:
        col = [c for c in coverage.columns if "possible" in c][0]
    else:
        col = None
    if col is not None:
        bars = ax.bar(coverage.index, coverage[col], color=["#3b6ea5", "#7a4fa0", "#c1442e"])
        for b, v in zip(bars, coverage[col]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v / 1e6:,.1f}M",
                    ha="center", va="bottom", fontsize=9)
        ax.set_yscale("log")
        ax.set_title("Every possible 30-day window, per split\n(stride 1 -- none skipped)")
        ax.set_ylabel("windows (log)")
    else:
        ax.axis("off")

    # -- bottom left: start position along the drive's own life
    ax = axes[1, 0]
    ax.hist(tl.start_fraction_of_life, bins=50, color="#7a4fa0", alpha=0.85)
    ax.set_title("Start position within the drive record\n"
                 "flat = not biased toward the end (the leaked signal)")
    ax.set_xlabel("start offset / (record length - window)")
    ax.set_ylabel("windows")

    # -- bottom centre: genuine readings per window
    ax = axes[1, 1]
    ax.hist(tl.n_real_days, bins=np.arange(0, cfg.window_days + 2) - 0.5,
            color="#c98b2e", alpha=0.9)
    ax.set_title(f"Genuine daily readings per {cfg.window_days}-day window\n"
                 "shortfall = forward-filled reporting gaps")
    ax.set_xlabel("observed days")
    ax.set_ylabel("windows")

    # -- bottom right: record length distribution
    ax = axes[1, 2]
    ax.hist(tl.drive_record_days, bins=80, color="#4f8a8b", alpha=0.85)
    ax.axvline(cfg.window_days, color="#c1442e", linestyle="--", linewidth=1.2,
               label=f"{cfg.window_days}-day minimum")
    ax.set_title("Drive record length\n(shorter than the window -> dropped)")
    ax.set_xlabel("days of record")
    ax.set_ylabel("windows")
    ax.legend(fontsize=8)

    for a in axes.ravel():
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


def plot_evaluation(results: dict, ks=(10, 25, 50, 100), max_points: int = 4_000_000,
                    save_path: str | None = None, show: bool = True):
    """`bw.plot_evaluation`, but on a uniform subsample of the window scores.

    The reported metrics all come from the full pass; only the *curves* are drawn from a
    subsample, because `precision_recall_curve` over 100 M+ points costs several GB and
    produces a line indistinguishable from the one a few million points give. The
    subsample is uniform, so it is unbiased -- it is not a stratified sample, which would
    distort the very base rate the plot is about.
    """
    y, p = results["test_targets"], results["test_probs"]
    if len(y) > max_points:
        rng = np.random.default_rng(0)
        take = rng.choice(len(y), max_points, replace=False)
        take.sort()
        sub = dict(results)
        sub["test_targets"], sub["test_probs"] = y[take], p[take]
        print(f"  curves drawn from {max_points:,} of {len(y):,} test windows "
              f"(uniform subsample; all printed metrics use the full pass)")
    else:
        sub = results
    return bw.plot_evaluation(sub, ks=ks, save_path=save_path, show=show)


# ---------------------------------------------------------------------------
# 11. Proving the label never reaches X
# ---------------------------------------------------------------------------

def verify_no_label_leakage(store: ArchiveSeriesStore, drives: np.ndarray,
                            n_probe: int = 20_000, seed: int = 42,
                            verbose: bool = True) -> dict:
    """Four checks that `failure` cannot be read out of the model input.

    Naming a column and dropping it is a claim. These are the tests of that claim:

    1. **Name check.** `failure` is not a channel, and no channel is derived from it.
    2. **Storage check.** `store.failure` and `store.values` are different arrays backed
       by different files -- not two views of one buffer -- so no write to one can
       surface in the other.
    3. **Independence check.** The label array is overwritten with noise and the same
       windows are gathered again. If any byte of X depended on the label, X would
       change. It must come back bit-identical.
    4. **Separability check.** Every channel is scored against the window label on its
       own. A channel that could reproduce the label would sit near AUC 1.0; real
       telemetry does not.

    Returns the per-channel scores so the notebook can show them rather than assert
    them silently.
    """
    from sklearn.metrics import roc_auc_score

    out: dict = {}
    names = list(store.feature_names)

    # -- 1. names ------------------------------------------------------------
    bad = [n for n in names if "failure" in n.lower() or "fail" in n.lower()]
    assert not bad, f"a failure-derived name is a model channel: {bad}"
    out["n_channels"] = len(names)
    out["channel_names"] = names

    # -- 2. storage ----------------------------------------------------------
    vals, fail = store.values, store.failure
    assert vals.dtype == np.float32 and fail.dtype == np.uint8
    assert np.shares_memory(vals, fail) is False, \
        "values and failure share memory -- they must be separate arrays"
    assert os.path.abspath(store._values_path) != os.path.abspath(store._failure_path)
    out["values_file"] = store._values_path
    out["failure_file"] = store._failure_path

    # -- 3. independence -----------------------------------------------------
    rng = np.random.default_rng(seed)
    pick = rng.choice(drives, size=min(len(drives), 4096), replace=False)
    starts = (rng.random(len(pick))
              * (store.lengths[pick] - store.cfg.window_days + 1)).astype(np.int64)
    np.minimum(starts, store.lengths[pick] - store.cfg.window_days, out=starts)
    before = store.gather(pick, starts).copy()

    # Swap the label array for pure noise, re-gather the same windows, then put the real
    # one back. The on-disk labels are mapped read-only, so this replaces the reference
    # rather than writing through it -- which is the same test from the gather's point of
    # view: if any byte of X came from the label, X would move. It must not.
    original = store._failure
    try:
        store._failure = rng.integers(0, 2, size=store.n_rows, dtype=np.uint8)
        after = store.gather(pick, starts)
    finally:
        store._failure = original
    identical = np.array_equal(np.nan_to_num(before), np.nan_to_num(after))
    assert identical, "X changed when the labels changed -- the label is reaching X"
    out["x_independent_of_label"] = bool(identical)
    out["n_independence_windows"] = int(len(before))

    # -- 4. separability -----------------------------------------------------
    pick = rng.choice(drives, size=min(len(drives), n_probe), replace=False)
    starts = (rng.random(len(pick))
              * (store.lengths[pick] - store.cfg.window_days + 1)).astype(np.int64)
    np.minimum(starts, store.lengths[pick] - store.cfg.window_days, out=starts)
    y = store.labels_for(pick, starts)
    rows = []
    if 0 < y.sum() < len(y):
        X = store.gather(pick, starts)
        last = X[:, -1, :]                     # the window's most recent day
        for j, name in enumerate(names):
            col = last[:, j]
            ok = np.isfinite(col)
            if ok.sum() < 100 or len(np.unique(col[ok])) < 2 or len(np.unique(y[ok])) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y[ok], col[ok]))
                auc = max(auc, 1.0 - auc)      # direction-free
            rows.append({"channel": name, "auc_vs_window_label": auc})
    scores = pd.DataFrame(rows).sort_values("auc_vs_window_label", ascending=False)
    out["channel_scores"] = scores
    out["n_probe_windows"] = int(len(pick))
    out["probe_positive_rate"] = float(y.mean())

    worst = scores.dropna(subset=["auc_vs_window_label"]).head(1)
    if len(worst):
        top_auc = float(worst.iloc[0]["auc_vs_window_label"])
        out["max_channel_auc"] = top_auc
        assert top_auc < 0.95, (
            f"channel {worst.iloc[0]['channel']} separates the label at AUC {top_auc:.3f} "
            f"-- that is a proxy, not telemetry")

    if verbose:
        print(_RULE)
        print("LABEL-LEAKAGE VERIFICATION")
        print(_RULE)
        print(f"  1. names ........... {len(names)} channels, none named after `failure`")
        print(f"  2. storage ......... values -> {os.path.basename(out['values_file'])}, "
              f"labels -> {os.path.basename(out['failure_file'])} (separate files, "
              f"no shared memory)")
        print(f"  3. independence .... labels overwritten with noise; X re-gathered "
              f"bit-identical over {len(before):,} windows  PASS")
        print(f"  4. separability .... {out['n_probe_windows']:,} probe windows, "
              f"{100 * out['probe_positive_rate']:.3f}% positive")
        print(f"     strongest single channel: "
              f"{worst.iloc[0]['channel']} at AUC {out.get('max_channel_auc', float('nan')):.4f} "
              f"(a proxy would sit at ~1.00)")
        print(f"\n  the model input is (batch, {store.cfg.window_days}, {len(names)}) and "
              f"contains SMART levels, their deltas and the observed mask -- nothing else")
    return out


# ---------------------------------------------------------------------------
# 12. Tiling the whole timeline with 30-day windows
# ---------------------------------------------------------------------------

def tiling_counts(store: ArchiveSeriesStore, drives: np.ndarray) -> np.ndarray:
    """Windows needed to cover each drive's whole record with `window_days` blocks.

    `ceil(length / window_days)`: every drive-day lands in exactly one window, and the
    last window of a short tail is pulled back so it still holds a full
    `window_days` of real calendar (drives below the window length were dropped at
    build time, so there is always room).
    """
    W = store.cfg.window_days
    return ((store.lengths[drives] + W - 1) // W).astype(np.int64)


class TilingWindows(Dataset):
    """Non-overlapping `window_days` windows covering every drive's whole timeline.

    `EnumeratedWindows` at stride 1 answers "score every possible window"; this answers
    "cover the timeline once, end to end". Every drive-day appears in exactly one window,
    so a pass over this dataset is a pass over all of the data, with no day counted twice
    and none skipped -- which is what makes it the right input for a model that should
    see the whole record rather than a sample of it.

    The final window of a drive whose length is not a multiple of `window_days` is
    aligned to its last day instead of running off the end, so it overlaps the previous
    one by the remainder rather than being short or padded.
    """

    def __init__(self, store: ArchiveSeriesStore, drives: np.ndarray,
                 batch_size: int = 4096, split_name: str = "tile"):
        self.store = store
        self.drives = np.asarray(drives, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.split_name = split_name

        self.n_win = tiling_counts(store, self.drives)
        self.win_offsets = np.zeros(len(self.drives) + 1, dtype=np.int64)
        np.cumsum(self.n_win, out=self.win_offsets[1:])
        self.n_windows = int(self.win_offsets[-1])
        self.labels = self._mark_labels()

    def locate(self, win_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Global window ids -> (drive index, within-drive start day)."""
        i = np.searchsorted(self.win_offsets, win_ids, side="right") - 1
        k = win_ids - self.win_offsets[i]
        W = self.store.cfg.window_days
        max_start = self.store.lengths[self.drives[i]] - W
        starts = np.minimum(k * W, max_start)      # last tile clamped to the record end
        return self.drives[i], starts

    def _mark_labels(self) -> np.ndarray:
        labels = np.zeros(self.n_windows, dtype=np.int8)
        ids = np.arange(self.n_windows, dtype=np.int64)
        step = 4_000_000
        for a in range(0, self.n_windows, step):
            chunk = ids[a: a + step]
            g, s = self.locate(chunk)
            labels[a: a + len(chunk)] = self.store.labels_for(g, s).astype(np.int8)
        return labels

    def __len__(self) -> int:
        return (self.n_windows + self.batch_size - 1) // self.batch_size

    def __getitem__(self, b: int):
        lo = b * self.batch_size
        hi = min(lo + self.batch_size, self.n_windows)
        ids = np.arange(lo, hi, dtype=np.int64)
        g, s = self.locate(ids)
        x = torch.from_numpy(self.store.gather(g, s))
        y = torch.from_numpy(self.labels[lo:hi].astype(np.float32)).unsqueeze(1)
        return x, y, torch.from_numpy(g.astype(np.int32))

    def coverage_report(self) -> pd.DataFrame:
        """Per-drive: record length, windows needed, days covered. The batch accounting."""
        W = self.store.cfg.window_days
        lengths = self.store.lengths[self.drives]
        return pd.DataFrame({
            "drive_index": self.drives,
            "serial_number": self.store.drive_ids[self.drives],
            "record_days": lengths,
            "windows_to_cover_timeline": self.n_win,
            "days_covered": self.n_win * W,
            "overlap_on_last_window": np.maximum(0, self.n_win * W - lengths),
            "drive_has_failure": self.store.drive_has_failure[self.drives],
        })

    def label_report(self) -> str:
        pos = int(self.labels.sum())
        return (f"{self.split_name:<5} | tiling W={self.store.cfg.window_days:<3} | "
                f"{self.n_windows:>12,} windows | {len(self.drives):>7,} drives | "
                f"positive rate {100 * pos / max(self.n_windows, 1):7.4f}% "
                f"({pos:,} positive windows) | {len(self):,} batches")


# ---------------------------------------------------------------------------
# 13. Per-window verdicts for the test phase
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_windows(model, dataset, device, threshold: float,
                    tag: str = "predict", verbose: bool = True) -> pd.DataFrame:
    """One row per window: which drive, which 30 days, predicted fail / no fail, truth."""
    model.eval()
    probs, codes = [], []
    n = len(dataset)
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    t0 = time.time()
    for i, (xb, yb, gb) in enumerate(loader):
        xb = xb.to(device, non_blocking=True)
        probs.append(torch.sigmoid(model(xb)).squeeze(1).float().cpu().numpy())
        codes.append(gb.numpy())
        if verbose and n and i % max(1, n // 10) == 0:
            print(f"    {tag}: {i + 1:,}/{n:,} batches "
                  f"({100 * (i + 1) / n:5.1f}%, {time.time() - t0:.0f}s)", flush=True)
    probs = np.concatenate(probs)
    codes = np.concatenate(codes)

    ids = np.arange(dataset.n_windows, dtype=np.int64)
    g, s = dataset.locate(ids)
    store = dataset.store
    start = (pd.Timestamp("1970-01-01")
             + pd.to_timedelta(store.start_days[g] + s, unit="D"))
    return pd.DataFrame({
        "drive_index": g,
        "serial_number": store.drive_ids[g],
        "window_in_drive": ids - dataset.win_offsets[
            np.searchsorted(dataset.win_offsets, ids, side="right") - 1],
        "window_start": start,
        "window_end": start + pd.Timedelta(days=store.cfg.window_days - 1),
        "risk": probs,
        "predicted_fail": (probs >= threshold).astype(int),
        "actual_fail": dataset.labels.astype(int),
    })


def plot_drive_histograms(pred: pd.DataFrame, n_drives: int = 24, threshold: float = 0.5,
                          seed: int = 42, save_path: str | None = None,
                          show: bool = True):
    """A risk histogram per drive, failing and healthy drives side by side.

    One panel is one drive: the distribution of its 30-day windows over predicted risk,
    with the decision threshold marked. A healthy drive should pile up against zero; a
    drive that failed should show a tail crossing the threshold in its last windows.
    """
    plt = bw._plt()
    rng = np.random.default_rng(seed)
    per_drive = pred.groupby("drive_index")["actual_fail"].max()
    failed = per_drive[per_drive == 1].index.to_numpy()
    healthy = per_drive[per_drive == 0].index.to_numpy()

    half = max(1, n_drives // 2)
    pick_f = rng.choice(failed, min(half, len(failed)), replace=False) if len(failed) else []
    pick_h = rng.choice(healthy, min(n_drives - len(pick_f), len(healthy)),
                        replace=False) if len(healthy) else []
    picks = list(pick_f) + list(pick_h)

    ncol = 6
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow),
                             squeeze=False)
    bins = np.linspace(0, 1, 26)
    n_right = n_total = 0
    for ax, gidx in zip(axes.ravel(), picks):
        sub = pred[pred.drive_index == gidx]
        did_fail = int(sub.actual_fail.max()) == 1
        colour = "#c1442e" if did_fail else "#7a9b57"
        ax.hist(sub.risk, bins=bins, color=colour, alpha=0.85)
        ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.1)
        flagged = int(sub.predicted_fail.sum())
        # Per-drive accuracy: the share of this drive's windows whose fail / no-fail
        # verdict matched the truth. On a healthy drive it is the share NOT flagged, so
        # it starts at 100% and every false alarm costs; on a failing drive the windows
        # inside the 30-day horizon have to be caught to keep it up.
        correct = int((sub.predicted_fail == sub.actual_fail).sum())
        acc = correct / max(len(sub), 1)
        n_right += correct
        n_total += len(sub)
        ax.set_title(f"{sub.serial_number.iloc[0]}\n"
                     f"{'FAILED' if did_fail else 'healthy'} | "
                     f"{len(sub)} windows, {flagged} flagged\n"
                     f"accuracy {100 * acc:.1f}%  ({correct}/{len(sub)})",
                     fontsize=7.5)
        ax.text(0.97, 0.90, f"{100 * acc:.0f}%", transform=ax.transAxes,
                ha="right", va="top", fontsize=11, fontweight="bold",
                color="#1f6f3f" if acc == 1.0 else ("#b8860b" if acc >= 0.9 else "#c1442e"),
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85))
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    for ax in axes.ravel():
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
    panel_acc = n_right / max(n_total, 1)
    all_acc = float((pred.predicted_fail == pred.actual_fail).mean())
    base = float(1.0 - pred.actual_fail.mean())
    fig.suptitle(
        f"Predicted risk per 30-day window, {len(picks)} test drives "
        f"(red = drive really failed, green = it did not; "
        f"dashed = threshold {threshold:.3f})\n"
        f"accuracy: {100 * panel_acc:.2f}% over these {n_total:,} windows  |  "
        f"{100 * all_acc:.2f}% across {len(pred):,} scored test windows  |  "
        f"always saying 'no fail' would score {100 * base:.2f}%",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, picks


# ---------------------------------------------------------------------------
# 14. Decision tree over every drive-day
# ---------------------------------------------------------------------------

AGGREGATES = ("last", "mean", "min", "max", "delta")


def window_table_features(store: ArchiveSeriesStore, dataset,
                          aggregates: Sequence[str] = ("last", "mean", "delta"),
                          chunk_windows: int = 500_000,
                          verbose: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Flatten each window into a fixed row of summary features, in chunks.

    A tree cannot take a (30, 46) tensor, and materialising every window in full would
    be tens of gigabytes. Each window is reduced to, per channel, some of: its last day,
    its mean, its minimum, its maximum and its first-to-last change -- the shape of the
    30 days rather than the 30 days themselves. Built a chunk at a time, so only the
    summary matrix is ever resident, and the number of aggregates is a knob because the
    tree's fit cost is linear in the column count.
    """
    names = list(store.feature_names)
    bad = [a for a in aggregates if a not in AGGREGATES]
    if bad:
        raise ValueError(f"unknown aggregates {bad}; pick from {AGGREGATES}")
    cols = [f"{n}__{a}" for a in aggregates for n in names]
    k = len(names)
    n_win = dataset.n_windows
    X = np.empty((n_win, k * len(aggregates)), dtype=np.float32)
    ids = np.arange(n_win, dtype=np.int64)

    t0 = time.time()
    for a0 in range(0, n_win, chunk_windows):
        b = min(a0 + chunk_windows, n_win)
        g, s = dataset.locate(ids[a0:b])
        W = store.gather(g, s)                      # (m, window_days, channels)
        for i, agg in enumerate(aggregates):
            sl = slice(i * k, (i + 1) * k)
            if agg == "last":
                X[a0:b, sl] = W[:, -1, :]
            elif agg == "mean":
                X[a0:b, sl] = W.mean(axis=1)
            elif agg == "min":
                X[a0:b, sl] = W.min(axis=1)
            elif agg == "max":
                X[a0:b, sl] = W.max(axis=1)
            else:                                    # delta
                X[a0:b, sl] = W[:, -1, :] - W[:, 0, :]
        del W
        if verbose and (a0 // chunk_windows) % 4 == 0:
            print(f"    features {b:,}/{n_win:,} ({100 * b / n_win:5.1f}%, "
                  f"{time.time() - t0:.0f}s)", flush=True)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = dataset.labels.astype(np.int8)
    if verbose:
        print(f"  feature table {X.shape[0]:,} x {X.shape[1]} float32 "
              f"({X.nbytes / 1e9:.1f} GB), {int(y.sum()):,} positive "
              f"({100 * y.mean():.4f}%)")
    return X, y, cols


def count_label_conflicts(X: np.ndarray, y: np.ndarray,
                          verbose: bool = True) -> dict:
    """Rows that are identical in every feature but disagree on the label.

    This is the exact ceiling on training accuracy. No decision tree, however deep, can
    separate two rows it cannot tell apart, so if this count is non-zero then 100% is
    not merely hard to reach -- it is unreachable, and the best any tree can do is to
    take the majority label in each conflicted group.
    """
    t0 = time.time()
    # Hash whole rows by viewing the (contiguous) buffer as one void field per row --
    # far cheaper than np.unique(..., axis=0), which sorts the full matrix.
    Xc = np.ascontiguousarray(X)
    rows = Xc.view(np.dtype((np.void, Xc.dtype.itemsize * Xc.shape[1]))).ravel()
    codes, _ = pd.factorize(pd.Series(rows), sort=False)

    df = pd.DataFrame({"g": codes, "y": y.astype(np.int8)})
    agg = df.groupby("g")["y"].agg(["min", "max", "size", "sum"])
    mixed = agg[agg["min"] != agg["max"]]
    n_groups = int(len(mixed))
    n_rows = int(mixed["size"].sum())
    # In a mixed group the tree can only get the majority right; the minority is lost.
    minority = int((mixed[["sum"]].assign(neg=mixed["size"] - mixed["sum"])
                    .min(axis=1)).sum()) if n_groups else 0
    ceiling = 1.0 - minority / len(y)
    out = {"conflict_groups": n_groups, "rows_in_conflict": n_rows,
           "unreachable_rows": minority, "max_train_accuracy": ceiling,
           "seconds": time.time() - t0}
    if verbose:
        print(f"  {n_groups:,} groups of identical feature rows carry both labels "
              f"({n_rows:,} rows, {time.time() - t0:.0f}s)")
        print(f"  {minority:,} rows can never be classified correctly by any tree")
        print(f"  => ceiling on training accuracy: {ceiling:.8f}")
    return out


def grow_tree_to_full_accuracy(X_tr, y_tr, X_te, y_te, depths=None,
                               max_depth_cap: int = 200, seed: int = 42,
                               verbose: bool = True):
    """Deepen a decision tree until it classifies its training set perfectly.

    The tree is grown in depth steps and every step is reported, because the gap between
    the two accuracy columns is the whole point: a tree with no depth limit and no leaf
    minimum will always reach 1.000 on the data it was fitted to -- it can carve a leaf
    per sample -- and that number says nothing about a disk it has not seen. The final
    unbounded fit is what "as big as needed for 100%" asks for; the test column next to
    it is what that costs.

    Accuracy is also the wrong lens at this base rate: predicting "never fails" for every
    window scores about 99.9%. Balanced accuracy is reported alongside for that reason.
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import balanced_accuracy_score

    rows = []
    depths = depths or [2, 4, 6, 8, 12, 16, 24, 32]
    best = None
    for d in list(depths) + [None]:
        t0 = time.time()
        clf = DecisionTreeClassifier(max_depth=d, random_state=seed,
                                     min_samples_leaf=1, min_samples_split=2)
        clf.fit(X_tr, y_tr)
        tr_acc = float(clf.score(X_tr, y_tr))
        te_acc = float(clf.score(X_te, y_te))
        tr_bal = float(balanced_accuracy_score(y_tr, clf.predict(X_tr)))
        te_pred = clf.predict(X_te)
        te_bal = float(balanced_accuracy_score(y_te, te_pred))
        rows.append({
            "max_depth": "none" if d is None else d,
            "actual_depth": int(clf.get_depth()),
            "leaves": int(clf.get_n_leaves()),
            "train_accuracy": tr_acc,
            "test_accuracy": te_acc,
            "train_balanced_acc": tr_bal,
            "test_balanced_acc": te_bal,
            "fit_seconds": time.time() - t0,
        })
        if verbose:
            r = rows[-1]
            print(f"  depth {str(r['max_depth']):>4} -> actual {r['actual_depth']:>3}, "
                  f"{r['leaves']:>9,} leaves | train acc {tr_acc:.6f} "
                  f"(bal {tr_bal:.4f}) | test acc {te_acc:.6f} (bal {te_bal:.4f}) "
                  f"| {r['fit_seconds']:.0f}s", flush=True)
        best = clf
        if tr_acc >= 1.0:
            if verbose:
                print(f"  reached 100.0000% training accuracy at depth "
                      f"{clf.get_depth()} with {clf.get_n_leaves():,} leaves")
            break
    return best, pd.DataFrame(rows)
