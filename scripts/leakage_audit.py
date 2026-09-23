"""Step 1 -- per-drive telemetry analysis and target-leakage detection.

The question this module answers before a single feature reaches a model: *is any
column in the frame a proxy for `failure` rather than a predictor of it?*

The analogy in the brief is a "failing grade of 40" used to predict pass/fail -- a
column that is not evidence about the outcome but a restatement of it. In Backblaze
drive telemetry that kind of column does exist, but it is almost never one of the SMART
attributes. It shows up in three other places, and this module probes all three:

1.  **Raw column association.**  Every telemetry column scored against the drive-day
    `failure` flag with a point-biserial correlation, a rank AUC, and the best
    single-threshold balanced accuracy (Youden's J). A column that separates the classes
    at AUC ~1.0 on its own is a giveaway, not a feature.

2.  **Missingness as a channel.**  A column can leak through *whether it is null* even
    when its values are innocuous. Operational pipelines routinely stop populating a
    field once a drive is pulled from service, which stamps the outcome onto the row.
    Scored as the gap P(null | failure) - P(null | healthy).

3.  **Record geometry.**  The single largest leak in any drive-stats dataset. Backblaze
    marks `failure = 1` on a drive's *last* reporting day, so any feature derived from
    where a row sits relative to the end of that drive's record -- days-to-last-record,
    record length, "still reporting at export end" -- reconstructs the label directly.
    These columns are not in the frame; they are things a feature-engineering step would
    *create*, so the module builds them itself and scores them as a warning.

The output is a `LeakageReport`: the scored tables, a `blocklist` of column names that
must never enter X, and `assert_clean()` to enforce it downstream.

Typical use
-----------
    from scripts.leakage_audit import audit_frame, select_feature_columns

    report = audit_frame(df)
    report.print_report()
    features = select_feature_columns(df, report, min_coverage=0.5)
    report.assert_clean(features)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ID_LIKE = ("date", "serial_number", "model", "failure")

# Columns that are identifiers or fleet topology rather than drive telemetry. They are
# not scored as features; several are separately checked as categorical leaks.
NON_TELEMETRY = (
    "datacenter", "cluster_id", "vault_id", "pod_id", "pod_slot_num",
    "is_legacy_format", "capacity_bytes",
)


@dataclass
class LeakageConfig:
    """Thresholds separating "genuine signal" from "restatement of the label"."""

    # A column that alone ranks failures above healthy days this well is not a
    # predictor. 0.95 is deliberately far above what real SMART telemetry reaches.
    severe_auc: float = 0.95
    suspect_auc: float = 0.85
    # Best single cut point: >= this balanced accuracy from one threshold is the
    # "failing grade of 40" signature.
    severe_balanced_acc: float = 0.90
    # |point-biserial r| that no honest telemetry column reaches on a 0.09% positive rate.
    severe_corr: float = 0.50
    # P(null | failure) - P(null | healthy). A field that stops being populated once a
    # drive is pulled leaks the outcome through its own absence.
    severe_missing_gap: float = 0.25
    suspect_missing_gap: float = 0.08
    # Categorical: highest per-level failure rate as a multiple of the base rate,
    # counting only levels with enough support to be real.
    severe_lift: float = 20.0
    suspect_lift: float = 5.0
    min_level_support: int = 1_000

    # Feature-eligibility (separate from leakage: these are "useless", not "cheating").
    min_coverage: float = 0.50
    min_unique: int = 3


@dataclass
class LeakageReport:
    columns: pd.DataFrame          # per-column numeric association
    categoricals: pd.DataFrame     # per-column categorical association
    geometry: pd.DataFrame         # derived record-geometry probes
    drives: pd.DataFrame           # per-drive telemetry summary
    blocklist: list[str]
    cfg: LeakageConfig = field(default_factory=LeakageConfig)
    base_rate: float = float("nan")
    n_rows: int = 0
    n_drives: int = 0

    # -- reporting ------------------------------------------------------------

    def _flagged(self, verdict: str) -> pd.DataFrame:
        frames = [t[t.verdict == verdict] for t in
                  (self.columns, self.categoricals, self.geometry) if len(t)]
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        # A geometry probe is scored at two levels; report it once, at its worst.
        order = out.get("auc_directional", pd.Series(0.0, index=out.index)).fillna(0.0)
        return (out.assign(_o=order).sort_values("_o", ascending=False)
                   .drop_duplicates("column").drop(columns="_o"))

    @property
    def severe(self) -> pd.DataFrame:
        return self._flagged("SEVERE LEAK")

    @property
    def suspect(self) -> pd.DataFrame:
        sus = self._flagged("suspect")
        if len(sus) and len(self.severe):
            sus = sus[~sus.column.isin(set(self.severe.column))]
        return sus

    def print_report(self, top_n: int = 12) -> None:
        rule = "=" * 92
        print(rule)
        print("STEP 1 -- TARGET LEAKAGE AND PROXY AUDIT")
        print(rule)
        print(f"  drive-days ........... {self.n_rows:,}")
        print(f"  distinct drives ...... {self.n_drives:,}")
        print(f"  failure events ....... {int(round(self.base_rate * self.n_rows)):,} "
              f"(base rate {100 * self.base_rate:.4f}%)")
        print()
        print(f"  thresholds: severe if rank-AUC >= {self.cfg.severe_auc:.2f}, or "
              f"1-threshold balanced acc >= {self.cfg.severe_balanced_acc:.2f},")
        print(f"              or |r| >= {self.cfg.severe_corr:.2f}, or missingness gap "
              f">= {self.cfg.severe_missing_gap:.2f}, or level lift >= "
              f"{self.cfg.severe_lift:.0f}x")

        print()
        print("-" * 92)
        print(f"A. TELEMETRY COLUMNS vs failure -- top {top_n} by rank AUC")
        print("-" * 92)
        cols = self.columns.sort_values("auc_directional", ascending=False)
        print(_fmt(cols.head(top_n), ["column", "coverage", "n_unique", "corr",
                                      "auc_directional", "balanced_acc", "verdict"]))

        print()
        print("-" * 92)
        print("B. MISSINGNESS AS A LEAK CHANNEL -- P(null | failure) vs P(null | healthy)")
        print("-" * 92)
        miss = self.columns.reindex(
            self.columns.missing_gap.abs().sort_values(ascending=False).index
        )
        print(_fmt(miss.head(6), ["column", "null_rate", "null_if_failure",
                                  "null_if_healthy", "missing_gap", "verdict"]))

        print()
        print("-" * 92)
        print("C. IDENTIFIER / TOPOLOGY COLUMNS -- categorical association")
        print("-" * 92)
        print(_fmt(self.categoricals.sort_values("max_lift", ascending=False),
                   ["column", "cardinality", "cramers_v", "max_lift", "verdict"]))

        print()
        print("-" * 92)
        print("D. RECORD-GEOMETRY PROBES -- features a naive engineering step would build")
        print("-" * 92)
        for lvl in ("per drive-day", "per drive"):
            sub = self.geometry[self.geometry.level == lvl]
            if not len(sub):
                continue
            tgt = "failure flag on that day" if lvl == "per drive-day" else "drive ever failed"
            print(f"  [{lvl}]  target = {tgt}")
            print(_fmt(sub.sort_values("auc_directional", ascending=False),
                       ["column", "auc_directional", "balanced_acc", "corr", "verdict"]))
            print()

        print()
        print(rule)
        print("VERDICT")
        print(rule)
        sev = self.severe
        if len(sev):
            print(f"  {len(sev)} column(s) flagged SEVERE -- dropped from X before "
                  f"preprocessing:")
            for _, r in sev.iterrows():
                print(f"    - {r['column']:<28} {r['reason']}")
        else:
            print("  no severe leak found")
        sus = self.suspect
        if len(sus):
            print(f"\n  {len(sus)} column(s) flagged SUSPECT -- also excluded, "
                  f"see reasons:")
            for _, r in sus.iterrows():
                print(f"    - {r['column']:<28} {r['reason']}")
        print(f"\n  blocklist ({len(self.blocklist)} names) enforced by "
              f"LeakageReport.assert_clean():")
        print("    " + ", ".join(self.blocklist))
        print(rule)

    # -- enforcement ----------------------------------------------------------

    def assert_clean(self, feature_columns) -> None:
        """Raise if any blocklisted name reached the feature set."""
        blocked = set(self.blocklist)
        hit = [c for c in feature_columns if c in blocked]
        if hit:
            raise ValueError(
                f"leaked proxy column(s) present in the feature set X: {hit}. "
                f"They must be dropped before preprocessing."
            )
        # Derived names too: `smart_5_raw_d7` is fine, `days_to_last_d7` is not.
        derived = [c for c in feature_columns
                   if any(c.startswith(b + "_") for b in blocked)]
        if derived:
            raise ValueError(f"feature(s) derived from a blocked column: {derived}")
        print(f"[leakage guard] {len(list(feature_columns))} feature columns checked "
              f"against a {len(blocked)}-name blocklist -- clean.")


# ---------------------------------------------------------------------------
# scoring primitives
# ---------------------------------------------------------------------------

def _fmt(df: pd.DataFrame, cols: list[str]) -> str:
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return out.to_string(index=False)


def _numeric_association(x: np.ndarray, y: np.ndarray) -> dict:
    """Point-biserial r, rank AUC, and the best single-threshold balanced accuracy.

    All three are computed on the rows where `x` is observed. A column can only be a
    proxy through the values it actually has; its absence is scored separately.
    """
    ok = ~np.isnan(x)
    xo, yo = x[ok], y[ok]
    out = {"coverage": float(ok.mean()), "n_unique": int(np.unique(xo).size) if xo.size else 0}
    if xo.size == 0 or out["n_unique"] < 2 or yo.sum() == 0 or yo.sum() == yo.size:
        out.update(corr=np.nan, auc=np.nan, auc_directional=np.nan, balanced_acc=np.nan)
        return out

    sd = xo.std()
    out["corr"] = float(np.corrcoef(xo, yo)[0, 1]) if sd > 0 else 0.0
    auc = float(roc_auc_score(yo, xo))
    out["auc"] = auc
    out["auc_directional"] = max(auc, 1.0 - auc)
    # Youden's J over every cut point, direction-agnostic: balanced accuracy of the best
    # single threshold. This is the direct test for "one rule reproduces the label".
    fpr, tpr, _ = roc_curve(yo, xo)
    out["balanced_acc"] = float((1.0 + np.max(np.abs(tpr - fpr))) / 2.0)
    return out


def _missingness(x: np.ndarray, y: np.ndarray) -> dict:
    null = np.isnan(x)
    if not null.any():
        return {"null_rate": 0.0, "null_if_failure": 0.0, "null_if_healthy": 0.0,
                "missing_gap": 0.0}
    p1 = float(null[y == 1].mean()) if (y == 1).any() else np.nan
    p0 = float(null[y == 0].mean()) if (y == 0).any() else np.nan
    return {"null_rate": float(null.mean()), "null_if_failure": p1,
            "null_if_healthy": p0, "missing_gap": p1 - p0}


def _cramers_v(x: pd.Series, y: np.ndarray) -> float:
    ct = pd.crosstab(x, y).to_numpy(dtype=float)
    if min(ct.shape) < 2:
        return 0.0
    n = ct.sum()
    expected = ct.sum(1, keepdims=True) @ ct.sum(0, keepdims=True) / n
    chi2 = ((ct - expected) ** 2 / np.maximum(expected, 1e-12)).sum()
    return float(np.sqrt((chi2 / n) / (min(ct.shape) - 1)))


def _verdict(row: dict, cfg: LeakageConfig) -> tuple[str, str]:
    """Classify one scored column and say why in one line."""
    auc = row.get("auc_directional", np.nan)
    bacc = row.get("balanced_acc", np.nan)
    corr = abs(row.get("corr", 0.0) or 0.0)
    gap = abs(row.get("missing_gap", 0.0) or 0.0)
    lift = row.get("max_lift", np.nan)

    if np.isfinite(auc) and auc >= cfg.severe_auc:
        return "SEVERE LEAK", f"rank AUC {auc:.3f} on its own -- reproduces the label"
    if np.isfinite(bacc) and bacc >= cfg.severe_balanced_acc:
        return "SEVERE LEAK", f"one threshold gives balanced accuracy {bacc:.3f}"
    if corr >= cfg.severe_corr:
        return "SEVERE LEAK", f"|point-biserial r| = {corr:.3f}"
    if gap >= cfg.severe_missing_gap:
        return ("SEVERE LEAK",
                f"null on {100 * row['null_if_failure']:.1f}% of failure days vs "
                f"{100 * row['null_if_healthy']:.1f}% of healthy days")
    if np.isfinite(lift) and lift >= cfg.severe_lift:
        return "SEVERE LEAK", f"a single level carries {lift:.1f}x the base failure rate"

    if np.isfinite(auc) and auc >= cfg.suspect_auc:
        return "suspect", f"rank AUC {auc:.3f} -- unusually high for telemetry"
    if gap >= cfg.suspect_missing_gap:
        return "suspect", f"missingness gap {gap:.3f} between failure and healthy days"
    if np.isfinite(lift) and lift >= cfg.suspect_lift:
        return "suspect", f"a single level carries {lift:.1f}x the base failure rate"
    return "clean", ""


# ---------------------------------------------------------------------------
# the three probes
# ---------------------------------------------------------------------------

def audit_numeric_columns(
    df: pd.DataFrame, y: np.ndarray, columns: list[str], cfg: LeakageConfig
) -> pd.DataFrame:
    rows = []
    for c in columns:
        v = df[c]
        if v.dtype == bool:
            v = v.astype("float64")
        if not np.issubdtype(v.dtype, np.number):
            continue
        x = v.to_numpy(dtype=np.float64)
        row = {"column": c, "level": "drive-day"}
        row.update(_numeric_association(x, y))
        row.update(_missingness(x, y))
        row["verdict"], row["reason"] = _verdict(row, cfg)
        rows.append(row)
    return pd.DataFrame(rows)


def audit_categoricals(
    df: pd.DataFrame, y: np.ndarray, columns: list[str], cfg: LeakageConfig
) -> pd.DataFrame:
    """Identifier and topology columns, scored by Cramer's V and per-level lift.

    Per-level lift only counts levels with `min_level_support` rows behind them: on a
    0.09% base rate a level of 20 drive-days containing one failure shows a lift of
    1000x that is pure sampling noise.
    """
    base = float(y.mean())
    rows = []
    for c in columns:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype.kind in "fiu":
            filled = s.fillna(-1)
        else:
            filled = s.astype("object").where(s.notna(), "<NA>")
        grouped = pd.DataFrame({"lvl": filled, "y": y}).groupby("lvl", observed=True)["y"]
        stats = grouped.agg(["size", "mean"])
        stats = stats[stats["size"] >= cfg.min_level_support]
        row = {
            "column": c,
            "level": "drive-day",
            "cardinality": int(s.nunique(dropna=True)),
            "cramers_v": _cramers_v(filled, y),
            "max_lift": float((stats["mean"] / base).max()) if len(stats) else np.nan,
        }
        # An identifier column leaks through absence as readily as through value: a slot
        # assignment that stops being written once the drive is pulled is a label stamp.
        row.update(_missingness(
            s.to_numpy(dtype=np.float64) if s.dtype.kind in "fiu"
            else np.where(s.isna().to_numpy(), np.nan, 0.0), y))
        row["verdict"], row["reason"] = _verdict(row, cfg)
        rows.append(row)
    return pd.DataFrame(rows)


def build_record_geometry(df: pd.DataFrame, id_col: str = "serial_number",
                          time_col: str = "date") -> pd.DataFrame:
    """The derived columns a naive feature step would build from record boundaries.

    None of these exist in the raw frame. They are constructed here precisely so the
    audit can score them and put them on the blocklist before anyone adds them.
    """
    g = df.groupby(id_col, sort=False)[time_col]
    first, last = g.transform("min"), g.transform("max")
    export_end = df[time_col].max()
    return pd.DataFrame({
        "days_since_first_record": (df[time_col] - first).dt.days,
        "days_to_last_record": (last - df[time_col]).dt.days,
        "record_length_days": (last - first).dt.days + 1,
        "still_reporting_at_export_end": (last == export_end).astype("int8"),
        "days_from_record_end_to_export_end": (export_end - last).dt.days,
        "calendar_day_index": (df[time_col] - df[time_col].min()).dt.days,
    })


def audit_record_geometry(df: pd.DataFrame, y: np.ndarray, cfg: LeakageConfig,
                          id_col: str = "serial_number",
                          time_col: str = "date") -> pd.DataFrame:
    """Score the geometry probes at both levels the task can be posed at.

    Drive-day level dilutes them: a failing drive contributes 800 healthy rows and one
    failure row, so a per-drive constant like `record_length_days` is attached to far
    more negatives than positives. Per *drive*, against "did this drive ever fail",
    the same columns are far sharper -- and that is exactly the level at which a
    grouped-by-drive split makes its decisions, so it is the level that matters.
    """
    geo = build_record_geometry(df, id_col, time_col)
    day_level = audit_numeric_columns(geo, y, list(geo.columns), cfg)
    day_level["level"] = "per drive-day"

    # Drive-level probes are per-drive constants, so aggregating is just taking one row
    # per drive. `days_to_last_record` and `days_since_first_record` are per-day
    # counters with no drive-level meaning and are left out rather than collapsed.
    drive_probes = ["record_length_days", "still_reporting_at_export_end",
                    "days_from_record_end_to_export_end"]
    per_drive = geo[drive_probes].copy()
    per_drive[id_col] = df[id_col].to_numpy()
    per_drive["_y"] = y
    agg = per_drive.groupby(id_col, sort=True).agg(
        {**{c: "first" for c in drive_probes}, "_y": "max"}
    )
    yd = agg.pop("_y").to_numpy(dtype=np.int8)
    drive_level = audit_numeric_columns(agg, yd, drive_probes, cfg)
    drive_level["level"] = "per drive"

    return pd.concat([day_level, drive_level], ignore_index=True)


def per_drive_summary(df: pd.DataFrame, id_col: str = "serial_number",
                      time_col: str = "date") -> pd.DataFrame:
    """One row per drive: coverage, span, calendar gaps, and the drive-level label.

    This is the table the windowing step in Step 2 draws from -- a drive can only offer
    a 90-day window if its record spans at least 90 days.
    """
    g = df.groupby(id_col)
    out = pd.DataFrame({
        "first_date": g[time_col].min(),
        "last_date": g[time_col].max(),
        "n_observed_days": g[time_col].size(),
        "ever_failed": g["failure"].max().astype("int8"),
    })
    if "model" in df.columns:
        out["model"] = g["model"].first()
    out["span_days"] = (out.last_date - out.first_date).dt.days + 1
    out["calendar_gap_days"] = out.span_days - out.n_observed_days
    out["coverage"] = out.n_observed_days / out.span_days
    out["n_windows_90d"] = (out.span_days - 90 + 1).clip(lower=0)
    return out.reset_index()


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------

def audit_frame(
    df: pd.DataFrame,
    cfg: LeakageConfig | None = None,
    id_col: str = "serial_number",
    time_col: str = "date",
    target_col: str = "failure",
    verbose: bool = True,
) -> LeakageReport:
    """Score every column, every missingness pattern, and every record-geometry probe."""
    cfg = cfg or LeakageConfig()
    y = df[target_col].to_numpy(dtype=np.int8)

    telemetry = [c for c in df.columns
                 if c not in ID_LIKE and c not in NON_TELEMETRY]
    categoricals = [c for c in NON_TELEMETRY if c in df.columns]

    if verbose:
        print(f"[audit] scoring {len(telemetry)} telemetry columns, "
              f"{len(categoricals)} identifier/topology columns, and "
              f"{len(build_record_geometry(df.head(1), id_col, time_col).columns)} "
              f"derived record-geometry probes over {len(df):,} drive-days ...")

    columns = audit_numeric_columns(df, y, telemetry, cfg)
    cats = audit_categoricals(df, y, categoricals, cfg)
    geometry = audit_record_geometry(df, y, cfg, id_col, time_col)
    drives = per_drive_summary(df, id_col, time_col)

    blocked = set()
    for table in (columns, cats, geometry):
        if len(table):
            blocked |= set(table.loc[table.verdict != "clean", "column"])
    # The identity column and the raw timestamp are proxies by construction rather than
    # by measurement: a per-drive model keyed on serial number memorises the outcome,
    # and the calendar date carries the cohort's construction (healthy drives were
    # sampled to survive to the export end, so "late date" implies "healthy").
    blocked |= {id_col, time_col, target_col}

    return LeakageReport(
        columns=columns, categoricals=cats, geometry=geometry, drives=drives,
        blocklist=sorted(blocked), cfg=cfg, base_rate=float(y.mean()),
        n_rows=len(df), n_drives=int(df[id_col].nunique()),
    )


def select_feature_columns(
    df: pd.DataFrame,
    report: LeakageReport,
    suffix: str = "_raw",
    min_coverage: float | None = None,
    min_unique: int | None = None,
) -> list[str]:
    """The surviving feature set X: well-covered, informative, and provably not leaked.

    Raw SMART attributes only. The `_normalized` twin of an attribute is the vendor's
    own rescaling of the same counter onto a 1-100 health scale; it carries no extra
    information and its quantisation (often 2-5 distinct values) hurts a sequence model.
    """
    cfg = report.cfg
    min_coverage = cfg.min_coverage if min_coverage is None else min_coverage
    min_unique = cfg.min_unique if min_unique is None else min_unique

    blocked = set(report.blocklist)
    tbl = report.columns
    keep = tbl[
        tbl.column.str.endswith(suffix)
        & (tbl.coverage >= min_coverage)
        & (tbl.n_unique >= min_unique)
        & ~tbl.column.isin(blocked)
    ]
    return sorted(keep.column.tolist(), key=lambda c: -float(
        tbl.loc[tbl.column == c, "auc_directional"].iloc[0]))


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib
    import matplotlib.pyplot as plt
    return plt


def plot_audit(report: LeakageReport, save_path: str | None = None, show: bool = True):
    """Four panels: what the audit found, and what the fleet it ran on looks like.

    The point of the top row is the contrast. Real telemetry columns sit in a band
    around AUC 0.5-0.8; the record-geometry probes sit against the right-hand edge.
    That gap is the difference between a predictor and a restatement of the label.
    """
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    BLUE, RED, AMBER, GREY = "#3b6ea5", "#c1442e", "#d99a2b", "#9aa0a6"
    colour_of = {"clean": BLUE, "suspect": AMBER, "SEVERE LEAK": RED}

    # -- top left: telemetry association, ranked
    ax = axes[0, 0]
    tel = (report.columns.dropna(subset=["auc_directional"])
           .sort_values("auc_directional", ascending=False).head(20).iloc[::-1])
    ax.barh(tel.column, tel.auc_directional,
            color=[colour_of.get(v, GREY) for v in tel.verdict])
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.axvline(report.cfg.suspect_auc, color=AMBER, linestyle="--", linewidth=1,
               label=f"suspect >= {report.cfg.suspect_auc}")
    ax.axvline(report.cfg.severe_auc, color=RED, linestyle="--", linewidth=1,
               label=f"severe >= {report.cfg.severe_auc}")
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Telemetry columns vs failure (top 20 by rank AUC)", fontsize=10)
    ax.set_xlabel("directional rank AUC, one column alone")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8, loc="lower right")

    # -- top right: the geometry probes, where the leak actually is
    ax = axes[0, 1]
    geo = (report.geometry.dropna(subset=["auc_directional"])
           .assign(lbl=lambda d: d.column + "  [" + d.level.str.replace("per ", "") + "]")
           .sort_values("auc_directional").tail(12))
    ax.barh(geo.lbl, geo.auc_directional,
            color=[colour_of.get(v, GREY) for v in geo.verdict])
    ax.axvline(0.5, color=GREY, linestyle=":", linewidth=1)
    ax.axvline(report.cfg.severe_auc, color=RED, linestyle="--", linewidth=1)
    ax.set_xlim(0.45, 1.0)
    ax.set_title("Record-geometry probes -- derived columns that reproduce the label",
                 fontsize=10)
    ax.set_xlabel("directional rank AUC, one column alone")
    ax.tick_params(axis="y", labelsize=7)

    # -- bottom left: the missingness channel
    ax = axes[1, 0]
    miss = pd.concat([report.columns, report.categoricals], ignore_index=True)
    miss = miss.dropna(subset=["missing_gap"])
    miss = miss.reindex(miss.missing_gap.abs().sort_values().index).tail(10)
    ax.barh(miss.column, miss.missing_gap,
            color=[colour_of.get(v, GREY) for v in miss.verdict])
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axvline(report.cfg.severe_missing_gap, color=RED, linestyle="--", linewidth=1,
               label=f"severe >= {report.cfg.severe_missing_gap}")
    ax.set_title("Leakage through absence:  P(null | failure) - P(null | healthy)",
                 fontsize=10)
    ax.set_xlabel("difference in null rate")
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8)

    # -- bottom right: the per-drive view the windowing step will draw from
    ax = axes[1, 1]
    d = report.drives
    ax.hist([d.loc[d.ever_failed == 0, "span_days"], d.loc[d.ever_failed == 1, "span_days"]],
            bins=40, stacked=True, color=["#7a9b57", RED],
            label=["never failed", "failed"], edgecolor="white", linewidth=0.3)
    ax.axvline(90, color="#333333", linestyle="--", linewidth=1.2,
               label="90-day window minimum")
    ax.set_title(f"Per-drive record span  ({len(d):,} drives, "
                 f"{int(d.ever_failed.sum()):,} with a failure)", fontsize=10)
    ax.set_xlabel("days of telemetry per drive")
    ax.set_ylabel("drives")
    ax.legend(fontsize=8)

    for a in axes.ravel():
        a.grid(alpha=0.25, linewidth=0.6)
        a.set_axisbelow(True)
    fig.suptitle("Step 1 -- telemetry association and target-leakage audit", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path:
        fig.savefig(save_path, dpi=130, bbox_inches="tight")
        print(f"  saved {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
