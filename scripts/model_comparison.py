"""Every model on the same windows: CNN variations, transformers, XGBoost, one tree.

The experiment log (`overfit_experiments.ExperimentLog`) scores the CNN and transformer
runs on stride-30 windows, and XGBoost on tiling windows -- so its rows are comparable
within a protocol but not across one. This module puts every model on **one** footing:

* the same 3.09M **tiling** test windows of the same 82k test drives,
* a threshold tuned the same way for every model -- best F1 on the **tiling validation**
  windows, frozen before test is read (exactly XGBoost's rule in the notebook),
* the same metrics: PR-AUC, ROC-AUC, precision / recall / F1 / balanced accuracy at that
  threshold, and Precision@K over windows and over drives (each drive by its worst
  window).

    rows = [score_probs("XGBoost", "xgboost", p_val, y_val, p_test, y_test, codes)]
    rows.append(score_network(model, "tf_summary", "transformer", tiles, device))
    table, pk = comparison_table(rows)
    write_comparison_workbook(table, pk, "results/model_comparison.xlsx",
                              exp_log_path="results/experiments_results.xlsx")

`write_comparison_workbook` is the Excel deliverable: the head-to-head ranking, the best
model of each family, the stride-30 leaderboard of every CNN and transformer run,
Precision@K side by side, and how each run trained -- with native Excel charts.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch

import archive_window_pipeline as ap
import backblaze_window_pipeline as bw
import overfit_experiments as ox

KS = (10, 25, 50, 100, 250, 1000)
FAMILY_ORDER = ("transformer", "xgboost", "cnn", "tree")
FAMILY_LABEL = {"cnn": "CNN", "transformer": "Transformer", "xgboost": "XGBoost",
                "tree": "Decision tree"}
# One fill per family, used on every sheet so a model's family reads at a glance.
FAMILY_FILL = {"cnn": "DCE6F1", "transformer": "E2EFDA", "xgboost": "FCE4D6",
               "tree": "EDEDED"}


# ---------------------------------------------------------------------------
# 0. The transformer runs
# ---------------------------------------------------------------------------

# The transformer's own settings; everything else (batch, epochs, patience, loss, the
# training stream) is inherited from the CNN's TrainConfig so the two differ only in the
# network.
TRANSFORMER_BASE = {"arch": "summary_transformer", "hidden_dim": 128, "n_layers": 3,
                    "n_heads": 4, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-5}


def transformer_variations() -> list[dict]:
    """One knob at a time around `TRANSFORMER_BASE`, like the CNN's variations.

    `tf_plain` is the control: the same encoder with day tokens only, no summary tokens
    and no [CLS] -- so the gap between it and `tf_summary` is what the summary tokens
    add, and the gap between it and the CNN is what attention alone adds.
    """
    return [
        {"name": "tf_summary"},
        {"name": "tf_plain", "arch": "transformer"},
        {"name": "tf_summary_6layers", "n_layers": 6},
        {"name": "tf_summary_width64", "hidden_dim": 64},
        {"name": "tf_summary_lr3e-04", "lr": 3e-4},
    ]


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(model, dataset, device, tag: str = "predict",
                  verbose: bool = False) -> np.ndarray:
    """P(fail) for every window of `dataset`, in dataset order."""
    _, _, probs, _, _ = ap.evaluate_archive_loader(
        model, ox._loader(dataset, device), torch.nn.BCEWithLogitsLoss(), device,
        tag=tag, verbose=verbose)
    return probs


def score_probs(name: str, family: str, p_val: np.ndarray, y_val: np.ndarray,
                p_test: np.ndarray, y_test: np.ndarray, codes_test: np.ndarray,
                ks=KS, threshold: float | None = None, note: str = "") -> dict:
    """One comparison row from validation and test scores.

    The threshold is the best-F1 cut on validation unless one is given (the decision
    tree never saw validation and is cut at 0.5). Validation PR-AUC is kept in the row
    because that -- never test -- is what picks each family's winner.
    """
    if threshold is None:
        threshold, _ = ap.sweep_threshold(np.asarray(y_val), np.asarray(p_val))
    val = bw.compute_metrics(y_val, p_val, threshold) if p_val is not None else {}
    m = bw.compute_metrics(y_test, p_test, threshold)
    pk_win = bw.precision_at_k_table(y_test, p_test, ks).assign(ranked="windows")
    _, p_drv, y_drv = ap.drive_level_scores(p_test, y_test, codes_test)
    pk_drv = bw.precision_at_k_table(y_drv, p_drv, ks).assign(ranked="drives")
    row = {
        "model": name, "family": family,
        "val_pr_auc": val.get("pr_auc", np.nan),
        "test_pr_auc": m["pr_auc"], "test_roc_auc": m["roc_auc"],
        "threshold": float(threshold),
        "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
        "bal_acc": ox.balanced_accuracy(m),
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "test_windows": m["n"], "test_positives": m["n_pos"],
        "test_drives": len(y_drv), "failed_drives": int(y_drv.sum()),
        "note": note,
    }
    for _, r in pk_drv.iterrows():
        row[f"drive_P@{int(r['k'])}"] = float(r["precision"])
        row[f"drive_hits@{int(r['k'])}"] = int(r["hits"])
    for _, r in pk_win.iterrows():
        row[f"window_P@{int(r['k'])}"] = float(r["precision"])
    row["_pk"] = pd.concat([pk_win, pk_drv], ignore_index=True).assign(model=name)
    return row


def score_network(model, name: str, family: str, tiles: dict, device,
                  note: str = "", ks=KS) -> dict:
    """Score a CNN or transformer on the tiling validation and test windows."""
    t0 = time.time()
    model.eval()
    p_val = predict_probs(model, tiles["val"], device, tag=f"{name} val")
    p_test = predict_probs(model, tiles["test"], device, tag=f"{name} test")
    codes, _ = tiles["test"].locate(np.arange(tiles["test"].n_windows, dtype=np.int64))
    row = score_probs(name, family, p_val, tiles["val"].labels, p_test,
                      tiles["test"].labels, codes, ks=ks, note=note)
    print(f"  {name:<28} {family:<12} val PR-AUC {row['val_pr_auc']:.4f}  "
          f"test PR-AUC {row['test_pr_auc']:.4f}  drive P@100 {row['drive_P@100']:.2f}  "
          f"({time.time() - t0:.0f}s)", flush=True)
    return row


def load_network(path: str, num_features: int, window_days: int, device,
                 tcfg_overrides: dict | None = None):
    """A model from an experiment checkpoint (`.pt`) or a saved artifact (`.pth`)."""
    if path.endswith(".pt"):
        model, tcfg = ox.load_checkpoint(path, num_features, window_days, device)
        return model, tcfg
    tcfg = bw.TrainConfig(**(tcfg_overrides or {}))
    model = bw.build_model(num_features, window_days, tcfg)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval(), tcfg


def comparison_table(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per model ranked by drive P@100 then PR-AUC, Precision@K long table)."""
    pk = pd.concat([r["_pk"] for r in rows], ignore_index=True)
    table = pd.DataFrame([{k: v for k, v in r.items() if k != "_pk"} for r in rows])
    return _rank(table), pk[["model", "ranked", "k", "hits", "precision"]]


def _rank(table: pd.DataFrame) -> pd.DataFrame:
    """Rank by drive P@100 then PR-AUC; mark each family's best on validation PR-AUC."""
    table = table.drop(columns=["rank", "family_winner"], errors="ignore")
    table = table.sort_values(["drive_P@100", "test_pr_auc"], ascending=False)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    is_best = table.groupby("family")["val_pr_auc"].transform(
        lambda s: s == s.max() if s.notna().any() else pd.Series(True, index=s.index))
    table.insert(3, "family_winner", np.where(is_best, "Yes", ""))
    return table.reset_index(drop=True)


TILING_CSV = "results/w2022x_tiling_comparison.csv"
TILING_PK_CSV = "results/w2022x_tiling_precision_at_k.csv"


def merge_into_csv(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add newly scored models to the saved comparison, replacing same-named rows."""
    new, new_pk = comparison_table(rows)
    old, old_pk = pd.read_csv(TILING_CSV), pd.read_csv(TILING_PK_CSV)
    table = _rank(pd.concat([old[~old["model"].isin(new["model"])], new],
                            ignore_index=True))
    pk = pd.concat([old_pk[~old_pk["model"].isin(new["model"])], new_pk],
                   ignore_index=True)
    table.to_csv(TILING_CSV, index=False)
    pk.to_csv(TILING_PK_CSV, index=False)
    return table, pk


# The second round: every network on XGBoost's training set -- all 10.5M tiling windows
# per epoch at natural prevalence -- so a gap between them is the architecture, not the
# data. The transformers get the transformer recipe (AdamW, warm-up + cosine, gradient
# clipping); the CNN keeps the recipe it was tuned with.
TF_RECIPE = {"optimizer": "adamw", "schedule": "warmup_cosine", "warmup_epochs": 0.5,
             "grad_clip": 1.0, "weight_decay": 0.05, "lr": 1e-3, "dropout": 0.1}
TILING_RUNS = {
    "transformer": [
        {"name": "tf_channel_tiling", "arch": "channel_transformer", "hidden_dim": 128,
         "n_layers": 3, "n_heads": 4, **TF_RECIPE},
        {"name": "tf_day_tiling", "arch": "summary_transformer", "hidden_dim": 64,
         "n_layers": 3, "n_heads": 4, **TF_RECIPE},
    ],
    "cnn": [
        # lr_3e-03, the best CNN on validation, unchanged except for the stream.
        {"name": "cnn_tiling", "arch": "cnn", "dilations": (1, 2, 4, 8, 16),
         "kernel_size": 3, "hidden_dim": 128, "dropout": 0.1, "lr": 3e-3,
         "weight_decay": 1e-5},
    ],
}


def family_winners(table: pd.DataFrame) -> pd.DataFrame:
    """Each family's model with the highest *validation* PR-AUC."""
    w = table[table["family_winner"] == "Yes"].copy()
    w["_o"] = w["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return w.sort_values(["drive_P@100", "test_pr_auc"], ascending=False).drop(columns="_o")


def verdict(table: pd.DataFrame, col: str, tie_pct: float = 5.0) -> str:
    """'A (0.99, next best B at 0.93)', or a tie when the top two are within 5%."""
    s = table.set_index("model")[col].astype(float).sort_values(ascending=False)
    (a_name, a), (b_name, b) = list(s.items())[:2]
    if a <= 0 or 100 * (a - b) / a < tie_pct:
        return f"{a_name} and {b_name} are tied ({a:.4f} vs {b:.4f})"
    return f"{a_name} ({a:.4f}, next best {b_name} at {b:.4f})"


# ---------------------------------------------------------------------------
# 2. The workbook
# ---------------------------------------------------------------------------

HEAD_COLS = ["rank", "model", "family", "family_winner", "drive_P@100", "drive_hits@100",
             "drive_P@1000", "drive_hits@1000", "test_pr_auc", "test_roc_auc",
             "val_pr_auc", "precision", "recall", "f1", "bal_acc", "threshold",
             "window_P@100", "window_P@1000", "tp", "fp", "fn", "note"]
HEAD_LABELS = {
    "rank": "Rank", "model": "Model", "family": "Family", "family_winner": "Family winner",
    "drive_P@100": "Drive precision @100", "drive_hits@100": "Drives hit @100",
    "drive_P@1000": "Drive precision @1000", "drive_hits@1000": "Drives hit @1000",
    "test_pr_auc": "Test PR-AUC", "test_roc_auc": "Test ROC-AUC",
    "val_pr_auc": "Val PR-AUC (selects)", "precision": "Precision", "recall": "Recall",
    "f1": "F1", "bal_acc": "Balanced accuracy", "threshold": "Threshold (from val)",
    "window_P@100": "Window precision @100", "window_P@1000": "Window precision @1000",
    "tp": "TP", "fp": "FP", "fn": "FN", "note": "Note",
}
# Higher is better for these; the best value in each is bolded and tinted green.
BEST_COLS = ("drive_P@100", "drive_P@1000", "test_pr_auc", "test_roc_auc", "precision",
             "recall", "f1", "bal_acc", "window_P@100", "window_P@1000")


def _styles():
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    return {
        "title": Font(bold=True, size=14), "bold": Font(bold=True),
        "head_font": Font(bold=True, color="FFFFFF"),
        "head_fill": PatternFill("solid", start_color="305496"),
        "band": PatternFill("solid", start_color="DCE6F1"),
        "best": PatternFill("solid", start_color="C6EFCE"),
        "wrap": Alignment(wrap_text=True, vertical="top"),
        "center": Alignment(horizontal="center", vertical="center", wrap_text=True),
        "thin": Border(bottom=Side(style="thin", color="BFBFBF")),
    }


def _fill(hex_):
    from openpyxl.styles import PatternFill
    return PatternFill("solid", start_color=hex_)


def _write_table(ws, df: pd.DataFrame, top: int = 1, labels: dict | None = None,
                 family_col: str | None = "family", best_cols=(), pct_cols=(),
                 int_cols=()) -> int:
    """Header + rows at row `top`; family-tinted rows, best-per-column highlighted.

    Returns the row after the table.
    """
    from openpyxl.utils import get_column_letter
    st = _styles()
    cols = list(df.columns)
    for j, c in enumerate(cols, 1):
        cell = ws.cell(top, j, (labels or {}).get(c, c))
        cell.font, cell.fill, cell.alignment = st["head_font"], st["head_fill"], st["center"]
    ws.row_dimensions[top].height = 32
    best = {c: df[c].astype(float).max() for c in best_cols if c in df and df[c].notna().any()}
    for i, row in enumerate(df.itertuples(index=False), top + 1):
        fam = getattr(row, family_col) if family_col and family_col in cols else None
        for j, c in enumerate(cols, 1):
            v = row[j - 1]
            if isinstance(v, (np.floating, float)) and np.isnan(v):
                v = None
            elif isinstance(v, np.generic):
                v = v.item()
            if c == family_col and v in FAMILY_LABEL:
                v = FAMILY_LABEL[v]
            cell = ws.cell(i, j, v)
            if fam in FAMILY_FILL:
                cell.fill = _fill(FAMILY_FILL[fam])
            if isinstance(v, float):
                cell.number_format = ("0.0%" if c in pct_cols else
                                      "0.0" if str(c).endswith("_pct") else "0.0000")
            elif isinstance(v, int) and c not in int_cols and c != "rank":
                cell.number_format = "#,##0"
            if c in best and v is not None and np.isclose(float(v), best[c]):
                cell.fill, cell.font = st["best"], st["bold"]
    for j, c in enumerate(cols, 1):
        head = str((labels or {}).get(c, c))
        vals = [str(x) for x in df[c].head(200).tolist()]
        width = max([min(len(head), 18)] + [len(v) if not isinstance(df[c].iloc[0], float)
                                             else 8 for v in vals])
        ws.column_dimensions[get_column_letter(j)].width = min(max(9, width + 2), 44)
    ws.freeze_panes = ws.cell(top + 1, 3)
    ws.auto_filter.ref = f"A{top}:{get_column_letter(len(cols))}{top + len(df)}"
    return top + len(df) + 1


def _bar_chart(ws, title: str, cats_ref, data_ref, anchor: str, y_title: str,
               height: float = 9, width: float = 22, y_max: float | None = None):
    from openpyxl.chart import BarChart
    ch = BarChart()
    ch.type, ch.style = "bar", 10
    ch.title, ch.y_axis.title = title, y_title
    ch.add_data(data_ref, titles_from_data=True)
    ch.set_categories(cats_ref)
    ch.height, ch.width = height, width
    ch.legend = None
    ch.x_axis.scaling.orientation = "maxMin"      # rank 1 at the top
    if y_max is not None:
        ch.y_axis.scaling.min, ch.y_axis.scaling.max = 0, y_max
    ws.add_chart(ch, anchor)


def select_models(table: pd.DataFrame, pk: pd.DataFrame,
                  families=("transformer", "xgboost", "cnn"),
                  best_only=("cnn",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The rows the workbook shows: every run of each family in `families`, except that
    a family in `best_only` contributes only its winner (chosen on validation)."""
    keep = table["family"].isin(families) & (
        ~table["family"].isin(best_only) | (table["family_winner"] == "Yes"))
    t = table[keep].sort_values(["drive_P@100", "test_pr_auc"], ascending=False).copy()
    t["rank"] = np.arange(1, len(t) + 1)
    return t.reset_index(drop=True), pk[pk["model"].isin(t["model"])]


def write_comparison_workbook(table: pd.DataFrame, pk: pd.DataFrame, path: str,
                              exp_log_path: str | None = None,
                              context: dict | None = None,
                              families=("transformer", "xgboost", "cnn"),
                              best_only=("cnn",)) -> str:
    """The comparison workbook: every transformer, XGBoost, and the best CNN.

    `families` / `best_only` choose the rows (see `select_models`); pass
    `best_only=()` to list every CNN variation too. Sheets:

    * `Overview` -- what was compared, how, the winners, and how to read the rest.
    * `Head_to_Head` -- every model on the same tiling test windows, ranked by drive
      precision @100 then PR-AUC, best value per column highlighted, with charts.
    * `Family_Winners` -- one model per family, each chosen on validation PR-AUC.
    * `PrecisionAtK` -- drive and window Precision@K for every model, side by side.
    * `Stride30_Leaderboard` -- every CNN and transformer run from the experiment log,
      on the stride-30 windows they were trained and selected on.
    * `Training` -- how each run trained: epochs, stop reason, overfitting verdict.
    * `Epoch_History` -- per-epoch validation PR-AUC of every run, with a chart.
    """
    from openpyxl import Workbook
    from openpyxl.chart import LineChart, Reference
    from openpyxl.utils import get_column_letter

    st = _styles()
    context = context or {}
    table, pk = select_models(table, pk, families, best_only)
    shown = set(table["model"])
    wb = Workbook()

    # -- Head_to_Head ---------------------------------------------------------
    ws = wb.active
    ws.title = "Head_to_Head"
    ws.cell(1, 1, "HEAD TO HEAD -- every transformer, XGBoost and the best CNN, on the "
                  "same test windows").font = st["title"]
    n_win = int(table["test_windows"].iloc[0])
    n_drv = int(table["test_drives"].iloc[0])
    ws.cell(2, 1, f"{n_win:,} test windows over {n_drv:,} test drives "
                  f"({int(table['failed_drives'].iloc[0]):,} failed). Ranked by drive "
                  f"precision @100, then test PR-AUC. Best value in each column in green. "
                  f"Thresholds tuned on the tiling validation windows.")
    cols = [c for c in HEAD_COLS if c in table]
    end = _write_table(ws, table[cols], top=4, labels=HEAD_LABELS, best_cols=BEST_COLS)
    n = len(table)
    cats = Reference(ws, min_col=cols.index("model") + 1, min_row=5, max_row=4 + n)
    for k, (col, title, ymax) in enumerate((
            ("drive_P@100", "Drive precision @100 (of 100 flagged drives, share that failed)", 1.0),
            ("test_pr_auc", "Test PR-AUC (whole ranking)", None))):
        j = cols.index(col) + 1
        _bar_chart(ws, title, cats, Reference(ws, min_col=j, min_row=4, max_row=4 + n),
                   f"{get_column_letter(2)}{end + 2 + k * 20}", HEAD_LABELS[col],
                   height=max(7, 0.55 * n + 3), y_max=ymax)

    # -- Family_Winners ---------------------------------------------------------
    ws = wb.create_sheet("Family_Winners")
    ws.cell(1, 1, "BEST MODEL OF EACH FAMILY").font = st["title"]
    ws.cell(2, 1, "Each family's winner is the run with the highest VALIDATION PR-AUC; "
                  "test is read only after the choice, so these test numbers are honest.")
    win = family_winners(table)
    end = _write_table(ws, win[cols].drop(columns=["rank", "family_winner"]), top=4,
                       labels=HEAD_LABELS, best_cols=BEST_COLS)
    r = end + 1
    ws.cell(r, 1, "Verdict").font = st["bold"]
    ws.cell(r + 1, 1, "Best at flagging drives (precision @100)")
    ws.cell(r + 1, 3, verdict(win, "drive_P@100"))
    ws.cell(r + 2, 1, "Best ranking overall (test PR-AUC)")
    ws.cell(r + 2, 3, verdict(win, "test_pr_auc"))
    ws.cell(r + 3, 1, "Ties: when the top two are within 5% of each other.")
    wcols = list(win[cols].drop(columns=["rank", "family_winner"]).columns)
    nw = len(win)
    cats = Reference(ws, min_col=wcols.index("model") + 1, min_row=5, max_row=4 + nw)
    for k, col in enumerate(("drive_P@100", "test_pr_auc", "f1")):
        j = wcols.index(col) + 1
        _bar_chart(ws, HEAD_LABELS[col], cats,
                   Reference(ws, min_col=j, min_row=4, max_row=4 + nw),
                   f"{get_column_letter(1 + 8 * k)}{r + 6}", HEAD_LABELS[col],
                   height=7, width=13, y_max=1.0 if "P@" in col else None)

    # -- PrecisionAtK -----------------------------------------------------------
    ws = wb.create_sheet("PrecisionAtK")
    ws.cell(1, 1, "PRECISION@K -- of the K items a model flags first, the share that "
                  "really failed").font = st["title"]
    order = table["model"].tolist()
    top = 3
    for ranked in ("drives", "windows"):
        sub = pk[pk["ranked"] == ranked].pivot(index="k", columns="model",
                                               values="precision")[order]
        sub = sub.reset_index().rename(columns={"k": "K"})
        ws.cell(top, 1, f"Ranking {ranked.upper()}"
                + (" (each drive by its worst window)" if ranked == "drives" else "")
                ).font = st["bold"]
        end = _write_table(ws, sub, top=top + 1, family_col=None)
        # best model per K, highlighted along the row
        for i in range(len(sub)):
            vals = sub.iloc[i, 1:].astype(float)
            for j, v in enumerate(vals, 2):
                if np.isclose(v, vals.max()):
                    c = ws.cell(top + 2 + i, j)
                    c.fill, c.font = st["best"], st["bold"]
        ch = LineChart()
        ch.title = f"Precision@K, ranking {ranked}"
        ch.y_axis.title, ch.x_axis.title = "precision", "K"
        ch.y_axis.scaling.min, ch.y_axis.scaling.max = 0, 1
        ch.height, ch.width = 9, 20
        ch.add_data(Reference(ws, min_col=2, max_col=1 + len(order), min_row=top + 1,
                              max_row=top + 1 + len(sub)), titles_from_data=True)
        ch.set_categories(Reference(ws, min_col=1, min_row=top + 2, max_row=top + 1 + len(sub)))
        ws.add_chart(ch, f"{get_column_letter(len(order) + 3)}{top}")
        top = max(end + 2, top + 20)
    ws.freeze_panes = None

    # -- stride-30 leaderboard, training, epochs (from the experiment log) -----
    if exp_log_path and os.path.exists(exp_log_path):
        log = ox.ExperimentLog(exp_log_path, resume=True)
        s = log.summary()
        s = s[s["model_type"].isin(["cnn", "transformer"])
              & s["variation"].isin(shown)].copy()
        if len(s):
            proto = s["eval_protocol"].mode().iloc[0]
            s = s[s["eval_protocol"] == proto].sort_values("val_pr_auc", ascending=False)
            s["rank_by_val"] = np.arange(1, len(s) + 1)
            cfgs = log.frames["Configs"].set_index("variation")
            s["arch"] = s["variation"].map(cfgs["arch"]) if "arch" in cfgs else s["model_type"]
            keep = ["rank_by_val", "variation", "model_type", "arch", "layers", "dilations",
                    "hidden_dim", "dropout", "lr", "parameters", "best_epoch",
                    "stopped_epoch", "overfitting_triggered", "val_pr_auc", "test_pr_auc",
                    "train_pr_auc", "pr_auc_gap_pct", "test_roc_auc", "test_bal_acc",
                    "test_f1", "test_P@100_drives", "test_P@1000_drives", "train_minutes"]
            lab = {"rank_by_val": "Rank (by val)", "variation": "Run",
                   "model_type": "Family", "test_P@100_drives": "Test drive P@100",
                   "test_P@1000_drives": "Test drive P@1000",
                   "pr_auc_gap_pct": "PR-AUC gap % (train over val)",
                   "overfitting_triggered": "Overfitting triggered"}
            ws = wb.create_sheet("Stride30_Leaderboard")
            ws.cell(1, 1, "EVERY TRANSFORMER AND THE BEST CNN -- the windows they "
                          "were selected on").font = st["title"]
            ws.cell(2, 1, f"Protocol: {proto}. Ranked by validation PR-AUC. XGBoost is "
                          f"not here: it was never scored on these windows -- see "
                          f"Head_to_Head for it.")
            sub = s[[c for c in keep if c in s]]
            end = _write_table(ws, sub, top=4, labels=lab, family_col="model_type",
                               best_cols=("val_pr_auc", "test_pr_auc", "test_roc_auc",
                                          "test_bal_acc", "test_f1", "test_P@100_drives",
                                          "test_P@1000_drives"))
            sc = list(sub.columns)
            _bar_chart(ws, "Validation PR-AUC by run (stride-30 windows)",
                       Reference(ws, min_col=sc.index("variation") + 1, min_row=5,
                                 max_row=4 + len(sub)),
                       Reference(ws, min_col=sc.index("val_pr_auc") + 1, min_row=4,
                                 max_row=4 + len(sub)),
                       f"B{end + 2}", "val PR-AUC", height=max(7, 0.5 * len(sub) + 3))

            ws = wb.create_sheet("Training")
            ws.cell(1, 1, "HOW EACH RUN TRAINED").font = st["title"]
            t = s[["variation", "model_type", "parameters", "best_epoch", "stopped_epoch",
                   "stop_reason", "status", "overfitting_triggered", "pr_auc_gap_pct",
                   "loss_gap_pct", "train_minutes"]].copy()
            _write_table(ws, t, top=3, family_col="model_type",
                         labels={"variation": "Run", "model_type": "Family",
                                 "status": "Overfitting verdict",
                                 "stop_reason": "Stopped because"})

            ep = log.frames["Epoch History"]
            runs = s["variation"].tolist()
            wide = ep[ep["variation"].isin(runs)].pivot_table(
                index="epoch", columns="variation", values="val_pr_auc")
            wide = wide[[r for r in runs if r in wide.columns]].reset_index()
            ws = wb.create_sheet("Epoch_History")
            ws.cell(1, 1, "VALIDATION PR-AUC BY EPOCH -- transformers and the best CNN").font = st["title"]
            _write_table(ws, wide, top=3, family_col=None)
            ch = LineChart()
            ch.title = "Validation PR-AUC by epoch"
            ch.x_axis.title, ch.y_axis.title = "epoch", "val PR-AUC"
            ch.height, ch.width = 11, 24
            ch.add_data(Reference(ws, min_col=2, max_col=wide.shape[1], min_row=3,
                                  max_row=3 + len(wide)), titles_from_data=True)
            ch.set_categories(Reference(ws, min_col=1, min_row=4, max_row=3 + len(wide)))
            ws.add_chart(ch, f"{get_column_letter(wide.shape[1] + 2)}3")

    # -- Overview (first sheet) ---------------------------------------------------
    ws = wb.create_sheet("Overview", 0)
    ws.column_dimensions["A"].width = 34
    ws.column_dimensions["B"].width = 100
    ws.cell(1, 1, "MODEL COMPARISON -- CNN vs XGBoost vs Transformer").font = st["title"]
    ws.cell(2, 1, context.get("subtitle", "Backblaze fleet, 2022 onward: does this drive "
                                          "fail within 30 days of a 30-day SMART window?"))
    r = 4

    def kv(k, v, bold=False):
        nonlocal r
        a = ws.cell(r, 1, k)
        a.font = st["bold"]
        b = ws.cell(r, 2, v)
        b.alignment = st["wrap"]
        if bold:
            b.font = st["bold"]
        r += 1

    def band(label):
        nonlocal r
        r += 1
        for j in (1, 2):
            ws.cell(r, j).fill = st["band"]
        ws.cell(r, 1, label).font = st["bold"]
        r += 1

    band("Result")
    kv("Best at flagging drives", verdict(win, "drive_P@100"), bold=True)
    kv("Best ranking (PR-AUC)", verdict(win, "test_pr_auc"), bold=True)
    for _, w in win.iterrows():
        kv(f"{FAMILY_LABEL.get(w['family'], w['family'])} winner",
           f"{w['model']}: drive P@100 {w['drive_P@100']:.2f} ({int(w['drive_hits@100'])}"
           f"/100), drive P@1000 {w['drive_P@1000']:.3f}, test PR-AUC "
           f"{w['test_pr_auc']:.4f}, ROC-AUC {w['test_roc_auc']:.4f}, F1 {w['f1']:.4f}")
    for k, v in context.get("findings", {}).items():
        kv(k, v)

    band("How the comparison is kept fair")
    kv("Same windows", f"Every model is scored on the same {n_win:,} tiling test windows "
                       f"of the same {n_drv:,} test drives. No drive is in two splits.")
    kv("Same threshold rule", "Each model's decision threshold is the best-F1 cut on the "
                              "tiling VALIDATION windows, frozen before test is read.")
    kv("Selection on validation", "Within a family the winner is the run with the highest "
                                  "validation PR-AUC. Picking on test would reward the "
                                  "run that got luckiest on this one test split.")
    kv("Same training protocol", "CNN and transformer runs share the training stream "
                                 "(4 random windows per drive per epoch, 25% forced "
                                 "positives), early stopping on validation PR-AUC "
                                 "(patience 3), the overfitting rule and seed 42. XGBoost "
                                 "trains on the flattened tiling windows (last, mean, "
                                 "delta per channel).")
    kv("Two window sets", "Head_to_Head uses tiling windows (each drive-day once, base rate "
                          "0.24%). Stride30_Leaderboard uses the stride-30 windows the "
                          "networks were selected on (base rate 0.13%). PR-AUC moves with "
                          "the base rate, so compare numbers only within one sheet.")

    band("Sheets")
    for name, what in (("Head_to_Head", "every transformer, XGBoost and the best CNN, ranked; charts"),
                       ("Family_Winners", "one model per family and the verdict"),
                       ("PrecisionAtK", "drive and window Precision@K for every model, with charts"),
                       ("Stride30_Leaderboard", "the transformers and the best CNN on their selection windows"),
                       ("Training", "best/stopped epoch, stop reason, overfitting verdict per run"),
                       ("Epoch_History", "validation PR-AUC per epoch for every run, charted")):
        if name in wb.sheetnames:
            kv(name, what)

    band("Colour key")
    for fam in FAMILY_ORDER:
        if fam in set(table["family"]):
            ws.cell(r, 1, FAMILY_LABEL[fam]).fill = _fill(FAMILY_FILL[fam])
            ws.cell(r, 2, "rows of this family are tinted this colour")
            r += 1
    ws.cell(r, 1, "Best in column").fill = st["best"]
    ws.cell(r, 2, "the best value in a metric column")
    r += 2
    ws.cell(r, 1, f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.xlsx"
    wb.save(tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:
        alt = path.replace(".xlsx", f"_{time.strftime('%Y%m%d_%H%M%S')}.xlsx")
        os.replace(tmp, alt)
        print(f"  {path} is locked (open in Excel?) -- wrote {alt} instead")
        return alt
    print(f"  wrote {path}")
    return path


def rebuild_from_csv(out: str = "results/model_comparison.xlsx",
                     exp_log_path: str = "results/experiments_results.xlsx",
                     **kwargs) -> str:
    """Re-write the workbook from the CSVs the notebook saved -- no model is re-scored."""
    table = pd.read_csv("results/w2022x_tiling_comparison.csv")
    table["family_winner"] = table["family_winner"].fillna("")
    pk = pd.read_csv("results/w2022x_tiling_precision_at_k.csv")
    n_cnn = int((table["family"] == "cnn").sum())
    best = table[(table["family"] == "cnn") & (table["family_winner"] == "Yes")]
    context = {"findings": {
        "Best CNN": f"{best['model'].iloc[0]} -- the highest validation PR-AUC of the "
                    f"{n_cnn} CNN runs (the other CNN variations are in "
                    f"experiments_results.xlsx)."}} if len(best) else None
    return write_comparison_workbook(table, pk, out, exp_log_path=exp_log_path,
                                     context=context, **kwargs)


if __name__ == "__main__":
    # python scripts/model_comparison.py [out.xlsx]   -- run from the project root
    import sys
    rebuild_from_csv(*sys.argv[1:2])
