"""Add the notebook's comparison tables to `results/experiments_results.xlsx`.

    python scripts/collect_results_to_excel.py [workbook.xlsx]

`Dataset_backblaze_Analysis_2022_XGBoost.ipynb` logs every CNN variation, transformer
and XGBoost to the workbook as it trains (`overfit_experiments.ExperimentLog`). The
cross-model tables -- the four-model summary, every model on the tiling windows, CNN vs
XGBoost, the tree depth ladder, the architecture search -- are written as CSVs at the end
of the run. This copies them in as extra sheets,
so one file holds every result.

Run it after the notebook finishes. `ExperimentLog.save()` rewrites only its own sheets,
so if the notebook runs again, run this again afterwards. Re-running replaces the sheets
it added; it never touches the experiment sheets.
"""
import os
import sys

import pandas as pd
from openpyxl import load_workbook

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import overfit_experiments as ox

WORKBOOK = os.path.join("results", "experiments_results.xlsx")

# sheet name -> (csv, keep the index column?)
TABLES = {
    "Final_Model_Summary": ("results/w2022x_model_summary.csv", False),
    "All_Models_Tiling": ("results/w2022x_tiling_comparison.csv", False),
    "CNN_vs_XGBoost": ("results/w2022x_cnn_vs_xgboost.csv", False),
    "CNN_vs_XGB_PrecisionAtK": ("results/w2022x_cnn_vs_xgboost_precision_at_k.csv", False),
    "Tree_Depth_Ladder": ("results/w2022x_tree_depth_ladder.csv", False),
    "Arch_Search_30_Trials": ("results/w2022x_cnn_search.csv", False),
    "CNN_Stride1_PrecisionAtK": ("results/w2022x_precision_at_k.csv", False),
    "CNN_Selected_Epochs": ("results/w2022x_training_history.csv", False),
    "XGB_Feature_Importance": ("results/w2022x_xgboost_importance.csv", False),
    "Top_100_Drives": ("results/w2022x_top_drives.csv", False),
}


def refresh_status(path: str) -> None:
    """Recompute each run's verdict with the current `ox.overfit_status`.

    Runs logged before the verdict wording was split ("gap over limit" vs "trend only")
    carry the older, coarser label; the diagnostics needed to redo it are in the sheet.
    """
    log = ox.ExperimentLog(path, resume=True)
    diag = log.frames["Diagnostics"]
    if not len(diag):
        return
    diag["status"] = [ox.overfit_status(r) for r in diag.to_dict("records")]
    status = diag.set_index("variation")["status"]
    tbl = log.frames["Summary Table"]
    tbl["status"] = tbl["variation"].map(status)
    log.save(path)


def main(path: str = WORKBOOK) -> None:
    if not os.path.exists(path):
        sys.exit(f"{path} not found -- run the notebook first")
    refresh_status(path)
    wb = load_workbook(path)
    added, missing = [], []
    for sheet, (csv, _) in TABLES.items():
        if not os.path.exists(csv):
            missing.append(csv)
            continue
        if sheet in wb.sheetnames:
            del wb[sheet]
        df = pd.read_csv(csv)
        ws = wb.create_sheet(sheet)
        ws.append(list(df.columns))
        for row in df.itertuples(index=False):
            ws.append([None if (isinstance(v, float) and pd.isna(v)) else v for v in row])
        ox._format_sheet(ws)
        added.append(f"{sheet} ({len(df)} rows)")
    # The cross-model verdict reads best right after the per-variation leaderboard.
    if "Final_3Model_Summary" in wb.sheetnames:      # the pre-transformer name
        del wb["Final_3Model_Summary"]
    if "Final_Model_Summary" in wb.sheetnames and "Summary Table" in wb.sheetnames:
        wb.move_sheet("Final_Model_Summary",
                      offset=wb.sheetnames.index("Summary Table") + 1
                      - wb.sheetnames.index("Final_Model_Summary"))
    wb.save(path)
    print(f"updated {path}")
    for a in added:
        print(f"  + {a}")
    for m in missing:
        print(f"  ! missing {m} (that part of the notebook has not run)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else WORKBOOK)
