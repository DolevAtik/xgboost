# Toshiba predictive-maintenance pipeline

The Toshiba counterpart to `Dataset_backblaze_Comparison.ipynb`, run against the local
`data/Toshiba/` export instead of downloaded Backblaze quarters, with a **90-day input
window** and a pooled model trained across all 14 Toshiba drive families.

## Files

| file | what it is |
|---|---|
| `scripts/build_toshiba_parquet.py` | converts the 105 xlsx workbooks into parquet (run once) |
| `data/toshiba_drivestats.parquet` | **3,224,289 drive-days x 197 columns, 34.5 MB** — the whole export in one file |
| `data/toshiba_parquet/model=<MODEL>/data.parquet` | the same data split per drive model |
| `scripts/make_toshiba_notebook.py` | generates the notebook (regenerate after a config change) |
| `Dataset_Toshiba_Comparison.ipynb` | the analysis + training notebook, with outputs |

Trained artifacts written by the notebook, into `models/`:
`toshiba_drive_failure_dnn.pth`, `toshiba_drive_failure_xgb.json`,
`toshiba_rul_dnn.pth`, `toshiba_rul_xgb.json` and the matching
`*_preprocessing_config.json` files; and into `results/`:
`toshiba_per_model_metrics.csv` / `toshiba_cross_model_comparison.csv`.

## Why parquet

`data/Toshiba/` is 105 Excel workbooks totalling 0.7 GB. A single `pd.read_excel` pass over
all of them takes about five minutes and materialises all 197 columns as float64.
The parquet is 34.5 MB, loads in under a second, and lets a reader pull only the
columns it needs. Rebuild it with:

```bash
python3 scripts/build_toshiba_parquet.py
```

Read it back with:

```python
import pyarrow.parquet as pq
df = pq.read_table("data/toshiba_drivestats.parquet",
                   columns=["date", "serial_number", "model", "failure", "smart_5_raw"]).to_pandas()
```

The schema is pinned in the build script rather than inferred, because a column that
happens to be all-null in one export window would otherwise land as float64 there and
int64 elsewhere, and the streaming writer rejects a mid-file schema change.

## The 90-day input window

Every prediction is made from the drive's trailing 90 days — 91 daily readings, today
plus the 90 before it. Per SMART attribute:

- the current reading
- `_d7`, `_d30`, `_d90` — change over 7 / 30 / 90 days
- `_mean90`, `_std90`, `_max90` — the window's level, volatility, and worst value

plus window span, drive age, capacity, and a one-hot of the drive model:
**129 features** over 16 SMART attributes.

Rows without a genuine full window are dropped rather than median-filled into looking
complete (2.74M of 3.22M rows survive). The window is built positionally because 99.9%
of consecutive readings are exactly one day apart, and every row carries the real
calendar span of its window so the 0.1% that are not get filtered out.

## Feature selection is Toshiba-specific

The Seagate feature list does not carry over. `smart_187`, `smart_188` and `smart_241` —
three of the nine attributes the Backblaze notebook is built on — are **not reported at
all** by these Toshiba families. Of the 93 raw attributes in the export, most are either
empty or constant across the entire fleet; 16 are both well-covered and informative.
Five of those (`smart_191`, `197`, `220`, `222`, `226`) are missing from the two "EY"
families, which is why the notebook separates `CORE_ATTRS` from `EXTENDED_ATTRS`.

Section 2a of the notebook derives this from the data — that is the step to repeat for
any new manufacturer.

## One bug worth knowing about

Both learners were being held back by the same mistake, reached two different ways:
**correcting the class imbalance twice.** The DNN drew class-balanced batches from a
`WeightedRandomSampler` *and* applied `pos_weight=37` to the loss. XGBoost was given
`scale_pos_weight` equal to the raw class ratio, which made validation PR-AUC peak after
about a dozen trees and then decline, so early stopping handed back a barely-trained
12-tree model.

Every headline metric here (PR-AUC, Precision@K) scores the **ranking** of drives, and
inflating the positive class distorts the gradient without improving that ranking. The
notebook now measures the imbalance strategy on validation instead of assuming one —
Section 10 for the DNN, Section 16b for XGBoost. It was the single change that moved
the numbers most.

## Caveat on the numbers

The export is a **cohort, not the fleet**: every failed drive plus a matched sample of
healthy drives. Ranking metrics (PR-AUC, Precision@K) transfer to production; the
absolute positive rate does not. Three families (`MD04ABA400V`, `HDWE160`,
`MG11ACA16TE`) have no failures in the export and contribute only negatives.
