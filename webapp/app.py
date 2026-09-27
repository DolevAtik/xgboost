"""Local prediction console -- drop in a telemetry workbook, get a ranked risk list.

    python webapp/app.py            then open http://127.0.0.1:5000

Drop in an `.xlsx` (or `.csv`) of daily SMART telemetry -- one drive or a whole family --
and the trained models score every drive in it. The file is parsed, checked against the
model's input contract, laid on a dense daily calendar, scaled with the statistics
**saved at training time**, and scored. Nothing is written to disk and nothing leaves the
machine.

The current run is `models/backblaze_2022x_w30`: a 30-day window over the Backblaze fleet
from 2022 on, predicting failure within 30 days of the window's last day. It carries two
estimators trained on the identical windows -- a dilated CNN over the sequence and
XGBoost over a flattened summary of it -- so the page can run either or put them side by
side. XGBoost wins on that test set (PR-AUC 0.388 against 0.323, 99 of the top 100 drives
correct against 92) and is the default.

`webapp/scoring.py` holds the artifact loading and everything model-shaped; this file is
parsing, routing and the shape of the JSON the page renders.

The workbooks in `data/Toshiba/<MODEL>/<start>_<end>.xlsx` and the purpose-built files in
`data/examples/` are exactly the right shape, and the page offers both as one-click chips.

Startup is a few seconds: only the artifacts are loaded, never the fleet.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import uuid

import numpy as np
import pandas as pd
import torch
from flask import Flask, jsonify, render_template, request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scoring  # noqa: E402

# Example files are addressed relative to data/, so the raw Toshiba export and the
# purpose-built examples can be offered from one endpoint behind one path check.
DATA_DIR = os.path.join(ROOT, "data")
SAMPLE_DIR = DATA_DIR
TOSHIBA_DIR = os.path.join(DATA_DIR, "Toshiba")
EXAMPLES_DIR = os.path.join(DATA_DIR, "examples")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))
DEFAULT_RUN = os.environ.get("RUN", "backblaze_2022x_w30")

# Uploads are held in memory only, and only the few most recent -- enough for the detail
# view to look back at the file you just scored, without growing without bound.
MAX_CACHED_UPLOADS = 3
# How far back the per-drive trajectory slides the window, in days.
TRAJECTORY_DAYS = 45
# Rows sent to the browser. The CSV export is built server-side from all of them.
MAX_TABLE_ROWS = 500

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024

STATE: dict = {}


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def boot() -> None:
    t0 = time.time()
    print("loading artifacts ...")
    runs = scoring.discover()
    if not runs:
        raise SystemExit(
            f"no usable model found in {scoring.MODELS_DIR}. Train one, or point the "
            f"app at a populated models/ directory.")
    active = DEFAULT_RUN if DEFAULT_RUN in runs else next(iter(runs))
    STATE.update(runs=runs, active=active, uploads={}, device=torch.device("cpu"))

    for key, run in runs.items():
        c = run.contract
        mark = "*" if key == active else " "
        kinds = "+".join(sorted(run.estimators, key=lambda k: k != run.default))
        print(f" {mark} {key:<22} {c.window_days:>3}d window / {c.horizon_days:>2}d "
              f"horizon | {len(c.base_features):>2} attributes -> {c.n_channels:>2} "
              f"channels | {kinds}")
    run = runs[active]
    for kind, est in run.estimators.items():
        m = run.metrics_for(kind)
        extra = (f" | test PR-AUC {m['pr_auc']:.3f}, ROC-AUC {m['roc_auc']:.3f}"
                 if m.get("pr_auc") is not None else "")
        print(f"     {est.label:<8} threshold {est.threshold:.4f}{extra}")
    print(f"\nready in {time.time() - t0:.1f}s -- http://{HOST}:{PORT}\n", flush=True)


def run_for(key: str | None) -> scoring.Run:
    runs = STATE["runs"]
    return runs.get(key or "", runs[STATE["active"]])


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

ALIASES = {
    "serial_number": ("serial_number", "serial", "id", "drive_id", "serialnumber"),
    "date": ("date", "time", "timestamp", "day"),
    "model": ("model", "drive_model", "model_name"),
    "failure": ("failure", "failed", "label"),
}


def read_table(stream, filename: str) -> pd.DataFrame:
    """Read an uploaded workbook or CSV into a frame.

    calamine over openpyxl for xlsx: on the 115k-row family workbooks in this project it
    is about six times faster, and those are exactly the files a user will drop in.
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".csv", ".txt"):
        return pd.read_csv(stream)
    if ext == ".parquet":
        return pd.read_parquet(stream)
    if ext not in (".xlsx", ".xlsm", ".xls"):
        raise ValueError(f"unsupported file type {ext!r} -- upload .xlsx, .csv or .parquet")
    try:
        return pd.read_excel(stream, engine="calamine")
    except ImportError:
        stream.seek(0)
        return pd.read_excel(stream)


def normalise(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Map whatever the file calls its columns onto the names the model expects."""
    notes = []
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    for canonical, options in ALIASES.items():
        if canonical in df.columns:
            continue
        for alt in options:
            if alt in df.columns:
                df = df.rename(columns={alt: canonical})
                notes.append(f"read column '{alt}' as '{canonical}'")
                break
    return df, notes


# ---------------------------------------------------------------------------
# scoring one file
# ---------------------------------------------------------------------------

def ranking_quality(rows: list[dict], key: str) -> dict | None:
    """How the ranking did against the ground truth the file happens to carry.

    Only meaningful on a historical file, and only as a description of *this* file --
    a handful of drives is not an evaluation. It is here because a retrospective run is
    the obvious first thing to try, and "4 of the 6 that died are in my top 6" is the
    answer to the question that prompts it.
    """
    y = np.array([r["recorded_failure"] for r in rows], dtype=np.int8)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    order = np.argsort([-r["scores"][key] for r in rows])
    hits = y[order].cumsum()
    n_pos = int(y.sum())
    ks = [k for k in (5, 10, 25, 50, 100) if k <= len(rows)] or [len(rows)]
    return {
        "n_failed": n_pos,
        "n_drives": len(rows),
        "at_k": [{"k": k, "hits": int(hits[k - 1]),
                  "precision": float(hits[k - 1] / k)} for k in ks],
        "recall_at_n_pos": float(hits[n_pos - 1] / n_pos),
        "mean_rank_of_failures": float(
            np.mean([i + 1 for i, r in enumerate(
                sorted(rows, key=lambda r: -r["scores"][key]))
                if r["recorded_failure"]])),
    }


def predict_frame(df: pd.DataFrame, source: str, run: scoring.Run,
                  kinds: list[str], pad_short: bool = False) -> dict:
    """Validate, window, score with each requested estimator. Returns the page payload.

    `pad_short` decides what happens to a drive with less than a full window of history.
    Off (the default) it is skipped, which is the honest answer: the model was trained on
    genuinely continuous days and that is what it is calibrated for. On, the drive is
    left-padded to the window length and scored anyway.

    That switch matters more than it looks. In a *historical* workbook the drives that
    failed are exactly the ones that stopped reporting partway through, so they are the
    short ones -- strict mode skips the very drives a retrospective run wants to see.
    Padded scores are an extrapolation and the page labels them as such.
    """
    t0 = time.time()
    contract = run.contract
    attrs = list(contract.base_features)
    window_days = contract.window_days
    floor_padded = max(10, window_days // 3)
    cfg = contract.data_config(
        pad_short=pad_short, min_days=floor_padded if pad_short else window_days)

    df, notes = normalise(df)
    missing_keys = [c for c in ("serial_number", "date") if c not in df.columns]
    if missing_keys:
        raise ValueError(
            f"the file has no {' and no '.join(missing_keys)} column. "
            f"Columns found: {', '.join(map(str, list(df.columns)[:12]))}...")

    present = [c for c in attrs if c in df.columns]
    absent = [c for c in attrs if c not in df.columns]
    for c in absent:
        df[c] = np.nan          # median-filled from the saved statistics below

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "serial_number"])
    if df.empty:
        raise ValueError("no rows with both a date and a serial number")
    for c in attrs:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["failure"] = (pd.to_numeric(df.get("failure"), errors="coerce")
                     .fillna(0).astype("int8") if "failure" in df.columns
                     else np.int8(0))
    df["serial_number"] = df["serial_number"].astype(str)

    # Which drives can even offer a window, measured before the store drops them.
    span = df.groupby("serial_number")["date"].agg(["min", "max", "size"])
    span["days"] = (span["max"] - span["min"]).dt.days + 1
    floor = cfg.min_days
    too_short = span[span["days"] < floor]

    scored = df[~df.serial_number.isin(too_short.index)]
    if scored.empty:
        longest = int(span["days"].max())
        raise ValueError(
            f"no drive in this file has {floor} days of telemetry -- the longest run is "
            f"{longest} days."
            + ("" if pad_short else
               f" The model needs a full {window_days}-day window; tick “also score "
               f"drives with a partial window” to score shorter runs anyway."))

    frame = scored[["serial_number", "date", "failure"] + attrs].rename(
        columns={"serial_number": "id", "date": "time"})
    store = contract.build_store(frame, cfg)
    windows = scoring.latest_windows(store)

    scores = {}
    timings = {}
    for kind in kinds:
        est = run.estimators[kind]
        t = time.time()
        out = np.zeros(store.n_drives, dtype=np.float32)
        for a in range(0, store.n_drives, 256):
            b = min(a + 256, store.n_drives)
            out[a:b] = est.score_windows(windows[a:b])
        scores[kind] = out
        timings[kind] = round(time.time() - t, 2)

    meta = scored.groupby("serial_number").agg(
        drive_model=("model", "first") if "model" in scored.columns
                    else ("serial_number", "size"),
        recorded_failure=("failure", "max"))
    if "model" not in scored.columns:
        meta["drive_model"] = "—"

    primary = kinds[0]
    thresholds = {k: run.estimators[k].threshold for k in kinds}
    rows = []
    for g in range(store.n_drives):
        serial = str(store.drive_ids[g])
        _, _, wm = store.get_window(g, store.max_start(g))
        per = {k: float(scores[k][g]) for k in kinds}
        rows.append({
            "serial": serial,
            "index": g,
            "scores": per,
            "risk": per[primary],
            "calls": {k: ("inspect" if per[k] >= thresholds[k] else "hold") for k in kinds},
            "action": "inspect" if per[primary] >= thresholds[primary] else "hold",
            "drive_model": str(meta.drive_model.get(serial, "—")),
            "window_start": wm["start_date"],
            "window_end": wm["end_date"],
            "n_real_days": wm["n_real_days"],
            "n_filled_days": wm["n_filled_days"],
            "n_padded_days": wm["n_padded_days"],
            "days_available": int(store.lengths[g]),
            "recorded_failure": int(meta.recorded_failure.get(serial, 0)),
        })
    rows.sort(key=lambda r: -r["risk"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i

    token = uuid.uuid4().hex[:12]
    uploads = STATE["uploads"]
    uploads[token] = {"raw": scored, "store": store, "run": run.key, "kinds": kinds,
                      "rows": {r["serial"]: r for r in rows}, "ts": time.time()}
    for old in sorted(uploads, key=lambda k: uploads[k]["ts"])[:-MAX_CACHED_UPLOADS]:
        uploads.pop(old, None)

    has_truth = bool((df["failure"] != 0).any())
    agreement = None
    if len(kinds) > 1:
        a, b = kinds[0], kinds[1]
        both = sum(1 for r in rows if r["calls"][a] == "inspect" == r["calls"][b])
        only_a = sum(1 for r in rows if r["calls"][a] == "inspect" != r["calls"][b])
        only_b = sum(1 for r in rows if r["calls"][b] == "inspect" != r["calls"][a])
        agreement = {"pair": [a, b], "both": both, "only_first": only_a,
                     "only_second": only_b,
                     "spearman": float(pd.Series(scores[a]).corr(
                         pd.Series(scores[b]), method="spearman"))
                     if store.n_drives > 2 else None}

    return {
        "token": token,
        "source": source,
        "run": run.key,
        "kinds": kinds,
        "primary": primary,
        "thresholds": thresholds,
        "seconds": round(time.time() - t0, 1),
        "seconds_by_model": timings,
        "pad_short": pad_short,
        "min_days": floor,
        "window_days": window_days,
        "horizon_days": contract.horizon_days,
        "diagnostics": {
            "rows": int(len(df)),
            "drives_in_file": int(span.shape[0]),
            "drives_scored": int(store.n_drives),
            "drives_skipped": int(len(too_short)),
            "drives_padded": int(sum(1 for r in rows if r["n_padded_days"] > 0)),
            "skipped_examples": [
                {"serial": s, "days": int(d)}
                for s, d in too_short["days"].head(5).items()],
            "date_min": df["date"].min().strftime("%Y-%m-%d"),
            "date_max": df["date"].max().strftime("%Y-%m-%d"),
            "attributes_found": present,
            "attributes_missing": absent,
            "has_failure_column": has_truth,
            "notes": notes,
        },
        "summary": {
            k: {"inspect": sum(1 for r in rows if r["calls"][k] == "inspect"),
                "hold": sum(1 for r in rows if r["calls"][k] == "hold"),
                "max_risk": max((r["scores"][k] for r in rows), default=0.0)}
            for k in kinds},
        "agreement": agreement,
        "quality": {k: ranking_quality(rows, k) for k in kinds} if has_truth else None,
        "rows": rows[:MAX_TABLE_ROWS],
        "n_rows_total": len(rows),
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def api_meta():
    runs = STATE["runs"]
    return jsonify({
        "active": STATE["active"],
        "runs": [runs[k].describe() for k in runs],
        "trajectory_days": TRAJECTORY_DAYS,
    })


@app.get("/api/samples")
def api_samples():
    """Ready-made test input, offered as chips on the page.

    `data/examples/` comes first when it exists: those files are built to exercise one
    behaviour each (see scripts/make_example_workbooks.py) and are the fastest way to see
    what the app does. After them, one workbook per Toshiba family from the raw export.

    Family workbooks are chosen from `manifest.csv`, not by file size. The naive pick --
    the smallest workbook per family -- lands on the short tail windows, which hold no
    full window at all and fail immediately.
    """
    out = []
    window_days = run_for(request.args.get("run")).contract.window_days

    examples_manifest = os.path.join(EXAMPLES_DIR, "manifest.json")
    if os.path.isfile(examples_manifest):
        try:
            with open(examples_manifest, encoding="utf-8") as fh:
                for e in json.load(fh):
                    if os.path.isfile(os.path.join(EXAMPLES_DIR, e["file"])):
                        out.append({
                            "kind": "example",
                            "label": e["label"],
                            "shows": e.get("shows", ""),
                            "rel": "examples/" + e["file"],
                            "drives": e.get("drives", 0),
                            "failures": None,
                        })
        except (OSError, ValueError, KeyError):      # chips are cosmetic; never fatal
            out = []

    manifest = os.path.join(TOSHIBA_DIR, "manifest.csv")
    if os.path.isfile(manifest):
        mf = pd.read_csv(manifest)
        span = (pd.to_datetime(mf.window_end) - pd.to_datetime(mf.window_start)).dt.days + 1
        mf = mf[span >= window_days]

        families = []
        for family, grp in mf.groupby("model"):
            pick = grp.sort_values("size_mb").iloc[0]
            rel = str(pick["file"]).replace("\\", "/")
            if not os.path.isfile(os.path.join(TOSHIBA_DIR, rel)):
                continue
            families.append({
                "kind": "family",
                "label": family.replace("TOSHIBA ", ""),
                "shows": "",
                "rel": "Toshiba/" + rel,
                "drives": int(pick["drives"]),
                "failures": int(pick["failures"]),
                "mb": round(float(pick["size_mb"]), 1),
            })
        # Workbooks holding a failure first -- those are the ones worth clicking to see
        # the model separate anything. Smallest first within each group, so it stays quick.
        families.sort(key=lambda x: (x["failures"] == 0, x["mb"]))
        out += families[:6]

    return jsonify({"samples": out, "dir": DATA_DIR})


def _requested(run: scoring.Run) -> list[str]:
    """Which estimators to score with, primary first."""
    asked = [k for k in request.values.get("models", "").split(",") if k]
    if not asked and request.is_json:
        asked = list((request.json or {}).get("models") or [])
    asked = [k for k in asked if k in run.estimators]
    if not asked:
        return [run.default]
    # The first one asked for is the one the table ranks by, but the default estimator
    # stays first when it is in the set, so the ranking does not silently change.
    return sorted(dict.fromkeys(asked), key=lambda k: k != run.default)


@app.post("/api/predict")
def api_predict():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file was uploaded"}), 400
    run = run_for(request.form.get("run"))
    try:
        pad = request.form.get("pad") == "1"
        df = read_table(io.BytesIO(f.read()), f.filename)
        return jsonify(predict_frame(df, f.filename, run, _requested(run), pad))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                       # noqa: BLE001 -- surfaced to the page
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.post("/api/predict-sample")
def api_predict_sample():
    """Score one of the bundled workbooks, addressed by its relative path.

    The path is resolved and checked to sit inside the sample directory, so a crafted
    `rel` cannot walk out of it and read something else off the disk.
    """
    body = request.json or {}
    rel, pad = body.get("rel", ""), bool(body.get("pad"))
    run = run_for(body.get("run"))
    target = os.path.realpath(os.path.join(SAMPLE_DIR, rel))
    if (not target.startswith(os.path.realpath(SAMPLE_DIR) + os.sep)
            or not os.path.isfile(target)):
        return jsonify({"error": "unknown example file"}), 400
    try:
        with open(target, "rb") as fh:
            df = read_table(io.BytesIO(fh.read()), target)
        return jsonify(predict_frame(df, os.path.basename(target), run,
                                     _requested(run), pad))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                       # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.get("/api/upload/<token>/drive/<serial>")
def api_drive(token: str, serial: str):
    """One scored drive in full: its telemetry, its trajectory and why it scored so.

    Three things the ranked table cannot show. The raw series in the units the file
    reported them; the same score recomputed with the window slid back day by day, which
    says whether the drive is deteriorating or merely unwell; and, for XGBoost, the exact
    signed contribution of each feature to this window's log-odds.
    """
    up = STATE["uploads"].get(token)
    if up is None:
        return jsonify({"error": "that upload is no longer in memory -- score the file "
                                 "again"}), 404
    row = up["rows"].get(serial)
    if row is None:
        return jsonify({"error": f"no drive {serial!r} in that upload"}), 404

    run = STATE["runs"][up["run"]]
    store, kinds = up["store"], up["kinds"]
    attrs = list(run.contract.base_features)
    g = row["index"]

    windows, ends = scoring.trailing_windows(store, g, TRAJECTORY_DAYS)
    trajectory = {"dates": ends}
    for kind in kinds:
        trajectory[kind] = [float(v) for v in run.estimators[kind].score_windows(windows)]

    explanations = {}
    for kind in kinds:
        ex = run.estimators[kind].explain(windows[-1:])
        if ex:
            explanations[kind] = ex

    hist = (up["raw"][up["raw"].serial_number == serial]
            .sort_values("date").tail(run.contract.window_days))
    return jsonify({
        **row,
        "dates": [d.strftime("%Y-%m-%d") for d in hist["date"]],
        "attributes": attrs,
        "series": {c: [None if pd.isna(v) else float(v) for v in hist[c]] for c in attrs},
        "latest": {c: (None if hist[c].dropna().empty
                       else float(hist[c].dropna().iloc[-1])) for c in attrs},
        "trajectory": trajectory,
        "explanations": explanations,
    })


if __name__ == "__main__":
    boot()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
