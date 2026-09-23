"""Local prediction console -- upload a 90-day telemetry workbook, get a risk score.

    python webapp/app.py            then open http://127.0.0.1:5000

Drop in an `.xlsx` (or `.csv`) holding 90 days of SMART telemetry -- one drive or a
whole family -- and the trained model scores every drive in it. The file is parsed,
checked against the model's input contract, laid on a dense daily calendar, scaled with
the statistics **saved at training time**, and scored. Nothing is written to disk and
nothing leaves the machine.

The workbooks in `data/Toshiba/<MODEL>/<start>_<end>.xlsx` are exactly this shape, so
they work as test input straight away -- the page offers a few of them as examples.

Startup is about a second: only the model is loaded, not the fleet.
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

import backblaze_window_pipeline as bw  # noqa: E402

PREFIX = os.environ.get("MODEL_PREFIX",
                        os.path.join(ROOT, "models", "backblaze_window90"))
# Example files are addressed relative to data/, so both the raw Toshiba export and the
# purpose-built examples can be offered from one endpoint and one path check.
DATA_DIR = os.path.join(ROOT, "data")
SAMPLE_DIR = DATA_DIR
TOSHIBA_DIR = os.path.join(DATA_DIR, "Toshiba")
EXAMPLES_DIR = os.path.join(DATA_DIR, "examples")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "5000"))

# Uploads are held in memory only, and only the few most recent -- enough for the detail
# view to look back at the file you just scored, without growing without bound.
MAX_CACHED_UPLOADS = 3
# Floor when partial windows are allowed: below this there is too little history for a
# 90-day model to say anything, padded or not.
MIN_PADDED_DAYS = 30

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 400 * 1024 * 1024

STATE: dict = {}


# ---------------------------------------------------------------------------
# startup
# ---------------------------------------------------------------------------

def boot() -> None:
    device = torch.device("cpu")
    model, config = bw.load_artifacts(PREFIX, device)
    attrs = list(config["feature_names"][: config["base_feature_count"]])
    STATE.update(model=model, config=config, attrs=attrs, device=device,
                 cfg=bw.data_config_from_saved(config), uploads={})
    print(f"model: {config['arch']} | {config['window_days']}-day window | "
          f"{len(attrs)} SMART attributes | {len(config['feature_names'])} channels")
    print(f"threshold {config['best_threshold']:.4f} (tuned on validation)")
    print(f"\nready -- http://{HOST}:{PORT}\n", flush=True)


# ---------------------------------------------------------------------------
# parsing and scoring
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
    if ext not in (".xlsx", ".xlsm", ".xls"):
        raise ValueError(f"unsupported file type {ext!r} -- upload .xlsx or .csv")
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


def predict_frame(df: pd.DataFrame, source: str, pad_short: bool = False) -> dict:
    """Validate, window, score. Returns the payload the page renders.

    `pad_short` decides what happens to a drive with less than a full window of history.
    Off (the default) it is skipped, which is the honest answer: the model was trained on
    90 genuinely continuous days and that is what it is calibrated for. On, the drive is
    left-padded to 90 and scored anyway.

    That switch matters more than it looks. In a *historical* workbook the drives that
    failed are exactly the ones that stopped reporting partway through, so they are the
    short ones -- strict mode skips the very drives a retrospective run wants to see.
    Padded scores are an extrapolation and the page labels them as such.
    """
    t0 = time.time()
    attrs = STATE["attrs"]
    config, model = STATE["config"], STATE["model"]
    window_days = config["window_days"]
    cfg = bw.data_config_from_saved(
        config,
        pad_short_drives=pad_short,
        min_days=MIN_PADDED_DAYS if pad_short else window_days,
    )

    df, notes = normalise(df)
    missing_keys = [c for c in ("serial_number", "date") if c not in df.columns]
    if missing_keys:
        raise ValueError(
            f"the file has no {' and no '.join(missing_keys)} column. "
            f"Columns found: {', '.join(map(str, list(df.columns)[:12]))}…")

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
            f"no drive in this file has {floor} days of telemetry — the longest run is "
            f"{longest} days."
            + ("" if pad_short else
               f" The model needs a full {window_days}-day window; tick “also score "
               f"drives with a partial window” to score shorter runs anyway."))

    frame = scored[["serial_number", "date", "failure"] + attrs].rename(
        columns={"serial_number": "id", "date": "time"})
    store = bw.DriveSeriesStore(frame, cfg)
    store.apply_scaling(config["feature_medians"], config["feature_mean"],
                        config["feature_std"])
    risk = bw.score_latest_windows(model, store, STATE["device"])

    meta = scored.groupby("serial_number").agg(
        drive_model=("model", "first") if "model" in scored.columns
                    else ("serial_number", "size"),
        recorded_failure=("failure", "max"))
    if "model" not in scored.columns:
        meta["drive_model"] = "—"

    threshold = float(config["best_threshold"])
    rows = []
    for g in range(store.n_drives):
        serial = str(store.drive_ids[g])
        _, _, wm = store.get_window(g, store.max_start(g))
        rows.append({
            "serial": serial,
            "risk": float(risk[g]),
            "action": "inspect" if risk[g] >= threshold else "hold",
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
    uploads[token] = {"raw": scored, "rows": {r["serial"]: r for r in rows},
                      "ts": time.time()}
    for old in sorted(uploads, key=lambda k: uploads[k]["ts"])[:-MAX_CACHED_UPLOADS]:
        uploads.pop(old, None)

    return {
        "token": token,
        "source": source,
        "pad_short": pad_short,
        "min_days": floor,
        "seconds": round(time.time() - t0, 1),
        "threshold": threshold,
        "window_days": window_days,
        "horizon_days": cfg.horizon_days,
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
            "has_failure_column": bool((df["failure"] != 0).any()),
            "notes": notes,
        },
        "summary": {
            "inspect": sum(1 for r in rows if r["action"] == "inspect"),
            "hold": sum(1 for r in rows if r["action"] == "hold"),
            "max_risk": max((r["risk"] for r in rows), default=0.0),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def api_meta():
    config = STATE["config"]
    m = config.get("test_metrics", {})
    return jsonify({
        "arch": config["arch"],
        "window_days": config["window_days"],
        "horizon_days": config.get("horizon_days", 0),
        "threshold": float(config["best_threshold"]),
        "attributes": STATE["attrs"],
        "n_channels": len(config["feature_names"]),
        "metrics": {k: m.get(k) for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")},
        "blocked_columns": [
            ["days_to_last_record", "AUC 0.99 — days until the drive's last record"],
            ["still_reporting_at_export_end", "AUC 0.97 per drive — survivorship"],
            ["days_from_record_end_to_export_end", "AUC 0.96 — same signal as a count"],
            ["record_length_days", "balanced acc 0.91 — cohort sampling artefact"],
            ["pod_slot_num", "null on 53.6% of failure days vs 2.2% healthy"],
            ["vault_id", "8.2× lift on one level — cohort artefact"],
            ["capacity_bytes", "5.5× lift — proxy for drive family"],
            ["serial_number", "identity — memorises the outcome"],
            ["date", "calendar position encodes how the cohort was built"],
        ],
    })


@app.get("/api/samples")
def api_samples():
    """Ready-made test input, offered as chips on the page.

    `data/examples/` comes first when it exists: those six files are built to exercise
    one behaviour each (see scripts/make_example_workbooks.py) and are the fastest way to
    see what the app does. After them, one workbook per Toshiba family from the raw
    export.

    Family workbooks are chosen from `manifest.csv`, not by file size. The naive pick --
    the smallest workbook per family -- lands on the 11-day tail windows, which contain
    no full 90-day window and fail immediately.
    """
    out = []

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
        window_days = STATE["config"]["window_days"]
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


@app.post("/api/predict")
def api_predict():
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file was uploaded"}), 400
    try:
        pad = request.form.get("pad") == "1"
        data = io.BytesIO(f.read())
        df = read_table(data, f.filename)
        return jsonify(predict_frame(df, f.filename, pad))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                       # noqa: BLE001 -- surfaced to the page
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.post("/api/predict-sample")
def api_predict_sample():
    """Score one of the bundled example workbooks, addressed by its relative path.

    The path is resolved and checked to sit inside the sample directory, so a crafted
    `rel` cannot walk out of it and read something else off the disk.
    """
    body = request.json or {}
    rel, pad = body.get("rel", ""), bool(body.get("pad"))
    target = os.path.realpath(os.path.join(SAMPLE_DIR, rel))
    if not target.startswith(os.path.realpath(SAMPLE_DIR) + os.sep) or not os.path.isfile(target):
        return jsonify({"error": "unknown example file"}), 400
    try:
        with open(target, "rb") as fh:
            df = read_table(io.BytesIO(fh.read()), target)
        return jsonify(predict_frame(df, os.path.basename(target), pad))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:                       # noqa: BLE001
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.get("/api/upload/<token>/drive/<serial>")
def api_drive(token: str, serial: str):
    """The 90 days behind one scored drive, in the units the file reported them."""
    up = STATE["uploads"].get(token)
    if up is None:
        return jsonify({"error": "that upload is no longer in memory — score the file "
                                 "again"}), 404
    row = up["rows"].get(serial)
    if row is None:
        return jsonify({"error": f"no drive {serial!r} in that upload"}), 404

    attrs = STATE["attrs"]
    hist = (up["raw"][up["raw"].serial_number == serial]
            .sort_values("date").tail(STATE["cfg"].window_days))
    return jsonify({
        **row,
        "dates": [d.strftime("%Y-%m-%d") for d in hist["date"]],
        "attributes": attrs,
        "series": {c: [None if pd.isna(v) else float(v) for v in hist[c]] for c in attrs},
        "latest": {c: (None if hist[c].dropna().empty
                       else float(hist[c].dropna().iloc[-1])) for c in attrs},
    })


if __name__ == "__main__":
    boot()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
