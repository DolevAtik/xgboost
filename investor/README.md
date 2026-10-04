# Drive Foresight — investor experience

A product site for the drive-failure prediction project: what the problem is, how the
system works, the evaluation results, and a live demo that scores real telemetry with the
saved models. It sits in front of the existing Flask engine (`../webapp/app.py`) and does
not replace it.

## Run it

Requirements: Node 22+, and the project's Python environment (the one `webapp/app.py`
already runs in, with `models/` and `results/` populated).

```bash
cd investor
npm install

# activate the project's Python env first, so `python` is the one with xgboost/torch
#   PowerShell: ..\.venv\Scripts\Activate.ps1      bash: source ../.venv/Scripts/activate

npm run dev          # engine (Flask :5000) + API (Node :4000) + site (Vite :5173)
```

Open <http://127.0.0.1:5173>. The nav shows "Engine live" once Flask is up.

To run the pieces separately: `npm run engine`, `npm run dev:api`, `npm run dev:web`.

Production build, served by the Node API on one port:

```bash
npm run build
npm run serve        # engine + API; site at http://127.0.0.1:4000
```

| variable | default | |
|---|---|---|
| `FLASK_URL` | `http://127.0.0.1:5000` | where the prediction engine runs |
| `API_PORT` | `4000` | Node API port |
| `HOST` | `127.0.0.1` | bind address for the Node API |
| `API_URL` | `http://127.0.0.1:4000` | where Vite proxies `/api` in dev |

The original Flask console still works on its own at <http://127.0.0.1:5000>.

## Architecture

```
browser ──► Vite / built site ──► Node API (Express, :4000) ──► Flask engine (:5000) ──► models/
                                        │
                                        └── reads results/w2022x_*  and demo/manifest.json
```

* **Flask** is untouched. It parses files, applies the saved scaler and scores with the
  saved XGBoost and CNN models. Every prediction on the site comes from it.
* **Node** (`server/`) is a thin typed boundary. It streams uploads to Flask without
  buffering or writing them to disk, keeps the same 400 MB limit, lists the demo
  datasets (only files named in a manifest, so an id cannot address anything else), and
  serves the evaluation results read from `results/`.
* **React** (`src/`) renders the page. Shared types live in `shared/types.ts`.

### API

| endpoint | what it returns | source |
|---|---|---|
| `GET /api/health` | API status and whether the engine answers | Flask `/api/meta` |
| `GET /api/model` | the served run: contract, estimators, held-out metrics, blocklist | Flask `/api/meta` |
| `GET /api/models` | every trained run found, with its files in `models/` | Flask + `models/` |
| `GET /api/evaluation` | scale, metrics, precision@K, lead time, tree ladder, leakage | `results/w2022x_*` |
| `GET /api/demos` | scoreable datasets: `demo/`, `data/examples/`, small 90-day Toshiba exports | manifests + Flask `/api/samples` |
| `GET /api/demos/template` | `demo/TEMPLATE.csv` as a download: the input schema with every column | `demo/` |
| `GET /api/demos/signal` | raw telemetry of the healthy and failing demo drives | `demo/01_*.csv`, `demo/02_*.csv` |
| `POST /api/demos/:id/predict?models=xgb,cnn&pad=1` | a prediction for a listed dataset | Flask `/api/predict` or `/api/predict-sample` |
| `POST /api/predict` | a prediction for an uploaded file (multipart field `file`) | Flask `/api/predict`, streamed |
| `GET /api/predictions/:token/drives/:serial` | one drive: telemetry, trajectory, SHAP | Flask `/api/upload/.../drive/...` |
| `.../trajectory`, `.../explanation` | the same, narrowed | as above |

No Flask capability had to be added.

## Page structure

| section | content | data |
|---|---|---|
| Hero | headline, live 3D drive, real 30-day telemetry of a failing drive, key figures | `/api/demos/signal`, `/api/evaluation` |
| The Problem | reactive vs predictive timeline (schematic, labelled) | failure count from `/api/evaluation` |
| Technology | six-stage pipeline; "Inside the engine" (two real drives scored live) behind a Go deeper toggle | `/api/model`, live predictions |
| Results | top-K precision, the accuracy trap, lead time; standard metrics and methodology behind a Go deeper toggle | `/api/evaluation` |
| Value | calculator: your fleet size and costs × the measured failure, catch and false-alarm rates | `/api/evaluation` |
| Vision | fleet illustration (labelled as such) and directions | — |
| Team | the three team members, roles and focus areas; photos from `public/team/` | `src/lib/team.ts` |
| Try it live (last, highlighted) | upload or sample, ranked drives, drive profile; XGBoost vs CNN behind a Go deeper toggle | live predictions, `/api/evaluation` |

Each visual is also viewable on its own in the artifact lab at `/#/lab`.

## Rules the content follows

* Every number is read from `results/` or computed live by the models; none is typed into
  a component. Sources are named on the page.
* Ranking performance is never presented as "accuracy"; the page shows why accuracy is the
  wrong measure at a 0.24% base rate.
* A retrospective hit rate on a demo file describes that file only. Toshiba cohort
  precision is not presented as performance.
* Scores are shown as risk scores, not calibrated probabilities. The "elevated / high"
  split of the model's `inspect` call at 0.5 is presentation, and the page says so.
* The vision section is labelled as direction, not results. No market or traction claims.

## Notes

* 3D scenes are code-split and fall back to a static backdrop without WebGL.
  `prefers-reduced-motion` stops all animation.
* Fonts are bundled (`@fontsource`); the site makes no external requests.
* "Get in touch" opens the visitor's mail client with the message filled in. Set the
  address in `src/lib/contact.ts` (`CONTACT.email`); until then the form says it is not configured.
* Team content and photos: `src/lib/team.ts` and `public/team/`.
* `STORYBOARD.md` is the original storyboard and content plan.
