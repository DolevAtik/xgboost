# Storyboard — "Know the failure before it happens."

The one sentence an investor should leave with:

> We built a system that learns the early warning signs of hard-drive failure, and on
> real-world test data it ranks the most at-risk drives with very high precision. The goal
> is to move maintenance from reacting to a failure to acting before it happens.

Not "an XGBoost model with ROC-AUC 0.919". The numbers are the evidence, not the pitch.

## Honesty rules (apply to every section)

* Every number on screen is read from a file in `../results/` or from the live model in
  `../models/`, never typed into a component. Source files are named in the UI.
* Ranking performance (precision@K) is never called "accuracy". The page shows, in
  numbers, why accuracy is the wrong yardstick at a 0.24% base rate.
* Demo files are illustrations. A retrospective hit-rate on a 60-drive demo file is
  labelled as describing that file, not as an evaluation. Toshiba cohort numbers are never
  shown as production performance.
* No market sizes, customers, revenue or traction. The vision chapter is labelled as
  direction, not as fact.
* The intelligence chapter describes the inference path that is actually served
  (XGBoost over 46 channels × last/mean/delta = 138 features, 470 trees), and the CNN as
  a separate, compared estimator.

## Chapters

| # | Chapter | What the investor understands | Visual | Interaction | Data |
|---|---|---|---|---|---|
| 01 | Hero | Something hidden inside the hardware can be seen before it breaks. | 3D drive in a dark room; a faint fault point glows on the platter while particles of telemetry stream off it. | Pointer parallax; CTA scrolls into the story. | — |
| 02 | The Hidden Problem | Today, the alert arrives *after* the outage. | Two timelines side by side: reactive (failure → alert → downtime) and predictive (signals → rising risk → inspection → no outage). | Plays on scroll; replay button. | Conceptual, labelled as such. |
| 03 | The Signal | Drives already report their own health every day; a dying one looks different. | Real 30-day telemetry of a healthy and a failing demo drive, side by side, plain-language attribute names. | Toggle attribute; hover a day. | `demo/01_healthy_drive.csv`, `demo/02_failing_drive.csv` |
| 04 | The Intelligence | Raw readings → cleaned → audited for cheating → summarised → model → ranked risk. | Living pipeline with flowing particles (Artifact C) and an engine view: 30-day window → 138 features → tree ensemble → risk (Artifact D). | Click any stage for a plain explanation + optional technical detail. | `models/backblaze_2022x_w30_*config.json`, `/api/model` |
| 05 | The Breakthrough | At real-world rarity, 99 of the 100 drives it worries about most really were about to fail. | Scale counters; 100-dot grid; precision@K ladder; "accuracy trap" explainer; how early it sees it. | Pick K; expand technical notes. | `results/w2022x_summary.json`, `w2022x_cnn_vs_xgboost_precision_at_k.csv`, `w2022x_lead_time_failing.csv` |
| 06 | The Product | This is a working product, not a slide. | Drive Explorer: pick a fleet file or drop your own, get a ranked queue, open a drive: risk, category, 30-day horizon, telemetry, trajectory, explanation. | Live calls through Node → Flask → saved models. | `demo/`, `data/examples/`, uploads |
| 06b | Model comparison | Why XGBoost is the default and what the CNN offers. | Head-to-head bars + ladder, plus interpretability and operational trade-offs. | Metric toggle. | `results/w2022x_model_summary.csv` |
| 08 | The Vision | Where this can go: from reactive maintenance to infrastructure that anticipates. | A fleet of drives in 3D shifting from red failures to calm, flagged-early units. | Scroll-driven. | Labelled as direction. |
| 09 | The Close | The memorable line, and a next step. | Headline + two CTAs: run the demo, open the engineering console. | — | — |

## Design system

Revised after review: organised and product-like rather than a slide deck.

* **Layout** a section nav at the top; each section has a small kicker, a compact
  heading with its lede beside it, and content on bordered panels (`.panel`). Max width
  80rem, consistent 16px gaps between panels.
* **Ground** near-black `#04060b` → navy `#0a1122`; panels are a faint navy gradient.
* **Accents** cyan `#3fe0ff` for interaction and emphasis. Green `#3ddc97`, amber
  `#ffb547` and red `#ff4d6a` are reserved for healthy / elevated / failure, and always
  come with a label. Chart series: XGBoost `#0ea5c6`, CNN `#8b6cf0` (validated for the
  dark surface).
* **Type** Inter Tight for headings, body and headline figures; JetBrains Mono for
  serials, codes and table numbers. Section headings clamp from 28px to 40px.
* **Motion** short fades (0.35–0.8s) on entry and state changes only. Everything is
  static under `prefers-reduced-motion`.

## Artifact-first build

Artifacts A–H are standalone components, each viewable alone at `/#/lab/<id>` before
being composed into the story at `/`:

| Artifact | Component |
|---|---|
| A Cinematic Hero | `HeroScene` + `DriveModel3D` |
| B Problem | `ProblemComparison` |
| C Pipeline | `DataPipeline` |
| D Intelligence Engine | `IntelligenceVisualization` |
| E Performance Proof | `PerformanceMetrics` |
| F Drive Explorer | `DriveExplorer`, `RiskTrajectoryChart`, `SHAPExplanation` |
| G Model Comparison | `ModelComparison` |
| H The Future | `ClosingScene` |
