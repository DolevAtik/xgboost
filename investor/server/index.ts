/**
 * Node API for the investor experience.
 *
 * A thin, typed boundary in front of the existing Flask prediction engine
 * (webapp/app.py). Predictions always come from Flask and the saved models; this layer
 * adds the evaluation results read from results/, a catalogue of the demo files, and
 * stable endpoint names for the React app. It never fabricates a score.
 */
import { existsSync } from "node:fs";
import path from "node:path";
import express, { type NextFunction, type Request, type Response } from "express";
import { DIST_DIR, FLASK_URL, HOST, MAX_UPLOAD_BYTES, MODELS_DIR, PORT } from "./config.ts";
import { listDemos, predictDemo, signal } from "./demos.ts";
import { evaluation } from "./evaluation.ts";
import { ApiError, flask, pipeUpload } from "./flask.ts";
import type { ArtifactSummary, DriveDetail, Health, ModelResponse, RunInfo } from "../shared/types.ts";

const app = express();
app.disable("x-powered-by");

type Handler = (req: Request, res: Response) => Promise<unknown>;
const route = (fn: Handler) => (req: Request, res: Response, next: NextFunction) => {
  fn(req, res).then((body) => body !== undefined && res.json(body), next);
};

interface FlaskMeta {
  active: string;
  runs: RunInfo[];
  trajectory_days: number;
}

const meta = () => flask<FlaskMeta>("/api/meta", {}, 15_000);

// A drive's detail is a single Flask call; trajectory and explanation are views of it.
const detailCache = new Map<string, DriveDetail>();
async function driveDetail(token: string, serial: string): Promise<DriveDetail> {
  const key = `${token}/${serial}`;
  const hit = detailCache.get(key);
  if (hit) return hit;
  const d = await flask<DriveDetail>(
    `/api/upload/${encodeURIComponent(token)}/drive/${encodeURIComponent(serial)}`,
  );
  detailCache.set(key, d);
  if (detailCache.size > 64) detailCache.delete(detailCache.keys().next().value!);
  return d;
}

// ---------------------------------------------------------------- routes

app.get(
  "/api/health",
  route(async () => {
    const t0 = Date.now();
    let engine: Health["engine"];
    try {
      await meta();
      engine = { ok: true, url: FLASK_URL, latencyMs: Date.now() - t0, error: null };
    } catch (e) {
      engine = { ok: false, url: FLASK_URL, latencyMs: null, error: (e as Error).message };
    }
    return { api: "ok", engine } satisfies Health;
  }),
);

/** The run the engine serves by default: contract, estimators, held-out metrics, blocklist. */
app.get(
  "/api/model",
  route(async () => {
    const m = await meta();
    const run = m.runs.find((r) => r.key === m.active) ?? m.runs[0];
    const { tables: _tables, ...rest } = run as RunInfo & { tables?: unknown };
    return { run: rest, trajectoryDays: m.trajectory_days } satisfies ModelResponse;
  }),
);

/** Every trained run the engine discovered in models/, with the files behind it. */
app.get(
  "/api/models",
  route(async () => {
    const m = await meta();
    const fs = await import("node:fs/promises");
    const files = await fs.readdir(MODELS_DIR).catch(() => [] as string[]);
    return m.runs.map(
      (r): ArtifactSummary => ({
        key: r.key,
        label: r.label,
        tag: r.tag,
        scope: r.scope,
        active: r.key === m.active,
        window_days: r.window_days,
        horizon_days: r.horizon_days,
        estimators: r.estimators.map((e) => ({
          kind: e.kind,
          label: e.label,
          size: e.size,
          pr_auc: e.metrics.pr_auc ?? null,
        })),
        files: files.filter((f) => f.startsWith(`${r.key}_`)).map((f) => `models/${f}`),
      }),
    );
  }),
);

/** Held-out evaluation results, read from results/w2022x_*. */
app.get("/api/evaluation", route(() => evaluation()));

app.get("/api/demos", route(() => listDemos()));
app.get("/api/demos/signal", route(() => signal()));

app.post(
  "/api/demos/:id/predict",
  route(async (req) => {
    const models = String(req.query.models ?? "xgb");
    const pad = req.query.pad !== "0";
    return predictDemo(String(req.params.id), models, pad);
  }),
);

/** Upload your own telemetry: streamed to Flask as-is, never buffered or written to disk. */
app.post(
  "/api/predict",
  route(async (req, res) => {
    const length = Number(req.headers["content-length"] ?? 0);
    if (length > MAX_UPLOAD_BYTES) throw new ApiError(413, "file is larger than the 400 MB limit");
    if (!String(req.headers["content-type"] ?? "").startsWith("multipart/form-data"))
      throw new ApiError(400, "send the file as multipart/form-data in a field named 'file'");
    const { status, body } = await pipeUpload(req, "/api/predict");
    res.status(status).json(body);
    return undefined;
  }),
);

app.get(
  "/api/predictions/:token/drives/:serial",
  route((req) => driveDetail(String(req.params.token), String(req.params.serial))),
);

app.get(
  "/api/predictions/:token/drives/:serial/trajectory",
  route(async (req) => {
    const d = await driveDetail(String(req.params.token), String(req.params.serial));
    return { serial: d.serial, horizonDays: 30, trajectory: d.trajectory };
  }),
);

app.get(
  "/api/predictions/:token/drives/:serial/explanation",
  route(async (req) => {
    const d = await driveDetail(String(req.params.token), String(req.params.serial));
    return { serial: d.serial, explanations: d.explanations };
  }),
);

app.use("/api", (_req, res) => {
  res.status(404).json({ error: "no such endpoint" });
});

// In production the API also serves the built site.
if (existsSync(DIST_DIR)) {
  app.use(express.static(DIST_DIR, { maxAge: "1h", index: false }));
  app.get(/^(?!\/api).*/, (_req, res) => res.sendFile(path.join(DIST_DIR, "index.html")));
}

app.use((err: unknown, _req: Request, res: Response, _next: NextFunction) => {
  const status = err instanceof ApiError ? err.status : 500;
  const message = err instanceof Error ? err.message : "unexpected error";
  if (status >= 500 && !(err instanceof ApiError)) console.error(err);
  res.status(status).json({ error: message });
});

app.listen(PORT, HOST, () => {
  console.log(`investor API on http://${HOST}:${PORT}  ->  prediction engine ${FLASK_URL}`);
  evaluation().catch((e) => console.error("could not read results/:", (e as Error).message));
});
