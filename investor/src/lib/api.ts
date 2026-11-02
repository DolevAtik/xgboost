import type {
  ArtifactSummary,
  DemoDataset,
  DriveDetail,
  EstimatorKind,
  Evaluation,
  Health,
  ModelResponse,
  Prediction,
  SignalResponse,
} from "../../shared/types";
import { snapshotDrivePath, snapshotPredictPath } from "../../shared/snapshot";

/**
 * Static build (GitHub Pages): every response is read from public/snapshot/, recorded
 * from the real engine by scripts/snapshot.ts. Only scoring an uploaded file needs the
 * live stack.
 */
export const STATIC = import.meta.env.VITE_STATIC === "1";
const snap = (rel: string) => `${import.meta.env.BASE_URL}snapshot/${rel}`;

/** Where the input-schema template downloads from. */
export const TEMPLATE_URL = STATIC ? snap("template.csv") : "/api/demos/template";

export class ApiFailure extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiFailure(0, "The API server is not reachable. Start it with: npm run dev");
  }
  const body = await res.json().catch(() => ({ error: `unexpected response (HTTP ${res.status})` }));
  if (!res.ok) throw new ApiFailure(res.status, body?.error ?? `HTTP ${res.status}`);
  return body as T;
}

const staticHealth = (): Promise<Health> =>
  Promise.resolve({ api: "ok", engine: { ok: true, url: "snapshot", latencyMs: null, error: null } });

export const api = {
  health: () => (STATIC ? staticHealth() : request<Health>("/api/health")),
  model: () => request<ModelResponse>(STATIC ? snap("model.json") : "/api/model"),
  models: () => request<ArtifactSummary[]>(STATIC ? snap("models.json") : "/api/models"),
  evaluation: () => request<Evaluation>(STATIC ? snap("evaluation.json") : "/api/evaluation"),
  demos: () => request<DemoDataset[]>(STATIC ? snap("demos.json") : "/api/demos"),
  signal: () => request<SignalResponse>(STATIC ? snap("demos-signal.json") : "/api/demos/signal"),

  predictDemo: (id: string, models: EstimatorKind[], pad = true) =>
    STATIC
      ? request<Prediction>(snap(snapshotPredictPath(id, models)))
      : request<Prediction>(
          `/api/demos/${encodeURIComponent(id)}/predict?models=${models.join(",")}&pad=${pad ? 1 : 0}`,
          { method: "POST" },
        ),

  /** Upload with progress, since a large workbook takes a moment to move. */
  predictFile: (file: File, models: EstimatorKind[], onProgress: (f: number) => void, pad = true) =>
    new Promise<Prediction>((resolve, reject) => {
      if (STATIC) {
        reject(new ApiFailure(0, "Scoring your own file needs the live engine. This online version shows recorded results for the sample datasets — pick one of them."));
        return;
      }
      const form = new FormData();
      form.append("file", file);
      form.append("models", models.join(","));
      form.append("pad", pad ? "1" : "0");
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/predict");
      xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
      xhr.onload = () => {
        let body: { error?: string } = {};
        try {
          body = JSON.parse(xhr.responseText);
        } catch {
          /* handled below */
        }
        if (xhr.status >= 200 && xhr.status < 300) resolve(body as Prediction);
        else reject(new ApiFailure(xhr.status, body.error ?? `upload failed (HTTP ${xhr.status})`));
      };
      xhr.onerror = () => reject(new ApiFailure(0, "The API server is not reachable."));
      xhr.send(form);
    }),

  drive: (token: string, serial: string) =>
    request<DriveDetail>(
      STATIC
        ? snap(snapshotDrivePath(token, serial))
        : `/api/predictions/${encodeURIComponent(token)}/drives/${encodeURIComponent(serial)}`,
    ),
};
