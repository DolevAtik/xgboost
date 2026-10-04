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

export const api = {
  health: () => request<Health>("/api/health"),
  model: () => request<ModelResponse>("/api/model"),
  models: () => request<ArtifactSummary[]>("/api/models"),
  evaluation: () => request<Evaluation>("/api/evaluation"),
  demos: () => request<DemoDataset[]>("/api/demos"),
  signal: () => request<SignalResponse>("/api/demos/signal"),

  predictDemo: (id: string, models: EstimatorKind[], pad = true) =>
    request<Prediction>(
      `/api/demos/${encodeURIComponent(id)}/predict?models=${models.join(",")}&pad=${pad ? 1 : 0}`,
      { method: "POST" },
    ),

  /** Upload with progress, since a large workbook takes a moment to move. */
  predictFile: (file: File, models: EstimatorKind[], onProgress: (f: number) => void, pad = true) =>
    new Promise<Prediction>((resolve, reject) => {
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
    request<DriveDetail>(`/api/predictions/${encodeURIComponent(token)}/drives/${encodeURIComponent(serial)}`),
};
