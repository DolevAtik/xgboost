import http from "node:http";
import type { Request } from "express";
import { FLASK_URL } from "./config.ts";

/** An error that carries the status the browser should see. */
export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function parse(res: Response): Promise<unknown> {
  const text = await res.text();
  let body: unknown;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    throw new ApiError(502, `the prediction engine returned a non-JSON response (HTTP ${res.status})`);
  }
  if (!res.ok) {
    const msg = (body as { error?: string })?.error ?? `prediction engine error (HTTP ${res.status})`;
    throw new ApiError(res.status, msg);
  }
  return body;
}

/** Call the Flask engine. Network failures become a 503 the page can explain. */
export async function flask<T>(pathname: string, init: RequestInit = {}, timeoutMs = 300_000): Promise<T> {
  let res: Response;
  try {
    res = await fetch(FLASK_URL + pathname, { ...init, signal: AbortSignal.timeout(timeoutMs) });
  } catch (e) {
    const reason = (e as Error).name === "TimeoutError" ? "timed out" : "is not reachable";
    throw new ApiError(503, `the prediction engine at ${FLASK_URL} ${reason}. Start it with: python webapp/app.py`);
  }
  return (await parse(res)) as T;
}

/**
 * Stream a multipart upload straight through to Flask without buffering it in Node.
 * The body is never written to disk on either side.
 */
export function pipeUpload(req: Request, pathname: string): Promise<{ status: number; body: unknown }> {
  const target = new URL(FLASK_URL + pathname);
  return new Promise((resolve, reject) => {
    const headers: http.OutgoingHttpHeaders = { "content-type": req.headers["content-type"] };
    if (req.headers["content-length"]) headers["content-length"] = req.headers["content-length"];
    const out = http.request(
      { hostname: target.hostname, port: target.port, path: target.pathname + target.search, method: "POST", headers },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          try {
            resolve({ status: res.statusCode ?? 502, body: JSON.parse(text) });
          } catch {
            // Flask answers 413 with an HTML page when its own size limit trips.
            const status = res.statusCode ?? 502;
            resolve({
              status,
              body: { error: status === 413 ? "file is larger than the 400 MB limit" : `prediction engine error (HTTP ${status})` },
            });
          }
        });
      },
    );
    out.on("error", () =>
      reject(new ApiError(503, `the prediction engine at ${FLASK_URL} is not reachable. Start it with: python webapp/app.py`)),
    );
    req.pipe(out);
  });
}
