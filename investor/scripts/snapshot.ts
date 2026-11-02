/**
 * Record every API response the site reads into public/snapshot/, so a static build
 * (VITE_STATIC=1, used for GitHub Pages) behaves like the live site without the engine.
 *
 * Run with the API and the Flask engine up (`npm run serve`), then: npm run snapshot
 * Every score in the snapshot comes from the real engine; only uploading a new file
 * needs the live stack.
 */
import { copyFile, mkdir, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { DemoDataset, Prediction } from "../shared/types.ts";
import { snapshotDrivePath, snapshotPredictPath } from "../shared/snapshot.ts";

const here = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.resolve(here, "..", "public", "snapshot");
const API = (process.env.API_URL ?? "http://127.0.0.1:4000").replace(/\/$/, "");
const MODEL_SETS = ["xgb", "xgb,cnn"];

async function get<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(API + url, init);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status} ${await res.text()}`);
  return (await res.json()) as T;
}

async function save(rel: string, body: unknown) {
  const file = path.join(OUT, rel);
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, JSON.stringify(body));
}

await rm(OUT, { recursive: true, force: true });

const health = await get<{ engine: { ok: boolean } }>("/api/health");
if (!health.engine.ok) throw new Error("the Flask engine is offline; start it with npm run serve");

for (const name of ["model", "models", "evaluation", "demos/signal"]) {
  await save(`${name.replace("/", "-")}.json`, await get(`/api/${name}`));
}
const demos = await get<DemoDataset[]>("/api/demos");
await save("demos.json", demos);
await copyFile(path.resolve(here, "..", "..", "demo", "TEMPLATE.csv"), path.join(OUT, "template.csv"));

let drives = 0;
for (const d of demos) {
  for (const models of MODEL_SETS) {
    const p = await get<Prediction>(
      `/api/demos/${encodeURIComponent(d.id)}/predict?models=${models}&pad=1`,
      { method: "POST" },
    );
    await save(snapshotPredictPath(d.id, models.split(",")), p);
    for (const row of p.rows) {
      const detail = await get(`/api/predictions/${p.token}/drives/${encodeURIComponent(row.serial)}`);
      await save(snapshotDrivePath(p.token, row.serial), detail);
      drives++;
    }
  }
  console.log(`${d.id}: done`);
}
console.log(`snapshot written to ${OUT}: ${demos.length} datasets, ${drives} drive details`);
