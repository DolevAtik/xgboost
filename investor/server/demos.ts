import { readFile } from "node:fs/promises";
import path from "node:path";
import { DEMO_DIR } from "./config.ts";
import { num, parseCsv } from "./csv.ts";
import { ApiError, flask } from "./flask.ts";
import type { DemoDataset, SignalResponse, SignalSeries } from "../shared/types.ts";

interface DemoManifest {
  attributes: string[];
  files: { file: string; rows: number; drives: number; what: string; expect: string }[];
}

/** Plain-language names for the demo/ files, keyed by file name. */
const DEMO_LABELS: Record<string, string> = {
  "03_fleet_60_drives.xlsx": "A 60-drive fleet",
  "04_live_snapshot.xlsx": "Live snapshot — no answer key",
  "02_failing_drive.csv": "One drive in its final month",
  "01_healthy_drive.csv": "One healthy drive",
  "05_short_windows.xlsx": "Drives with short history",
  "06_renamed_and_sparse.csv": "Messy headers, sparse data",
};

async function manifest(): Promise<DemoManifest> {
  return JSON.parse(await readFile(path.join(DEMO_DIR, "manifest.json"), "utf8")) as DemoManifest;
}

/**
 * Every dataset the explorer can score: the files in demo/ (scored by streaming them to
 * Flask's /api/predict) and the engineering examples Flask itself offers from data/.
 * Only files named in a manifest are listed, so an id can never address anything else.
 */
export async function listDemos(): Promise<DemoDataset[]> {
  const out: DemoDataset[] = [];
  const m = await manifest().catch(() => null);
  for (const f of m?.files ?? []) {
    if (!(f.file in DEMO_LABELS)) continue; // smartctl_log.csv needs convert.py first
    out.push({
      id: `demo:${f.file}`,
      group: "demo",
      label: DEMO_LABELS[f.file],
      description: f.what,
      file: `demo/${f.file}`,
      drives: f.drives,
      rows: f.rows,
      expect: f.expect,
    });
  }
  out.sort((a, b) => Object.keys(DEMO_LABELS).indexOf(a.file.slice(5)) - Object.keys(DEMO_LABELS).indexOf(b.file.slice(5)));

  try {
    const s = await flask<{
      samples: { kind: string; label: string; shows: string; rel: string; drives: number; failures: number | null; mb?: number }[];
    }>(
      "/api/samples",
      {},
      10_000,
    );
    for (const e of s.samples.filter((x) => x.kind === "example")) {
      out.push({
        id: `example:${e.rel}`,
        group: "example",
        label: e.label.replace(/^\d+\s*-\s*/, ""),
        description: e.shows,
        file: `data/${e.rel}`,
        drives: e.drives,
        rows: null,
        expect: null,
      });
    }
    // Real 90-day workbooks from the Toshiba export: long enough for a full trajectory.
    for (const f of s.samples.filter((x) => x.kind === "family" && (x.mb ?? 99) <= 4)) {
      const span = f.rel.split("/").pop()?.replace(".xlsx", "").replace("_", " → ") ?? "";
      out.push({
        id: `example:${f.rel}`,
        group: "export",
        label: `Toshiba ${f.label} · 90 days`,
        description: `real export, ${span}: ${f.drives} drives, ${f.failures} recorded failures`,
        file: `data/${f.rel}`,
        drives: f.drives,
        rows: null,
        expect: null,
      });
    }
  } catch {
    // The engineering examples come from Flask; the demo/ files are still listed without it.
  }
  return out;
}

/** Score a listed dataset with the live engine. */
export async function predictDemo(id: string, models: string, pad: boolean): Promise<unknown> {
  const all = await listDemos();
  const demo = all.find((d) => d.id === id);
  if (!demo) throw new ApiError(404, `unknown dataset ${id}`);

  // Engineering examples and the real exports live under data/ and are read by Flask itself.
  if (demo.group !== "demo") {
    return flask("/api/predict-sample", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ rel: id.slice("example:".length), pad, models: models.split(",") }),
    });
  }
  const name = demo.file.slice("demo/".length);
  const bytes = await readFile(path.join(DEMO_DIR, name));
  const form = new FormData();
  form.append("file", new Blob([bytes]), name);
  form.append("models", models);
  form.append("pad", pad ? "1" : "0");
  return flask("/api/predict", { method: "POST", body: form });
}

function series(text: string, attributes: string[]): SignalSeries {
  const rows = parseCsv(text);
  const s: Record<string, (number | null)[]> = {};
  for (const a of attributes) s[a] = rows.map((r) => (r[a] === "" ? null : num(r[a])));
  return {
    serial: rows[0]?.serial_number ?? "",
    model: rows[0]?.model ?? "",
    dates: rows.map((r) => r.date),
    series: s,
    failedOnLastDay: rows.at(-1)?.failure === "1",
  };
}

/** The raw telemetry of the healthy and the failing demo drive, for "The Signal". */
export async function signal(): Promise<SignalResponse> {
  const m = await manifest();
  const [healthy, failing] = await Promise.all([
    readFile(path.join(DEMO_DIR, "01_healthy_drive.csv"), "utf8"),
    readFile(path.join(DEMO_DIR, "02_failing_drive.csv"), "utf8"),
  ]);
  return {
    source: ["demo/01_healthy_drive.csv", "demo/02_failing_drive.csv"],
    healthy: series(healthy, m.attributes),
    failing: series(failing, m.attributes),
  };
}
