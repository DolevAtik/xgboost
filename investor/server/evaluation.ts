import { readFile } from "node:fs/promises";
import path from "node:path";
import { RESULTS_DIR, RESULTS_PREFIX } from "./config.ts";
import { num, readCsv } from "./csv.ts";
import type { Evaluation, LeadPoint, ModelResult } from "../shared/types.ts";

const file = (name: string) => path.join(RESULTS_DIR, `${RESULTS_PREFIX}_${name}`);

interface SummaryJson {
  n_drives: number;
  n_drive_days: number;
  shards: string[];
  blocklist: string[];
  leakage_checks_passed: boolean;
  max_channel_auc_vs_label: number;
  xgboost: {
    best_iteration: number;
    test_at_tuned: { threshold: number; tp: number; fp: number; fn: number; tn: number; n: number; n_pos: number; base_rate: number };
  };
}

/**
 * Failing test drives scored at each day before their failure, from
 * w2022x_lead_time_failing.csv. Only leads inside the 30-day horizon (0..29) count as
 * a correct early alert: a window ending 30+ days out is not labelled positive.
 */
async function leadTime(horizon: number): Promise<Evaluation["leadTime"]> {
  let rows;
  try {
    rows = await readCsv(file("lead_time_failing.csv"));
  } catch {
    return null;
  }
  const byLead = new Map<number, { n: number; xgb: number; cnn: number; xf: number; cf: number }>();
  const xgbAny = new Map<string, boolean>();
  const cnnAny = new Map<string, boolean>();
  const xgbWeek = new Map<string, boolean>();
  for (const r of rows) {
    const lead = num(r.lead);
    const xf = r.xgb_flag === "True";
    const cf = r.cnn_flag === "True";
    const agg = byLead.get(lead) ?? { n: 0, xgb: 0, cnn: 0, xf: 0, cf: 0 };
    agg.n++;
    agg.xgb += num(r.xgb);
    agg.cnn += num(r.cnn);
    agg.xf += xf ? 1 : 0;
    agg.cf += cf ? 1 : 0;
    byLead.set(lead, agg);
    if (!xgbAny.has(r.drive)) {
      xgbAny.set(r.drive, false);
      cnnAny.set(r.drive, false);
      xgbWeek.set(r.drive, false);
    }
    if (lead < horizon) {
      if (xf) xgbAny.set(r.drive, true);
      if (cf) cnnAny.set(r.drive, true);
      if (xf && lead >= 7) xgbWeek.set(r.drive, true);
    }
  }
  const share = (m: Map<string, boolean>) => [...m.values()].filter(Boolean).length / Math.max(1, m.size);
  const curve: LeadPoint[] = [...byLead.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([lead, a]) => ({
      lead,
      xgbMean: a.xgb / a.n,
      cnnMean: a.cnn / a.n,
      xgbFlagRate: a.xf / a.n,
      cnnFlagRate: a.cf / a.n,
    }));
  return {
    drives: xgbAny.size,
    curve,
    xgbFlaggedWithinHorizon: share(xgbAny),
    cnnFlaggedWithinHorizon: share(cnnAny),
    xgbFlaggedWeekAhead: share(xgbWeek),
  };
}

async function build(): Promise<Evaluation> {
  const summary = JSON.parse(await readFile(file("summary.json"), "utf8")) as SummaryJson;
  const [modelRows, patk, coverage, importance, channelAuc, ladder] = await Promise.all([
    readCsv(file("model_summary.csv")),
    readCsv(file("cnn_vs_xgboost_precision_at_k.csv")),
    readCsv(file("window_coverage.csv")),
    readCsv(file("xgboost_importance.csv")),
    readCsv(file("channel_label_auc.csv")),
    readCsv(file("tree_depth_ladder.csv")),
  ]);

  const models: ModelResult[] = modelRows.map((r) => ({
    model: r.model,
    prAuc: num(r["PR-AUC"]),
    rocAuc: num(r["ROC-AUC"]),
    precision: num(r.precision),
    recall: num(r.recall),
    f1: num(r.F1),
    threshold: num(r.threshold),
    hitsAt100: num(r["drives hit @100"]),
    hitsAt1000: num(r["drives hit @1000"]),
  }));

  const t = summary.xgboost.test_at_tuned;
  const test = coverage.find((r) => r.split === "test");

  return {
    sources: [
      "results/w2022x_summary.json",
      "results/w2022x_model_summary.csv",
      "results/w2022x_cnn_vs_xgboost_precision_at_k.csv",
      "results/w2022x_window_coverage.csv",
      "results/w2022x_lead_time_failing.csv",
      "results/w2022x_xgboost_importance.csv",
      "results/w2022x_channel_label_auc.csv",
      "results/w2022x_tree_depth_ladder.csv",
    ],
    scale: {
      drives: summary.n_drives,
      driveDays: summary.n_drive_days,
      quarters: summary.shards,
      testWindows: test ? num(test["tiling 30-day windows"]) : t.n,
      testPositives: t.n_pos,
      baseRate: t.base_rate,
      splits: coverage.map((r) => ({
        split: r.split,
        drives: num(r.drives),
        drivesWithFailure: num(r["drives with a failure"]),
        windows: num(r["tiling 30-day windows"]),
      })),
    },
    models,
    precisionAtK: patk.map((r) => ({
      k: num(r.K),
      xgbHits: num(r["XGBoost drive hits"]),
      cnnHits: num(r["CNN drive hits"]),
    })),
    confusion: { tp: t.tp, fp: t.fp, fn: t.fn, tn: t.tn, n: t.n, threshold: t.threshold },
    accuracy: { alwaysHealthy: 1 - t.base_rate, xgboost: (t.tp + t.tn) / t.n },
    leadTime: await leadTime(30),
    importance: importance.map((r) => ({ feature: r.feature, gain: num(r.gain) })),
    channelAuc: channelAuc.map((r) => ({ channel: r.channel, auc: num(r.auc_vs_window_label) })),
    treeLadder: ladder.map((r) => ({
      maxDepth: r.max_depth,
      actualDepth: num(r.actual_depth),
      leaves: num(r.leaves),
      trainAccuracy: num(r.train_accuracy),
      testAccuracy: num(r.test_accuracy),
      trainBalanced: num(r.train_balanced_acc),
      testBalanced: num(r.test_balanced_acc),
    })),
    leakage: {
      passed: summary.leakage_checks_passed,
      maxChannelAuc: summary.max_channel_auc_vs_label,
      blocklist: summary.blocklist,
    },
    xgbTrees: summary.xgboost.best_iteration + 1,
  };
}

let cached: Promise<Evaluation> | null = null;

/** Read once; the results files do not change while the server runs. */
export function evaluation(): Promise<Evaluation> {
  cached ??= build().catch((e) => {
    cached = null;
    throw e;
  });
  return cached;
}
