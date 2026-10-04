/**
 * The contract between the Node API and the React app.
 *
 * Prediction shapes mirror what webapp/app.py returns; evaluation shapes are built by
 * server/evaluation.ts from the files in results/. Nothing here is a mock.
 */

export type EstimatorKind = "xgb" | "cnn";

// ---------------------------------------------------------------- model metadata

export interface Metrics {
  pr_auc?: number;
  roc_auc?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  threshold?: number;
  tp?: number;
  fp?: number;
  fn?: number;
  n?: number;
  n_pos?: number;
  base_rate?: number;
}

export interface EstimatorInfo {
  kind: EstimatorKind;
  label: string;
  blurb: string;
  threshold: number;
  metrics: Metrics;
  is_default: boolean;
  explains: boolean;
  size: string;
}

export interface BlockedColumn {
  column: string;
  why: string;
}

/** One trained run, as described by the Flask app's /api/meta. */
export interface RunInfo {
  key: string;
  label: string;
  tag: string;
  scope: string;
  note: string;
  window_days: number;
  horizon_days: number;
  attributes: string[];
  n_channels: number;
  delta_lags: number[];
  default: EstimatorKind;
  estimators: EstimatorInfo[];
  n_drives: number | null;
  n_drive_days: number | null;
  shards: string[] | null;
  blocklist: BlockedColumn[];
  sparse_dropped: string[];
  max_channel_auc: number | null;
  leakage_checks_passed: boolean | null;
}

export interface ModelResponse {
  run: RunInfo;
  trajectoryDays: number;
}

export interface ArtifactSummary {
  key: string;
  label: string;
  tag: string;
  scope: string;
  active: boolean;
  window_days: number;
  horizon_days: number;
  estimators: { kind: EstimatorKind; label: string; size: string; pr_auc: number | null }[];
  files: string[];
}

// ---------------------------------------------------------------- evaluation

export interface ModelResult {
  model: string;
  prAuc: number;
  rocAuc: number;
  precision: number;
  recall: number;
  f1: number;
  threshold: number;
  hitsAt100: number;
  hitsAt1000: number;
}

export interface PrecisionAtK {
  k: number;
  xgbHits: number;
  cnnHits: number;
}

export interface LeadPoint {
  lead: number;
  xgbMean: number;
  cnnMean: number;
  xgbFlagRate: number;
  cnnFlagRate: number;
}

export interface Evaluation {
  sources: string[];
  scale: {
    drives: number;
    driveDays: number;
    quarters: string[];
    testWindows: number;
    testPositives: number;
    baseRate: number;
    splits: { split: string; drives: number; drivesWithFailure: number; windows: number }[];
  };
  models: ModelResult[];
  precisionAtK: PrecisionAtK[];
  confusion: { tp: number; fp: number; fn: number; tn: number; n: number; threshold: number };
  accuracy: { alwaysHealthy: number; xgboost: number };
  leadTime: {
    drives: number;
    curve: LeadPoint[];
    xgbFlaggedWithinHorizon: number;
    cnnFlaggedWithinHorizon: number;
    xgbFlaggedWeekAhead: number;
  } | null;
  importance: { feature: string; gain: number }[];
  channelAuc: { channel: string; auc: number }[];
  treeLadder: {
    maxDepth: string;
    actualDepth: number;
    leaves: number;
    trainAccuracy: number;
    testAccuracy: number;
    trainBalanced: number;
    testBalanced: number;
  }[];
  leakage: { passed: boolean; maxChannelAuc: number; blocklist: string[] };
  xgbTrees: number;
}

// ---------------------------------------------------------------- demos

export interface DemoDataset {
  id: string;
  group: "demo" | "example" | "export";
  label: string;
  description: string;
  file: string;
  drives: number;
  rows: number | null;
  expect: string | null;
}

export interface SignalSeries {
  serial: string;
  model: string;
  dates: string[];
  series: Record<string, (number | null)[]>;
  failedOnLastDay: boolean;
}

export interface SignalResponse {
  source: string[];
  healthy: SignalSeries;
  failing: SignalSeries;
}

// ---------------------------------------------------------------- predictions

export type Call = "inspect" | "hold";

export interface DriveRow {
  serial: string;
  index: number;
  rank: number;
  scores: Partial<Record<EstimatorKind, number>>;
  risk: number;
  calls: Partial<Record<EstimatorKind, Call>>;
  action: Call;
  drive_model: string;
  window_start: string;
  window_end: string;
  n_real_days: number;
  n_filled_days: number;
  n_padded_days: number;
  days_available: number;
  recorded_failure: number;
}

export interface RankingQuality {
  n_failed: number;
  n_drives: number;
  at_k: { k: number; hits: number; precision: number }[];
  recall_at_n_pos: number;
  mean_rank_of_failures: number;
}

export interface Prediction {
  token: string;
  source: string;
  run: string;
  kinds: EstimatorKind[];
  primary: EstimatorKind;
  thresholds: Partial<Record<EstimatorKind, number>>;
  seconds: number;
  pad_short: boolean;
  min_days: number;
  window_days: number;
  horizon_days: number;
  diagnostics: {
    rows: number;
    drives_in_file: number;
    drives_scored: number;
    drives_skipped: number;
    drives_padded: number;
    date_min: string;
    date_max: string;
    attributes_found: string[];
    attributes_missing: string[];
    has_failure_column: boolean;
    notes: string[];
  };
  summary: Partial<Record<EstimatorKind, { inspect: number; hold: number; max_risk: number }>>;
  agreement: { pair: EstimatorKind[]; both: number; only_first: number; only_second: number; spearman: number | null } | null;
  quality: Partial<Record<EstimatorKind, RankingQuality | null>> | null;
  rows: DriveRow[];
  n_rows_total: number;
}

export interface Contribution {
  feature: string;
  contribution: number;
  value: number;
}

export interface DriveDetail extends DriveRow {
  dates: string[];
  attributes: string[];
  series: Record<string, (number | null)[]>;
  latest: Record<string, number | null>;
  trajectory: { dates: string[] } & Partial<Record<EstimatorKind, number[]>>;
  explanations: Partial<Record<EstimatorKind, Contribution[]>>;
}

export interface Health {
  api: "ok";
  engine: { ok: boolean; url: string; latencyMs: number | null; error: string | null };
}
