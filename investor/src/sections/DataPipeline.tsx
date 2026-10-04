import { useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Brain, ChevronRight, Database, Filter, Gauge, ShieldCheck, Sparkles } from "lucide-react";
import type { Evaluation, RunInfo } from "../../shared/types";
import { useStoryData } from "../lib/data";
import { fmtCompact, fmtInt } from "../lib/format";
import { EASE } from "../lib/motion";

interface Stage {
  id: string;
  label: string;
  short: string;
  icon: ReactNode;
  plain: string;
  fact: (e: Evaluation | null, r: RunInfo | null) => ReactNode;
  tech: string;
}

const STAGES: Stage[] = [
  {
    id: "raw",
    short: "Daily SMART reports from every drive in the fleet",
    label: "Raw telemetry",
    icon: <Database size={18} />,
    plain: "Every drive reports a set of health counters once a day. We start from those reports, exactly as they were recorded.",
    fact: (e) =>
      e ? (
        <>
          <b>{fmtCompact(e.scale.driveDays)}</b> daily reports from <b>{fmtInt(e.scale.drives)}</b> drives, {e.scale.quarters.length} quarterly
          public archives ({e.scale.quarters[0].replace("_", " ")} – {e.scale.quarters.at(-1)?.replace("_", " ")}).
        </>
      ) : null,
    tech: "Backblaze quarterly drive-stats archives, reduced to parquet shards without extracting CSVs to disk (scripts/build_backblaze_archive.py).",
  },
  {
    id: "clean",
    short: "Continuous calendar, gaps marked, comparable scales",
    label: "Cleaning",
    icon: <Filter size={18} />,
    plain: "Each drive's history is laid on a continuous calendar. Missing days are marked as missing, never invented, and counters are put on a comparable scale.",
    fact: (_, r) =>
      r ? (
        <>
          <b>{r.sparse_dropped.length}</b> attributes reported by only part of the fleet were dropped, leaving <b>{r.attributes.length}</b> that every
          drive reports.
        </>
      ) : null,
    tech: "Dense daily calendar per drive, log1p on counters, an observed-mask channel for gaps, and the scaler fitted at training time is replayed at serving time (no train/serve skew).",
  },
  {
    id: "audit",
    short: "Remove any column that leaks the answer",
    label: "Leakage audit",
    icon: <ShieldCheck size={18} />,
    plain: "Before any model sees the data, every column is tested for whether it secretly gives away the answer. The ones that do are removed.",
    fact: (e, r) =>
      e && r ? (
        <>
          <b>{r.blocklist.length}</b> columns blocked as leaks or identifiers. The strongest remaining single attribute scores{" "}
          <b>{e.leakage.maxChannelAuc.toFixed(2)}</b> AUC on its own: a real signal, not an echo of the answer.
        </>
      ) : null,
    tech: "scripts/leakage_audit.py scores every column against the label on whole drives; LeakageReport.assert_clean() raises if a blocked column, or anything derived from it, reaches the feature matrix.",
  },
  {
    id: "features",
    short: "Summarise each 30-day window per signal",
    label: "Feature engineering",
    icon: <Sparkles size={18} />,
    plain: "For each drive we look at its last 30 days and summarise how every counter behaved: where it ended, its average, and how much it changed.",
    fact: (_, r) =>
      r ? (
        <>
          <b>{r.attributes.length}</b> attributes → <b>{r.n_channels}</b> signals (value, 1-day and 7-day change, reporting completeness) →{" "}
          <b>{r.n_channels * 3}</b> numbers per drive.
        </>
      ) : null,
    tech: "Channels: 15 log-scaled attributes, their lag-1 and lag-7 differences, plus an observed mask. XGBoost reads each channel's last value, mean and first-to-last delta over the window.",
  },
  {
    id: "model",
    short: "Gradient-boosted trees learn failure patterns",
    label: "Machine learning",
    icon: <Brain size={18} />,
    plain: "A model learns, from millions of past examples, which patterns came before a failure and which did not.",
    fact: (e) =>
      e ? (
        <>
          <b>{e.xgbTrees}</b> gradient-boosted decision trees, trained on <b>{fmtCompact(e.scale.splits[0]?.windows ?? 0, 1)}</b> windows from{" "}
          <b>{fmtInt(e.scale.splits[0]?.drives ?? 0)}</b> drives, and tested on different drives it never saw.
        </>
      ) : null,
    tech: "XGBoost, binary:logistic, early-stopped on validation PR-AUC. A dilated 1-D CNN was trained on the identical windows for comparison (Interactive demo).",
  },
  {
    id: "risk",
    short: "A 30-day risk score; riskiest drives first",
    label: "Risk prediction",
    icon: <Gauge size={18} />,
    plain: "Every drive gets a risk score: the chance it fails in the next 30 days. The fleet is sorted so the riskiest drives are at the top of the list.",
    fact: (e, r) =>
      e && r ? (
        <>
          Drives scoring above <b>{e.confusion.threshold.toFixed(3)}</b> are marked <b>inspect</b>, within a <b>{r.horizon_days}-day</b> horizon.
        </>
      ) : null,
    tech: "The threshold is tuned on validation F1 and drawn on every risk bar in the product. Ranking quality, not the threshold, is what the Results section measures.",
  },
];

/** Artifact C: raw telemetry to ranked risk, one clickable stage card at a time. */
export function DataPipeline() {
  const [active, setActive] = useState(0);
  const reduce = useReducedMotion();
  const { evaluation, model } = useStoryData();
  const stage = STAGES[active];

  return (
    <div>
      <ol className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {STAGES.map((s, i) => {
          const on = i === active;
          return (
            <li key={s.id} className="relative">
              <button
                type="button"
                aria-pressed={on}
                onClick={() => setActive(i)}
                className={`group relative flex h-full w-full flex-col overflow-hidden rounded-xl border p-4 text-left transition-all duration-300 ${
                  on ? "border-signal/60 bg-signal/[0.07] shadow-[0_0_32px_-12px_rgba(63,224,255,0.55)]" : "border-line bg-night/60 hover:border-line-strong hover:bg-deep/50"
                }`}
              >
                {on && !reduce && (
                  <motion.span
                    aria-hidden
                    className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-signal to-transparent"
                    initial={{ x: "-100%" }}
                    animate={{ x: "100%" }}
                    transition={{ duration: 2.4, repeat: Infinity, ease: "linear" }}
                  />
                )}
                <span
                  className={`mb-4 flex h-10 w-10 items-center justify-center rounded-lg border transition ${
                    on ? "border-signal/50 bg-signal/15 text-signal" : "border-line-strong text-muted group-hover:text-soft"
                  }`}
                >
                  {s.icon}
                </span>
                <span className={`text-sm font-medium ${on ? "text-ink" : "text-soft"}`}>
                  {i + 1}. {s.label}
                </span>
                <span className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-muted">{s.short}</span>
              </button>
              {i < STAGES.length - 1 && (
                <span aria-hidden className="absolute -right-[11px] top-1/2 z-10 hidden -translate-y-1/2 text-faint lg:block">
                  <ChevronRight size={14} />
                </span>
              )}
            </li>
          );
        })}
      </ol>

      <div className="panel mt-4" aria-live="polite">
        <AnimatePresence mode="wait">
          <motion.div
            key={stage.id}
            initial={reduce ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? undefined : { opacity: 0, y: -6 }}
            transition={{ duration: 0.35, ease: EASE }}
            className="grid gap-6 md:grid-cols-3 md:gap-10"
          >
            <div>
              <p className="kicker mb-2 text-[10px]">Stage {active + 1} of {STAGES.length}</p>
              <h3 className="text-lg font-semibold">{stage.label}</h3>
              <p className="mt-2 text-sm leading-relaxed text-soft">{stage.plain}</p>
            </div>
            <div>
              <p className="mb-2 text-xs font-medium text-muted">In this project</p>
              <p className="text-sm leading-relaxed text-soft [&_b]:num [&_b]:font-medium [&_b]:text-ink">
                {stage.fact(evaluation.data, model.data?.run ?? null) ?? <span className="text-muted">Loading the numbers…</span>}
              </p>
            </div>
            <div>
              <p className="mb-2 text-xs font-medium text-muted">Under the hood</p>
              <p className="text-sm leading-relaxed text-muted">{stage.tech}</p>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
