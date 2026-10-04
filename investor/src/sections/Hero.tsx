import { lazy, Suspense, useMemo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { AlertTriangle, ArrowRight, BarChart3, CalendarDays, Clock, Database, HardDrive, Play } from "lucide-react";
import type { SignalSeries } from "../../shared/types";
import { api } from "../lib/api";
import { useStoryData } from "../lib/data";
import { fmtCompact, fmtInt } from "../lib/format";
import { EASE, hasWebGL, useIsCompact } from "../lib/motion";
import { smartName } from "../lib/smart";
import { useApi } from "../lib/useApi";

const HeroScene = lazy(() => import("../scenes/HeroScene"));

/** Shown instantly, and instead of WebGL where there is none. */
function StaticBackdrop() {
  return (
    <div aria-hidden className="absolute inset-0">
      <div className="absolute left-1/2 top-1/2 h-64 w-64 -translate-x-1/2 -translate-y-1/2 rounded-full border border-signal/20" />
      <div className="absolute left-1/2 top-1/2 h-40 w-40 -translate-x-1/2 -translate-y-1/2 rounded-full border border-signal/10" />
      <div className="absolute left-[58%] top-[42%] h-2 w-2 rounded-full bg-risk shadow-[0_0_24px_6px_rgba(255,77,106,0.45)]" />
    </div>
  );
}

function Spark({ values, tone }: { values: (number | null)[]; tone: string }) {
  const nums = values.map((v) => v ?? 0);
  const lo = Math.min(...nums);
  const hi = Math.max(...nums);
  const span = hi - lo || 1;
  const d = nums.map((v, i) => `${i ? "L" : "M"}${((i / Math.max(1, nums.length - 1)) * 100).toFixed(1)},${(26 - ((v - lo) / span) * 22).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 100 28" preserveAspectRatio="none" className="h-7 w-24" aria-hidden>
      <path d={d} fill="none" stroke={tone} strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

const SPARK_ATTRS = ["smart_197_raw", "smart_198_raw", "smart_9_raw", "smart_194_raw", "smart_193_raw"];

/** Real telemetry of the failing demo drive, as small multiples beside the 3D drive. */
function TelemetryCard({ s }: { s: SignalSeries }) {
  return (
    <div className="glass w-60 rounded-xl p-4">
      <p className="text-[11px] text-muted">Last 30 days · demo drive</p>
      <p className="num mb-3 truncate text-xs text-soft">{s.serial}</p>
      <ul className="space-y-2">
        {SPARK_ATTRS.map((a) => {
          const v = s.series[a] ?? [];
          const moved = Math.max(...v.map((x) => x ?? 0)) !== Math.min(...v.map((x) => x ?? 0));
          const rising = (v.at(-1) ?? 0) > (v[0] ?? 0) && (a === "smart_197_raw" || a === "smart_198_raw");
          return (
            <li key={a} className="flex items-center justify-between gap-3">
              <span className="truncate text-[11px] text-muted">{smartName(a)}</span>
              <Spark values={v} tone={rising ? "var(--color-risk)" : moved ? "var(--color-signal)" : "var(--color-faint)"} />
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function StatsStrip() {
  const { evaluation } = useStoryData();
  const e = evaluation.data;
  const xgb = e?.models.find((m) => m.model === "XGBoost");
  const items = [
    { icon: <HardDrive size={22} />, v: e ? fmtInt(e.scale.drives) : "—", l: "Drives analysed" },
    { icon: <CalendarDays size={22} />, v: e ? `${fmtCompact(e.scale.driveDays)}+` : "—", l: "Drive-days of telemetry" },
    { icon: <BarChart3 size={22} />, v: xgb ? `${xgb.hitsAt100} / 100` : "—", l: "Top-100 precision (test)", hint: "Of the 100 held-out test drives ranked riskiest, how many really failed within 30 days." },
    { icon: <Clock size={22} />, v: "30 days", l: "Prediction horizon" },
    { icon: <Database size={22} />, v: e ? `${e.scale.quarters.length} quarters` : "—", l: "Real-world fleet data" },
  ];
  return (
    <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-2xl border hairline bg-line sm:grid-cols-3 lg:grid-cols-5">
      {items.map((it, i) => (
        <div key={it.l} className={`flex items-center gap-4 bg-night/95 px-5 py-5 ${i === 4 ? "col-span-2 sm:col-span-1" : ""}`} title={it.hint}>
          <span className="text-signal/80" aria-hidden>
            {it.icon}
          </span>
          <div className="min-w-0">
            <dd className="text-xl font-semibold tabular-nums tracking-tight text-ink">{it.v}</dd>
            <dt className="truncate text-xs text-muted">{it.l}</dt>
          </div>
        </div>
      ))}
    </dl>
  );
}

/** Hero. Artifact A. */
export function Hero() {
  const reduce = !!useReducedMotion();
  const compact = useIsCompact();
  const webgl = useMemo(hasWebGL, []);
  const signal = useApi(api.signal);
  const failing = signal.data?.failing;
  const pending = failing?.series.smart_197_raw ?? [];
  const pendingFrom = pending.find((v) => v !== null) ?? 0;
  const pendingTo = [...pending].reverse().find((v) => v !== null) ?? 0;

  const t = (d: number) =>
    reduce ? {} : { initial: { opacity: 0, y: 18 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.8, ease: EASE, delay: d } };

  return (
    <section id="hook" data-chapter="hook" aria-labelledby="hook-title" className="relative overflow-hidden pb-14 pt-28 md:pt-32">
      <div aria-hidden className="absolute inset-0" style={{ background: "radial-gradient(70% 60% at 70% 30%, #0d1a38 0%, #070b14 50%, #04060b 85%)" }} />
      <div className="relative mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="grid items-center gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <div>
            <motion.p {...t(0.05)} className="kicker mb-5">
              Predictive maintenance for data infrastructure
            </motion.p>
            <motion.h1 {...t(0.12)} id="hook-title" className="text-[clamp(2.6rem,5.4vw,4.5rem)] font-semibold leading-[1.02] tracking-[-0.03em]">
              Know the failure <span className="text-signal">before</span> it happens.
            </motion.h1>
            <motion.p {...t(0.2)} className="mt-6 max-w-xl text-base leading-relaxed text-soft md:text-lg">
              We read the health telemetry hard drives already report every day, and rank which drives are most likely to fail within the next 30
              days, turning reactive maintenance into planned action.
            </motion.p>
            <motion.div {...t(0.28)} className="mt-8 flex flex-wrap items-center gap-3">
              <a href="#intelligence" className="btn btn-ghost">
                Explore the technology <ArrowRight size={15} aria-hidden />
              </a>
              <a href="#product" className="btn btn-primary">
                <Play size={14} aria-hidden /> Try the live demo
              </a>
            </motion.div>
          </div>

          <motion.div {...t(0.15)} className="relative h-[340px] sm:h-[420px] lg:h-[480px]">
            {webgl ? (
              <Suspense fallback={<StaticBackdrop />}>
                <HeroScene animate={!reduce} compact={compact} />
              </Suspense>
            ) : (
              <StaticBackdrop />
            )}
            {failing && (
              <>
                <div className="absolute left-0 top-4 flex items-center gap-2.5 rounded-lg border border-warn/50 bg-void/70 px-3 py-2 backdrop-blur sm:left-4">
                  <AlertTriangle size={16} className="text-warn" aria-hidden />
                  <div className="leading-tight">
                    <p className="text-[11px] text-warn">Pending sectors rising</p>
                    <p className="num text-xs text-ink">
                      {pendingFrom} → {pendingTo} in 30 days
                    </p>
                  </div>
                </div>
                <div className="absolute right-0 top-4 hidden xl:block">
                  <TelemetryCard s={failing} />
                </div>
              </>
            )}
          </motion.div>
        </div>

        <motion.div {...t(0.35)} className="mt-10 md:mt-12">
          <StatsStrip />
        </motion.div>
      </div>
    </section>
  );
}
