import { useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Check, Circle } from "lucide-react";
import type { Evaluation } from "../../shared/types";
import { Expandable } from "../components/Expandable";
import { StorySection } from "../components/StorySection";
import { ErrorState, Loading, Source } from "../components/States";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { useStoryData } from "../lib/data";
import { fmtInt, fmtPct } from "../lib/format";
import { useRise } from "../lib/motion";

const tooltipStyle = { background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 };

/** K dots, one per ranked drive: filled = really failed, hollow = did not. */
function TopK({ e }: { e: Evaluation }) {
  const ks = e.precisionAtK.map((p) => p.k);
  const [k, setK] = useState(ks.includes(100) ? 100 : ks[0]);
  const row = e.precisionAtK.find((p) => p.k === k)!;
  const reduce = useReducedMotion();
  const dot = k <= 100 ? "h-3.5 w-3.5 sm:h-4 sm:w-4" : k <= 250 ? "h-2.5 w-2.5" : "h-1.5 w-1.5 sm:h-2 sm:w-2";

  return (
    <div className="panel grid gap-10 lg:grid-cols-[1fr_1.1fr] lg:gap-16">
      <div>
        <p className="kicker mb-4 text-[10px]">Ranking precision · held-out test drives</p>
        <p className="text-[clamp(3.5rem,8vw,6rem)] font-semibold leading-[0.9] tracking-[-0.05em] tabular-nums">
          {row.xgbHits}
          <span className="text-faint">/{k}</span>
        </p>
        <p className="mt-5 max-w-md text-base leading-relaxed text-soft">
          Of the <b className="font-medium text-ink">{k}</b> test drives the system ranked as most at risk,{" "}
          <b className="font-medium text-ok">{row.xgbHits}</b> really did fail within 30 days.
        </p>
        <div role="radiogroup" aria-label="How many top-ranked drives" className="mt-8 flex flex-wrap gap-2">
          {ks.map((kk) => (
            <button
              key={kk}
              role="radio"
              aria-checked={kk === k}
              onClick={() => setK(kk)}
              className={`num rounded-full border px-3.5 py-1.5 text-xs transition ${
                kk === k ? "border-signal bg-signal/10 text-ink" : "border-line text-muted hover:text-soft"
              }`}
            >
              top {fmtInt(kk)}
            </button>
          ))}
        </div>
      </div>
      <div>
        <div className="flex flex-wrap gap-[5px]" role="img" aria-label={`${row.xgbHits} of the top ${k} ranked drives failed; ${k - row.xgbHits} did not.`}>
          {Array.from({ length: k }, (_, i) => {
            const hit = i < row.xgbHits;
            return (
              <motion.span
                key={`${k}-${i}`}
                initial={reduce ? false : { opacity: 0, scale: 0.4 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: reduce ? 0 : Math.min(1.2, i * (1.2 / k)), duration: 0.3 }}
                className={`${dot} rounded-full ${hit ? "bg-ok" : "border border-muted"}`}
              />
            );
          })}
        </div>
        <div className="mt-5 flex flex-wrap gap-5 text-xs text-muted">
          <span className="flex items-center gap-2">
            <Check size={12} className="text-ok" aria-hidden /> failed within 30 days ({row.xgbHits})
          </span>
          <span className="flex items-center gap-2">
            <Circle size={11} aria-hidden /> did not fail ({k - row.xgbHits})
          </span>
        </div>
      </div>
    </div>
  );
}

/** Why accuracy is the wrong yardstick, in this project's own numbers. */
function AccuracyTrap({ e }: { e: Evaluation }) {
  const rise = useRise();
  const xgb = e.models.find((m) => m.model === "XGBoost");
  const rows = [
    {
      who: "A “model” that always answers healthy",
      acc: e.accuracy.alwaysHealthy,
      found: 0,
      top: null as number | null,
    },
    {
      who: "Our model",
      acc: e.accuracy.xgboost,
      found: e.confusion.tp,
      top: (xgb?.hitsAt100 ?? null) as number | null,
    },
  ];
  return (
    <motion.div {...rise} className="panel grid gap-10 lg:grid-cols-[1fr_1.4fr] lg:gap-16">
      <div>
        <h3 className="text-xl font-semibold tracking-tight md:text-2xl">
          Why we don't lead with <span>“accuracy”</span>.
        </h3>
        <p className="mt-4 text-sm leading-relaxed text-soft md:text-[15px]">
          Failures are rare: {fmtPct(e.scale.baseRate, 2)} of cases. A system that never raises an alarm is already{" "}
          {fmtPct(e.accuracy.alwaysHealthy, 2)} “accurate” and completely useless. What matters is whether the drives at
          the top of the list are the ones that will really fail.
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-left text-sm">
          <thead>
            <tr className="text-xs text-muted">
              <th className="pb-4 pr-4 font-normal" />
              <th className="pb-4 pr-4 font-normal">Accuracy</th>
              <th className="pb-4 pr-4 font-normal">Failing cases caught</th>
              <th className="pb-4 font-normal">Right in top 100</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.who} className="border-t hairline">
                <td className={`py-5 pr-4 ${i ? "text-ink" : "text-muted"}`}>{r.who}</td>
                <td className="num py-5 text-soft">{fmtPct(r.acc, 2)}</td>
                <td className={`num py-5 ${i ? "text-ok" : "text-risk"}`}>
                  {fmtInt(r.found)} <span className="text-faint">/ {fmtInt(e.scale.testPositives)}</span>
                </td>
                <td className={`num py-5 ${i ? "text-ok" : "text-risk"}`}>{r.top ?? "no ranking"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-4 text-sm text-muted">Same accuracy. Completely different value.</p>
      </div>
    </motion.div>
  );
}

function LeadTime({ e }: { e: Evaluation }) {
  const lt = e.leadTime;
  const data = useMemo(
    () => (lt ? lt.curve.map((p) => ({ lead: p.lead, flagged: p.xgbFlagRate * 100 })) : []),
    [lt],
  );
  if (!lt) return null;
  return (
    <div className="panel grid gap-10 lg:grid-cols-[1fr_1.4fr] lg:gap-16">
      <div>
        <h3 className="text-xl font-semibold tracking-tight md:text-2xl">
          How <span className="text-signal">early</span> it sees it.
        </h3>
        <p className="mt-4 text-sm leading-relaxed text-soft md:text-[15px]">
          We replayed {fmtInt(lt.drives)} drives that failed during the test period, day by day, through the model.
        </p>
        <dl className="mt-10 grid grid-cols-2 gap-8">
          <div>
            <dd className="num text-4xl text-ink">{fmtPct(lt.xgbFlaggedWithinHorizon, 0)}</dd>
            <dt className="mt-2 text-sm text-muted">were flagged for inspection at least once in their final 30 days</dt>
          </div>
          <div>
            <dd className="num text-4xl text-ink">{fmtPct(lt.xgbFlaggedWeekAhead, 0)}</dd>
            <dt className="mt-2 text-sm text-muted">were flagged at least a week before they failed</dt>
          </div>
        </dl>
      </div>
      <figure className="min-w-0">
        <figcaption className="mb-3 text-sm text-soft">Share of failing drives flagged, by days before failure</figcaption>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
              <defs>
                <linearGradient id="leadFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-series-xgb)" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="var(--color-series-xgb)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="lead"
                reversed
                tickLine={false}
                axisLine={false}
                ticks={[90, 60, 30, 14, 7, 0]}
                type="number"
                domain={[0, 90]}
                tickFormatter={(d) => (d === 0 ? "fail" : `${d}d`)}
              />
              <YAxis tickLine={false} axisLine={false} unit="%" width={48} domain={[0, 60]} />
              <ReferenceLine x={30} stroke="var(--color-faint)" strokeDasharray="4 4" label={{ value: "30-day horizon", fill: "var(--color-muted)", fontSize: 11, position: "insideTopLeft" }} />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(l) => (Number(l) === 0 ? "on the day of failure" : `${l} days before failure`)}
                formatter={(v) => [`${Number(v).toFixed(1)}% flagged`, "XGBoost"]}
              />
              <Area type="monotone" dataKey="flagged" stroke="var(--color-series-xgb)" strokeWidth={2} fill="url(#leadFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-muted">The closer a drive gets to failure, the more likely the model is to flag it on that day.</p>
      </figure>
    </div>
  );
}

/** Artifact E. */
export function PerformanceProof() {
  const { evaluation } = useStoryData();
  const { data: e, error, loading, reload } = evaluation;
  if (loading) return <Loading label="Reading the evaluation results" />;
  if (error || !e) return <ErrorState message={error ?? "no evaluation results"} onRetry={reload} />;
  const xgb = e.models.find((m) => m.model === "XGBoost");

  return (
    <div className="space-y-6">
      <TopK e={e} />
      <AccuracyTrap e={e} />
      <LeadTime e={e} />

      <p className="text-xs leading-relaxed text-muted">
        Scope: tested on one large public fleet (Backblaze, {e.scale.quarters.length} quarters). Any new fleet should be validated on its
        own history before relying on these numbers.
      </p>
      <Expandable
        title="Standard ML metrics and methodology"
        hint={`PR-AUC ${xgb?.prAuc.toFixed(3)} · ROC-AUC ${xgb?.rocAuc.toFixed(3)}, and how precision, thresholds and lead time were measured.`}
      >
          {xgb && (
            <div className="grid gap-px overflow-hidden rounded-2xl border hairline bg-line sm:grid-cols-2">
              {[
                {
                  v: xgb.prAuc,
                  base: e.scale.baseRate,
                  t: "Ranking quality on rare events (PR-AUC)",
                  d: `A random ranking would score ${e.scale.baseRate.toFixed(4)}. Ours is about ${Math.round(xgb.prAuc / e.scale.baseRate)}× that.`,
                },
                {
                  v: xgb.rocAuc,
                  base: 0.5,
                  t: "Separation of failing from healthy (ROC-AUC)",
                  d: `Pick one failing and one healthy case at random: ${fmtPct(xgb.rocAuc, 0)} of the time the failing one is ranked higher. Random is 50%.`,
                },
              ].map((m) => (
                <div key={m.t} className="bg-night/95 p-6 md:p-7">
                  <p className="num text-4xl">{m.v.toFixed(3)}</p>
                  <p className="mt-3 text-soft">{m.t}</p>
                  <p className="mt-2 text-sm text-muted">{m.d}</p>
                </div>
              ))}
            </div>
          )}

          <div className="pt-2">
            <TechnicalDisclosure title="Why precision@K is the right measure for a rare-event problem">
              <p>
                With a {fmtPct(e.scale.baseRate, 2)} base rate, accuracy and ROC-AUC are dominated by the {fmtInt(e.confusion.tn)}{" "}
                easy healthy cases. An operator does not act on every drive. They act on a short list. <b>Precision@K</b> asks
                exactly that question: of the K drives at the top of the list, how many really fail? The drive-level ladder is
                read from <code>w2022x_cnn_vs_xgboost_precision_at_k.csv</code>.
              </p>
              <p>
                <b>PR-AUC</b> summarises precision across every possible list length. Its baseline is the base rate (
                {e.scale.baseRate.toFixed(4)}), not 0.5, which is why {xgb?.prAuc.toFixed(3)} is a strong result here even
                though it looks modest next to the ROC-AUC.
              </p>
            </TechnicalDisclosure>
            <TechnicalDisclosure title="Ranking performance versus the decision threshold">
              <p>
                The ranking is the product: sort the fleet, work from the top. The threshold ({e.confusion.threshold.toFixed(3)},
                tuned on validation F1) decides only where “inspect” starts. At that threshold, on {fmtInt(e.confusion.n)} test
                windows, {fmtPct(xgb?.precision ?? 0)} of flagged windows were real failures and{" "}
                {fmtPct(xgb?.recall ?? 0)} of failing windows were flagged ({fmtInt(e.confusion.tp)} true alerts,{" "}
                {fmtInt(e.confusion.fp)} false alarms, {fmtInt(e.confusion.fn)} missed).
              </p>
              <p>
                These figures are measured at the fleet's real failure rate, on drives held out from training. They are not from
                a balanced sample, where the same model would look far better. Precision figures from the separate Toshiba
                cohort study are inflated by construction and are not shown here as performance.
              </p>
            </TechnicalDisclosure>
            <TechnicalDisclosure title="How the lead-time curve was measured">
              <p>
                Each failing test drive is scored with its 30-day window ending 0 to 90 days before its failure (
                <code>w2022x_lead_time_failing.csv</code>). The curve is the share of drives scoring above the threshold at each
                distance. The headline shares count a drive once if it was flagged on any day inside the 30-day horizon (or,
                for “a week before”, on any day 7 to 29 days out).
              </p>
            </TechnicalDisclosure>
          </div>
      </Expandable>
      <Source>{e.sources.slice(0, 5).join(" · ")}</Source>
    </div>
  );
}

/** 05 — The Breakthrough. */
export default function PerformanceMetrics() {
  return (
    <StorySection
      id="breakthrough"
      kicker="Results"
      title={
        <>
          At the top of its list, it is <span className="text-ok">almost always</span> right.
        </>
      }
      band
      lede="Measured on real fleet data at its true failure rate, on drives the model never saw during training."
    >
      <PerformanceProof />
    </StorySection>
  );
}
