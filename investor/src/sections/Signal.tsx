import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, HeartPulse } from "lucide-react";
import type { SignalSeries } from "../../shared/types";
import { StorySection } from "../components/StorySection";
import { ErrorState, Loading, Source } from "../components/States";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { api } from "../lib/api";
import { useRise } from "../lib/motion";
import { SIGNAL_ATTRS, SMART } from "../lib/smart";
import { useApi } from "../lib/useApi";

function Panel({ s, attr, max, tone, title, icon }: { s: SignalSeries; attr: string; max: number; tone: string; title: string; icon: React.ReactNode }) {
  const data = s.dates.map((d, i) => ({ day: i + 1, date: d, v: s.series[attr]?.[i] ?? null }));
  const first = data.find((d) => d.v !== null)?.v ?? 0;
  const last = [...data].reverse().find((d) => d.v !== null)?.v ?? 0;
  return (
    <figure className="panel min-w-0">
      <figcaption className="mb-4 flex items-center justify-between gap-3">
        <span className="flex items-center gap-2 text-sm text-soft">
          {icon}
          {title}
        </span>
        <span className="num text-xs text-muted">
          day 1 <span className="text-soft">{first}</span> → day {data.length} <span style={{ color: tone }}>{last}</span>
        </span>
      </figcaption>
      <div className="h-56 md:h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -18 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="day" tickLine={false} axisLine={false} interval={6} tickFormatter={(d) => `d${d}`} />
            <YAxis domain={[0, max]} tickLine={false} axisLine={false} allowDecimals={false} width={48} />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              contentStyle={{ background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 }}
              labelFormatter={(_, p) => (p?.[0]?.payload?.date as string) ?? ""}
              formatter={(v) => [String(v ?? "—"), SMART[attr]?.name ?? attr]}
            />
            <Line type="monotone" dataKey="v" stroke={tone} strokeWidth={2} dot={false} activeDot={{ r: 4 }} connectNulls isAnimationActive />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

/** Real telemetry: one healthy drive and one in its final month, on the same scale. */
export function SignalVisual() {
  const { data, error, loading, reload } = useApi(api.signal);
  const [attr, setAttr] = useState<string>(SIGNAL_ATTRS[0]);
  const rise = useRise();

  const max = useMemo(() => {
    if (!data) return 1;
    const vals = [...(data.healthy.series[attr] ?? []), ...(data.failing.series[attr] ?? [])].filter((v): v is number => v !== null);
    return Math.max(1, ...vals) * 1.1;
  }, [data, attr]);

  if (loading) return <Loading label="Reading telemetry" />;
  if (error || !data) return <ErrorState message={error ?? "no data"} onRetry={reload} />;

  return (
    <motion.div {...rise}>
      <div role="tablist" aria-label="SMART attribute" className="mb-5 flex flex-wrap gap-2">
        {SIGNAL_ATTRS.map((a) => (
          <button
            key={a}
            role="tab"
            aria-selected={a === attr}
            onClick={() => setAttr(a)}
            className={`rounded-full border px-3.5 py-1.5 text-[13px] transition ${
              a === attr ? "border-signal bg-signal/10 text-ink" : "border-line text-muted hover:border-line-strong hover:text-soft"
            }`}
          >
            {SMART[a].name}
          </button>
        ))}
      </div>
      <p className="mb-6 max-w-3xl text-sm text-soft">
        <b className="font-medium text-ink">{SMART[attr].name}</b> — {SMART[attr].plain}.
        {max <= 1.1 && (
          <span className="text-muted">
            {" "}
            Flat on both drives here: no single counter tells the whole story, which is why the model weighs all fifteen
            together.
          </span>
        )}
      </p>
      <div className="grid gap-6 md:grid-cols-2">
        <Panel
          s={data.healthy}
          attr={attr}
          max={max}
          tone="var(--color-ok)"
          title="Healthy drive"
          icon={<HeartPulse size={15} className="text-ok" aria-hidden />}
        />
        <Panel
          s={data.failing}
          attr={attr}
          max={max}
          tone="var(--color-risk)"
          title={data.failing.failedOnLastDay ? "Drive that failed on day 30" : "Failing drive"}
          icon={<Activity size={15} className="text-risk" aria-hidden />}
        />
      </div>
      <Source>
        {data.source.join(", ")} — {data.healthy.model} and {data.failing.model}, 30 consecutive days each, as recorded.
      </Source>
      <TechnicalDisclosure title="What is SMART telemetry, and which attributes does the model read?" className="mt-8">
        <p>
          SMART (Self-Monitoring, Analysis and Reporting Technology) is a set of counters every modern hard drive keeps
          about itself. Operators already collect it daily, for example with <code>smartctl</code>. No new hardware or
          sensors are required.
        </p>
        <p>The model reads fifteen of them:</p>
        <ul className="grid gap-x-8 gap-y-1 sm:grid-cols-2">
          {Object.entries(SMART).map(([k, v]) => (
            <li key={k}>
              <b>{v.name}</b> <code>{v.smartctl}</code>
            </li>
          ))}
        </ul>
        <p>
          Attributes that only part of the fleet reports were dropped before training, so a missing column on one vendor
          cannot be mistaken for a signal.
        </p>
      </TechnicalDisclosure>
    </motion.div>
  );
}

/** 03 — The Signal. */
export default function Signal() {
  return (
    <StorySection
      id="signal"
      kicker="The data"
      title={
        <>
          Drives already report their own <span className="text-signal">health</span>.
        </>
      }
      band
      lede="Every drive logs its own health daily: errors, damaged spots, temperature, age. One reading says little; the pattern over weeks says a lot. Below, the real 30-day record of a healthy drive and one that failed."
    >
      <SignalVisual />
    </StorySection>
  );
}
