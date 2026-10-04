import { useState } from "react";
import { motion } from "framer-motion";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Ban, Eye, Lock, Split } from "lucide-react";
import type { Evaluation } from "../../shared/types";
import { StorySection } from "../components/StorySection";
import { ErrorState, Loading, Source } from "../components/States";
import { useStoryData } from "../lib/data";
import { fmtInt, fmtPct } from "../lib/format";
import { useRise } from "../lib/motion";

function Pillar({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  const rise = useRise();
  return (
    <motion.div {...rise} className="panel">
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg border border-signal/40 bg-signal/10 text-signal">{icon}</div>
      <h3 className="text-base font-semibold">{title}</h3>
      <div className="mt-2 text-sm leading-relaxed text-soft">{children}</div>
    </motion.div>
  );
}

function Splits({ e }: { e: Evaluation }) {
  const total = e.scale.splits.reduce((a, s) => a + s.drives, 0);
  const tone: Record<string, string> = { train: "var(--color-faint)", val: "var(--color-violet)", test: "var(--color-signal)" };
  const name: Record<string, string> = { train: "Learn from", val: "Tune on", test: "Judge on" };
  return (
    <div>
      <div className="flex h-3 w-full gap-[2px] overflow-hidden rounded-full" role="img" aria-label="Drive split between training, validation and test">
        {e.scale.splits.map((s) => (
          <span key={s.split} style={{ width: `${(s.drives / total) * 100}%`, background: tone[s.split] }} />
        ))}
      </div>
      <dl className="mt-5 grid grid-cols-3 gap-4">
        {e.scale.splits.map((s) => (
          <div key={s.split}>
            <dt className="flex items-center gap-2 text-xs text-muted">
              <span className="h-2 w-2 rounded-full" style={{ background: tone[s.split] }} />
              {name[s.split] ?? s.split}
            </dt>
            <dd className="num mt-1 text-lg text-ink">{fmtInt(s.drives)}</dd>
            <dd className="text-xs text-faint">drives</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Memorisation({ e }: { e: Evaluation }) {
  const data = e.treeLadder.map((t) => ({
    depth: t.maxDepth === "none" ? "no limit" : String(t.actualDepth),
    train: +(t.trainBalanced * 100).toFixed(1),
    test: +(t.testBalanced * 100).toFixed(1),
  }));
  const deepest = e.treeLadder.at(-1);
  return (
    <div className="panel grid gap-10 lg:grid-cols-[1fr_1.3fr] lg:gap-16">
      <div>
        <h3 className="text-xl font-semibold tracking-tight md:text-2xl">
          Memorising is not <span>predicting.</span>
        </h3>
        <p className="mt-4 text-sm leading-relaxed text-soft md:text-[15px]">
          As a control, we let a single decision tree grow without limit. It memorised its training drives almost perfectly
          {deepest && <> ({fmtPct(deepest.trainBalanced, 1)})</>}, then fell to {deepest && fmtPct(deepest.testBalanced, 0)} on new drives,
          not much better than a coin flip (50%). Every number we report is measured on drives held out from training.
        </p>
      </div>
      <figure className="min-w-0">
        <figcaption className="mb-3 text-sm text-soft">Single decision tree: balanced accuracy by depth</figcaption>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 8, right: 12, left: -14, bottom: 0 }}>
              <CartesianGrid vertical={false} />
              <XAxis dataKey="depth" tickLine={false} axisLine={false} />
              <YAxis domain={[50, 100]} unit="%" tickLine={false} axisLine={false} width={50} />
              <Tooltip
                contentStyle={{ background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 }}
                labelFormatter={(l) => (l === "no limit" ? `no depth limit (grew to ${deepest?.actualDepth})` : `depth ${l}`)}
                formatter={(v, n) => [`${v}%`, n === "train" ? "on training drives" : "on new drives"]}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => (v === "train" ? "on training drives" : "on new drives")} />
              <Line dataKey="train" stroke="var(--color-muted)" strokeDasharray="5 4" strokeWidth={2} dot={{ r: 4 }} />
              <Line dataKey="test" stroke="var(--color-series-xgb)" strokeWidth={2} dot={{ r: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </figure>
    </div>
  );
}

function Ledger() {
  const { model } = useStoryData();
  const [open, setOpen] = useState(false);
  const run = model.data?.run;
  if (!run) return null;
  const rows = open ? run.blocklist : run.blocklist.slice(0, 4);
  return (
    <div>
      <ul className="divide-y divide-line border-y hairline">
        {rows.map((b) => (
          <li key={b.column} className="grid gap-1 py-4 sm:grid-cols-[minmax(0,260px)_1fr] sm:gap-8">
            <span className="flex items-center gap-2 font-mono text-[13px] text-soft">
              <Ban size={13} className="shrink-0 text-risk" aria-hidden /> {b.column}
            </span>
            <span className="text-sm text-muted">{b.why}</span>
          </li>
        ))}
      </ul>
      {run.blocklist.length > 4 && (
        <button type="button" onClick={() => setOpen((o) => !o)} className="chip mt-4 hover:border-signal hover:text-ink" aria-expanded={open}>
          {open ? "Show fewer" : `Show all ${run.blocklist.length} blocked columns`}
        </button>
      )}
      {run.sparse_dropped.length > 0 && (
        <p className="mt-4 text-sm text-muted">
          Also dropped for patchy coverage across vendors: <span className="font-mono text-[12px] text-soft">{run.sparse_dropped.join(", ")}</span>.
        </p>
      )}
    </div>
  );
}

/** 07 — The Trust. */
export default function Trust() {
  const { evaluation } = useStoryData();
  const e = evaluation.data;
  return (
    <StorySection
      id="trust"
      kicker="Methodology"
      title={
        <>
          Built to be <span className="text-signal">checked</span>.
        </>
      }
      band
      lede="Strong machine-learning results are often too good to be true, because the model quietly saw the answer. This is how the evaluation rules that out."
    >
      {evaluation.loading && <Loading />}
      {evaluation.error && <ErrorState message={evaluation.error} onRetry={evaluation.reload} />}
      {e && (
        <div className="space-y-4">
          <div className="grid gap-4 md:grid-cols-3">
            <Pillar icon={<Lock size={20} />} title="No peeking at the answer">
              Every column was tested for whether it gives the outcome away before training began. Leaks were removed, and the pipeline refuses to
              run if one slips back in. The best remaining single signal scores {e.leakage.maxChannelAuc.toFixed(2)} on its own: useful, not a giveaway.
            </Pillar>
            <Pillar icon={<Split size={20} />} title="Judged on strangers">
              Drives were divided before training. The model never saw a single day from the {fmtInt(e.scale.splits.find((s) => s.split === "test")?.drives ?? 0)}{" "}
              drives it was tested on, and each of those drives' full recorded history was scored, 30 days at a time.
            </Pillar>
            <Pillar icon={<Eye size={20} />} title="Every call explained">
              For each drive, the product shows the exact contribution of each signal to its score, the telemetry behind it, and how its risk evolved.
              Nothing is a black box to the operator.
            </Pillar>
          </div>

          <div className="panel grid gap-10 lg:grid-cols-[1fr_1.3fr] lg:gap-16">
            <div>
              <h3 className="text-xl font-semibold tracking-tight md:text-2xl">What the model is not allowed to see.</h3>
              <p className="mt-4 text-sm leading-relaxed text-soft md:text-[15px]">
                Some columns look harmless and give the answer away. A drive that stops reporting has, by definition, failed. Rack position goes
                missing once a disk is pulled. These were found by the audit and blocked.
              </p>
              <div className="mt-10">
                <Splits e={e} />
              </div>
            </div>
            <Ledger />
          </div>

          <Memorisation e={e} />

          <div className="panel">
            <h3 className="text-base font-semibold">What we don't claim</h3>
            <ul className="mt-4 grid list-disc gap-3 pl-5 text-sm leading-relaxed text-soft marker:text-faint md:grid-cols-2">
              <li>The results come from one large public fleet (Backblaze, {e.scale.quarters.length} quarters). Any new fleet should be validated on its own history first.</li>
              <li>Risk scores rank drives. They are not a guarantee, and the inspect threshold was tuned for this fleet.</li>
              <li>Drives with less than 30 days of history are padded to score at all; those scores are marked as extrapolations.</li>
              <li>An earlier Toshiba study used a matched cohort, which inflates precision by design. Its numbers are not presented as performance here.</li>
            </ul>
          </div>
          <Source>/api/model (blocklist, from webapp/scoring.py) · results/w2022x_window_coverage.csv · results/w2022x_tree_depth_ladder.csv</Source>
        </div>
      )}
    </StorySection>
  );
}
