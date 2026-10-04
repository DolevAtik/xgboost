import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, CheckCircle2, FileSpreadsheet, Info, Upload, Wrench, X } from "lucide-react";
import type { DemoDataset, DriveDetail, DriveRow, EstimatorKind, Prediction } from "../../shared/types";
import { RiskTrajectoryChart } from "../components/RiskTrajectoryChart";
import { SHAPExplanation } from "../components/SHAPExplanation";
import { ErrorState, Loading } from "../components/States";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { api } from "../lib/api";
import { fmtInt, riskCategory, toneColor } from "../lib/format";
import { EASE } from "../lib/motion";
import { SMART, smartName } from "../lib/smart";
import { useApi, type Async } from "../lib/useApi";

const plural = (n: number, word: string) => `${fmtInt(n)} ${word}${n === 1 ? "" : "s"}`;

const GROUPS: { id: DemoDataset["group"]; title: string }[] = [
  { id: "demo", title: "Demo fleets" },
  { id: "export", title: "Real 90-day exports" },
  { id: "example", title: "Engineering test files" },
];

// ---------------------------------------------------------------- dataset picker

function DatasetPicker({
  demos,
  selected,
  busy,
  onPick,
  onFile,
}: {
  demos: Async<DemoDataset[]>;
  selected: string | null;
  busy: boolean;
  onPick: (d: DemoDataset) => void;
  onFile: (f: File) => void;
}) {
  const { data, error, loading, reload } = demos;
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  return (
    <div className="space-y-8">
      {loading && <Loading label="Listing datasets" />}
      {error && <ErrorState message={error} onRetry={reload} />}
      {data &&
        GROUPS.map((g) => {
          const items = data.filter((d) => d.group === g.id);
          if (!items.length) return null;
          return (
            <div key={g.id}>
              <p className="kicker mb-3 text-[10px]">{g.title}</p>
              <ul className="space-y-1">
                {items.map((d) => {
                  const on = d.id === selected;
                  return (
                    <li key={d.id}>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => onPick(d)}
                        aria-pressed={on}
                        className={`w-full rounded-xl px-3 py-2.5 text-left transition disabled:opacity-60 ${
                          on ? "bg-signal/10 ring-1 ring-signal/50" : "hover:bg-deep/60"
                        }`}
                      >
                        <span className="flex items-baseline justify-between gap-3">
                          <span className={`text-sm ${on ? "text-ink" : "text-soft"}`}>{d.label}</span>
                          <span className="num shrink-0 text-[11px] text-faint">{plural(d.drives, "drive")}</span>
                        </span>
                        <span className="mt-0.5 block truncate text-xs text-muted">{d.description}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          );
        })}

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files[0];
          if (f && !busy) onFile(f);
        }}
        className={`rounded-2xl border border-dashed p-5 text-center transition ${drag ? "border-signal bg-signal/5" : "border-line-strong"}`}
      >
        <Upload size={18} className="mx-auto text-muted" aria-hidden />
        <p className="mt-2 text-sm text-soft">Upload your own telemetry</p>
        <p className="mt-1 text-xs text-muted">.xlsx, .csv or .parquet · up to 400 MB · never stored</p>
        <button type="button" disabled={busy} onClick={() => input.current?.click()} className="chip mt-3 hover:border-signal hover:text-ink">
          Choose a file
        </button>
        <input
          ref={input}
          type="file"
          accept=".xlsx,.xlsm,.xls,.csv,.txt,.parquet"
          className="sr-only"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onFile(f);
            e.target.value = "";
          }}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- ranked queue

function RiskBar({ score, threshold, tone }: { score: number; threshold: number; tone: string }) {
  return (
    <span className="relative block h-1.5 w-full rounded-full bg-deep">
      <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.max(2, score * 100)}%`, background: tone }} />
      <span className="absolute -top-1 h-3.5 w-px bg-soft/70" style={{ left: `${threshold * 100}%` }} title={`threshold ${threshold.toFixed(3)}`} />
    </span>
  );
}

function Queue({ p, selected, onSelect }: { p: Prediction; selected: string | null; onSelect: (r: DriveRow) => void }) {
  const thr = p.thresholds[p.primary] ?? 0.5;
  const truth = p.diagnostics.has_failure_column;
  return (
    <div className="max-h-[560px] overflow-y-auto pr-1" role="listbox" aria-label="Drives ranked by risk">
      {p.rows.map((r) => {
        const cat = riskCategory(r.risk, thr);
        const on = r.serial === selected;
        return (
          <button
            key={r.serial}
            role="option"
            aria-selected={on}
            onClick={() => onSelect(r)}
            className={`grid w-full grid-cols-[2.2rem_1fr_auto] items-center gap-3 rounded-lg px-3 py-2.5 text-left transition ${
              on ? "bg-deep ring-1 ring-line-strong" : "hover:bg-deep/50"
            }`}
          >
            <span className="num text-xs text-faint">{r.rank}</span>
            <span className="min-w-0">
              <span className="flex items-center gap-2">
                <span className="num truncate text-[13px] text-soft">{r.serial}</span>
                {truth && r.recorded_failure === 1 && <span className="rounded bg-risk/15 px-1.5 text-[10px] text-risk">failed</span>}
                {r.n_padded_days > 0 && <span className="rounded bg-warn/10 px-1.5 text-[10px] text-warn">{r.n_real_days}d</span>}
              </span>
              <span className="mt-1.5 block">
                <RiskBar score={r.risk} threshold={thr} tone={toneColor[cat.tone]} />
              </span>
            </span>
            <span className="w-16 text-right">
              <span className="num block text-sm" style={{ color: toneColor[cat.tone] }}>
                {r.risk.toFixed(3)}
              </span>
              <span className="block text-[10px] uppercase tracking-wider text-faint">{cat.call}</span>
            </span>
          </button>
        );
      })}
      {p.n_rows_total > p.rows.length && (
        <p className="px-3 py-3 text-xs text-muted">
          Showing the top {p.rows.length} of {fmtInt(p.n_rows_total)} drives.
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- drive profile

function Telemetry({ d }: { d: DriveDetail }) {
  const moving = d.attributes.filter((a) => {
    const v = (d.series[a] ?? []).filter((x): x is number => x !== null);
    return v.length && Math.max(...v) !== Math.min(...v);
  });
  const options = [...new Set([...["smart_197_raw", "smart_198_raw", "smart_5_raw"].filter((a) => d.attributes.includes(a)), ...moving])];
  const [attr, setAttr] = useState(options[0] ?? d.attributes[0]);
  const data = d.dates.map((date, i) => ({ date, v: d.series[attr]?.[i] ?? null }));
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <label className="flex items-center gap-2 text-xs text-muted">
          Attribute
          <select
            value={attr}
            onChange={(e) => setAttr(e.target.value)}
            className="rounded-lg border border-line bg-night px-2 py-1 text-xs text-ink"
          >
            {d.attributes.map((a) => (
              <option key={a} value={a}>
                {smartName(a)}
                {moving.includes(a) ? " ·" : ""}
              </option>
            ))}
          </select>
        </label>
        <span className="text-[11px] text-faint">{SMART[attr]?.plain} · “·” = changed in this window</span>
      </div>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, left: -14, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={40} tickFormatter={(x: string) => x.slice(5)} />
            <YAxis tickLine={false} axisLine={false} width={50} allowDecimals={false} />
            <Tooltip
              contentStyle={{ background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 }}
              formatter={(v) => [v === null ? "not reported" : String(v), smartName(attr)]}
            />
            <Line type="stepAfter" dataKey="v" stroke="var(--color-signal)" strokeWidth={2} dot={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Profile({ p, serial }: { p: Prediction; serial: string }) {
  const [state, setState] = useState<{ d: DriveDetail | null; error: string | null }>({ d: null, error: null });
  const load = useCallback(() => {
    setState({ d: null, error: null });
    api.drive(p.token, serial).then(
      (d) => setState({ d, error: null }),
      (e: Error) => setState({ d: null, error: e.message }),
    );
  }, [p.token, serial]);
  useEffect(load, [load]);

  const { d, error } = state;
  if (error) return <ErrorState message={error} onRetry={load} />;
  if (!d) return <Loading label="Opening drive" className="py-10" />;

  const thr = p.thresholds[p.primary] ?? 0.5;
  const cat = riskCategory(d.risk, thr);
  const Icon = cat.tone === "ok" ? CheckCircle2 : cat.tone === "warn" ? Wrench : AlertTriangle;

  return (
    <motion.div key={serial} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: EASE }} className="space-y-10">
      <div className="space-y-5">
        <div>
          <p className="num text-xs text-faint">
            #{d.rank} · {d.drive_model}
          </p>
          <p className="num mt-1 text-lg text-soft">{d.serial}</p>
          <p className="num mt-4 text-6xl leading-none" style={{ color: toneColor[cat.tone] }}>
            {d.risk.toFixed(3)}
          </p>
          <p className="mt-2 text-sm text-muted">risk score · failure within the next {p.horizon_days} days</p>
        </div>
        <div className="space-y-2">
          <span
            className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm"
            style={{ borderColor: toneColor[cat.tone], color: toneColor[cat.tone] }}
          >
            <Icon size={14} aria-hidden /> {cat.label} · {cat.call}
          </span>
          <p className="num text-xs text-muted">
            window {d.window_start} → {d.window_end}
          </p>
          {p.kinds.length > 1 && (
            <p className="num text-xs text-muted">
              {p.kinds.map((k) => `${k === "xgb" ? "XGBoost" : "CNN"} ${d.scores[k]?.toFixed(3)}`).join(" · ")}
            </p>
          )}
          {d.n_padded_days > 0 && (
            <p className="text-xs text-warn">
              only {d.n_real_days} real days; {d.n_padded_days} padded — an extrapolation
            </p>
          )}
          {p.diagnostics.has_failure_column && (
            <p className="text-xs text-muted">recorded outcome in file: {d.recorded_failure ? "failed" : "did not fail"}</p>
          )}
        </div>
      </div>

      <section aria-label="Risk trajectory">
        <h4 className="mb-1 text-sm font-medium text-ink">Risk trajectory</h4>
        <p className="mb-4 text-xs text-muted">The same model, re-scored as if it were asked on each earlier day.</p>
        <RiskTrajectoryChart trajectory={d.trajectory} kinds={p.kinds} thresholds={p.thresholds} />
      </section>

      <section aria-label="Why this score">
        <h4 className="mb-1 text-sm font-medium text-ink">Why this score</h4>
        {d.explanations.xgb ? (
          <>
            <p className="mb-4 text-xs text-muted">The signals that moved this drive's XGBoost score most, exact for this prediction.</p>
            <SHAPExplanation contributions={d.explanations.xgb} />
          </>
        ) : (
          <p className="text-xs text-muted">The CNN does not offer per-feature explanations. Add XGBoost to see one.</p>
        )}
      </section>

      <section aria-label="Telemetry">
        <h4 className="mb-1 text-sm font-medium text-ink">Telemetry behind the score</h4>
        <p className="mb-4 text-xs text-muted">The raw daily readings from your file, in the units it reported.</p>
        <Telemetry d={d} />
      </section>
    </motion.div>
  );
}

// ---------------------------------------------------------------- the explorer

/** Artifact F: a real prediction workflow against the live engine. */
export function DriveExplorer() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [kinds, setKinds] = useState<EstimatorKind[]>(["xgb"]);
  const [pending, setPending] = useState<{ label: string; progress: number | null } | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [last, setLast] = useState<{ kind: "demo"; d: DemoDataset } | { kind: "file"; f: File } | null>(null);
  const profileRef = useRef<HTMLDivElement>(null);

  const run = useCallback(async (src: { kind: "demo"; d: DemoDataset } | { kind: "file"; f: File }, models: EstimatorKind[]) => {
    setLast(src);
    setError(null);
    setDatasetId(src.kind === "demo" ? src.d.id : null);
    setPending({ label: src.kind === "demo" ? src.d.label : src.f.name, progress: src.kind === "file" ? 0 : null });
    try {
      const p =
        src.kind === "demo"
          ? await api.predictDemo(src.d.id, models)
          : await api.predictFile(src.f, models, (f) => setPending((s) => (s ? { ...s, progress: f } : s)));
      setResult(p);
      setSelected(p.rows[0]?.serial ?? null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(null);
    }
  }, []);

  // Open on the fleet demo so the explorer is never empty.
  const booted = useRef(false);
  const demos = useApi(api.demos);
  useEffect(() => {
    if (booted.current || !demos.data?.length) return;
    booted.current = true;
    const first = demos.data.find((d) => d.id === "demo:03_fleet_60_drives.xlsx") ?? demos.data[0];
    void run({ kind: "demo", d: first }, ["xgb"]);
  }, [demos.data, run]);

  const toggleCnn = () => {
    const next: EstimatorKind[] = kinds.includes("cnn") ? ["xgb"] : ["xgb", "cnn"];
    setKinds(next);
    if (last) void run(last, next);
  };

  const pick = (r: DriveRow) => {
    setSelected(r.serial);
    if (window.matchMedia("(max-width: 1023px)").matches) profileRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const p = result;
  const q = p?.quality?.[p.primary];

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[300px_1fr]">
      <aside className="panel min-w-0 lg:max-h-[1100px] lg:overflow-y-auto">
        <DatasetPicker demos={demos} selected={datasetId} busy={!!pending} onPick={(d) => run({ kind: "demo", d }, kinds)} onFile={(f) => run({ kind: "file", f }, kinds)} />
      </aside>

      <div className="panel min-w-0">
        {/* toolbar */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b hairline pb-5">
          <div className="min-w-0">
            <p className="flex items-center gap-2 truncate text-sm text-soft">
              <FileSpreadsheet size={14} className="shrink-0 text-muted" aria-hidden />
              {pending ? pending.label : (p?.source ?? "No file scored yet")}
            </p>
            {p && !pending && (
              <p className="num mt-1 text-xs text-muted">
                {fmtInt(p.diagnostics.drives_scored)} drives scored in {p.seconds}s · {p.summary[p.primary]?.inspect ?? 0} to inspect ·{" "}
                {p.diagnostics.date_min} → {p.diagnostics.date_max}
              </p>
            )}
          </div>
          <label className="flex cursor-pointer items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={kinds.includes("cnn")} onChange={toggleCnn} disabled={!!pending} className="accent-[#8b6cf0]" />
            Also run the CNN side by side
          </label>
        </div>

        <AnimatePresence>
          {pending && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mb-6">
              <Loading label={pending.progress !== null && pending.progress < 1 ? `Uploading ${Math.round(pending.progress * 100)}%` : "Scoring with the live engine"} />
              {pending.progress !== null && (
                <div className="mt-3 h-0.5 rounded bg-deep">
                  <div className="h-0.5 rounded bg-signal transition-all" style={{ width: `${pending.progress * 100}%` }} />
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <div className="mb-6 flex items-start gap-3">
            <ErrorState message={error} className="flex-1" />
            <button type="button" aria-label="Dismiss" onClick={() => setError(null)} className="mt-3 text-muted hover:text-ink">
              <X size={16} />
            </button>
          </div>
        )}

        {p && (
          <>
            {(p.diagnostics.attributes_missing.length > 0 || p.diagnostics.drives_skipped > 0 || p.diagnostics.drives_padded > 0 || p.diagnostics.notes.length > 0) && (
              <div className="mb-6 flex gap-3 rounded-xl border border-warn/25 bg-warn/5 p-4 text-xs text-soft">
                <Info size={14} className="mt-0.5 shrink-0 text-warn" aria-hidden />
                <div className="space-y-1">
                  {p.diagnostics.attributes_missing.length > 0 && (
                    <p>
                      {p.diagnostics.attributes_missing.length} of 15 attributes were missing and filled with the training median:{" "}
                      {p.diagnostics.attributes_missing.map(smartName).join(", ")}.
                    </p>
                  )}
                  {p.diagnostics.drives_padded > 0 && (
                    <p>
                      {plural(p.diagnostics.drives_padded, "drive")} had less than {p.window_days} days and {p.diagnostics.drives_padded === 1 ? "was" : "were"}{" "}
                      padded; {p.diagnostics.drives_padded === 1 ? "its score is an extrapolation" : "their scores are extrapolations"}.
                    </p>
                  )}
                  {p.diagnostics.drives_skipped > 0 && (
                    <p>
                      {plural(p.diagnostics.drives_skipped, "drive")} had too little history to score and {p.diagnostics.drives_skipped === 1 ? "was" : "were"} skipped.
                    </p>
                  )}
                  {p.diagnostics.notes.map((n) => (
                    <p key={n}>{n}.</p>
                  ))}
                </div>
              </div>
            )}

            {q && (
              <div className="mb-8 rounded-xl border hairline bg-night/60 p-5">
                <p className="text-sm text-soft">
                  This file records which drives later failed ({q.n_failed} of {q.n_drives}). Ranked by the model:{" "}
                  {q.at_k.map((a, i) => (
                    <span key={a.k} className="num text-ink">
                      {i ? " · " : ""}
                      {a.hits}/{a.k}
                    </span>
                  ))}{" "}
                  of the top-K were real failures.
                </p>
                <p className="mt-2 text-xs text-muted">
                  The model never sees that column. It describes this one file only and is not an evaluation; the Results section is.
                </p>
              </div>
            )}

            <div className="grid grid-cols-1 gap-10 xl:grid-cols-[minmax(0,320px)_1fr]">
              <Queue key={p.token} p={p} selected={selected} onSelect={pick} />
              <div ref={profileRef} className="min-w-0 scroll-mt-20">
                {selected ? <Profile key={`${p.token}-${selected}`} p={p} serial={selected} /> : <p className="text-sm text-muted">Select a drive.</p>}
              </div>
            </div>
          </>
        )}

        <TechnicalDisclosure title="What happens when a file is scored" className="mt-8">
          <p>
            The browser sends the file to the Node API, which streams it to the existing Flask engine (<code>webapp/app.py</code>).
            Column names are matched to the model's contract, each drive's history is laid on a daily calendar, the scaler
            saved at training time is replayed, and each drive's most recent 30-day window is scored by the saved models in{" "}
            <code>models/</code>. Nothing is written to disk and nothing leaves the machine.
          </p>
          <p>
            Risk categories: <b>inspect</b> is the model's own call (score at or above its tuned threshold). The page splits
            “inspect” at 0.5 into <b>elevated</b> and <b>high</b> so a 0.25 and a 0.99 do not read alike; that split is
            presentation, not part of the model.
          </p>
        </TechnicalDisclosure>
      </div>
    </div>
  );
}
