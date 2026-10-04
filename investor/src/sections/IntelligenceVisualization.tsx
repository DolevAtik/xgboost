import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useInView, useReducedMotion } from "framer-motion";
import { ChevronRight } from "lucide-react";
import type { DriveDetail, Prediction, SignalSeries } from "../../shared/types";
import { ErrorState, Loading, Source } from "../components/States";
import { api } from "../lib/api";
import { useStoryData } from "../lib/data";
import { riskCategory, toneColor } from "../lib/format";
import { EASE } from "../lib/motion";
import { featureLabel, smartName } from "../lib/smart";

type Which = "failing" | "healthy";
const DEMO_ID: Record<Which, string> = { failing: "demo:02_failing_drive.csv", healthy: "demo:01_healthy_drive.csv" };

interface Scored {
  series: SignalSeries;
  prediction: Prediction;
  detail: DriveDetail;
}

/** Score both demo drives with the live engine once, and keep them. */
function useScoredDrives() {
  const [state, setState] = useState<{ data: Partial<Record<Which, Scored>>; error: string | null }>({ data: {}, error: null });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const sig = await api.signal();
        const out: Partial<Record<Which, Scored>> = {};
        for (const w of ["failing", "healthy"] as Which[]) {
          const prediction = await api.predictDemo(DEMO_ID[w], ["xgb"]);
          const row = prediction.rows[0];
          const detail = await api.drive(prediction.token, row.serial);
          out[w] = { series: sig[w], prediction, detail };
        }
        if (live) setState({ data: out, error: null });
      } catch (e) {
        if (live) setState({ data: {}, error: (e as Error).message });
      }
    })();
    return () => {
      live = false;
    };
  }, [tick]);
  return { ...state, retry: () => setTick((t) => t + 1) };
}

const STEPS = [
  { title: "30 days of history", text: "The engine takes the drive's last 30 daily reports: fifteen counters per day." },
  { title: "Summarise the month", text: "Each counter, and how fast it is changing, is reduced to three numbers: where it ended, its average, and its change." },
  { title: "Ask the trees", text: "Hundreds of small decision trees each look at those numbers and nudge the risk up or down." },
  { title: "One risk score", text: "The nudges add up to one risk score between 0 and 1: how likely is this drive to fail in the next 30 days?" },
];

/**
 * Scale each attribute row to its own 30-day range, so what lights up is change. A
 * counter that never moved stays dim however large its value is.
 */
function heat(series: SignalSeries, attrs: string[]) {
  return attrs.map((a) => {
    const v = series.series[a] ?? [];
    const nums = v.filter((x): x is number => x !== null);
    const lo = Math.min(...nums);
    const hi = Math.max(...nums);
    return v.map((x) => (x === null ? null : hi - lo < 1e-9 ? 0 : (x - lo) / (hi - lo)));
  });
}

/** Artifact D: the inference path that is actually served — window, features, trees, score. */
export function IntelligenceVisualization() {
  const { data, error, retry } = useScoredDrives();
  const { model, evaluation } = useStoryData();
  const [which, setWhich] = useState<Which>("failing");
  const [step, setStep] = useState(0);
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { margin: "-20% 0px" });
  const reduce = useReducedMotion();
  const [auto, setAuto] = useState(true);

  useEffect(() => {
    if (!inView || !auto || reduce) return;
    const id = setInterval(() => setStep((s) => (s + 1) % STEPS.length), 3800);
    return () => clearInterval(id);
  }, [inView, auto, reduce]);

  const run = model.data?.run;
  const attrs = run?.attributes ?? [];
  const scored = data[which];
  const grid = useMemo(() => (scored ? heat(scored.series, attrs) : []), [scored, attrs]);
  const nTrees = evaluation.data?.xgbTrees ?? 470;

  if (error) return <ErrorState message={error} onRetry={retry} />;
  if (!scored || !run) return <Loading label="Scoring two real drives with the live engine" />;

  const risk = scored.prediction.rows[0].risk;
  const threshold = scored.prediction.thresholds.xgb ?? 0.5;
  const cat = riskCategory(risk, threshold);
  const contribs = scored.detail.explanations.xgb ?? [];
  const pick = (i: number) => {
    setAuto(false);
    setStep(i);
  };

  return (
    <div ref={ref}>
      <div className="mb-8 flex flex-wrap items-center justify-between gap-4 border-b hairline pb-5">
        <div role="radiogroup" aria-label="Drive" className="inline-flex rounded-full border border-line p-1">
          {(["failing", "healthy"] as Which[]).map((w) => (
            <button
              key={w}
              role="radio"
              aria-checked={which === w}
              onClick={() => setWhich(w)}
              className={`rounded-full px-4 py-1.5 text-sm transition ${which === w ? "bg-ink text-void" : "text-muted hover:text-ink"}`}
            >
              {w === "failing" ? "Drive in its final month" : "Healthy drive"}
            </button>
          ))}
        </div>
        <span className="num text-xs text-faint">
          {scored.detail.drive_model} · {scored.detail.serial}
        </span>
      </div>

      <div className="grid gap-8 lg:grid-cols-[260px_1fr] lg:gap-12">
        {/* steps */}
        <ol className="space-y-1">
          {STEPS.map((s, i) => (
            <li key={s.title}>
              <button
                type="button"
                onClick={() => pick(i)}
                aria-current={i === step ? "step" : undefined}
                className={`relative w-full rounded-lg px-3.5 py-2.5 text-left text-sm transition ${i === step ? "bg-deep/70" : "hover:bg-deep/40"}`}
              >
                <span className="flex items-center gap-3">
                  <span className={`num text-xs ${i === step ? "text-violet" : "text-faint"}`}>0{i + 1}</span>
                  <span className={i === step ? "text-ink" : "text-muted"}>{s.title}</span>
                  {i === step && <ChevronRight size={14} className="ml-auto text-violet" aria-hidden />}
                </span>
                {i === step && auto && !reduce && (
                  <motion.span
                    key={`${step}-bar`}
                    className="absolute bottom-0 left-4 h-px bg-violet"
                    initial={{ width: 0 }}
                    animate={{ width: "calc(100% - 2rem)" }}
                    transition={{ duration: 3.8, ease: "linear" }}
                  />
                )}
              </button>
            </li>
          ))}
        </ol>

        {/* stage */}
        <div className="min-h-[380px] min-w-0" aria-live="polite">
          <p className="mb-6 max-w-xl text-sm text-soft">{STEPS[step].text}</p>
          <AnimatePresence mode="wait">
            <motion.div
              key={`${step}-${which}`}
              initial={reduce ? false : { opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduce ? undefined : { opacity: 0 }}
              transition={{ duration: 0.5, ease: EASE }}
            >
              {step === 0 && (
                <div className="overflow-x-auto">
                  <div className="min-w-[520px] space-y-[3px]">
                    {grid.map((row, r) => (
                      <div key={attrs[r]} className="flex items-center gap-3">
                        <span className="w-40 shrink-0 truncate text-right text-[11px] text-muted">{smartName(attrs[r])}</span>
                        <div className="grid flex-1 gap-[2px]" style={{ gridTemplateColumns: `repeat(${row.length}, minmax(0,1fr))` }}>
                          {row.map((v, d) => (
                            <motion.span
                              key={d}
                              initial={reduce ? false : { opacity: 0 }}
                              animate={{ opacity: 1 }}
                              transition={{ delay: reduce ? 0 : d * 0.015 }}
                              className="h-3 rounded-[2px]"
                              style={{
                                background:
                                  v === null ? "transparent" : `color-mix(in oklab, var(--color-signal) ${Math.round(8 + v * 82)}%, #0a1122)`,
                                outline: v === null ? "1px dashed var(--color-line)" : undefined,
                              }}
                              title={`${smartName(attrs[r])}, day ${d + 1}`}
                            />
                          ))}
                        </div>
                      </div>
                    ))}
                    <p className="pl-[172px] pt-2 font-mono text-[10px] text-faint">day 1 → day 30 · brighter = higher than that counter's own 30-day low · dim rows did not change</p>
                  </div>
                </div>
              )}

              {step === 1 && (
                <div className="grid items-center gap-6 sm:grid-cols-[1fr_auto_1fr]">
                  <div className="space-y-2 text-sm">
                    <p className="num text-5xl text-ink">{attrs.length}</p>
                    <p className="text-muted">counters × 30 days</p>
                    <p className="pt-4 text-muted">
                      + 1-day and 7-day change of each, + how many days were reported = <b className="num text-ink">{run.n_channels}</b> signals
                    </p>
                  </div>
                  <ChevronRight className="hidden text-violet sm:block" aria-hidden />
                  <div className="space-y-3">
                    {["latest", "30-day average", "change over the window"].map((a, i) => (
                      <motion.div
                        key={a}
                        initial={reduce ? false : { opacity: 0, x: 16 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: reduce ? 0 : 0.15 * i }}
                        className="flex items-center justify-between rounded-lg border border-violet/30 bg-violet/5 px-4 py-3"
                      >
                        <span className="text-sm text-soft">{a}</span>
                        <span className="num text-sm text-violet">× {run.n_channels}</span>
                      </motion.div>
                    ))}
                    <p className="pt-2 text-right text-sm text-muted">
                      = <b className="num text-2xl text-ink">{run.n_channels * 3}</b> numbers describe this drive's month
                    </p>
                  </div>
                </div>
              )}

              {step === 2 && (
                <div>
                  <div className="grid gap-[3px]" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(10px, 1fr))" }} aria-hidden>
                    {Array.from({ length: nTrees }, (_, i) => (
                      <motion.span
                        key={i}
                        initial={reduce ? false : { opacity: 0.1 }}
                        animate={{ opacity: [0.15, 1, 0.55] }}
                        transition={{ delay: reduce ? 0 : (i % 47) * 0.02 + Math.floor(i / 47) * 0.05, duration: 0.6 }}
                        className="aspect-square rounded-[2px]"
                        style={{ background: risk >= threshold ? "var(--color-warn)" : "var(--color-electric)" }}
                      />
                    ))}
                  </div>
                  <p className="mt-5 text-sm text-muted">
                    <b className="num text-ink">{nTrees}</b> trees, each a short chain of yes/no questions such as “has the number of
                    pending sectors grown this week?”. No single tree decides; their small votes are added up.
                  </p>
                </div>
              )}

              {step === 3 && (
                <div className="grid gap-10 md:grid-cols-[auto_1fr] md:items-center">
                  <div>
                    <p className="num text-[clamp(3.25rem,7vw,5rem)] leading-none" style={{ color: toneColor[cat.tone] }}>
                      {risk.toFixed(3)}
                    </p>
                    <p className="mt-3 text-sm text-soft">
                      risk score for failure within {run.horizon_days} days ·{" "}
                      <b className="font-medium" style={{ color: toneColor[cat.tone] }}>
                        {cat.call}
                      </b>
                    </p>
                  </div>
                  <div>
                    <p className="kicker mb-3 text-[10px]">What pushed the score {risk >= threshold ? "up" : "down"} most</p>
                    <ul className="space-y-2">
                      {contribs.slice(0, 4).map((c) => {
                        const f = featureLabel(c.feature);
                        return (
                          <li key={c.feature} className="flex items-baseline justify-between gap-4 border-b hairline pb-2 text-sm">
                            <span className="min-w-0">
                              <span className="text-soft">{f.title}</span> <span className="text-faint">· {f.detail}</span>
                            </span>
                            <span className="num shrink-0" style={{ color: c.contribution > 0 ? "var(--color-warn)" : "var(--color-ok)" }}>
                              {c.contribution > 0 ? "▲ raises" : "▼ lowers"}
                            </span>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
      <Source>
        live scores from the saved model models/backblaze_2022x_w30_xgb.json for demo/01_healthy_drive.csv and
        demo/02_failing_drive.csv; explanations are exact per-prediction SHAP values.
      </Source>
    </div>
  );
}
