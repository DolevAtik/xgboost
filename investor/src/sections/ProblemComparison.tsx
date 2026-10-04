import { useEffect, useRef, useState } from "react";
import { animate, motion, useInView, useReducedMotion } from "framer-motion";
import { AlertOctagon, BellRing, RotateCcw, ShieldCheck, Wrench } from "lucide-react";
import { StorySection } from "../components/StorySection";
import { useStoryData } from "../lib/data";
import { fmtInt } from "../lib/format";

/** Points on the shared timeline, as fractions of its width. */
const FAIL_AT = 0.78;
const ALERT_AT = 0.5;

function Track({
  progress,
  tone,
  children,
}: {
  progress: number;
  tone: "risk" | "signal";
  children: React.ReactNode;
}) {
  return (
    <div className="relative h-24">
      <div className="absolute inset-x-0 top-1/2 h-px bg-line-strong" />
      <div
        className="absolute left-0 top-1/2 h-px"
        style={{ width: `${progress * 100}%`, background: tone === "risk" ? "var(--color-faint)" : "var(--color-signal)" }}
      />
      <div
        className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{ left: `${progress * 100}%`, background: tone === "risk" ? "var(--color-soft)" : "var(--color-signal)" }}
      />
      {children}
    </div>
  );
}

function Marker({
  at,
  show,
  icon,
  label,
  tone,
  below = false,
}: {
  at: number;
  show: boolean;
  icon: React.ReactNode;
  label: string;
  tone: string;
  below?: boolean;
}) {
  return (
    <motion.div
      initial={false}
      animate={{ opacity: show ? 1 : 0.0, scale: show ? 1 : 0.9 }}
      transition={{ duration: 0.4 }}
      className={`absolute flex -translate-x-1/2 flex-col items-center gap-2 ${below ? "top-[calc(50%+12px)]" : "bottom-[calc(50%+12px)]"}`}
      style={{ left: `${at * 100}%` }}
    >
      {below && <span className="h-3 w-px" style={{ background: tone }} />}
      <span className="flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-xs" style={{ borderColor: tone, color: tone }}>
        {icon}
        {label}
      </span>
      {!below && <span className="h-3 w-px" style={{ background: tone }} />}
    </motion.div>
  );
}

/** A schematic risk curve: flat, then climbing toward the failure point. Illustrative. */
function RiskSketch({ progress }: { progress: number }) {
  const pts = Array.from({ length: 60 }, (_, i) => {
    const x = (i / 59) * FAIL_AT;
    const y = 0.08 + 0.85 * Math.pow(Math.max(0, (x - 0.15) / (FAIL_AT - 0.15)), 1.6);
    return [x, y] as const;
  });
  const visible = pts.filter(([x]) => x <= progress);
  const d = visible.map(([x, y], i) => `${i ? "L" : "M"}${(x * 1000).toFixed(1)},${(100 - y * 100).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 1000 100" preserveAspectRatio="none" className="absolute inset-x-0 top-0 h-24 w-full overflow-visible" aria-hidden>
      <line x1="0" x2="1000" y1={100 - 0.41 * 100} y2={100 - 0.41 * 100} stroke="var(--color-warn)" strokeDasharray="4 6" strokeOpacity="0.5" vectorEffect="non-scaling-stroke" />
      <path d={d} fill="none" stroke="var(--color-signal)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/** Artifact B: the same drive, the same failure, two different operating models. */
export function ProblemVisual() {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: false, margin: "-25% 0px" });
  const reduce = useReducedMotion();
  const [p, setP] = useState(reduce ? 1 : 0);
  const [run, setRun] = useState(0);

  useEffect(() => {
    if (reduce) return setP(1);
    if (!inView) return;
    setP(0);
    const c = animate(0, 1, { duration: 6, ease: "linear", onUpdate: setP });
    return () => c.stop();
  }, [inView, run, reduce]);

  const failed = p >= FAIL_AT;
  const alerted = p >= ALERT_AT;

  return (
    <div ref={ref} className="grid gap-6 lg:grid-cols-2">
      {/* Reactive */}
      <div className="panel">
        <div className="flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h3 className="panel-title">Reactive maintenance</h3>
            <p className="panel-sub">What most fleets do today</p>
          </div>
          <span className="rounded-full border border-risk/40 px-2.5 py-0.5 text-[11px] text-risk">After failure</span>
        </div>
        <div className="relative pt-24">
          <p className="absolute left-0 top-8 font-mono text-[10px] text-faint">no early-warning signal is watched</p>
        <Track progress={Math.min(p, failed ? FAIL_AT : p)} tone="risk">
          <Marker at={FAIL_AT} show={failed} icon={<AlertOctagon size={12} />} label="Drive fails" tone="var(--color-risk)" />
          <Marker at={Math.min(0.97, FAIL_AT + 0.08)} show={p > FAIL_AT + 0.04} icon={<BellRing size={12} />} label="Alert" tone="var(--color-risk)" below />
          <motion.div
            aria-hidden
            className="absolute top-1/2 h-8 -translate-y-1/2 rounded bg-risk/15"
            style={{ left: `${FAIL_AT * 100}%`, width: `${Math.max(0, Math.min(p, 1) - FAIL_AT) * 100}%` }}
          />
        </Track>
        </div>
        <motion.p initial={false} animate={{ opacity: failed ? 1 : 0.35 }} className="mt-6 text-sm leading-relaxed text-soft">
          Nothing looks wrong until the drive stops. The alert arrives <b className="font-medium text-risk">after</b> the
          failure — data is at risk, the service degrades, and the replacement is unplanned.
        </motion.p>
      </div>

      {/* Predictive */}
      <div className="panel border-signal/25">
        <div className="mb-[-1rem] flex flex-wrap items-baseline justify-between gap-3">
          <div>
            <h3 className="panel-title">Predictive maintenance</h3>
            <p className="panel-sub">What this system enables</p>
          </div>
          <span className="rounded-full border border-signal/50 px-2.5 py-0.5 text-[11px] text-signal">Before failure</span>
        </div>
        <div className="relative pt-28">
          <span className="absolute right-0 top-[2.6rem] font-mono text-[10px] text-warn/70">inspect threshold</span>
          <RiskSketch progress={Math.min(p, ALERT_AT + 0.02)} />
          <Track progress={Math.min(p, FAIL_AT)} tone="signal">
            <Marker at={ALERT_AT} show={alerted} icon={<Wrench size={12} />} label="Inspect this drive" tone="var(--color-warn)" />
            <Marker at={FAIL_AT} show={failed} icon={<ShieldCheck size={12} />} label="Replaced in time" tone="var(--color-ok)" below />
          </Track>
        </div>
        <motion.p initial={false} animate={{ opacity: alerted ? 1 : 0.35 }} className="mt-6 text-sm leading-relaxed text-soft">
          Daily telemetry is read continuously. As the warning signs build, the drive's risk climbs, and an inspection
          is scheduled <b className="font-medium text-signal">before</b> anything breaks.
        </motion.p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4 lg:col-span-2">
        <p className="font-mono text-[11px] text-faint">Schematic. The measured version of this curve is in the Results section.</p>
        <button type="button" onClick={() => setRun((r) => r + 1)} className="chip hover:border-signal hover:text-ink">
          <RotateCcw size={12} aria-hidden /> Replay
        </button>
      </div>
    </div>
  );
}

/** 02 — The Hidden Problem. */
export function ProblemComparison() {
  const { data } = useStoryData().evaluation;
  // Drives with a recorded failure across every split, over the quarters the data spans.
  const failures = data?.scale.splits.reduce((a, s) => a + s.drivesWithFailure, 0);
  const perDay = data && failures ? failures / (data.scale.quarters.length * 91.3) : null;
  return (
    <StorySection
      id="problem"
      kicker="The problem"
      title={
        <>
          Failures are discovered <span className="text-risk">after</span> the damage is done.
        </>
      }
      lede={
        data && failures && perDay ? (
          <>
            In the fleet we studied, {fmtInt(failures)} of {fmtInt(data.scale.drives)} drives failed over{" "}
            {(data.scale.quarters.length / 4).toFixed(1)} years — about {Math.floor(perDay)} every day. Most are replaced
            only once they have already stopped working.
          </>
        ) : (
          "Large storage fleets lose drives every day. Most are replaced only once they have already stopped working."
        )
      }
    >
      <ProblemVisual />
    </StorySection>
  );
}
