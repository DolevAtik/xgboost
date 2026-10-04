import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Activity, AlertOctagon, BellRing, HardDrive, Pause, Play, RotateCcw, ShieldCheck, Wrench } from "lucide-react";
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

const DAYS = 60;
const DURATION_S = 10;
const TICK_MS = 50;

/** The schematic risk curve's value at a point on the timeline. */
const riskAt = (x: number) => 0.08 + 0.85 * Math.pow(Math.max(0, (Math.min(x, FAIL_AT) - 0.15) / (FAIL_AT - 0.15)), 1.6);

type Phase = "idle" | "running" | "paused" | "done";

function Status({ tone, children }: { tone: string; children: React.ReactNode }) {
  return (
    <p
      className="mt-4 flex min-h-[2.25rem] flex-wrap items-center gap-2 rounded-lg border px-3 py-2 text-xs"
      style={{ borderColor: `color-mix(in oklab, ${tone} 35%, transparent)`, color: tone }}
    >
      {children}
    </p>
  );
}

/** Artifact B: the same drive, the same failure, two operating models, played day by day. */
export function ProblemVisual() {
  const [p, setP] = useState(0);
  const [phase, setPhase] = useState<Phase>("idle");
  const pRef = useRef(0);

  // Driven by a timer rather than a motion library, so it plays under reduced motion
  // too: it changes state (a day counter, a status line), it is not decoration.
  useEffect(() => {
    if (phase !== "running") return;
    const id = setInterval(() => {
      const next = Math.min(1, pRef.current + TICK_MS / 1000 / DURATION_S);
      pRef.current = next;
      setP(next);
      if (next >= 1) setPhase("done");
    }, TICK_MS);
    return () => clearInterval(id);
  }, [phase]);

  const restart = () => {
    pRef.current = 0;
    setP(0);
    setPhase("running");
  };
  const start = () => (phase === "paused" ? setPhase("running") : restart());

  const failed = p >= FAIL_AT;
  const alerted = p >= ALERT_AT;
  const day = Math.round(p * DAYS);
  const failDay = Math.round(FAIL_AT * DAYS);
  const warnDays = Math.round((FAIL_AT - ALERT_AT) * DAYS);
  const downDays = Math.max(0, Math.round((p - FAIL_AT) * DAYS));
  const live = phase === "running";

  return (
    <div className="space-y-4">
      {/* playback strip */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3 rounded-xl border hairline bg-night/60 px-4 py-3">
        <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-ink">
          <span className="relative flex h-2.5 w-2.5">
            {live && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-risk opacity-70" />}
            <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${live ? "bg-risk" : "bg-faint"}`} />
          </span>
          {phase === "idle" ? "Ready" : live ? "Live" : phase === "paused" ? "Paused" : "Finished"}
        </span>
        <span className="num text-xs text-muted">
          day <span className="text-ink">{String(day).padStart(2, "0")}</span> / {DAYS}
        </span>
        <div className="h-1 min-w-[8rem] flex-1 overflow-hidden rounded bg-deep" aria-hidden>
          <div className="h-1 bg-gradient-to-r from-signal to-warn" style={{ width: `${p * 100}%` }} />
        </div>
        <div className="flex items-center gap-2">
          {live ? (
            <button type="button" onClick={() => setPhase("paused")} className="chip hover:border-signal hover:text-ink">
              <Pause size={12} aria-hidden /> Pause
            </button>
          ) : (
            <button type="button" onClick={start} className="btn btn-primary !px-4 !py-1.5 !text-[13px]">
              {phase === "done" ? <RotateCcw size={13} aria-hidden /> : <Play size={13} aria-hidden />}
              {phase === "idle" ? "Start" : phase === "paused" ? "Resume" : "Replay"}
            </button>
          )}
          {(live || phase === "paused") && (
            <button type="button" onClick={restart} className="chip hover:border-signal hover:text-ink">
              <RotateCcw size={12} aria-hidden /> Restart
            </button>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Reactive */}
        <div className="panel">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <div>
              <h3 className="panel-title">Reactive maintenance</h3>
              <p className="panel-sub">What most fleets do today</p>
            </div>
            <span className="rounded-full border border-risk/40 px-2.5 py-0.5 text-[11px] text-risk">After failure</span>
          </div>
          {!failed ? (
            <Status tone="var(--color-muted)">
              <HardDrive size={13} aria-hidden />
              {phase === "idle" ? "Press Start to follow one drive through its last 60 days." : "Running normally · no warning signs are watched"}
            </Status>
          ) : p < FAIL_AT + 0.04 ? (
            <Status tone="var(--color-risk)">
              <AlertOctagon size={13} aria-hidden /> Day {failDay}: the drive failed without warning
            </Status>
          ) : (
            <Status tone="var(--color-risk)">
              <BellRing size={13} aria-hidden /> Outage · the alert came after the failure · down {downDays} day{downDays === 1 ? "" : "s"}
            </Status>
          )}
          <div className="relative pt-16">
            <p className="absolute left-0 top-6 font-mono text-[10px] text-faint">no early-warning signal is watched</p>
            <Track progress={Math.min(p, FAIL_AT)} tone="risk">
              <Marker at={FAIL_AT} show={failed} icon={<AlertOctagon size={12} />} label="Drive fails" tone="var(--color-risk)" />
              <Marker
                at={Math.min(0.97, FAIL_AT + 0.08)}
                show={p > FAIL_AT + 0.04}
                icon={<BellRing size={12} />}
                label="Alert"
                tone="var(--color-risk)"
                below
              />
              <div
                aria-hidden
                className="absolute top-1/2 h-8 -translate-y-1/2 rounded bg-risk/15"
                style={{ left: `${FAIL_AT * 100}%`, width: `${Math.max(0, Math.min(p, 1) - FAIL_AT) * 100}%` }}
              />
            </Track>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-soft">
            Nothing looks wrong until the drive stops. The alert arrives <b className="font-medium text-risk">after</b> the failure — data is at risk,
            the service degrades, and the replacement is unplanned.
          </p>
        </div>

        {/* Predictive */}
        <div className="panel border-signal/25">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <div>
              <h3 className="panel-title">Predictive maintenance</h3>
              <p className="panel-sub">What this system enables</p>
            </div>
            <span className="rounded-full border border-signal/50 px-2.5 py-0.5 text-[11px] text-signal">Before failure</span>
          </div>
          {!alerted ? (
            <Status tone="var(--color-signal)">
              <Activity size={13} aria-hidden />
              {phase === "idle" ? (
                "Telemetry is read and scored every day."
              ) : (
                <>
                  Monitoring · risk score <span className="num">{riskAt(p).toFixed(2)}</span>
                </>
              )}
            </Status>
          ) : !failed ? (
            <Status tone="var(--color-warn)">
              <Wrench size={13} aria-hidden /> Flagged {warnDays} days before failure · risk <span className="num">{riskAt(p).toFixed(2)}</span> ·
              inspection scheduled
            </Status>
          ) : (
            <Status tone="var(--color-ok)">
              <ShieldCheck size={13} aria-hidden /> Replaced in a planned window · no outage
            </Status>
          )}
          <div className="relative pt-24">
            <span className="absolute right-0 top-[1.6rem] font-mono text-[10px] text-warn/70">inspect threshold</span>
            <RiskSketch progress={Math.min(p, FAIL_AT)} />
            <Track progress={Math.min(p, FAIL_AT)} tone="signal">
              <Marker at={ALERT_AT} show={alerted} icon={<Wrench size={12} />} label="Inspect this drive" tone="var(--color-warn)" />
              <Marker at={FAIL_AT} show={failed} icon={<ShieldCheck size={12} />} label="Replaced in time" tone="var(--color-ok)" below />
            </Track>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-soft">
            Daily telemetry is read continuously. As the warning signs build, the drive's risk climbs, and an inspection is scheduled{" "}
            <b className="font-medium text-signal">before</b> anything breaks.
          </p>
        </div>
      </div>

      <p className="font-mono text-[11px] text-faint">Schematic: one illustrative drive, not data. The measured version of this curve is in the Results section.</p>
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
