import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { ArrowRight, ArrowUpRight, CalendarCheck, Cpu, ListOrdered, Plug } from "lucide-react";
import { StorySection } from "../components/StorySection";
import { WhenVisible } from "../components/WhenVisible";
import type { FleetMode } from "../scenes/FleetScene";
import { api } from "../lib/api";
import { hasWebGL, useIsCompact, useRise } from "../lib/motion";
import { useApi } from "../lib/useApi";

const FleetScene = lazy(() => import("../scenes/FleetScene"));

const DIRECTIONS = [
  {
    icon: <ListOrdered size={18} />,
    title: "A daily triage list for every fleet",
    text: "Every drive scored every morning, with the few that need attention at the top.",
  },
  {
    icon: <CalendarCheck size={18} />,
    title: "Planned, not emergency, replacement",
    text: "Swap at-risk drives during scheduled maintenance instead of after an outage.",
  },
  {
    icon: <Plug size={18} />,
    title: "Fits the tools operators already run",
    text: "Telemetry already collected with smartctl feeds straight in. A daily collector is part of the project.",
  },
  {
    icon: <Cpu size={18} />,
    title: "Beyond hard drives",
    text: "The same method of self-reported health, audited for leaks and ranked by risk could extend to other equipment that monitors itself.",
  },
];

/** Artifact H: the same wear events, seen too late or seen in time. */
export function FleetVision() {
  const reduce = useReducedMotion();
  const compact = useIsCompact();
  const webgl = useMemo(hasWebGL, []);
  const [mode, setMode] = useState<FleetMode>("reactive");
  const [auto, setAuto] = useState(!reduce);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => setMode((m) => (m === "reactive" ? "predictive" : "reactive")), 7000);
    return () => clearInterval(id);
  }, [auto]);

  const choose = (m: FleetMode) => {
    setAuto(false);
    setMode(m);
  };

  return (
    <div className="relative">
      <div className="relative h-[380px] overflow-hidden rounded-2xl border hairline bg-night md:h-[460px]">
        {webgl ? (
          <WhenVisible className="absolute inset-0" unmountWhenHidden rootMargin="200px 0px">
            <Suspense fallback={null}>
              <FleetScene mode={mode} compact={compact} animate={!reduce} />
            </Suspense>
          </WhenVisible>
        ) : (
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,#0e1730,transparent_70%)]" />
        )}
        <div aria-hidden className="pointer-events-none absolute inset-0 bg-gradient-to-t from-night via-transparent to-night/40" />
        <div className="absolute inset-x-0 bottom-0 flex flex-wrap items-end justify-between gap-4 p-6 md:p-8">
          <AnimatePresence mode="wait">
            <motion.div
              key={mode}
              initial={reduce ? false : { opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduce ? undefined : { opacity: 0, y: -6 }}
              className="max-w-md"
              aria-live="polite"
            >
              <p className="text-xl font-medium md:text-2xl">{mode === "reactive" ? "Reacting" : "Anticipating"}</p>
              <p className="mt-1 text-sm text-soft">
                {mode === "reactive" ? (
                  <>
                    Drives fail (<span className="text-risk">red</span>), go dark, and are replaced after the outage.
                  </>
                ) : (
                  <>
                    The same drives are flagged early (<span className="text-warn">amber</span>), serviced (<span className="text-signal">cyan</span>),
                    and never go dark.
                  </>
                )}
              </p>
            </motion.div>
          </AnimatePresence>
          <div role="radiogroup" aria-label="Operating model" className="inline-flex rounded-full border border-line bg-void/60 p-1 backdrop-blur">
            {(["reactive", "predictive"] as FleetMode[]).map((m) => (
              <button
                key={m}
                role="radio"
                aria-checked={mode === m}
                onClick={() => choose(m)}
                className={`rounded-full px-4 py-1.5 text-sm capitalize transition ${mode === m ? "bg-ink text-void" : "text-muted hover:text-ink"}`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
      </div>
      <p className="mt-2 mb-4 font-mono text-[11px] text-faint">Illustration of the operating model. Event timing is not data.</p>
    </div>
  );
}

/** 08 — The Vision. */
export function Vision() {
  const rise = useRise();
  return (
    <StorySection
      id="vision"
      kicker="The vision"
      title={
        <>
          From reactive repair to infrastructure that <span className="text-signal">anticipates</span>.
        </>
      }
      lede="Data centers already collect this telemetry. What has been missing is a trustworthy way to turn it into a decision before the failure. Toggle between the two operating models."
    >
      <FleetVision />
      <div className="mt-4">
        <p className="kicker mb-4 text-[10px]">Where this can go · direction, not results</p>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {DIRECTIONS.map((d) => (
            <motion.div key={d.title} {...rise} className="panel">
              <span className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg border border-signal/40 bg-signal/10 text-signal">{d.icon}</span>
              <h3 className="text-[15px] font-semibold">{d.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-soft">{d.text}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </StorySection>
  );
}

/** The close: one line, the takeaway, and the next steps. */
export function Close() {
  const rise = useRise(0, 16);
  const health = useApi(api.health);
  const consoleUrl = health.data?.engine.url ?? "http://127.0.0.1:5000";
  return (
    <section id="close" data-chapter="close" aria-labelledby="close-title" className="border-t hairline py-20 md:py-24">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.div
          {...rise}
          className="relative overflow-hidden rounded-2xl border border-signal/25 p-8 md:p-12"
          style={{ background: "radial-gradient(80% 120% at 100% 0%, rgba(63,224,255,0.12), transparent 60%), linear-gradient(180deg, #0b1428, #070b14)" }}
        >
          <div className="grid items-center gap-8 lg:grid-cols-[1.3fr_1fr]">
            <div>
              <p className="kicker mb-3">Next step</p>
              <h2 id="close-title" className="text-[clamp(1.75rem,3.4vw,2.75rem)] font-semibold leading-tight tracking-tight">
                Know the failure <span className="text-signal">before</span> it happens.
              </h2>
              <p className="mt-4 max-w-2xl text-[15px] leading-relaxed text-soft">
                A system that learns the early warning signs of drive failure, and on real-world test data ranks the riskiest drives with very high
                precision. Its job is to turn maintenance from a reaction into a plan.
              </p>
            </div>
            <div className="flex flex-wrap gap-3 lg:justify-end">
              <a href="#product" className="btn btn-primary">
                Run the live demo <ArrowRight size={15} aria-hidden />
              </a>
              <a href="#breakthrough" className="btn btn-ghost">
                Review the results
              </a>
              <a href={consoleUrl} target="_blank" rel="noreferrer" className="btn btn-ghost">
                Engineering console <ArrowUpRight size={14} aria-hidden />
              </a>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

export default function ClosingScene() {
  return (
    <>
      <Vision />
      <Close />
    </>
  );
}
