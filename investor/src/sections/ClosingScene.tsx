import { motion } from "framer-motion";
import { ArrowRight, ArrowUpRight, CalendarCheck, Cpu, ListOrdered, Plug } from "lucide-react";
import { StorySection } from "../components/StorySection";
import { FleetComparison } from "../components/FleetComparison";
import { ErrorState, Loading } from "../components/States";
import { useStoryData } from "../lib/data";
import { api } from "../lib/api";
import { useRise } from "../lib/motion";
import { useApi } from "../lib/useApi";
import Team from "./Team";

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

/** Artifact H: the same failures, discovered too late or replaced in time. */
export function FleetVision() {
  const { evaluation } = useStoryData();
  const rate = evaluation.data?.leadTime?.xgbFlaggedWithinHorizon;
  if (evaluation.loading) return <Loading label="Loading the measured catch rate" />;
  if (rate === undefined) return <ErrorState message={evaluation.error ?? "lead-time results are missing"} onRetry={evaluation.reload} />;
  return (
    <div className="mb-4">
      <FleetComparison catchRate={rate} />
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
      lede="Data centers already collect this telemetry. What has been missing is a trustworthy way to turn it into a decision before the failure. Below, two identical fleets face the same failures."
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
      <Team />
      <Close />
    </>
  );
}
