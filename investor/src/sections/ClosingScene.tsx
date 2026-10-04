import { motion } from "framer-motion";
import { Mail } from "lucide-react";
import { StorySection } from "../components/StorySection";
import { FleetComparison } from "../components/FleetComparison";
import { ErrorState, Loading } from "../components/States";
import { useStoryData } from "../lib/data";
import { openContact } from "../lib/contact";
import { useRise } from "../lib/motion";
import Team from "./Team";

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
    </StorySection>
  );
}

/** The close: one line, the takeaway, and the next steps. */
export function Close() {
  const rise = useRise(0, 16);
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
              <button type="button" onClick={openContact} className="btn btn-primary">
                <Mail size={15} aria-hidden /> Get in touch
              </button>
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
    </>
  );
}
