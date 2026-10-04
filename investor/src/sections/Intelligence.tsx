import { StorySection } from "../components/StorySection";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { DataPipeline } from "./DataPipeline";
import { IntelligenceVisualization } from "./IntelligenceVisualization";

/** 04 — The Intelligence. Artifacts C and D. */
export default function Intelligence() {
  return (
    <StorySection
      id="intelligence"
      kicker="How it works"
      title={
        <>
          From raw telemetry to a <span className="text-signal">ranked</span> risk list.
        </>
      }
      lede="A six-stage pipeline turns daily counters into one decision per drive, with a leakage audit so the model never sees the answer. Select a stage for details."
    >
      <DataPipeline />
      <div className="mt-14">
        <h3 className="text-xl font-semibold tracking-tight">Inside the engine</h3>
        <p className="mb-5 mt-1 max-w-3xl text-sm text-soft">
          Two real drives, scored live by the model the product runs. Watch one month of telemetry become one number.
        </p>
        <div className="panel">
          <IntelligenceVisualization />
        </div>
        <TechnicalDisclosure title="Why trees on summaries, and not a neural network on the raw sequence?" className="mt-6">
          <p>
            The product's default engine is <b>XGBoost</b>: gradient-boosted decision trees reading the 138 window
            summaries shown above. A tree cannot take a 30 × 46 grid directly, which is why the month is summarised first.
          </p>
          <p>
            We also trained a <b>dilated 1-D convolutional network (CNN)</b> on the exact same windows. It reads the raw
            day-by-day sequence, so it sees the order of events and not only their summary. On held-out drives it ranked
            less well, and it cannot be explained feature by feature, so XGBoost is the default. Both can be run side by
            side in the product (Interactive demo).
          </p>
        </TechnicalDisclosure>
      </div>
    </StorySection>
  );
}
