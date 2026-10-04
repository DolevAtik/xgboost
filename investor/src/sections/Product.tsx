import { StorySection } from "../components/StorySection";
import { DriveExplorer } from "./DriveExplorer";
import { ModelComparisonView } from "./ModelComparison";

/** 06 — The Product. Artifacts F and G. */
export default function Product() {
  return (
    <StorySection
      id="product"
      kicker="Interactive demo"
      title={
        <>
          Explore a real fleet.
        </>
      }
      lede="Pick a dataset or upload your own telemetry. Every drive is scored live by the saved model, ranked by risk, and explained. Select a drive to see its history, trajectory and the signals behind its score."
    >
      <DriveExplorer />
      <div id="models" className="mt-16 scroll-mt-20">
        <p className="kicker mb-3">Model comparison</p>
        <h3 className="mb-6 text-xl font-semibold tracking-tight md:text-2xl">
          Why XGBoost is the default engine.
        </h3>
        <ModelComparisonView />
      </div>
    </StorySection>
  );
}
