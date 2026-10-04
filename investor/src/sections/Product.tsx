import { Upload } from "lucide-react";
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
          Try it on your own data.
        </>
      }
      lede="Upload your own drive telemetry, or try one of the samples. Every drive is scored live by the saved model, ranked by risk and explained."
      action={
        <a href="#upload" className="btn btn-primary">
          <Upload size={15} aria-hidden /> Upload your file
        </a>
      }
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
