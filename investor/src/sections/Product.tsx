import { Upload } from "lucide-react";
import { Expandable } from "../components/Expandable";
import { StorySection } from "../components/StorySection";
import { useStoryData } from "../lib/data";
import { DriveExplorer } from "./DriveExplorer";
import { ModelComparisonView } from "./ModelComparison";

/** 06 — The Product. Artifacts F and G. */
export default function Product() {
  const e = useStoryData().evaluation.data;
  const xgb = e?.models.find((m) => m.model === "XGBoost");
  const cnn = e?.models.find((m) => m.model === "CNN");
  return (
    <StorySection
      id="product"
      kicker="The main feature · live demo"
      featured
      title={
        <>
          Try it on <span className="text-signal">your own data</span>.
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
      <Expandable
        className="mt-6"
        title="Why XGBoost, and not a neural network?"
        hint={
          xgb && cnn
            ? `We also built a neural network (CNN) on the same data. XGBoost found more real failures at the top of the list (${xgb.hitsAt100} vs ${cnn.hitsAt100} of 100) and explains every score, so it runs by default.`
            : "We also built a neural network on the same data. XGBoost ranked better and explains every score, so it runs by default."
        }
      >
        <ModelComparisonView />
      </Expandable>
    </StorySection>
  );
}
