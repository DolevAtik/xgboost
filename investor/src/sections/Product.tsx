import { Upload } from "lucide-react";
import { Expandable } from "../components/Expandable";
import { StorySection } from "../components/StorySection";
import { useStoryData } from "../lib/data";
import { useLang } from "../lib/i18n";
import { DriveExplorer } from "./DriveExplorer";
import { ModelComparisonView } from "./ModelComparison";

/** 06 — The Product. Artifacts F and G. */
export default function Product() {
  const e = useStoryData().evaluation.data;
  const xgb = e?.models.find((m) => m.model === "XGBoost");
  const cnn = e?.models.find((m) => m.model === "CNN");
  const { t, he } = useLang();
  return (
    <StorySection
      id="product"
      kicker={t("The main feature · live demo", "הפיצ׳ר המרכזי · הדגמה חיה")}
      featured
      title={
        he ? (
          <>
            נסו על <span className="text-signal">הנתונים שלכם</span>.
          </>
        ) : (
          <>
            Try it on <span className="text-signal">your own data</span>.
          </>
        )
      }
      lede={t(
        "Upload your own drive telemetry, or try one of the samples. Every drive is scored live by the saved model, ranked by risk and explained.",
        "העלו נתונים של הכוננים שלכם, או נסו אחת מהדוגמאות. כל כונן מקבל ציון בזמן אמת מהמודל השמור, מדורג לפי סיכון, ומוסבר.",
      )}
      action={
        <a href="#upload" className="btn btn-primary">
          <Upload size={15} aria-hidden /> {t("Upload your file", "העלאת קובץ")}
        </a>
      }
    >
      <DriveExplorer />
      <Expandable
        className="mt-6"
        title={t("Why XGBoost, and not a neural network?", "למה XGBoost ולא רשת נוירונים?")}
        hint={
          xgb && cnn
            ? t(
                `We also built a neural network (CNN) on the same data. XGBoost found more real failures at the top of the list (${xgb.hitsAt100} vs ${cnn.hitsAt100} of 100) and explains every score, so it runs by default.`,
                `בנינו גם רשת נוירונים (CNN) על אותם נתונים. XGBoost מצא יותר תקלות אמיתיות בראש הרשימה (${xgb.hitsAt100} לעומת ${cnn.hitsAt100} מתוך 100) ומסביר כל ציון, ולכן הוא ברירת המחדל.`,
              )
            : t(
                "We also built a neural network on the same data. XGBoost ranked better and explains every score, so it runs by default.",
                "בנינו גם רשת נוירונים על אותם נתונים. XGBoost דירג טוב יותר ומסביר כל ציון, ולכן הוא ברירת המחדל.",
              )
        }
      >
        <ModelComparisonView />
      </Expandable>
    </StorySection>
  );
}
