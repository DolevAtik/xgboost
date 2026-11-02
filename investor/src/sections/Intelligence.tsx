import { StorySection } from "../components/StorySection";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { Expandable } from "../components/Expandable";
import { useLang } from "../lib/i18n";
import { DataPipeline } from "./DataPipeline";
import { IntelligenceVisualization } from "./IntelligenceVisualization";

/** 04 — The Intelligence. Artifacts C and D. */
export default function Intelligence() {
  const { t, he } = useLang();
  return (
    <StorySection
      id="intelligence"
      kicker={t("How it works", "איך זה עובד")}
      title={
        he ? (
          <>
            מנתונים גולמיים לרשימת סיכון <span className="text-signal">מדורגת</span>.
          </>
        ) : (
          <>
            From raw telemetry to a <span className="text-signal">ranked</span> risk list.
          </>
        )
      }
      lede={t(
        "A six-stage pipeline turns daily counters into one decision per drive, with a leakage audit so the model never sees the answer. Select a stage for details.",
        "תהליך בן שישה שלבים הופך מונים יומיים להחלטה אחת לכל כונן, כולל בדיקת דליפות שמוודאת שהמודל אף פעם לא רואה את התשובה. בחרו שלב לפרטים.",
      )}
    >
      <DataPipeline />
      <Expandable
        className="mt-6"
        title={t("Inside the engine", "מבט אל תוך המנוע")}
        hint={t(
          "Watch two real drives scored live: one month of telemetry becoming one risk score.",
          "צפו בשני כוננים אמיתיים שמקבלים ציון בזמן אמת: חודש של נתונים הופך לציון סיכון אחד.",
        )}
      >
        <div className="panel">
          <IntelligenceVisualization />
        </div>
        <TechnicalDisclosure
          title={t("Why trees on summaries, and not a neural network on the raw sequence?", "למה עצי החלטה על סיכומים, ולא רשת נוירונים על הרצף הגולמי?")}
          className="mt-6"
        >
          {he ? (
            <>
              <p>
                מנוע ברירת המחדל של המוצר הוא <b>XGBoost</b>: עצי החלטה שקוראים את 138 הסיכומים של החלון שמוצגים למעלה. עץ לא יכול לקרוא טבלה של 30 × 46
                ישירות, ולכן החודש מסוכם קודם.
              </p>
              <p>
                אימנו גם <b>רשת נוירונים קונבולוציונית (CNN)</b> על אותם חלונות בדיוק. היא קוראת את הרצף יום אחרי יום, ולכן רואה גם את סדר האירועים ולא רק
                את הסיכום. על כוננים שלא נראו באימון היא דירגה פחות טוב, ואי אפשר להסביר את ההחלטות שלה מאפיין אחרי מאפיין, ולכן XGBoost הוא ברירת המחדל.
              </p>
            </>
          ) : (
            <>
              <p>
                The product's default engine is <b>XGBoost</b>: gradient-boosted decision trees reading the 138 window summaries shown above. A tree
                cannot take a 30 × 46 grid directly, which is why the month is summarised first.
              </p>
              <p>
                We also trained a <b>dilated 1-D convolutional network (CNN)</b> on the exact same windows. It reads the raw day-by-day sequence, so it
                sees the order of events and not only their summary. On held-out drives it ranked less well, and it cannot be explained feature by
                feature, so XGBoost is the default.
              </p>
            </>
          )}
        </TechnicalDisclosure>
      </Expandable>
    </StorySection>
  );
}
