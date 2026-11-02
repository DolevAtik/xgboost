import { ArrowDown, Play, Upload } from "lucide-react";
import { useLang } from "../lib/i18n";

/**
 * The live demo is the last chapter; this points to it while the reader is still
 * looking at the results it proves.
 */
export function DemoTeaser() {
  const { t } = useLang();
  return (
    <div className="mx-auto max-w-7xl px-4 pb-4 sm:px-6 lg:px-8">
      <div
        className="flex flex-wrap items-center justify-between gap-5 rounded-2xl border border-signal/40 px-6 py-5 md:px-8"
        style={{ background: "linear-gradient(90deg, rgba(63,224,255,0.10), rgba(63,224,255,0.02))" }}
      >
        <div className="flex items-center gap-4">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-signal/15 text-signal">
            <Play size={18} aria-hidden className="rtl:rotate-180" />
          </span>
          <div>
            <p className="text-base font-semibold text-ink">{t("Seen the numbers? Now run it yourself.", "ראיתם את המספרים? עכשיו נסו בעצמכם.")}</p>
            <p className="text-sm text-muted">
              {t(
                "Upload your own drive telemetry or pick a sample, and get a ranked risk list in seconds.",
                "העלו נתונים של הכוננים שלכם או בחרו דוגמה, וקבלו רשימת סיכון מדורגת תוך שניות.",
              )}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          <a href="#upload" className="btn btn-primary">
            <Upload size={15} aria-hidden /> {t("Upload your file", "העלאת קובץ")}
          </a>
          <a href="#product" className="btn btn-ghost">
            {t("Go to the live demo", "להדגמה החיה")} <ArrowDown size={15} aria-hidden />
          </a>
        </div>
      </div>
    </div>
  );
}
