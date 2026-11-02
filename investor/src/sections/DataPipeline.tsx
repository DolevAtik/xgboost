import { useState, type ReactNode } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Brain, ChevronRight, Database, Filter, Gauge, ShieldCheck, Sparkles } from "lucide-react";
import type { Evaluation, RunInfo } from "../../shared/types";
import { useStoryData } from "../lib/data";
import { fmtCompact, fmtInt } from "../lib/format";
import { useLang, type Bi, type Lang } from "../lib/i18n";
import { EASE } from "../lib/motion";

interface Stage {
  id: string;
  label: Bi;
  short: Bi;
  icon: ReactNode;
  plain: Bi;
  fact: (e: Evaluation | null, r: RunInfo | null, lang: Lang) => ReactNode;
  tech: Bi;
}

const quarter = (q: string) => q.replace("_", " ");

const STAGES: Stage[] = [
  {
    id: "raw",
    label: { en: "Raw telemetry", he: "נתונים גולמיים" },
    short: { en: "Daily SMART reports from every drive in the fleet", he: "דיווח SMART יומי מכל כונן בצי" },
    icon: <Database size={18} />,
    plain: {
      en: "Every drive reports a set of health counters once a day. We start from those reports, exactly as they were recorded.",
      he: "כל כונן מדווח פעם ביום על סדרה של מוני בריאות. אנחנו מתחילים מהדיווחים האלה, בדיוק כפי שנרשמו.",
    },
    fact: (e, _, lang) =>
      e ? (
        lang === "he" ? (
          <>
            <b>{fmtCompact(e.scale.driveDays)}</b> דיווחים יומיים מ־<b>{fmtInt(e.scale.drives)}</b> כוננים, {e.scale.quarters.length} ארכיונים רבעוניים
            ציבוריים ({quarter(e.scale.quarters[0])} – {quarter(e.scale.quarters.at(-1) ?? "")}).
          </>
        ) : (
          <>
            <b>{fmtCompact(e.scale.driveDays)}</b> daily reports from <b>{fmtInt(e.scale.drives)}</b> drives, {e.scale.quarters.length} quarterly
            public archives ({quarter(e.scale.quarters[0])} – {quarter(e.scale.quarters.at(-1) ?? "")}).
          </>
        )
      ) : null,
    tech: {
      en: "Backblaze quarterly drive-stats archives, reduced to parquet shards without extracting CSVs to disk (scripts/build_backblaze_archive.py).",
      he: "הארכיונים הרבעוניים של Backblaze, מומרים לקובצי parquet בלי לחלץ קובצי CSV לדיסק (scripts/build_backblaze_archive.py).",
    },
  },
  {
    id: "clean",
    label: { en: "Cleaning", he: "ניקוי" },
    short: { en: "Continuous calendar, gaps marked, comparable scales", he: "לוח ימים רציף, ימים חסרים מסומנים, סקאלה אחידה" },
    icon: <Filter size={18} />,
    plain: {
      en: "Each drive's history is laid on a continuous calendar. Missing days are marked as missing, never invented, and counters are put on a comparable scale.",
      he: "ההיסטוריה של כל כונן נפרסת על לוח ימים רציף. ימים חסרים מסומנים כחסרים ולא ממציאים להם ערכים, והמונים מובאים לסקאלה אחידה.",
    },
    fact: (_, r, lang) =>
      r ? (
        lang === "he" ? (
          <>
            <b>{r.sparse_dropped.length}</b> מדדים שרק חלק מהצי מדווח עליהם הוסרו, ונשארו <b>{r.attributes.length}</b> שכל כונן מדווח.
          </>
        ) : (
          <>
            <b>{r.sparse_dropped.length}</b> attributes reported by only part of the fleet were dropped, leaving <b>{r.attributes.length}</b> that every
            drive reports.
          </>
        )
      ) : null,
    tech: {
      en: "Dense daily calendar per drive, log1p on counters, an observed-mask channel for gaps, and the scaler fitted at training time is replayed at serving time (no train/serve skew).",
      he: "לוח יומי רציף לכל כונן, log1p על המונים, ערוץ שמסמן ימים חסרים, ואותו נרמול שנקבע באימון מופעל גם בזמן החיזוי.",
    },
  },
  {
    id: "audit",
    label: { en: "Leakage audit", he: "בדיקת דליפות" },
    short: { en: "Remove any column that leaks the answer", he: "הסרת כל עמודה שחושפת את התשובה" },
    icon: <ShieldCheck size={18} />,
    plain: {
      en: "Before any model sees the data, every column is tested for whether it secretly gives away the answer. The ones that do are removed.",
      he: "לפני שמודל כלשהו רואה את הנתונים, כל עמודה נבדקת: האם היא מסגירה בסתר את התשובה. עמודות כאלה מוסרות.",
    },
    fact: (e, r, lang) =>
      e && r ? (
        lang === "he" ? (
          <>
            <b>{r.blocklist.length}</b> עמודות נחסמו כדליפות או כמזהים. המדד החזק ביותר שנשאר מגיע לבדו ל־AUC של{" "}
            <b>{e.leakage.maxChannelAuc.toFixed(2)}</b>: אות אמיתי, לא הד של התשובה.
          </>
        ) : (
          <>
            <b>{r.blocklist.length}</b> columns blocked as leaks or identifiers. The strongest remaining single attribute scores{" "}
            <b>{e.leakage.maxChannelAuc.toFixed(2)}</b> AUC on its own: a real signal, not an echo of the answer.
          </>
        )
      ) : null,
    tech: {
      en: "scripts/leakage_audit.py scores every column against the label on whole drives; LeakageReport.assert_clean() raises if a blocked column, or anything derived from it, reaches the feature matrix.",
      he: "scripts/leakage_audit.py בודק כל עמודה מול התשובה על כוננים שלמים; LeakageReport.assert_clean() עוצר את הריצה אם עמודה חסומה, או משהו שנגזר ממנה, מגיעה למודל.",
    },
  },
  {
    id: "features",
    label: { en: "Feature engineering", he: "הנדסת מאפיינים" },
    short: { en: "Summarise each 30-day window per signal", he: "סיכום כל חלון של 30 יום לכל אות" },
    icon: <Sparkles size={18} />,
    plain: {
      en: "For each drive we look at its last 30 days and summarise how every counter behaved: where it ended, its average, and how much it changed.",
      he: "לכל כונן אנחנו מסתכלים על 30 הימים האחרונים ומסכמים איך התנהג כל מונה: איפה הוא הסתיים, מה הממוצע שלו, וכמה הוא השתנה.",
    },
    fact: (_, r, lang) =>
      r ? (
        lang === "he" ? (
          <>
            <b>{r.attributes.length}</b> מדדים ← <b>{r.n_channels}</b> אותות (ערך, שינוי יומי ושבועי, שלמות הדיווח) ← <b>{r.n_channels * 3}</b> מספרים לכל
            כונן.
          </>
        ) : (
          <>
            <b>{r.attributes.length}</b> attributes → <b>{r.n_channels}</b> signals (value, 1-day and 7-day change, reporting completeness) →{" "}
            <b>{r.n_channels * 3}</b> numbers per drive.
          </>
        )
      ) : null,
    tech: {
      en: "Channels: 15 log-scaled attributes, their lag-1 and lag-7 differences, plus an observed mask. XGBoost reads each channel's last value, mean and first-to-last delta over the window.",
      he: "ערוצים: 15 מדדים בסקאלה לוגריתמית, ההפרשים שלהם ליום ולשבוע, וסימון ימים חסרים. XGBoost קורא לכל ערוץ את הערך האחרון, הממוצע והשינוי לאורך החלון.",
    },
  },
  {
    id: "model",
    label: { en: "Machine learning", he: "למידת מכונה" },
    short: { en: "Gradient-boosted trees learn failure patterns", he: "עצי החלטה לומדים דפוסים שקודמים לתקלה" },
    icon: <Brain size={18} />,
    plain: {
      en: "A model learns, from millions of past examples, which patterns came before a failure and which did not.",
      he: "המודל לומד ממיליוני דוגמאות מהעבר אילו דפוסים הופיעו לפני תקלה ואילו לא.",
    },
    fact: (e, _, lang) =>
      e ? (
        lang === "he" ? (
          <>
            <b>{e.xgbTrees}</b> עצי החלטה (gradient boosting), שאומנו על <b>{fmtCompact(e.scale.splits[0]?.windows ?? 0, 1)}</b> חלונות מ־
            <b>{fmtInt(e.scale.splits[0]?.drives ?? 0)}</b> כוננים, ונבדקו על כוננים אחרים שהמודל לא ראה מעולם.
          </>
        ) : (
          <>
            <b>{e.xgbTrees}</b> gradient-boosted decision trees, trained on <b>{fmtCompact(e.scale.splits[0]?.windows ?? 0, 1)}</b> windows from{" "}
            <b>{fmtInt(e.scale.splits[0]?.drives ?? 0)}</b> drives, and tested on different drives it never saw.
          </>
        )
      ) : null,
    tech: {
      en: "XGBoost, binary:logistic, early-stopped on validation PR-AUC. A dilated 1-D CNN was trained on the identical windows for comparison.",
      he: "XGBoost עם binary:logistic, עם עצירה מוקדמת לפי PR-AUC על סט האימות. לצורך השוואה אומנה גם רשת CNN על אותם חלונות בדיוק.",
    },
  },
  {
    id: "risk",
    label: { en: "Risk prediction", he: "חיזוי סיכון" },
    short: { en: "A 30-day risk score; riskiest drives first", he: "ציון סיכון ל־30 יום; המסוכנים ביותר ראשונים" },
    icon: <Gauge size={18} />,
    plain: {
      en: "Every drive gets a risk score: the chance it fails in the next 30 days. The fleet is sorted so the riskiest drives are at the top of the list.",
      he: "כל כונן מקבל ציון סיכון: הסיכוי שיתקלקל ב־30 הימים הקרובים. הצי ממוין כך שהכוננים המסוכנים ביותר נמצאים בראש הרשימה.",
    },
    fact: (e, r, lang) =>
      e && r ? (
        lang === "he" ? (
          <>
            כוננים עם ציון מעל <b>{e.confusion.threshold.toFixed(3)}</b> מסומנים <b>לבדיקה</b>, בטווח של <b>{r.horizon_days} יום</b>.
          </>
        ) : (
          <>
            Drives scoring above <b>{e.confusion.threshold.toFixed(3)}</b> are marked <b>inspect</b>, within a <b>{r.horizon_days}-day</b> horizon.
          </>
        )
      ) : null,
    tech: {
      en: "The threshold is tuned on validation F1 and drawn on every risk bar in the product. Ranking quality, not the threshold, is what the Results section measures.",
      he: "הסף נקבע לפי F1 על סט האימות ומסומן על כל פס סיכון במוצר. חלק התוצאות מודד את איכות הדירוג, לא את הסף.",
    },
  },
];

/** Artifact C: raw telemetry to ranked risk, one clickable stage card at a time. */
export function DataPipeline() {
  const [active, setActive] = useState(0);
  const reduce = useReducedMotion();
  const { evaluation, model } = useStoryData();
  const { lang, t } = useLang();
  const stage = STAGES[active];

  return (
    <div>
      <ol className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {STAGES.map((s, i) => {
          const on = i === active;
          return (
            <li key={s.id} className="relative">
              <button
                type="button"
                aria-pressed={on}
                onClick={() => setActive(i)}
                className={`group relative flex h-full w-full flex-col overflow-hidden rounded-xl border p-4 text-start transition-all duration-300 ${
                  on ? "border-signal/60 bg-signal/[0.07] shadow-[0_0_32px_-12px_rgba(63,224,255,0.55)]" : "border-line bg-night/60 hover:border-line-strong hover:bg-deep/50"
                }`}
              >
                {on && !reduce && (
                  <motion.span
                    aria-hidden
                    className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-signal to-transparent"
                    initial={{ x: "-100%" }}
                    animate={{ x: "100%" }}
                    transition={{ duration: 2.4, repeat: Infinity, ease: "linear" }}
                  />
                )}
                <span
                  className={`mb-4 flex h-10 w-10 items-center justify-center rounded-lg border transition ${
                    on ? "border-signal/50 bg-signal/15 text-signal" : "border-line-strong text-muted group-hover:text-soft"
                  }`}
                >
                  {s.icon}
                </span>
                <span className={`text-sm font-medium ${on ? "text-ink" : "text-soft"}`}>
                  {i + 1}. {s.label[lang]}
                </span>
                <span className="mt-1.5 line-clamp-3 text-xs leading-relaxed text-muted">{s.short[lang]}</span>
              </button>
              {i < STAGES.length - 1 && (
                <span aria-hidden className="absolute -end-[11px] top-1/2 z-10 hidden -translate-y-1/2 text-faint lg:block">
                  <ChevronRight size={14} className="rtl:rotate-180" />
                </span>
              )}
            </li>
          );
        })}
      </ol>

      <div className="panel mt-4" aria-live="polite">
        <AnimatePresence mode="wait">
          <motion.div
            key={stage.id}
            initial={reduce ? false : { opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduce ? undefined : { opacity: 0, y: -6 }}
            transition={{ duration: 0.35, ease: EASE }}
            className="grid gap-6 md:grid-cols-3 md:gap-10"
          >
            <div>
              <p className="kicker mb-2 text-[10px]">{t(`Stage ${active + 1} of ${STAGES.length}`, `שלב ${active + 1} מתוך ${STAGES.length}`)}</p>
              <h3 className="text-lg font-semibold">{stage.label[lang]}</h3>
              <p className="mt-2 text-sm leading-relaxed text-soft">{stage.plain[lang]}</p>
            </div>
            <div>
              <p className="mb-2 text-xs font-medium text-muted">{t("In this project", "בפרויקט הזה")}</p>
              <p className="text-sm leading-relaxed text-soft [&_b]:num [&_b]:font-medium [&_b]:text-ink">
                {stage.fact(evaluation.data, model.data?.run ?? null, lang) ?? <span className="text-muted">{t("Loading the numbers…", "טוען את המספרים…")}</span>}
              </p>
            </div>
            <div>
              <p className="mb-2 text-xs font-medium text-muted">{t("Under the hood", "מאחורי הקלעים")}</p>
              <p className="text-sm leading-relaxed text-muted">{stage.tech[lang]}</p>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
