import { useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Check, Circle } from "lucide-react";
import type { Evaluation } from "../../shared/types";
import { Expandable } from "../components/Expandable";
import { StorySection } from "../components/StorySection";
import { ErrorState, Loading, Source } from "../components/States";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { useStoryData } from "../lib/data";
import { fmtInt, fmtPct } from "../lib/format";
import { useRise } from "../lib/motion";
import { useLang } from "../lib/i18n";

const tooltipStyle = { background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 };

/** K dots, one per ranked drive: filled = really failed, hollow = did not. */
function TopK({ e }: { e: Evaluation }) {
  const ks = e.precisionAtK.map((p) => p.k);
  const [k, setK] = useState(ks.includes(100) ? 100 : ks[0]);
  const row = e.precisionAtK.find((p) => p.k === k)!;
  const reduce = useReducedMotion();
  const { t, he } = useLang();
  const dot = k <= 100 ? "h-3.5 w-3.5 sm:h-4 sm:w-4" : k <= 250 ? "h-2.5 w-2.5" : "h-1.5 w-1.5 sm:h-2 sm:w-2";

  return (
    <div className="panel grid gap-10 lg:grid-cols-[1fr_1.1fr] lg:gap-16">
      <div>
        <p className="kicker mb-4 text-[10px]">{t("Ranking precision · held-out test drives", "דיוק הדירוג · כוננים שלא נראו באימון")}</p>
        <p dir="ltr" className="text-start text-[clamp(3.5rem,8vw,6rem)] font-semibold leading-[0.9] tracking-[-0.05em] tabular-nums rtl:text-end">
          {row.xgbHits}
          <span className="text-faint">/{k}</span>
        </p>
        <p className="mt-5 max-w-md text-base leading-relaxed text-soft">
          {he ? (
            <>
              מתוך <b className="font-medium text-ink">{k}</b> כונני המבחן שהמערכת דירגה כמסוכנים ביותר,{" "}
              <b className="font-medium text-ok">{row.xgbHits}</b> באמת התקלקלו תוך 30 יום.
            </>
          ) : (
            <>
              Of the <b className="font-medium text-ink">{k}</b> test drives the system ranked as most at risk,{" "}
              <b className="font-medium text-ok">{row.xgbHits}</b> really did fail within 30 days.
            </>
          )}
        </p>
        <div role="radiogroup" aria-label={t("How many top-ranked drives", "כמה כוננים מראש הרשימה")} className="mt-8 flex flex-wrap gap-2">
          {ks.map((kk) => (
            <button
              key={kk}
              role="radio"
              aria-checked={kk === k}
              onClick={() => setK(kk)}
              className={`num rounded-full border px-3.5 py-1.5 text-xs transition ${
                kk === k ? "border-signal bg-signal/10 text-ink" : "border-line text-muted hover:text-soft"
              }`}
            >
              {t(`top ${fmtInt(kk)}`, `${fmtInt(kk)} המובילים`)}
            </button>
          ))}
        </div>
      </div>
      <div>
        <div className="flex flex-wrap gap-[5px]" role="img" aria-label={t(`${row.xgbHits} of the top ${k} ranked drives failed; ${k - row.xgbHits} did not.`, `${row.xgbHits} מתוך ${k} הכוננים המובילים התקלקלו; ${k - row.xgbHits} לא.`)}>
          {Array.from({ length: k }, (_, i) => {
            const hit = i < row.xgbHits;
            return (
              <motion.span
                key={`${k}-${i}`}
                initial={reduce ? false : { opacity: 0, scale: 0.4 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: reduce ? 0 : Math.min(1.2, i * (1.2 / k)), duration: 0.3 }}
                className={`${dot} rounded-full ${hit ? "bg-ok" : "border border-muted"}`}
              />
            );
          })}
        </div>
        <div className="mt-5 flex flex-wrap gap-5 text-xs text-muted">
          <span className="flex items-center gap-2">
            <Check size={12} className="text-ok" aria-hidden /> {t("failed within 30 days", "התקלקלו תוך 30 יום")} ({row.xgbHits})
          </span>
          <span className="flex items-center gap-2">
            <Circle size={11} aria-hidden /> {t("did not fail", "לא התקלקלו")} ({k - row.xgbHits})
          </span>
        </div>
      </div>
    </div>
  );
}

/** Why accuracy is the wrong yardstick, in this project's own numbers. */
function AccuracyTrap({ e }: { e: Evaluation }) {
  const rise = useRise();
  const { t, he } = useLang();
  const xgb = e.models.find((m) => m.model === "XGBoost");
  const rows = [
    {
      who: t("A “model” that always answers healthy", "“מודל” שתמיד עונה שהכל תקין"),
      acc: e.accuracy.alwaysHealthy,
      found: 0,
      top: null as number | null,
    },
    {
      who: t("Our model", "המודל שלנו"),
      acc: e.accuracy.xgboost,
      found: e.confusion.tp,
      top: (xgb?.hitsAt100 ?? null) as number | null,
    },
  ];
  return (
    <motion.div {...rise} className="panel grid gap-10 lg:grid-cols-[1fr_1.4fr] lg:gap-16">
      <div>
        <h3 className="text-xl font-semibold tracking-tight md:text-2xl">
          {t("Why we don't lead with “accuracy”.", "למה אנחנו לא מתחילים מ“דיוק”.")}
        </h3>
        <p className="mt-4 text-sm leading-relaxed text-soft md:text-[15px]">
          {he
            ? `תקלות הן נדירות: ${fmtPct(e.scale.baseRate, 2)} מהמקרים. מערכת שאף פעם לא מתריעה כבר “מדויקת” ב־${fmtPct(e.accuracy.alwaysHealthy, 2)}, ולגמרי חסרת תועלת. מה שחשוב הוא אם הכוננים בראש הרשימה הם אלה שבאמת עומדים להתקלקל.`
            : `Failures are rare: ${fmtPct(e.scale.baseRate, 2)} of cases. A system that never raises an alarm is already ${fmtPct(e.accuracy.alwaysHealthy, 2)} “accurate” and completely useless. What matters is whether the drives at the top of the list are the ones that will really fail.`}
        </p>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-start text-sm">
          <thead>
            <tr className="text-xs text-muted">
              <th className="pb-4 pe-4 font-normal" />
              <th className="pb-4 pe-4 font-normal">{t("Accuracy", "דיוק")}</th>
              <th className="pb-4 pe-4 font-normal">{t("Failing cases caught", "תקלות שזוהו")}</th>
              <th className="pb-4 font-normal">{t("Right in top 100", "צדק ב־100 המובילים")}</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.who} className="border-t hairline">
                <td className={`py-5 pe-4 ${i ? "text-ink" : "text-muted"}`}>{r.who}</td>
                <td className="num py-5 text-soft">{fmtPct(r.acc, 2)}</td>
                <td className={`num py-5 ${i ? "text-ok" : "text-risk"}`}>
                  <span dir="ltr" className="inline-block">
                    {fmtInt(r.found)} <span className="text-faint">/ {fmtInt(e.scale.testPositives)}</span>
                  </span>
                </td>
                <td className={`num py-5 ${i ? "text-ok" : "text-risk"}`}>{r.top ?? t("no ranking", "אין דירוג")}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-4 text-sm text-muted">{t("Same accuracy. Completely different value.", "אותו דיוק. ערך שונה לגמרי.")}</p>
      </div>
    </motion.div>
  );
}

function LeadTime({ e }: { e: Evaluation }) {
  const lt = e.leadTime;
  const { t, he } = useLang();
  const data = useMemo(
    () => (lt ? lt.curve.map((p) => ({ lead: p.lead, flagged: p.xgbFlagRate * 100 })) : []),
    [lt],
  );
  if (!lt) return null;
  return (
    <div className="panel grid gap-10 lg:grid-cols-[1fr_1.4fr] lg:gap-16">
      <div>
        <h3 className="text-xl font-semibold tracking-tight md:text-2xl">
          {he ? (
            <>
              כמה <span className="text-signal">מוקדם</span> המערכת מזהה.
            </>
          ) : (
            <>
              How <span className="text-signal">early</span> it sees it.
            </>
          )}
        </h3>
        <p className="mt-4 text-sm leading-relaxed text-soft md:text-[15px]">
          {t(
            `We replayed ${fmtInt(lt.drives)} drives that failed during the test period, day by day, through the model.`,
            `הרצנו מחדש דרך המודל, יום אחרי יום, ${fmtInt(lt.drives)} כוננים שהתקלקלו בתקופת המבחן.`,
          )}
        </p>
        <dl className="mt-10 grid grid-cols-2 gap-8">
          <div>
            <dd className="num text-4xl text-ink">{fmtPct(lt.xgbFlaggedWithinHorizon, 0)}</dd>
            <dt className="mt-2 text-sm text-muted">{t("were flagged for inspection at least once in their final 30 days", "סומנו לבדיקה לפחות פעם אחת ב־30 הימים האחרונים שלהם")}</dt>
          </div>
          <div>
            <dd className="num text-4xl text-ink">{fmtPct(lt.xgbFlaggedWeekAhead, 0)}</dd>
            <dt className="mt-2 text-sm text-muted">{t("were flagged at least a week before they failed", "סומנו לפחות שבוע לפני שהתקלקלו")}</dt>
          </div>
        </dl>
      </div>
      <figure className="min-w-0">
        <figcaption className="mb-3 text-sm text-soft">{t("Share of failing drives flagged, by days before failure", "שיעור הכוננים שסומנו, לפי מספר הימים לפני התקלה")}</figcaption>
        <div className="h-72" dir="ltr">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}>
              <defs>
                <linearGradient id="leadFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="var(--color-series-xgb)" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="var(--color-series-xgb)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid vertical={false} />
              <XAxis
                dataKey="lead"
                reversed
                tickLine={false}
                axisLine={false}
                ticks={[90, 60, 30, 14, 7, 0]}
                type="number"
                domain={[0, 90]}
                tickFormatter={(d) => (d === 0 ? t("fail", "תקלה") : t(`${d}d`, `${d} י׳`))}
              />
              <YAxis tickLine={false} axisLine={false} unit="%" width={48} domain={[0, 60]} />
              <ReferenceLine x={30} stroke="var(--color-faint)" strokeDasharray="4 4" label={{ value: t("30-day horizon", "טווח 30 יום"), fill: "var(--color-muted)", fontSize: 11, position: "insideTopLeft" }} />
              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(l) => (Number(l) === 0 ? t("on the day of failure", "ביום התקלה") : t(`${l} days before failure`, `${l} ימים לפני התקלה`))}
                formatter={(v) => [t(`${Number(v).toFixed(1)}% flagged`, `${Number(v).toFixed(1)}% סומנו`), "XGBoost"]}
              />
              <Area type="monotone" dataKey="flagged" stroke="var(--color-series-xgb)" strokeWidth={2} fill="url(#leadFill)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-2 text-xs text-muted">
          {t(
            "The closer a drive gets to failure, the more likely the model is to flag it on that day.",
            "ככל שכונן מתקרב לתקלה, כך גדל הסיכוי שהמודל יסמן אותו באותו יום.",
          )}
        </p>
      </figure>
    </div>
  );
}

/** Artifact E. */
export function PerformanceProof() {
  const { evaluation } = useStoryData();
  const { data: e, error, loading, reload } = evaluation;
  const { t, he } = useLang();
  if (loading) return <Loading label={t("Reading the evaluation results", "קורא את תוצאות ההערכה")} />;
  if (error || !e) return <ErrorState message={error ?? "no evaluation results"} onRetry={reload} />;
  const xgb = e.models.find((m) => m.model === "XGBoost");

  return (
    <div className="space-y-6">
      <TopK e={e} />
      <AccuracyTrap e={e} />
      <LeadTime e={e} />

      <p className="text-xs leading-relaxed text-muted">
        {t(
          `Scope: tested on one large public fleet (Backblaze, ${e.scale.quarters.length} quarters). Any new fleet should be validated on its own history before relying on these numbers.`,
          `היקף: נבדק על צי ציבורי גדול אחד (Backblaze, ${e.scale.quarters.length} רבעונים). צי חדש צריך להיבדק על ההיסטוריה שלו לפני שסומכים על המספרים האלה.`,
        )}
      </p>
      <Expandable
        title={t("Standard ML metrics and methodology", "מדדי למידת מכונה סטנדרטיים ומתודולוגיה")}
        hint={t(
          `PR-AUC ${xgb?.prAuc.toFixed(3)} · ROC-AUC ${xgb?.rocAuc.toFixed(3)}, and how precision, thresholds and lead time were measured.`,
          `PR-AUC ${xgb?.prAuc.toFixed(3)} · ROC-AUC ${xgb?.rocAuc.toFixed(3)}, ואיך נמדדו הדיוק, הסף וזמן ההתרעה.`,
        )}
      >
          {xgb && (
            <div className="grid gap-px overflow-hidden rounded-2xl border hairline bg-line sm:grid-cols-2">
              {[
                {
                  v: xgb.prAuc,
                  base: e.scale.baseRate,
                  t: t("Ranking quality on rare events (PR-AUC)", "איכות הדירוג באירועים נדירים (PR-AUC)"),
                  d: t(
                    `A random ranking would score ${e.scale.baseRate.toFixed(4)}. Ours is about ${Math.round(xgb.prAuc / e.scale.baseRate)}× that.`,
                    `דירוג אקראי היה מקבל ${e.scale.baseRate.toFixed(4)}. שלנו גבוה פי ${Math.round(xgb.prAuc / e.scale.baseRate)} בערך.`,
                  ),
                },
                {
                  v: xgb.rocAuc,
                  base: 0.5,
                  t: t("Separation of failing from healthy (ROC-AUC)", "הפרדה בין כוננים תקינים לתקולים (ROC-AUC)"),
                  d: t(
                    `Pick one failing and one healthy case at random: ${fmtPct(xgb.rocAuc, 0)} of the time the failing one is ranked higher. Random is 50%.`,
                    `בוחרים באקראי כונן תקול אחד ותקין אחד: ב־${fmtPct(xgb.rocAuc, 0)} מהמקרים התקול מדורג גבוה יותר. ניחוש אקראי: 50%.`,
                  ),
                },
              ].map((m) => (
                <div key={m.t} className="bg-night/95 p-6 md:p-7">
                  <p className="num text-4xl">{m.v.toFixed(3)}</p>
                  <p className="mt-3 text-soft">{m.t}</p>
                  <p className="mt-2 text-sm text-muted">{m.d}</p>
                </div>
              ))}
            </div>
          )}

          <div className="pt-2">
            <TechnicalDisclosure title={t("Why precision@K is the right measure for a rare-event problem", "למה precision@K הוא המדד הנכון לאירועים נדירים")}>
              {he ? (
                <>
                  <p>
                    כששיעור הבסיס הוא {fmtPct(e.scale.baseRate, 2)}, הדיוק ו־ROC-AUC נשלטים על ידי {fmtInt(e.confusion.tn)} המקרים התקינים הקלים. מפעיל לא
                    מטפל בכל כונן; הוא מטפל ברשימה קצרה. <b>Precision@K</b> שואל בדיוק את זה: מתוך K הכוננים בראש הרשימה, כמה באמת מתקלקלים?
                  </p>
                  <p>
                    <b>PR-AUC</b> מסכם את הדיוק לאורך כל אורכי הרשימה האפשריים. קו הבסיס שלו הוא שיעור הבסיס ({e.scale.baseRate.toFixed(4)}) ולא 0.5, ולכן{" "}
                    {xgb?.prAuc.toFixed(3)} הוא תוצאה חזקה כאן, גם אם הוא נראה צנוע לצד ה־ROC-AUC.
                  </p>
                </>
              ) : (
                <>
              <p>
                With a {fmtPct(e.scale.baseRate, 2)} base rate, accuracy and ROC-AUC are dominated by the {fmtInt(e.confusion.tn)}{" "}
                easy healthy cases. An operator does not act on every drive. They act on a short list. <b>Precision@K</b> asks
                exactly that question: of the K drives at the top of the list, how many really fail? The drive-level ladder is
                read from <code>w2022x_cnn_vs_xgboost_precision_at_k.csv</code>.
              </p>
              <p>
                <b>PR-AUC</b> summarises precision across every possible list length. Its baseline is the base rate (
                {e.scale.baseRate.toFixed(4)}), not 0.5, which is why {xgb?.prAuc.toFixed(3)} is a strong result here even
                though it looks modest next to the ROC-AUC.
              </p>
                </>
              )}
            </TechnicalDisclosure>
            <TechnicalDisclosure title={t("Ranking performance versus the decision threshold", "איכות הדירוג מול סף ההחלטה")}>
              {he ? (
                <>
                  <p>
                    הדירוג הוא המוצר: ממיינים את הצי ועובדים מלמעלה. הסף ({e.confusion.threshold.toFixed(3)}, שנקבע לפי F1 על סט האימות) קובע רק איפה
                    מתחיל “לבדיקה”. בסף הזה, על {fmtInt(e.confusion.n)} חלונות מבחן, {fmtPct(xgb?.precision ?? 0)} מהחלונות שסומנו היו תקלות אמיתיות,
                    ו־{fmtPct(xgb?.recall ?? 0)} מהחלונות התקולים סומנו ({fmtInt(e.confusion.tp)} התראות נכונות, {fmtInt(e.confusion.fp)} התראות שווא,{" "}
                    {fmtInt(e.confusion.fn)} פספוסים).
                  </p>
                  <p>
                    המספרים נמדדו בשיעור התקלות האמיתי של הצי, על כוננים שלא נכללו באימון, ולא על מדגם מאוזן שבו אותו מודל היה נראה הרבה יותר טוב. נתוני
                    הדיוק ממחקר Toshiba הנפרד מנופחים מעצם הבנייה שלהם, ולכן לא מוצגים כאן.
                  </p>
                </>
              ) : (
                <>
              <p>
                The ranking is the product: sort the fleet, work from the top. The threshold ({e.confusion.threshold.toFixed(3)},
                tuned on validation F1) decides only where “inspect” starts. At that threshold, on {fmtInt(e.confusion.n)} test
                windows, {fmtPct(xgb?.precision ?? 0)} of flagged windows were real failures and{" "}
                {fmtPct(xgb?.recall ?? 0)} of failing windows were flagged ({fmtInt(e.confusion.tp)} true alerts,{" "}
                {fmtInt(e.confusion.fp)} false alarms, {fmtInt(e.confusion.fn)} missed).
              </p>
              <p>
                These figures are measured at the fleet's real failure rate, on drives held out from training. They are not from
                a balanced sample, where the same model would look far better. Precision figures from the separate Toshiba
                cohort study are inflated by construction and are not shown here as performance.
              </p>
                </>
              )}
            </TechnicalDisclosure>
            <TechnicalDisclosure title={t("How the lead-time curve was measured", "איך נמדדה עקומת זמן ההתרעה")}>
              {he ? (
                <p>
                  כל כונן תקול בסט המבחן קיבל ציון על חלון של 30 יום שמסתיים 0 עד 90 ימים לפני התקלה. העקומה מראה את שיעור הכוננים שעברו את הסף בכל מרחק.
                  המספרים הבולטים סופרים כונן פעם אחת אם סומן בכל יום בתוך טווח 30 הימים (או, עבור “שבוע מראש”, בכל יום 7 עד 29 ימים לפני).
                </p>
              ) : (
              <p>
                Each failing test drive is scored with its 30-day window ending 0 to 90 days before its failure (
                <code>w2022x_lead_time_failing.csv</code>). The curve is the share of drives scoring above the threshold at each
                distance. The headline shares count a drive once if it was flagged on any day inside the 30-day horizon (or,
                for “a week before”, on any day 7 to 29 days out).
              </p>
              )}
            </TechnicalDisclosure>
          </div>
      </Expandable>
      <Source>{e.sources.slice(0, 5).join(" · ")}</Source>
    </div>
  );
}

/** 05 — The Breakthrough. */
export default function PerformanceMetrics() {
  const { t, he } = useLang();
  return (
    <StorySection
      id="breakthrough"
      kicker={t("Results", "תוצאות")}
      title={
        he ? (
          <>
            בראש הרשימה שלה, המערכת צודקת <span className="text-ok">כמעט תמיד</span>.
          </>
        ) : (
          <>
            At the top of its list, it is <span className="text-ok">almost always</span> right.
          </>
        )
      }
      band
      lede={t(
        "Measured on real fleet data at its true failure rate, on drives the model never saw during training.",
        "נמדד על נתונים אמיתיים של צי כוננים, בשיעור התקלות האמיתי שלו, על כוננים שהמודל לא ראה באימון.",
      )}
    >
      <PerformanceProof />
    </StorySection>
  );
}
