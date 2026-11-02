import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Activity, AlertOctagon, BellRing, HardDrive, Pause, Play, RotateCcw, ShieldCheck, Wrench } from "lucide-react";
import { StorySection } from "../components/StorySection";
import { useStoryData } from "../lib/data";
import { fmtInt } from "../lib/format";
import { useLang } from "../lib/i18n";

/** Points on the shared timeline, as fractions of its width. */
const FAIL_AT = 0.78;
const ALERT_AT = 0.5;

function Track({
  progress,
  tone,
  children,
}: {
  progress: number;
  tone: "risk" | "signal";
  children: React.ReactNode;
}) {
  return (
    <div className="relative h-24">
      <div className="absolute inset-x-0 top-1/2 h-px bg-line-strong" />
      <div
        className="absolute left-0 top-1/2 h-px"
        style={{ width: `${progress * 100}%`, background: tone === "risk" ? "var(--color-faint)" : "var(--color-signal)" }}
      />
      <div
        className="absolute top-1/2 h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{ left: `${progress * 100}%`, background: tone === "risk" ? "var(--color-soft)" : "var(--color-signal)" }}
      />
      {children}
    </div>
  );
}

function Marker({
  at,
  show,
  icon,
  label,
  tone,
  below = false,
}: {
  at: number;
  show: boolean;
  icon: React.ReactNode;
  label: string;
  tone: string;
  below?: boolean;
}) {
  return (
    <motion.div
      initial={false}
      animate={{ opacity: show ? 1 : 0.0, scale: show ? 1 : 0.9 }}
      transition={{ duration: 0.4 }}
      className={`absolute flex -translate-x-1/2 flex-col items-center gap-2 ${below ? "top-[calc(50%+12px)]" : "bottom-[calc(50%+12px)]"}`}
      style={{ left: `${at * 100}%` }}
    >
      {below && <span className="h-3 w-px" style={{ background: tone }} />}
      <span dir="auto" className="flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-xs" style={{ borderColor: tone, color: tone }}>
        {icon}
        {label}
      </span>
      {!below && <span className="h-3 w-px" style={{ background: tone }} />}
    </motion.div>
  );
}

/** A schematic risk curve: flat, then climbing toward the failure point. Illustrative. */
function RiskSketch({ progress }: { progress: number }) {
  const pts = Array.from({ length: 60 }, (_, i) => {
    const x = (i / 59) * FAIL_AT;
    const y = 0.08 + 0.85 * Math.pow(Math.max(0, (x - 0.15) / (FAIL_AT - 0.15)), 1.6);
    return [x, y] as const;
  });
  const visible = pts.filter(([x]) => x <= progress);
  const d = visible.map(([x, y], i) => `${i ? "L" : "M"}${(x * 1000).toFixed(1)},${(100 - y * 100).toFixed(1)}`).join(" ");
  return (
    <svg viewBox="0 0 1000 100" preserveAspectRatio="none" className="absolute inset-x-0 top-0 h-24 w-full overflow-visible" aria-hidden>
      <line x1="0" x2="1000" y1={100 - 0.41 * 100} y2={100 - 0.41 * 100} stroke="var(--color-warn)" strokeDasharray="4 6" strokeOpacity="0.5" vectorEffect="non-scaling-stroke" />
      <path d={d} fill="none" stroke="var(--color-signal)" strokeWidth="2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

const DAYS = 60;
const DURATION_S = 10;
const TICK_MS = 50;

/** The schematic risk curve's value at a point on the timeline. */
const riskAt = (x: number) => 0.08 + 0.85 * Math.pow(Math.max(0, (Math.min(x, FAIL_AT) - 0.15) / (FAIL_AT - 0.15)), 1.6);

type Phase = "idle" | "running" | "paused" | "done";

function Status({ tone, children }: { tone: string; children: React.ReactNode }) {
  return (
    <p
      className="mt-4 flex min-h-[2.25rem] flex-wrap items-center gap-2 rounded-lg border px-3 py-2 text-xs"
      style={{ borderColor: `color-mix(in oklab, ${tone} 35%, transparent)`, color: tone }}
    >
      {children}
    </p>
  );
}

/** Artifact B: the same drive, the same failure, two operating models, played day by day. */
export function ProblemVisual() {
  const [p, setP] = useState(0);
  const [phase, setPhase] = useState<Phase>("idle");
  const pRef = useRef(0);

  // Driven by a timer rather than a motion library, so it plays under reduced motion
  // too: it changes state (a day counter, a status line), it is not decoration.
  useEffect(() => {
    if (phase !== "running") return;
    const id = setInterval(() => {
      const next = Math.min(1, pRef.current + TICK_MS / 1000 / DURATION_S);
      pRef.current = next;
      setP(next);
      if (next >= 1) setPhase("done");
    }, TICK_MS);
    return () => clearInterval(id);
  }, [phase]);

  const restart = () => {
    pRef.current = 0;
    setP(0);
    setPhase("running");
  };
  const start = () => (phase === "paused" ? setPhase("running") : restart());

  const failed = p >= FAIL_AT;
  const alerted = p >= ALERT_AT;
  const day = Math.round(p * DAYS);
  const failDay = Math.round(FAIL_AT * DAYS);
  const warnDays = Math.round((FAIL_AT - ALERT_AT) * DAYS);
  const downDays = Math.max(0, Math.round((p - FAIL_AT) * DAYS));
  const live = phase === "running";
  const { t, he } = useLang();

  return (
    <div className="space-y-4">
      {/* playback strip */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-3 rounded-xl border hairline bg-night/60 px-4 py-3">
        <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-ink">
          <span className="relative flex h-2.5 w-2.5">
            {live && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-risk opacity-70" />}
            <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${live ? "bg-risk" : "bg-faint"}`} />
          </span>
          {phase === "idle" ? t("Ready", "מוכן") : live ? t("Live", "בשידור חי") : phase === "paused" ? t("Paused", "מושהה") : t("Finished", "הסתיים")}
        </span>
        <span className="num text-xs text-muted">
          {t("day", "יום")} <span className="text-ink">{String(day).padStart(2, "0")}</span> / {DAYS}
        </span>
        <div dir="ltr" className="h-1 min-w-[8rem] flex-1 overflow-hidden rounded bg-deep" aria-hidden>
          <div className="h-1 bg-gradient-to-r from-signal to-warn" style={{ width: `${p * 100}%` }} />
        </div>
        <div className="flex items-center gap-2">
          {live ? (
            <button type="button" onClick={() => setPhase("paused")} className="chip hover:border-signal hover:text-ink">
              <Pause size={12} aria-hidden /> {t("Pause", "השהה")}
            </button>
          ) : (
            <button type="button" onClick={start} className="btn btn-primary !px-4 !py-1.5 !text-[13px]">
              {phase === "done" ? <RotateCcw size={13} aria-hidden /> : <Play size={13} aria-hidden className="rtl:rotate-180" />}
              {phase === "idle" ? t("Start", "התחל") : phase === "paused" ? t("Resume", "המשך") : t("Replay", "הפעל שוב")}
            </button>
          )}
          {(live || phase === "paused") && (
            <button type="button" onClick={restart} className="chip hover:border-signal hover:text-ink">
              <RotateCcw size={12} aria-hidden /> {t("Restart", "מההתחלה")}
            </button>
          )}
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        {/* Reactive */}
        <div className="panel">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <div>
              <h3 className="panel-title">{t("Reactive maintenance", "תחזוקה תגובתית")}</h3>
              <p className="panel-sub">{t("What most fleets do today", "מה שרוב הארגונים עושים היום")}</p>
            </div>
            <span className="rounded-full border border-risk/40 px-2.5 py-0.5 text-[11px] text-risk">{t("After failure", "אחרי התקלה")}</span>
          </div>
          {!failed ? (
            <Status tone="var(--color-muted)">
              <HardDrive size={13} aria-hidden />
              {phase === "idle"
                ? t("Press Start to follow one drive through its last 60 days.", "לחצו על התחל כדי לעקוב אחרי כונן אחד ב־60 הימים האחרונים שלו.")
                : t("Running normally · no warning signs are watched", "עובד כרגיל · אף אחד לא עוקב אחרי סימני אזהרה")}
            </Status>
          ) : p < FAIL_AT + 0.04 ? (
            <Status tone="var(--color-risk)">
              <AlertOctagon size={13} aria-hidden /> {t(`Day ${failDay}: the drive failed without warning`, `יום ${failDay}: הכונן התקלקל בלי שום אזהרה`)}
            </Status>
          ) : (
            <Status tone="var(--color-risk)">
              <BellRing size={13} aria-hidden />{" "}
              {t(
                `Outage · the alert came after the failure · down ${downDays} day${downDays === 1 ? "" : "s"}`,
                `השבתה · ההתראה הגיעה אחרי התקלה · ${downDays === 1 ? "יום אחד" : `${downDays} ימים`} בלי שירות`,
              )}
            </Status>
          )}
          <div dir="ltr" className="relative pt-16">
            <p dir="auto" className="absolute left-0 top-6 font-mono text-[10px] text-faint">
              {t("no early-warning signal is watched", "אין מעקב אחרי סימני אזהרה")}
            </p>
            <Track progress={Math.min(p, FAIL_AT)} tone="risk">
              <Marker at={FAIL_AT} show={failed} icon={<AlertOctagon size={12} />} label={t("Drive fails", "הכונן מתקלקל")} tone="var(--color-risk)" />
              <Marker
                at={Math.min(0.97, FAIL_AT + 0.08)}
                show={p > FAIL_AT + 0.04}
                icon={<BellRing size={12} />}
                label={t("Alert", "התראה")}
                tone="var(--color-risk)"
                below
              />
              <div
                aria-hidden
                className="absolute top-1/2 h-8 -translate-y-1/2 rounded bg-risk/15"
                style={{ left: `${FAIL_AT * 100}%`, width: `${Math.max(0, Math.min(p, 1) - FAIL_AT) * 100}%` }}
              />
            </Track>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-soft">
            {he ? (
              <>
                הכל נראה תקין עד שהכונן מפסיק לעבוד. ההתראה מגיעה <b className="font-medium text-risk">אחרי</b> התקלה: המידע בסיכון, השירות נפגע,
                וההחלפה לא מתוכננת.
              </>
            ) : (
              <>
                Nothing looks wrong until the drive stops. The alert arrives <b className="font-medium text-risk">after</b> the failure — data is at
                risk, the service degrades, and the replacement is unplanned.
              </>
            )}
          </p>
        </div>

        {/* Predictive */}
        <div className="panel border-signal/25">
          <div className="flex flex-wrap items-baseline justify-between gap-3">
            <div>
              <h3 className="panel-title">{t("Predictive maintenance", "תחזוקה חזויה")}</h3>
              <p className="panel-sub">{t("What this system enables", "מה שהמערכת שלנו מאפשרת")}</p>
            </div>
            <span className="rounded-full border border-signal/50 px-2.5 py-0.5 text-[11px] text-signal">{t("Before failure", "לפני התקלה")}</span>
          </div>
          {!alerted ? (
            <Status tone="var(--color-signal)">
              <Activity size={13} aria-hidden />
              {phase === "idle" ? (
                t("Telemetry is read and scored every day.", "נתוני הכונן נקראים ומנוקדים בכל יום.")
              ) : (
                <>
                  {t("Monitoring · risk score", "בניטור · ציון סיכון")} <span className="num">{riskAt(p).toFixed(2)}</span>
                </>
              )}
            </Status>
          ) : !failed ? (
            <Status tone="var(--color-warn)">
              <Wrench size={13} aria-hidden /> {t(`Flagged ${warnDays} days before failure · risk`, `סומן ${warnDays} ימים לפני התקלה · סיכון`)}{" "}
              <span className="num">{riskAt(p).toFixed(2)}</span> · {t("inspection scheduled", "נקבעה בדיקה")}
            </Status>
          ) : (
            <Status tone="var(--color-ok)">
              <ShieldCheck size={13} aria-hidden /> {t("Replaced in a planned window · no outage", "הוחלף בתחזוקה מתוכננת · בלי השבתה")}
            </Status>
          )}
          <div dir="ltr" className="relative pt-24">
            <span dir="auto" className="absolute right-0 top-[1.6rem] font-mono text-[10px] text-warn/70">
              {t("inspect threshold", "סף בדיקה")}
            </span>
            <RiskSketch progress={Math.min(p, FAIL_AT)} />
            <Track progress={Math.min(p, FAIL_AT)} tone="signal">
              <Marker at={ALERT_AT} show={alerted} icon={<Wrench size={12} />} label={t("Inspect this drive", "לבדוק את הכונן")} tone="var(--color-warn)" />
              <Marker at={FAIL_AT} show={failed} icon={<ShieldCheck size={12} />} label={t("Replaced in time", "הוחלף בזמן")} tone="var(--color-ok)" below />
            </Track>
          </div>
          <p className="mt-4 text-sm leading-relaxed text-soft">
            {he ? (
              <>
                נתוני הכונן נקראים כל יום. כשסימני האזהרה מצטברים, ציון הסיכון עולה, ונקבעת בדיקה{" "}
                <b className="font-medium text-signal">לפני</b> שמשהו מתקלקל.
              </>
            ) : (
              <>
                Daily telemetry is read continuously. As the warning signs build, the drive's risk climbs, and an inspection is scheduled{" "}
                <b className="font-medium text-signal">before</b> anything breaks.
              </>
            )}
          </p>
        </div>
      </div>

      <p className="text-[11px] text-faint">
        {t(
          "Schematic: one illustrative drive, not data. The measured version of this curve is in the Results section.",
          "המחשה: כונן אחד לדוגמה, לא נתונים אמיתיים. הגרסה המדודה של העקומה נמצאת בחלק התוצאות.",
        )}
      </p>
    </div>
  );
}

/** 02 — The Hidden Problem. */
export function ProblemComparison() {
  const { data } = useStoryData().evaluation;
  // Drives with a recorded failure across every split, over the quarters the data spans.
  const failures = data?.scale.splits.reduce((a, s) => a + s.drivesWithFailure, 0);
  const perDay = data && failures ? failures / (data.scale.quarters.length * 91.3) : null;
  const { t, he } = useLang();
  return (
    <StorySection
      id="problem"
      kicker={t("The problem", "הבעיה")}
      title={
        he ? (
          <>
            על תקלות מגלים רק <span className="text-risk">אחרי</span> שהנזק כבר נגרם.
          </>
        ) : (
          <>
            Failures are discovered <span className="text-risk">after</span> the damage is done.
          </>
        )
      }
      lede={
        data && failures && perDay ? (
          he ? (
            <>
              בצי הכוננים שחקרנו, {fmtInt(failures)} מתוך {fmtInt(data.scale.drives)} כוננים התקלקלו במשך {(data.scale.quarters.length / 4).toFixed(1)}{" "}
              שנים, כ־{Math.floor(perDay)} בכל יום. רובם מוחלפים רק אחרי שכבר הפסיקו לעבוד.
            </>
          ) : (
            <>
              In the fleet we studied, {fmtInt(failures)} of {fmtInt(data.scale.drives)} drives failed over{" "}
              {(data.scale.quarters.length / 4).toFixed(1)} years — about {Math.floor(perDay)} every day. Most are replaced only once they have
              already stopped working.
            </>
          )
        ) : (
          t(
            "Large storage fleets lose drives every day. Most are replaced only once they have already stopped working.",
            "בציי אחסון גדולים כוננים מתקלקלים כל יום. רובם מוחלפים רק אחרי שכבר הפסיקו לעבוד.",
          )
        )
      }
    >
      <ProblemVisual />
    </StorySection>
  );
}
