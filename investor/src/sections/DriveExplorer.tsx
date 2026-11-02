import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertTriangle, CheckCircle2, Download, FileSpreadsheet, HelpCircle, Info, Upload, Wrench, X } from "lucide-react";
import type { DemoDataset, DriveDetail, DriveRow, EstimatorKind, Prediction } from "../../shared/types";
import { RiskTrajectoryChart } from "../components/RiskTrajectoryChart";
import { SHAPExplanation } from "../components/SHAPExplanation";
import { ErrorState, Loading } from "../components/States";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { api, TEMPLATE_URL } from "../lib/api";
import { fmtInt, riskCategory, toneColor } from "../lib/format";
import { EASE } from "../lib/motion";
import { smartName, smartPlain } from "../lib/smart";
import { useLang, type Lang } from "../lib/i18n";
import { useApi, type Async } from "../lib/useApi";

const plural = (n: number, word: string) => `${fmtInt(n)} ${word}${n === 1 ? "" : "s"}`;
/** "1 drive" / "12 drives", or in Hebrew "כונן אחד" / "12 כוננים". */
const drives = (n: number, lang: Lang) => (lang === "he" ? (n === 1 ? "כונן אחד" : `${fmtInt(n)} כוננים`) : plural(n, "drive"));

const GROUPS: { id: DemoDataset["group"]; title: { en: string; he: string } }[] = [
  { id: "demo", title: { en: "Demo fleets", he: "ציי הדגמה" } },
  { id: "export", title: { en: "Real 90-day exports", he: "ייצוא אמיתי של 90 יום" } },
  { id: "example", title: { en: "Engineering test files", he: "קובצי בדיקה הנדסיים" } },
];

/** Hebrew names for the datasets the server lists (it speaks English only). */
const HE_DATASETS: Record<string, [string, string]> = {
  "demo:03_fleet_60_drives.xlsx": ["צי של 60 כוננים", "60 כוננים, 20 מהם התקלקלו"],
  "demo:04_live_snapshot.xlsx": ["תמונת מצב חיה, בלי תשובות", "40 כוננים נכון ל־2026-03-31, בלי עמודת תקלות"],
  "demo:02_failing_drive.csv": ["כונן אחד בחודש האחרון שלו", "כונן אחד, 30 הימים האחרונים שלו"],
  "demo:01_healthy_drive.csv": ["כונן תקין אחד", "כונן אחד, 30 ימים תקינים"],
  "demo:05_short_windows.xlsx": ["כוננים עם היסטוריה קצרה", "14 כוננים עם 10 עד 30 ימים"],
  "demo:06_renamed_and_sparse.csv": ["כותרות שונות ונתונים חלקיים", "כותרות timestamp/serial, רק 6 מתוך 15 מדדים"],
  "example:examples/01_single_drive_healthy.xlsx": ["כונן תקין", "המקרה השקט: כונן אחד, 30 ימים תקינים, ציון קרוב לאפס"],
  "example:examples/02_single_drive_failing.xlsx": ["כונן שהתקלקל", "המקרה הרועש: 30 הימים האחרונים של כונן שהתקלקל, ציון קרוב לאחד"],
  "example:examples/03_small_fleet_mixed.xlsx": ["צי מעורב", "דירוג של 12 כוננים; הקובץ כולל גם עמודות חסומות שהמודל לא קורא"],
  "example:examples/04_partial_windows.xlsx": ["חלונות חלקיים", "כוננים שהדיווח שלהם נקטע באמצע החלון"],
  "example:examples/05_missing_smart_columns.xlsx": ["עמודות חסרות", "אותם 12 כוננים בלי 10 מתוך 15 המדדים: הציון עדיין מחושב, עם אזהרה"],
  "example:examples/06_renamed_columns.xlsx": ["כותרות בשמות אחרים", "עמודות time/serial ובלי תשובות, כמו קובץ חדש לגמרי"],
};

function dsText(d: DemoDataset, lang: Lang): { label: string; description: string } {
  if (lang !== "he") return { label: d.label, description: d.description };
  const known = HE_DATASETS[d.id];
  if (known) return { label: known[0], description: known[1] };
  const label = d.label.replace(/ · 90 days$/, " · 90 יום");
  const m = d.description.match(/^real export, (\S+) → (\S+): (\d+) drives, (\d+) recorded failures$/);
  const description = m ? `ייצוא אמיתי, ${m[1]} עד ${m[2]}: ${m[3]} כוננים, ${m[4]} תקלות מתועדות` : d.description;
  return { label, description };
}

/** The engine's notes about the file ("read column 'serial' as 'serial_number'"). */
function noteText(n: string, lang: Lang): string {
  if (lang !== "he") return n;
  const m = n.match(/^read column '(.+)' as '(.+)'$/);
  return m ? `העמודה '${m[1]}' נקראה כ־'${m[2]}'` : n;
}

// ---------------------------------------------------------------- choosing data

const ACCEPT = ".xlsx,.xlsm,.xls,.csv,.txt,.parquet";

/** The primary way in: drop or choose your own telemetry file. */
function UploadCard({
  busy,
  progress,
  lastUpload,
  onFile,
}: {
  busy: boolean;
  progress: number | null;
  lastUpload: { name: string; drives: number } | null;
  onFile: (f: File) => void;
}) {
  const [drag, setDrag] = useState(false);
  const [help, setHelp] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const choose = () => !busy && input.current?.click();
  const { t, lang } = useLang();

  return (
    <div id="upload" className="panel scroll-mt-24 border-signal/30">
      <p className="kicker mb-1.5 text-[10px]">{t("Option A · your own data", "אפשרות א׳ · הנתונים שלכם")}</p>
      <h3 className="text-lg font-semibold">{t("Upload your drive telemetry", "העלו את נתוני הכוננים שלכם")}</h3>
      <p className="mt-1 text-sm text-muted">{t("Daily SMART readings for one drive or a whole fleet. Results in seconds.", "קריאות SMART יומיות של כונן אחד או של צי שלם. תוצאות תוך שניות.")}</p>

      <div
        role="button"
        tabIndex={0}
        aria-label={t("Upload a telemetry file: drop it here, or press Enter to choose one", "העלאת קובץ נתונים: גררו אותו לכאן, או לחצו Enter כדי לבחור")}
        onClick={choose}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            choose();
          }
        }}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          const f = e.dataTransfer.files[0];
          if (f && !busy) onFile(f);
        }}
        className={`mt-4 flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-5 py-8 text-center transition ${
          drag ? "border-signal bg-signal/10" : "border-signal/35 bg-signal/[0.03] hover:border-signal/70 hover:bg-signal/[0.06]"
        } ${busy ? "pointer-events-none opacity-80" : ""}`}
      >
        <span className="flex h-12 w-12 items-center justify-center rounded-full bg-signal/15 text-signal">
          <Upload size={22} aria-hidden />
        </span>
        {progress !== null ? (
          <div className="mt-4 w-full max-w-xs">
            <p className="text-sm text-ink">{progress < 1 ? t(`Uploading… ${Math.round(progress * 100)}%`, `מעלה… ${Math.round(progress * 100)}%`) : t("Scoring with the live engine…", "מחשב ציונים במנוע החי…")}</p>
            <div dir="ltr" className="mt-3 h-1 rounded bg-deep">
              <div className="h-1 rounded bg-signal transition-all" style={{ width: `${progress * 100}%` }} />
            </div>
          </div>
        ) : (
          <>
            <p className="mt-4 text-[15px] font-medium text-ink">{drag ? t("Drop to score this file", "שחררו כדי לנתח את הקובץ") : t("Drag a file here", "גררו קובץ לכאן")}</p>
            <p className="mt-1 text-sm text-muted">{t("or", "או")}</p>
            <span className="btn btn-primary pointer-events-none mt-3">{t("Choose a file", "בחירת קובץ")}</span>
            <p className="mt-4 text-xs text-muted">
              {t(".xlsx · .csv · .parquet · up to 400 MB · processed in memory, never stored", ".xlsx · .csv · .parquet · עד 400MB · מעובד בזיכרון, לא נשמר")}
            </p>
          </>
        )}
      </div>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        className="sr-only"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />

      {lastUpload && progress === null && (
        <p className="mt-3 flex items-center gap-2 text-sm text-ok">
          <CheckCircle2 size={15} aria-hidden /> <span dir="ltr">{lastUpload.name}</span>{" "}
          {t(`scored · ${plural(lastUpload.drives, "drive")}. Results are below.`, `נותח · ${drives(lastUpload.drives, "he")}. התוצאות למטה.`)}
        </p>
      )}

      <div className="mt-4 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
        <a href={TEMPLATE_URL} download className="inline-flex items-center gap-1.5 text-signal hover:underline">
          <Download size={14} aria-hidden /> {t("Download a template", "הורדת תבנית")}
        </a>
        <button type="button" onClick={() => setHelp((h) => !h)} aria-expanded={help} className="inline-flex items-center gap-1.5 text-soft hover:text-ink">
          <HelpCircle size={14} aria-hidden /> {t("What should the file contain?", "מה צריך להיות בקובץ?")}
        </button>
      </div>
      {help && (
        <div className="mt-4 space-y-2 rounded-lg border hairline bg-void/40 p-4 text-xs leading-relaxed text-soft">
          {lang === "he" ? (
            <>
              <p>
                <b className="text-ink">שורה אחת לכל כונן בכל יום.</b> עמודות חובה: <code className="text-signal">date</code> ו־
                <code className="text-signal">serial_number</code> (מתקבלים גם time או timestamp, serial או drive_id).
              </p>
              <p>
                <b className="text-ink">מדדי SMART</b> כעמודות בשם <code>smart_N_raw</code>, למשל <code>smart_5_raw</code> ו־<code>smart_197_raw</code>.
                המודל קורא 15 מדדים; מדדים חסרים מושלמים אוטומטית, והעמוד מציין אילו.
              </p>
              <p>
                <b className="text-ink">היסטוריה:</b> 30 ימים לכל כונן נותנים חיזוי מלא. כוננים עם 10 עד 29 ימים עדיין מקבלים ציון ומסומנים כחלקיים.
                עמודות רשות: <code>model</code>, ו־<code>failure</code> (0/1) כדי לבדוק את הדירוג מול מה שקרה בפועל.
              </p>
              <p className="text-muted">אוספים עם smartctl? בתיקייה demo/from_smartctl/ יש סקריפט יומי שכותב בדיוק את הפורמט הזה.</p>
            </>
          ) : (
            <>
          <p>
            <b className="text-ink">One row per drive per day.</b> Required columns: <code className="text-signal">date</code> and{" "}
            <code className="text-signal">serial_number</code> (also accepted: time or timestamp, serial or drive_id).
          </p>
          <p>
            <b className="text-ink">SMART attributes</b> as <code>smart_N_raw</code> columns, such as <code>smart_5_raw</code> and{" "}
            <code>smart_197_raw</code>. The model reads 15; any that are missing are filled automatically, and the page says which.
          </p>
          <p>
            <b className="text-ink">History:</b> 30 days per drive gives a full prediction. Drives with 10 to 29 days are still scored and marked as
            partial. Optional columns: <code>model</code>, and <code>failure</code> (0/1) to check the ranking against what really happened.
          </p>
          <p className="text-muted">Collecting with smartctl? demo/from_smartctl/ has a daily collector that writes this format.</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** The other way in: a ready-made dataset, for anyone without a file at hand. */
function SamplePicker({
  demos,
  selected,
  busy,
  onPick,
}: {
  demos: Async<DemoDataset[]>;
  selected: string | null;
  busy: boolean;
  onPick: (d: DemoDataset) => void;
}) {
  const { data, error, loading, reload } = demos;
  // Three picks cover the story: a fleet to rank, one drive that failed, and a real
  // 90-day export with a long enough history for risk trajectories.
  const quickIds = [
    "demo:03_fleet_60_drives.xlsx",
    "demo:02_failing_drive.csv",
    data?.find((d) => d.group === "export" && d.drives >= 20)?.id ?? data?.find((d) => d.group === "export")?.id,
  ];
  const quick = quickIds.map((id) => data?.find((d) => d.id === id)).filter((d): d is DemoDataset => !!d);
  const more = data?.filter((d) => !quick.includes(d)) ?? [];
  const moreSelected = more.find((d) => d.id === selected)?.id ?? "";
  const { t, lang } = useLang();

  return (
    <div className="panel flex min-w-0 flex-col">
      <p className="kicker mb-1.5 text-[10px]">{t("Option B · sample data", "אפשרות ב׳ · נתוני דוגמה")}</p>
      <h3 className="text-lg font-semibold">{t("No file at hand? Try a sample", "אין לכם קובץ? נסו דוגמה")}</h3>
      <p className="mt-1 text-sm text-muted">{t("Real drive telemetry, scored with one click.", "נתונים אמיתיים של כוננים, ניתוח בלחיצה אחת.")}</p>
      {loading && <Loading label={t("Listing datasets", "טוען רשימת קבצים")} className="mt-5" />}
      {error && <ErrorState message={error} onRetry={reload} className="mt-5" />}
      <ul className="mt-4 grid min-w-0 grid-cols-1 gap-2">
        {quick.map((d) => {
          const on = d.id === selected;
          return (
            <li key={d.id} className="min-w-0">
              <button
                type="button"
                disabled={busy}
                onClick={() => onPick(d)}
                aria-pressed={on}
                className={`h-full w-full rounded-lg border px-3 py-2.5 text-start transition disabled:opacity-60 ${
                  on ? "border-signal/60 bg-signal/10" : "border-line hover:border-line-strong hover:bg-deep/60"
                }`}
              >
                <span className="flex items-baseline justify-between gap-2">
                  <span className={`text-sm ${on ? "text-ink" : "text-soft"}`}>{dsText(d, lang).label}</span>
                  {on && <CheckCircle2 size={14} className="shrink-0 text-signal" aria-hidden />}
                </span>
                <span className="mt-0.5 block truncate text-xs text-muted">{dsText(d, lang).description}</span>
              </button>
            </li>
          );
        })}
      </ul>
      {more.length > 0 && (
        <label className="mt-4 block text-xs text-muted">
          {t("More datasets", "קבצים נוספים")}
          <select
            value={moreSelected}
            disabled={busy}
            onChange={(e) => {
              const d = more.find((m) => m.id === e.target.value);
              if (d) onPick(d);
            }}
            className="mt-1.5 block w-full rounded-lg border border-line bg-night px-3 py-2 text-sm text-ink"
          >
            <option value="">{t("Choose another dataset…", "בחרו קובץ אחר…")}</option>
            {GROUPS.map((g) => (
              <optgroup key={g.id} label={g.title[lang]}>
                {more
                  .filter((d) => d.group === g.id)
                  .map((d) => (
                    <option key={d.id} value={d.id}>
                      {dsText(d, lang).label} ({drives(d.drives, lang)})
                    </option>
                  ))}
              </optgroup>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}

const HOW = [
  { n: 1, t: { en: "Choose data", he: "בוחרים נתונים" }, d: { en: "Upload a file or pick a sample", he: "מעלים קובץ או בוחרים דוגמה" } },
  { n: 2, t: { en: "See the ranking", he: "רואים את הדירוג" }, d: { en: "Every drive scored, riskiest first", he: "כל כונן מקבל ציון, המסוכנים ראשונים" } },
  { n: 3, t: { en: "Open a drive", he: "פותחים כונן" }, d: { en: "Its trend and why it was flagged", he: "המגמה שלו ולמה הוא סומן" } },
];

// ---------------------------------------------------------------- ranked queue

function RiskBar({ score, threshold, tone }: { score: number; threshold: number; tone: string }) {
  return (
    <span dir="ltr" className="relative block h-1.5 w-full rounded-full bg-deep">
      <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.max(2, score * 100)}%`, background: tone }} />
      <span className="absolute -top-1 h-3.5 w-px bg-soft/70" style={{ left: `${threshold * 100}%` }} title={`threshold ${threshold.toFixed(3)}`} />
    </span>
  );
}

function Queue({ p, selected, onSelect }: { p: Prediction; selected: string | null; onSelect: (r: DriveRow) => void }) {
  const thr = p.thresholds[p.primary] ?? 0.5;
  const truth = p.diagnostics.has_failure_column;
  const { t, lang } = useLang();
  return (
    <div className="max-h-[560px] overflow-y-auto pe-1" role="listbox" aria-label={t("Drives ranked by risk", "כוננים מדורגים לפי סיכון")}>
      {p.rows.map((r) => {
        const cat = riskCategory(r.risk, thr, lang);
        const on = r.serial === selected;
        return (
          <button
            key={r.serial}
            role="option"
            aria-selected={on}
            onClick={() => onSelect(r)}
            className={`grid w-full grid-cols-[2.2rem_1fr_auto] items-center gap-3 rounded-lg px-3 py-2.5 text-start transition ${
              on ? "bg-deep ring-1 ring-line-strong" : "hover:bg-deep/50"
            }`}
          >
            <span className="num text-xs text-faint">{r.rank}</span>
            <span className="min-w-0">
              <span className="flex items-center gap-2">
                <span className="num truncate text-[13px] text-soft">{r.serial}</span>
                {truth && r.recorded_failure === 1 && <span className="rounded bg-risk/15 px-1.5 text-[10px] text-risk">{t("failed", "התקלקל")}</span>}
                {r.n_padded_days > 0 && <span className="rounded bg-warn/10 px-1.5 text-[10px] text-warn">{t(`${r.n_real_days}d`, `${r.n_real_days} י׳`)}</span>}
              </span>
              <span className="mt-1.5 block">
                <RiskBar score={r.risk} threshold={thr} tone={toneColor[cat.tone]} />
              </span>
            </span>
            <span className="w-16 text-end">
              <span className="num block text-sm" style={{ color: toneColor[cat.tone] }}>
                {r.risk.toFixed(3)}
              </span>
              <span className="block text-[10px] uppercase tracking-wider text-faint">{cat.call}</span>
            </span>
          </button>
        );
      })}
      {p.n_rows_total > p.rows.length && (
        <p className="px-3 py-3 text-xs text-muted">
          {t(`Showing the top ${p.rows.length} of ${fmtInt(p.n_rows_total)} drives.`, `מוצגים ${p.rows.length} המובילים מתוך ${fmtInt(p.n_rows_total)} כוננים.`)}
        </p>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- drive profile

function Telemetry({ d }: { d: DriveDetail }) {
  const moving = d.attributes.filter((a) => {
    const v = (d.series[a] ?? []).filter((x): x is number => x !== null);
    return v.length && Math.max(...v) !== Math.min(...v);
  });
  const options = [...new Set([...["smart_197_raw", "smart_198_raw", "smart_5_raw"].filter((a) => d.attributes.includes(a)), ...moving])];
  const [attr, setAttr] = useState(options[0] ?? d.attributes[0]);
  const data = d.dates.map((date, i) => ({ date, v: d.series[attr]?.[i] ?? null }));
  const { t, lang } = useLang();
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <label className="flex items-center gap-2 text-xs text-muted">
          {t("Attribute", "מדד")}
          <select
            value={attr}
            onChange={(e) => setAttr(e.target.value)}
            className="rounded-lg border border-line bg-night px-2 py-1 text-xs text-ink"
          >
            {d.attributes.map((a) => (
              <option key={a} value={a}>
                {smartName(a, lang)}
                {moving.includes(a) ? " ·" : ""}
              </option>
            ))}
          </select>
        </label>
        <span className="text-[11px] text-faint">
          {smartPlain(attr, lang)} · {t("“·” = changed in this window", "“·” = השתנה בחלון הזה")}
        </span>
      </div>
      <div className="h-40" dir="ltr">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, left: -14, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={40} tickFormatter={(x: string) => x.slice(5)} />
            <YAxis tickLine={false} axisLine={false} width={50} allowDecimals={false} />
            <Tooltip
              contentStyle={{ background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 }}
              formatter={(v) => [v === null ? t("not reported", "לא דווח") : String(v), smartName(attr, lang)]}
            />
            <Line type="stepAfter" dataKey="v" stroke="var(--color-signal)" strokeWidth={2} dot={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Profile({ p, serial }: { p: Prediction; serial: string }) {
  const [state, setState] = useState<{ d: DriveDetail | null; error: string | null }>({ d: null, error: null });
  const load = useCallback(() => {
    setState({ d: null, error: null });
    api.drive(p.token, serial).then(
      (d) => setState({ d, error: null }),
      (e: Error) => setState({ d: null, error: e.message }),
    );
  }, [p.token, serial]);
  useEffect(load, [load]);
  const { t, lang } = useLang();

  const { d, error } = state;
  if (error) return <ErrorState message={error} onRetry={load} />;
  if (!d) return <Loading label={t("Opening drive", "פותח את הכונן")} className="py-10" />;

  const thr = p.thresholds[p.primary] ?? 0.5;
  const cat = riskCategory(d.risk, thr, lang);
  const Icon = cat.tone === "ok" ? CheckCircle2 : cat.tone === "warn" ? Wrench : AlertTriangle;

  return (
    <motion.div key={serial} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4, ease: EASE }} className="space-y-10">
      <div className="space-y-5">
        <div>
          <p className="num text-xs text-faint">
            #{d.rank} · {d.drive_model}
          </p>
          <p className="num mt-1 text-lg text-soft">{d.serial}</p>
          <p className="num mt-4 text-6xl leading-none" style={{ color: toneColor[cat.tone] }}>
            {d.risk.toFixed(3)}
          </p>
          <p className="mt-2 text-sm text-muted">{t(`risk score · failure within the next ${p.horizon_days} days`, `ציון סיכון · תקלה ב־${p.horizon_days} הימים הקרובים`)}</p>
        </div>
        <div className="space-y-2">
          <span
            className="inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-sm"
            style={{ borderColor: toneColor[cat.tone], color: toneColor[cat.tone] }}
          >
            <Icon size={14} aria-hidden /> {cat.label} · {cat.call}
          </span>
          <p className="num text-xs text-muted">
            {t("window", "חלון")} <span dir="ltr">{d.window_start} → {d.window_end}</span>
          </p>
          {p.kinds.length > 1 && (
            <p className="num text-xs text-muted">
              {p.kinds.map((k) => `${k === "xgb" ? "XGBoost" : "CNN"} ${d.scores[k]?.toFixed(3)}`).join(" · ")}
            </p>
          )}
          {d.n_padded_days > 0 && (
            <p className="text-xs text-warn">
              {t(
                `only ${d.n_real_days} real days; ${d.n_padded_days} padded — an extrapolation`,
                `רק ${d.n_real_days} ימים אמיתיים; ${d.n_padded_days} הושלמו — זו הערכה`,
              )}
            </p>
          )}
          {p.diagnostics.has_failure_column && (
            <p className="text-xs text-muted">
              {t("recorded outcome in file:", "התוצאה שמתועדת בקובץ:")} {d.recorded_failure ? t("failed", "התקלקל") : t("did not fail", "לא התקלקל")}
            </p>
          )}
        </div>
      </div>

      <section aria-label={t("Risk trajectory", "מגמת הסיכון")}>
        <h4 className="mb-1 text-sm font-medium text-ink">{t("Risk trajectory", "מגמת הסיכון")}</h4>
        <p className="mb-4 text-xs text-muted">{t("The same model, re-scored as if it were asked on each earlier day.", "אותו מודל, שמחשב ציון כאילו נשאל בכל אחד מהימים הקודמים.")}</p>
        <RiskTrajectoryChart trajectory={d.trajectory} kinds={p.kinds} thresholds={p.thresholds} />
      </section>

      <section aria-label={t("Why this score", "למה הציון הזה")}>
        <h4 className="mb-1 text-sm font-medium text-ink">{t("Why this score", "למה הציון הזה")}</h4>
        {d.explanations.xgb ? (
          <>
            <p className="mb-4 text-xs text-muted">
              {t("The signals that moved this drive's XGBoost score most, exact for this prediction.", "האותות שהכי השפיעו על ציון ה־XGBoost של הכונן הזה, מדויק לחיזוי הזה.")}
            </p>
            <SHAPExplanation contributions={d.explanations.xgb} />
          </>
        ) : (
          <p className="text-xs text-muted">{t("The CNN does not offer per-feature explanations. Add XGBoost to see one.", "ה־CNN לא נותן הסבר לפי מאפיינים. הוסיפו XGBoost כדי לראות הסבר.")}</p>
        )}
      </section>

      <section aria-label={t("Telemetry", "נתוני הכונן")}>
        <h4 className="mb-1 text-sm font-medium text-ink">{t("Telemetry behind the score", "הנתונים שמאחורי הציון")}</h4>
        <p className="mb-4 text-xs text-muted">{t("The raw daily readings from your file, in the units it reported.", "הקריאות היומיות הגולמיות מהקובץ, ביחידות שבהן דווחו.")}</p>
        <Telemetry d={d} />
      </section>
    </motion.div>
  );
}

// ---------------------------------------------------------------- the explorer

/** Artifact F: a real prediction workflow against the live engine. */
export function DriveExplorer() {
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [kinds, setKinds] = useState<EstimatorKind[]>(["xgb"]);
  const [pending, setPending] = useState<{ label: string; progress: number | null } | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [last, setLast] = useState<{ kind: "demo"; d: DemoDataset } | { kind: "file"; f: File } | null>(null);
  const profileRef = useRef<HTMLDivElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);
  const [lastUpload, setLastUpload] = useState<{ name: string; drives: number } | null>(null);

  const run = useCallback(async (src: { kind: "demo"; d: DemoDataset } | { kind: "file"; f: File }, models: EstimatorKind[]) => {
    setLast(src);
    setError(null);
    setDatasetId(src.kind === "demo" ? src.d.id : null);
    setPending({ label: src.kind === "demo" ? src.d.label : src.f.name, progress: src.kind === "file" ? 0 : null });
    try {
      const p =
        src.kind === "demo"
          ? await api.predictDemo(src.d.id, models)
          : await api.predictFile(src.f, models, (f) => setPending((s) => (s ? { ...s, progress: f } : s)));
      setResult(p);
      setSelected(p.rows[0]?.serial ?? null);
      if (src.kind === "file") {
        setLastUpload({ name: src.f.name, drives: p.diagnostics.drives_scored });
        // Someone who just uploaded wants to see the answer, not the upload card.
        requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      }
    } catch (e) {
      setError((e as Error).message);
      if (src.kind === "file") requestAnimationFrame(() => resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
    } finally {
      setPending(null);
    }
  }, []);

  // Open on the fleet demo so the explorer is never empty.
  const booted = useRef(false);
  const demos = useApi(api.demos);
  useEffect(() => {
    if (booted.current || !demos.data?.length) return;
    booted.current = true;
    const first = demos.data.find((d) => d.id === "demo:03_fleet_60_drives.xlsx") ?? demos.data[0];
    void run({ kind: "demo", d: first }, ["xgb"]);
  }, [demos.data, run]);

  const toggleCnn = () => {
    const next: EstimatorKind[] = kinds.includes("cnn") ? ["xgb"] : ["xgb", "cnn"];
    setKinds(next);
    if (last) void run(last, next);
  };

  const pick = (r: DriveRow) => {
    setSelected(r.serial);
    if (window.matchMedia("(max-width: 1023px)").matches) profileRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const p = result;
  const q = p?.quality?.[p.primary];
  const { t, lang } = useLang();

  return (
    <div className="space-y-4">
      <ol className="grid gap-2 sm:grid-cols-3" aria-label={t("How to use the demo", "איך משתמשים בהדגמה")}>
        {HOW.map((h) => (
          <li key={h.n} className="flex items-center gap-3 rounded-xl border hairline bg-night/60 px-4 py-3">
            <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-signal/50 text-xs font-semibold text-signal">
              {h.n}
            </span>
            <span className="min-w-0">
              <span className="block text-sm font-medium text-ink">{h.t[lang]}</span>
              <span className="block truncate text-xs text-muted">{h.d[lang]}</span>
            </span>
          </li>
        ))}
      </ol>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1.1fr_1fr]">
        <UploadCard
          busy={!!pending}
          progress={pending?.progress ?? null}
          lastUpload={lastUpload}
          onFile={(f) => run({ kind: "file", f }, kinds)}
        />
        <SamplePicker demos={demos} selected={datasetId} busy={!!pending} onPick={(d) => run({ kind: "demo", d }, kinds)} />
      </div>

      <div ref={resultsRef} className="panel min-w-0 scroll-mt-20">
        {/* toolbar */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-4 border-b hairline pb-5">
          <div className="min-w-0">
            <p className="flex flex-wrap items-center gap-2 text-sm text-soft">
              <FileSpreadsheet size={14} className="shrink-0 text-muted" aria-hidden />
              <span className="truncate" dir={pending || p ? "ltr" : undefined}>{pending ? pending.label : (p?.source ?? t("No file scored yet", "עדיין לא נותח קובץ"))}</span>
              {p && !pending && (
                <span className={`rounded-full px-2 py-0.5 text-[11px] ${last?.kind === "file" ? "bg-signal/15 text-signal" : "bg-deep text-muted"}`}>
                  {last?.kind === "file" ? t("Your file", "הקובץ שלכם") : t("Sample data", "נתוני דוגמה")}
                </span>
              )}
            </p>
            {p && !pending && (
              <p className="mt-1 text-xs text-muted">
                {t(
                  `${fmtInt(p.diagnostics.drives_scored)} drives scored in ${p.seconds < 1 ? "under a second" : `${p.seconds}s`} · ${p.summary[p.primary]?.inspect ?? 0} to inspect`,
                  `${fmtInt(p.diagnostics.drives_scored)} כוננים נותחו ${p.seconds < 1 ? "בפחות משנייה" : `ב־${p.seconds} שניות`} · ${p.summary[p.primary]?.inspect ?? 0} לבדיקה`,
                )}{" "}
                · <span dir="ltr">{p.diagnostics.date_min} → {p.diagnostics.date_max}</span>
              </p>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-4">
          <a href="#upload" className="inline-flex items-center gap-1.5 rounded-full border border-signal/40 px-3 py-1.5 text-xs text-signal hover:bg-signal/10">
            <Upload size={13} aria-hidden />{" "}
            {last?.kind === "file" ? t("Upload another file", "העלאת קובץ נוסף") : t("Upload your own file", "העלאת קובץ משלכם")}
          </a>
          <label className="flex cursor-pointer items-center gap-2 text-xs text-muted">
            <input type="checkbox" checked={kinds.includes("cnn")} onChange={toggleCnn} disabled={!!pending} className="accent-[#8b6cf0]" />
            {t("Also run the CNN side by side", "להריץ גם את ה־CNN במקביל")}
          </label>
          </div>
        </div>

        <AnimatePresence>
          {pending && pending.progress === null && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mb-6">
              <Loading label={t("Scoring with the live engine", "מחשב ציונים במנוע החי")} />
              {pending.progress !== null && (
                <div className="mt-3 h-0.5 rounded bg-deep">
                  <div className="h-0.5 rounded bg-signal transition-all" style={{ width: `${pending.progress * 100}%` }} />
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>

        {error && (
          <div className="mb-6 flex items-start gap-3">
            <ErrorState message={error} className="flex-1" />
            <button type="button" aria-label={t("Dismiss", "סגירה")} onClick={() => setError(null)} className="mt-3 text-muted hover:text-ink">
              <X size={16} />
            </button>
          </div>
        )}

        {p && (
          <>
            {(p.diagnostics.attributes_missing.length > 0 || p.diagnostics.drives_skipped > 0 || p.diagnostics.drives_padded > 0 || p.diagnostics.notes.length > 0) && (
              <div className="mb-6 flex gap-3 rounded-xl border border-warn/25 bg-warn/5 p-4 text-xs text-soft">
                <Info size={14} className="mt-0.5 shrink-0 text-warn" aria-hidden />
                <div className="space-y-1">
                  {p.diagnostics.attributes_missing.length > 0 && (
                    <p>
                      {t(
                        `${p.diagnostics.attributes_missing.length} of 15 attributes were missing and filled with the training median:`,
                        `${p.diagnostics.attributes_missing.length} מתוך 15 המדדים היו חסרים והושלמו לפי החציון מהאימון:`,
                      )}{" "}
                      {p.diagnostics.attributes_missing.map((a) => smartName(a, lang)).join(", ")}.
                    </p>
                  )}
                  {p.diagnostics.drives_padded > 0 && (
                    <p>
                      {lang === "he"
                        ? `${drives(p.diagnostics.drives_padded, "he")} עם פחות מ־${p.window_days} ימים הושלמו; הציון שלהם הוא הערכה.`
                        : `${plural(p.diagnostics.drives_padded, "drive")} had less than ${p.window_days} days and ${p.diagnostics.drives_padded === 1 ? "was" : "were"} padded; ${p.diagnostics.drives_padded === 1 ? "its score is an extrapolation" : "their scores are extrapolations"}.`}
                    </p>
                  )}
                  {p.diagnostics.drives_skipped > 0 && (
                    <p>
                      {lang === "he"
                        ? `${drives(p.diagnostics.drives_skipped, "he")} עם היסטוריה קצרה מדי לא נותחו.`
                        : `${plural(p.diagnostics.drives_skipped, "drive")} had too little history to score and ${p.diagnostics.drives_skipped === 1 ? "was" : "were"} skipped.`}
                    </p>
                  )}
                  {p.diagnostics.notes.map((n) => (
                    <p key={n}>{noteText(n, lang)}.</p>
                  ))}
                </div>
              </div>
            )}

            {q && (
              <div className="mb-8 rounded-xl border hairline bg-night/60 p-5">
                <p className="text-sm text-soft">
                  {t(
                    `This file records which drives later failed (${q.n_failed} of ${q.n_drives}). Ranked by the model:`,
                    `הקובץ הזה מתעד אילו כוננים התקלקלו בהמשך (${q.n_failed} מתוך ${q.n_drives}). בדירוג של המודל:`,
                  )}{" "}
                  <span dir="ltr" className="inline-block">
                    {q.at_k.map((a, i) => (
                      <span key={a.k} className="num text-ink">
                        {i ? " · " : ""}
                        {a.hits}/{a.k}
                      </span>
                    ))}
                  </span>{" "}
                  {t("of the top-K were real failures.", "מהמובילים היו תקלות אמיתיות.")}
                </p>
                <p className="mt-2 text-xs text-muted">
                  {t(
                    "The model never sees that column. It describes this one file only and is not an evaluation; the Results section is.",
                    "המודל אף פעם לא רואה את העמודה הזו. זה מתאר רק את הקובץ הזה ואינו הערכה; ההערכה נמצאת בחלק התוצאות.",
                  )}
                </p>
              </div>
            )}

            <p className="mb-3 text-xs text-muted">
              {t(
                "Drives are ranked from highest to lowest risk. The tick on each bar is the model's inspect threshold. Select a drive to see why it scored as it did.",
                "הכוננים מדורגים מהסיכון הגבוה לנמוך. הקו הקטן על כל פס הוא סף הבדיקה של המודל. בחרו כונן כדי לראות למה קיבל את הציון שלו.",
              )}
            </p>
            <div className="grid grid-cols-1 gap-10 xl:grid-cols-[minmax(0,320px)_1fr]">
              <Queue key={p.token} p={p} selected={selected} onSelect={pick} />
              <div ref={profileRef} className="min-w-0 scroll-mt-20">
                {selected ? <Profile key={`${p.token}-${selected}`} p={p} serial={selected} /> : <p className="text-sm text-muted">{t("Select a drive.", "בחרו כונן.")}</p>}
              </div>
            </div>
          </>
        )}

        <TechnicalDisclosure title={t("What happens when a file is scored", "מה קורה כשקובץ מנותח")} className="mt-8">
          {lang === "he" ? (
            <>
              <p>
                הדפדפן שולח את הקובץ לשרת ה־Node, שמעביר אותו כזרם למנוע ה־Flask הקיים (<code>webapp/app.py</code>). שמות העמודות מותאמים למה שהמודל
                מצפה לו, ההיסטוריה של כל כונן נפרסת על לוח יומי, הנרמול שנשמר באימון מופעל שוב, וחלון 30 הימים האחרון של כל כונן מקבל ציון מהמודלים
                השמורים בתיקייה <code>models/</code>. שום דבר לא נכתב לדיסק ושום דבר לא יוצא מהמחשב.
              </p>
              <p>
                קטגוריות הסיכון: <b>לבדיקה</b> היא ההחלטה של המודל עצמו (ציון בסף שנקבע או מעליו). העמוד מפצל את “לבדיקה” ב־0.5 ל<b>מוגבר</b> ו<b>גבוה</b>{" "}
                כדי ש־0.25 ו־0.99 לא ייראו אותו דבר; הפיצול הוא רק לתצוגה ולא חלק מהמודל.
              </p>
            </>
          ) : (
            <>
          <p>
            The browser sends the file to the Node API, which streams it to the existing Flask engine (<code>webapp/app.py</code>).
            Column names are matched to the model's contract, each drive's history is laid on a daily calendar, the scaler
            saved at training time is replayed, and each drive's most recent 30-day window is scored by the saved models in{" "}
            <code>models/</code>. Nothing is written to disk and nothing leaves the machine.
          </p>
          <p>
            Risk categories: <b>inspect</b> is the model's own call (score at or above its tuned threshold). The page splits
            “inspect” at 0.5 into <b>elevated</b> and <b>high</b> so a 0.25 and a 0.99 do not read alike; that split is
            presentation, not part of the model.
          </p>
            </>
          )}
        </TechnicalDisclosure>
      </div>
    </div>
  );
}
