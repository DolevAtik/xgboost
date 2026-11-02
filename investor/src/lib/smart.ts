import type { Lang } from "./i18n";

/** Plain-language names for the fifteen SMART attributes the model reads, in both languages. */
export const SMART: Record<string, { name: string; plain: string; heName: string; hePlain: string; smartctl: string }> = {
  smart_1_raw: { name: "Read errors", plain: "how often reading data needed a retry", heName: "שגיאות קריאה", hePlain: "כמה פעמים קריאת נתונים דרשה ניסיון חוזר", smartctl: "Raw_Read_Error_Rate" },
  smart_3_raw: { name: "Spin-up time", plain: "how long the platters take to reach speed", heName: "זמן הגעה למהירות", hePlain: "כמה זמן לוקח לדיסקים להגיע למהירות", smartctl: "Spin_Up_Time" },
  smart_4_raw: { name: "Start/stop count", plain: "how many times the motor has started", heName: "הפעלות מנוע", hePlain: "כמה פעמים המנוע הופעל", smartctl: "Start_Stop_Count" },
  smart_5_raw: { name: "Reallocated sectors", plain: "damaged spots the drive has already retired", heName: "סקטורים שהוחלפו", hePlain: "אזורים פגומים שהכונן כבר הוציא משימוש", smartctl: "Reallocated_Sector_Ct" },
  smart_7_raw: { name: "Seek errors", plain: "misses while moving the read head", heName: "שגיאות חיפוש", hePlain: "החטאות בהזזת ראש הקריאה", smartctl: "Seek_Error_Rate" },
  smart_9_raw: { name: "Power-on hours", plain: "the drive's age in running hours", heName: "שעות פעולה", hePlain: "גיל הכונן בשעות פעולה", smartctl: "Power_On_Hours" },
  smart_10_raw: { name: "Spin retries", plain: "failed attempts to spin up", heName: "ניסיונות סיבוב חוזרים", hePlain: "ניסיונות כושלים להתחיל להסתובב", smartctl: "Spin_Retry_Count" },
  smart_12_raw: { name: "Power cycles", plain: "how many times it was switched on", heName: "מחזורי הדלקה", hePlain: "כמה פעמים הכונן הודלק", smartctl: "Power_Cycle_Count" },
  smart_192_raw: { name: "Emergency retracts", plain: "head parked because power was cut", heName: "החזרות חירום של הראש", hePlain: "הראש הוחנה כי החשמל נותק", smartctl: "Power-Off_Retract_Count" },
  smart_193_raw: { name: "Load cycles", plain: "times the head was parked and unparked", heName: "מחזורי טעינת ראש", hePlain: "כמה פעמים הראש הוחנה והוחזר לעבודה", smartctl: "Load_Cycle_Count" },
  smart_194_raw: { name: "Temperature", plain: "degrees Celsius", heName: "טמפרטורה", hePlain: "מעלות צלזיוס", smartctl: "Temperature_Celsius" },
  smart_197_raw: { name: "Pending sectors", plain: "unstable spots waiting to be retired", heName: "סקטורים ממתינים", hePlain: "אזורים לא יציבים שממתינים להחלפה", smartctl: "Current_Pending_Sector" },
  smart_198_raw: { name: "Uncorrectable sectors", plain: "spots that could not be read at all", heName: "סקטורים שאינם ניתנים לתיקון", hePlain: "אזורים שלא ניתן היה לקרוא בכלל", smartctl: "Offline_Uncorrectable" },
  smart_199_raw: { name: "Cable/interface errors", plain: "data corrupted on the way to the host", heName: "שגיאות כבל וממשק", hePlain: "נתונים ששובשו בדרך למחשב", smartctl: "UDMA_CRC_Error_Count" },
  smart_240_raw: { name: "Head flying hours", plain: "time the read head spent in motion", heName: "שעות תנועת ראש", hePlain: "הזמן שראש הקריאה היה בתנועה", smartctl: "Head_Flying_Hours" },
};

export const smartName = (attr: string, lang: Lang = "en") => (lang === "he" ? SMART[attr]?.heName : SMART[attr]?.name) ?? attr;
export const smartPlain = (attr: string, lang: Lang = "en") => (lang === "he" ? SMART[attr]?.hePlain : SMART[attr]?.plain) ?? "";

const AGG: Record<string, { en: string; he: string }> = {
  last: { en: "latest", he: "ערך אחרון" },
  mean: { en: "30-day average", he: "ממוצע 30 יום" },
  delta: { en: "change over the window", he: "שינוי לאורך החלון" },
};

/**
 * "smart_197_raw_d7__last" -> "Pending sectors — 7-day growth, latest".
 * Channels are the raw attribute (log-scaled), its 1-day difference (_d1), its 7-day
 * difference (_d7), and `observed`, the share of days the drive actually reported.
 */
export function featureLabel(feature: string, lang: Lang = "en"): { title: string; detail: string } {
  const he = lang === "he";
  const [channel, agg = ""] = feature.split("__");
  const aggText = AGG[agg]?.[lang] ?? agg;
  if (channel === "observed")
    return he
      ? { title: "שלמות הדיווח", detail: `חלק הימים שדווחו, ${aggText}` }
      : { title: "Reporting completeness", detail: `share of days reported, ${aggText}` };
  const m = channel.match(/^(smart_\d+_raw)(?:_d(\d+))?$/);
  if (!m) return { title: channel, detail: aggText };
  const base = smartName(m[1], lang);
  const diff = m[2] ? (he ? `גידול ב־${m[2]} ימים, ` : `${m[2]}-day growth, `) : "";
  return { title: base, detail: `${diff}${aggText}` };
}
