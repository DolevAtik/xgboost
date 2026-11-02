import type { Bi } from "./i18n";

/**
 * The team section's content, in English and Hebrew. Names are shown as written in
 * both languages. Edit roles, bios and focus areas here.
 *
 * Photos: drop a square image (ideally 600×600 or larger) at
 * investor/public/team/<photo>, e.g. public/team/dolev-atik.jpg. Until the file exists
 * the card shows the person's initials instead.
 */
export interface Member {
  name: string;
  role: Bi;
  summary: Bi;
  focus: Bi[];
  photo: string;
}

export const TEAM: Member[] = [
  {
    name: "Nave Dan",
    role: { en: "Data & Pipeline Lead", he: "אחראי דאטה ותהליכי עיבוד" },
    summary: {
      en: "Built the data foundation: ingesting the full Backblaze archive into a fleet-scale dataset, and the leakage audit that keeps the model from seeing the answer.",
      he: "בנה את תשתית הנתונים: קליטת כל ארכיון Backblaze למאגר בהיקף של צי שלם, ובדיקת הדליפות שמונעת מהמודל לראות את התשובה.",
    },
    focus: [
      { en: "Archive ingestion", he: "קליטת הארכיון" },
      { en: "Data cleaning", he: "ניקוי נתונים" },
      { en: "Leakage audit", he: "בדיקת דליפות" },
      { en: "30-day windowing", he: "חלונות של 30 יום" },
    ],
    photo: "nave-dan.jpg",
  },
  {
    name: "Teddy Boliasny",
    role: { en: "Machine Learning Lead", he: "אחראי למידת מכונה" },
    summary: {
      en: "Designed and trained the prediction models, XGBoost and the CNN, and the evaluation that measures them at the fleet's real failure rate.",
      he: "תכנן ואימן את מודלי החיזוי, XGBoost ו־CNN, ואת ההערכה שמודדת אותם בשיעור התקלות האמיתי של הצי.",
    },
    focus: [
      { en: "XGBoost & CNN", he: "XGBoost ו־CNN" },
      { en: "Feature engineering", he: "הנדסת מאפיינים" },
      { en: "Evaluation & precision@K", he: "הערכה ו־precision@K" },
      { en: "Explainability (SHAP)", he: "הסבר החלטות (SHAP)" },
    ],
    photo: "teddy-boliasny.jpg",
  },
  {
    name: "Dolev Atik",
    role: { en: "Product & Platform Lead", he: "אחראי מוצר ופלטפורמה" },
    summary: {
      en: "Turned the models into a working product: the prediction engine that scores uploaded telemetry, the demo datasets, and this site.",
      he: "הפך את המודלים למוצר עובד: מנוע החיזוי שמנקד קבצים שמועלים, קובצי ההדגמה, והאתר הזה.",
    },
    focus: [
      { en: "Prediction engine", he: "מנוע החיזוי" },
      { en: "Drive explorer", he: "סייר הכוננים" },
      { en: "Web platform & API", he: "פלטפורמת ווב ו־API" },
      { en: "Product experience", he: "חוויית מוצר" },
    ],
    photo: "dolev-atik.jpg",
  },
];
