/**
 * The team section's content. Edit names, roles and focus areas here.
 *
 * Photos: drop a square image (ideally 600×600 or larger) at
 * investor/public/team/<photo>, e.g. public/team/dolev-atik.jpg. Until the file exists
 * the card shows the person's initials instead.
 */
export interface Member {
  name: string;
  role: string;
  summary: string;
  focus: string[];
  photo: string;
}

export const TEAM: Member[] = [
  {
    name: "Nave Dan",
    role: "Data & Pipeline Lead",
    summary:
      "Built the data foundation: ingesting the full Backblaze archive into a fleet-scale dataset, and the leakage audit that keeps the model from seeing the answer.",
    focus: ["Archive ingestion", "Data cleaning", "Leakage audit", "30-day windowing"],
    photo: "nave-dan.jpg",
  },
  {
    name: "Teddy Boliasny",
    role: "Machine Learning Lead",
    summary:
      "Designed and trained the prediction models, XGBoost and the CNN, and the evaluation that measures them at the fleet's real failure rate.",
    focus: ["XGBoost & CNN", "Feature engineering", "Evaluation & precision@K", "Explainability (SHAP)"],
    photo: "teddy-boliasny.jpg",
  },
  {
    name: "Dolev Atik",
    role: "Product & Platform Lead",
    summary:
      "Turned the models into a working product: the prediction engine that scores uploaded telemetry, the demo datasets, and this site.",
    focus: ["Prediction engine", "Drive explorer", "Web platform & API", "Product experience"],
    photo: "dolev-atik.jpg",
  },
];
