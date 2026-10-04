export const fmtInt = (n: number) => Math.round(n).toLocaleString("en-US");

/** 459048144 -> "459M", 3088649 -> "3.09M", 413045 -> "413K". */
export function fmtCompact(n: number, digits = 0): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(digits)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(digits)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(digits)}K`;
  return String(Math.round(n));
}

export const fmtPct = (f: number, digits = 1) => `${(f * 100).toFixed(digits)}%`;
export const fmt3 = (f: number) => f.toFixed(3);

export type Tone = "risk" | "warn" | "ok";

/**
 * The model's own decision is binary: at or above its tuned threshold the drive is
 * "inspect", below it "hold". The page splits "inspect" once more at 0.5 so a 0.25 and a
 * 0.99 do not read the same; that split is presentation, and the UI says so.
 */
export function riskCategory(score: number, threshold: number): { label: string; call: string; tone: Tone } {
  if (score >= Math.max(0.5, threshold)) return { label: "High risk", call: "inspect", tone: "risk" };
  if (score >= threshold) return { label: "Elevated", call: "inspect", tone: "warn" };
  return { label: "Low risk", call: "hold", tone: "ok" };
}

export const toneColor: Record<Tone, string> = {
  risk: "var(--color-risk)",
  warn: "var(--color-warn)",
  ok: "var(--color-ok)",
};
