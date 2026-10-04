import { motion, useReducedMotion } from "framer-motion";
import type { Contribution } from "../../shared/types";
import { featureLabel } from "../lib/smart";

/**
 * Exact per-prediction SHAP contributions (log-odds) from the XGBoost engine.
 * Right = pushed the risk up, left = pushed it down. They sum to the score's logit.
 */
export function SHAPExplanation({ contributions, top = 8 }: { contributions: Contribution[]; top?: number }) {
  const reduce = useReducedMotion();
  const rows = contributions.slice(0, top);
  const max = Math.max(...rows.map((r) => Math.abs(r.contribution)), 1e-6);

  if (!rows.length) return <p className="text-sm text-muted">No explanation available for this model.</p>;

  return (
    <div>
      <div className="mb-3 flex justify-between font-mono text-[10px] uppercase tracking-widest text-faint">
        <span>▼ lowers risk</span>
        <span>raises risk ▲</span>
      </div>
      <ul className="space-y-2.5">
        {rows.map((r, i) => {
          const f = featureLabel(r.feature);
          const w = (Math.abs(r.contribution) / max) * 50;
          const up = r.contribution > 0;
          return (
            <li key={r.feature} className="group">
              <div className="mb-1 flex items-baseline justify-between gap-3 text-xs">
                <span className="min-w-0 truncate">
                  <span className="text-soft">{f.title}</span> <span className="text-faint">· {f.detail}</span>
                </span>
                <span className="num shrink-0 text-muted">
                  {up ? "+" : "−"}
                  {Math.abs(r.contribution).toFixed(2)}
                </span>
              </div>
              <div className="relative h-2 rounded-full bg-deep/60" title={`${r.feature}: ${r.contribution.toFixed(3)} log-odds`}>
                <span className="absolute left-1/2 top-[-3px] h-[14px] w-px bg-line-strong" />
                <motion.span
                  initial={reduce ? false : { width: 0 }}
                  animate={{ width: `${w}%` }}
                  transition={{ duration: 0.7, delay: reduce ? 0 : i * 0.05, ease: [0.22, 1, 0.36, 1] }}
                  className="absolute top-0 h-2 rounded-full"
                  style={{
                    background: up ? "var(--color-warn)" : "var(--color-signal)",
                    ...(up ? { left: "50%" } : { right: "50%" }),
                  }}
                />
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
