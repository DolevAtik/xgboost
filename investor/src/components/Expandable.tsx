import { useId, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { EASE } from "../lib/motion";

/**
 * A closed "go deeper" block for content a technical reader wants and an investor can
 * skip. Its children are not mounted until it is opened, so closed content costs nothing.
 */
export function Expandable({ title, hint, children, className = "" }: { title: string; hint: string; children: ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className={className}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
        className={`group flex w-full items-center justify-between gap-4 rounded-xl border px-5 py-4 text-left transition ${
          open ? "border-signal/40 bg-signal/[0.05]" : "border-line bg-night/60 hover:border-line-strong hover:bg-deep/50"
        }`}
      >
        <span className="min-w-0">
          <span className="kicker mb-1 block text-[10px]">Go deeper</span>
          <span className="block text-[15px] font-medium text-ink">{title}</span>
          <span className="mt-0.5 block text-sm text-muted">{hint}</span>
        </span>
        <span className="flex shrink-0 items-center gap-2 text-xs text-muted group-hover:text-soft">
          {open ? "Hide" : "Show"}
          <ChevronDown size={16} aria-hidden className={`transition-transform duration-300 ${open ? "rotate-180" : ""}`} />
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.4, ease: EASE }}
            className="overflow-hidden"
          >
            <div className="pt-4">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
