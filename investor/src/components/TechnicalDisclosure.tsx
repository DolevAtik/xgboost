import { useId, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Plus } from "lucide-react";
import { EASE } from "../lib/motion";

/** Optional depth for the technical reader. Closed by default; the story reads without it. */
export function TechnicalDisclosure({ title, children, className = "" }: { title: string; children: ReactNode; className?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <div className={`border-t hairline ${className}`}>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((o) => !o)}
        className="group flex w-full items-center justify-between gap-4 py-4 text-left"
      >
        <span className="flex items-center gap-3">
          <span className="kicker text-[10px] text-faint">Technical</span>
          <span className="text-sm text-soft transition group-hover:text-ink">{title}</span>
        </span>
        <Plus aria-hidden size={16} className={`shrink-0 text-muted transition-transform duration-300 ${open ? "rotate-45" : ""}`} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.45, ease: EASE }}
            className="overflow-hidden"
          >
            <div className="max-w-3xl space-y-3 pb-6 text-sm leading-relaxed text-muted [&_b]:font-medium [&_b]:text-soft [&_code]:font-mono [&_code]:text-[12px] [&_code]:text-soft">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
