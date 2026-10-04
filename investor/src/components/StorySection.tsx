import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { useRise } from "../lib/motion";

interface Props {
  id: string;
  kicker: string;
  title: ReactNode;
  lede?: ReactNode;
  /** Right-aligned header slot, e.g. a link to the next step. */
  action?: ReactNode;
  children?: ReactNode;
  className?: string;
  /** Alternate band background, so sections read as distinct blocks. */
  band?: boolean;
}

/** A page section: small kicker, a compact heading with its lede beside it, then content. */
export function StorySection({ id, kicker, title, lede, action, children, className = "", band = false }: Props) {
  const rise = useRise(0, 16);
  return (
    <section
      id={id}
      data-chapter={id}
      aria-labelledby={`${id}-title`}
      className={`relative scroll-mt-16 border-t hairline py-20 md:py-24 ${band ? "bg-night/70" : ""} ${className}`}
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.header {...rise} className="mb-10 grid gap-x-12 gap-y-4 md:mb-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)_auto] lg:items-end">
          <div>
            <p className="kicker mb-3">{kicker}</p>
            <h2 id={`${id}-title`} className="text-[clamp(1.75rem,3vw,2.5rem)] font-semibold leading-tight tracking-tight">
              {title}
            </h2>
          </div>
          {lede ? <p className="max-w-2xl text-[15px] leading-relaxed text-soft md:text-base">{lede}</p> : <span />}
          {action && <div className="lg:justify-self-end">{action}</div>}
        </motion.header>
        {children}
      </div>
    </section>
  );
}

/** The bordered surface every visual sits on. */
export function Panel({ children, className = "", as: Tag = "div" }: { children: ReactNode; className?: string; as?: "div" | "figure" | "article" }) {
  return <Tag className={`panel ${className}`}>{children}</Tag>;
}
