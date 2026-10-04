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
  /** The page's main feature: highlighted frame, badge and a larger heading. */
  featured?: boolean;
}

/** A page section: small kicker, a compact heading with its lede beside it, then content. */
export function StorySection({ id, kicker, title, lede, action, children, className = "", band = false, featured = false }: Props) {
  const rise = useRise(0, 16);
  return (
    <section
      id={id}
      data-chapter={id}
      aria-labelledby={`${id}-title`}
      className={`relative scroll-mt-16 py-20 md:py-24 ${featured ? "border-y border-signal/30" : "border-t hairline"} ${band ? "bg-night/70" : ""} ${className}`}
      style={
        featured
          ? { background: "radial-gradient(70% 45% at 50% 0%, rgba(63,224,255,0.10), transparent 70%), linear-gradient(180deg, #08101f, #04060b 60%)" }
          : undefined
      }
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <motion.header {...rise} className="mb-10 grid gap-x-12 gap-y-4 md:mb-12 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)_auto] lg:items-end">
          <div>
            {featured ? (
              <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-signal/50 bg-signal/10 px-3 py-1 text-xs font-medium text-signal">
                <span className="relative flex h-2 w-2">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal opacity-60" />
                  <span className="relative inline-flex h-2 w-2 rounded-full bg-signal" />
                </span>
                {kicker}
              </p>
            ) : (
              <p className="kicker mb-3">{kicker}</p>
            )}
            <h2
              id={`${id}-title`}
              className={`${featured ? "text-[clamp(2.1rem,4vw,3.25rem)]" : "text-[clamp(1.75rem,3vw,2.5rem)]"} font-semibold leading-tight tracking-tight`}
            >
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
