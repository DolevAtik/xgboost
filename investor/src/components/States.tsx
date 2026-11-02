import { AlertTriangle, Info, RotateCw } from "lucide-react";
import type { ReactNode } from "react";
import { useLang } from "../lib/i18n";

export function Loading({ label, className = "" }: { label?: string; className?: string }) {
  const { t } = useLang();
  return (
    <div role="status" className={`flex items-center gap-3 text-sm text-muted ${className}`}>
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-signal" />
      </span>
      {label ?? t("Loading", "טוען")}…
    </div>
  );
}

export function ErrorState({ message, onRetry, className = "" }: { message: string; onRetry?: () => void; className?: string }) {
  const { t } = useLang();
  return (
    <div role="alert" className={`flex flex-wrap items-start gap-3 rounded-xl border border-risk/30 bg-risk/5 p-4 text-sm ${className}`}>
      <AlertTriangle aria-hidden size={16} className="mt-0.5 shrink-0 text-risk" />
      <p className="min-w-0 flex-1 text-soft">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="chip hover:border-signal hover:text-ink">
          <RotateCw size={12} aria-hidden /> {t("Retry", "נסה שוב")}
        </button>
      )}
    </div>
  );
}

/**
 * Names the files a figure came from. Kept out of the reader's way: a small "Source"
 * marker that reveals the detail on hover or focus.
 */
export function Source({ children }: { children: ReactNode }) {
  const { t } = useLang();
  return (
    <div className="group relative mt-3 inline-block">
      <button type="button" className="inline-flex items-center gap-1 text-[11px] text-faint hover:text-muted focus-visible:text-muted">
        <Info size={11} aria-hidden /> {t("Source", "מקור")}
      </button>
      <p
        role="tooltip"
        dir="ltr"
        className="pointer-events-none absolute bottom-full start-0 z-20 mb-2 w-max max-w-[min(32rem,80vw)] rounded-lg border hairline bg-void px-3 py-2 font-mono text-[11px] leading-relaxed text-muted shadow-xl hidden group-focus-within:block group-hover:block"
      >
        {children}
      </p>
    </div>
  );
}
