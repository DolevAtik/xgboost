import { AlertTriangle, RotateCw } from "lucide-react";
import type { ReactNode } from "react";

export function Loading({ label = "Loading", className = "" }: { label?: string; className?: string }) {
  return (
    <div role="status" className={`flex items-center gap-3 text-sm text-muted ${className}`}>
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-signal opacity-60" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-signal" />
      </span>
      {label}…
    </div>
  );
}

export function ErrorState({ message, onRetry, className = "" }: { message: string; onRetry?: () => void; className?: string }) {
  return (
    <div role="alert" className={`flex flex-wrap items-start gap-3 rounded-xl border border-risk/30 bg-risk/5 p-4 text-sm ${className}`}>
      <AlertTriangle aria-hidden size={16} className="mt-0.5 shrink-0 text-risk" />
      <p className="min-w-0 flex-1 text-soft">{message}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="chip hover:border-signal hover:text-ink">
          <RotateCw size={12} aria-hidden /> Retry
        </button>
      )}
    </div>
  );
}

/** Names the file a number came from. Every figure on the page has one. */
export function Source({ children }: { children: ReactNode }) {
  return <p className="mt-4 font-mono text-[11px] leading-relaxed text-faint">Source: {children}</p>;
}
