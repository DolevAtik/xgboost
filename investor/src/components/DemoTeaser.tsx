import { ArrowDown, Play, Upload } from "lucide-react";

/**
 * The live demo is the last chapter; this points to it while the reader is still
 * looking at the results it proves.
 */
export function DemoTeaser() {
  return (
    <div className="mx-auto max-w-7xl px-4 pb-4 sm:px-6 lg:px-8">
      <div
        className="flex flex-wrap items-center justify-between gap-5 rounded-2xl border border-signal/40 px-6 py-5 md:px-8"
        style={{ background: "linear-gradient(90deg, rgba(63,224,255,0.10), rgba(63,224,255,0.02))" }}
      >
        <div className="flex items-center gap-4">
          <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-full bg-signal/15 text-signal">
            <Play size={18} aria-hidden />
          </span>
          <div>
            <p className="text-base font-semibold text-ink">Seen the numbers? Now run it yourself.</p>
            <p className="text-sm text-muted">Upload your own drive telemetry or pick a sample, and get a ranked risk list in seconds.</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-3">
          <a href="#upload" className="btn btn-primary">
            <Upload size={15} aria-hidden /> Upload your file
          </a>
          <a href="#product" className="btn btn-ghost">
            Go to the live demo <ArrowDown size={15} aria-hidden />
          </a>
        </div>
      </div>
    </div>
  );
}
