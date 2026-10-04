import { useEffect, useState } from "react";
import { ArrowRight, Menu, X } from "lucide-react";
import { api } from "../lib/api";
import { useApi } from "../lib/useApi";

export const NAV_LINKS = [
  { id: "problem", label: "The Problem" },
  { id: "signal", label: "The Data" },
  { id: "intelligence", label: "Technology" },
  { id: "breakthrough", label: "Results" },
  { id: "product", label: "Try It" },
  { id: "vision", label: "Vision" },
  { id: "team", label: "Team" },
];

function Logo() {
  return (
    <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden>
      <circle cx="16" cy="16" r="12" fill="none" stroke="var(--color-signal)" strokeWidth="2" />
      <circle cx="16" cy="16" r="6" fill="none" stroke="var(--color-signal)" strokeOpacity=".5" strokeWidth="1.5" />
      <circle cx="16" cy="16" r="2.4" fill="var(--color-signal)" />
      <circle cx="23.5" cy="9" r="2.6" fill="var(--color-risk)" />
    </svg>
  );
}

/** Brand, section links that track the scroll position, engine status, and the demo CTA. */
export function Nav() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState<string | null>(null);
  const health = useApi(api.health);

  useEffect(() => {
    const on = () => setScrolled(window.scrollY > 24);
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);

  // Highlight the link of the section in the middle of the viewport. Sections are
  // code-split, so look them up on each pass rather than once.
  useEffect(() => {
    let frame = 0;
    const update = () => {
      const mid = window.innerHeight * 0.4;
      let current: string | null = null;
      for (const l of NAV_LINKS) {
        const el = document.getElementById(l.id);
        if (el && el.getBoundingClientRect().top <= mid) current = l.id;
      }
      setActive(current);
    };
    const on = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", on, { passive: true });
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("scroll", on);
    };
  }, []);

  const ok = health.data?.engine.ok;
  const status = health.loading ? "checking engine" : ok ? "Engine live" : "Engine offline";

  return (
    <header
      className={`fixed inset-x-0 top-0 z-40 border-b transition-colors duration-300 ${
        scrolled || open ? "border-line bg-void/85 backdrop-blur-xl" : "border-transparent bg-transparent"
      }`}
    >
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-6 px-4 sm:px-6 lg:px-8">
        <a href="#hook" className="flex shrink-0 items-center gap-2.5" aria-label="Drive Foresight, back to top" onClick={() => setOpen(false)}>
          <Logo />
          <span className="text-[15px] font-semibold uppercase tracking-[0.08em]">Drive Foresight</span>
        </a>

        <nav aria-label="Sections" className="hidden lg:block">
          <ul className="flex items-center gap-7">
            {NAV_LINKS.map((l) => (
              <li key={l.id}>
                <a
                  href={`#${l.id}`}
                  aria-current={active === l.id ? "true" : undefined}
                  className={`relative py-5 text-sm transition-colors ${active === l.id ? "text-ink" : "text-muted hover:text-ink"}`}
                >
                  {l.label}
                  {active === l.id && <span className="absolute inset-x-0 -bottom-px h-px bg-signal" />}
                </a>
              </li>
            ))}
          </ul>
        </nav>

        <div className="flex items-center gap-3">
          <span
            className="hidden items-center gap-2 text-xs text-muted xl:flex"
            title={health.data?.engine.error ?? health.data?.engine.url ?? ""}
          >
            <span className={`h-1.5 w-1.5 rounded-full ${health.loading ? "bg-faint" : ok ? "bg-ok" : "bg-risk"}`} />
            {status}
          </span>
          <a href="#product" className="hidden items-center gap-2 rounded-full border border-signal/50 px-4 py-2 text-sm transition hover:border-signal hover:bg-signal/10 sm:inline-flex">
            Try the demo <ArrowRight size={14} aria-hidden />
          </a>
          <button
            type="button"
            className="rounded-lg p-2 text-soft hover:text-ink lg:hidden"
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
          >
            {open ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {open && (
        <nav aria-label="Sections" className="border-t hairline px-4 pb-4 lg:hidden">
          <ul className="grid gap-1 pt-2">
            {NAV_LINKS.map((l) => (
              <li key={l.id}>
                <a href={`#${l.id}`} onClick={() => setOpen(false)} className="block rounded-lg px-3 py-2.5 text-soft hover:bg-deep hover:text-ink">
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      )}
    </header>
  );
}

export function Footer() {
  return (
    <footer className="border-t hairline py-10">
      <div className="mx-auto flex max-w-7xl flex-wrap items-start justify-between gap-6 px-4 text-xs text-faint sm:px-6 lg:px-8">
        <div className="flex items-center gap-2.5">
          <Logo />
          <span className="text-sm font-semibold uppercase tracking-[0.08em] text-soft">Drive Foresight</span>
        </div>
        <p className="max-w-xl leading-relaxed">
          Performance figures are read from the project's evaluation files in <span className="font-mono">results/</span>. Predictions come from the
          saved models in <span className="font-mono">models/</span>, served by the local engine. Training data: Backblaze public drive-stats archives,
          Q1 2022 – Q2 2026. Runs entirely on this machine.
        </p>
        <a href="#/lab" className="font-mono hover:text-soft">
          artifact lab →
        </a>
      </div>
    </footer>
  );
}
