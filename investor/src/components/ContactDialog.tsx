import { useEffect, useRef, useState, type FormEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Mail, Send, X } from "lucide-react";
import { CONTACT } from "../lib/contact";

const field =
  "mt-1.5 block w-full rounded-lg border border-line bg-void/60 px-3 py-2.5 text-sm text-ink placeholder:text-faint focus:border-signal focus:outline-none";

/** "Get in touch": a short form that hands the message to the visitor's mail client. */
export function ContactDialog() {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sent, setSent] = useState(false);
  const first = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const on = () => {
      setOpen(true);
      setSent(false);
      setError(null);
    };
    window.addEventListener("open-contact", on);
    return () => window.removeEventListener("open-contact", on);
  }, []);

  useEffect(() => {
    if (!open) return;
    const t = setTimeout(() => first.current?.focus(), 50);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    document.body.style.overflow = "hidden";
    return () => {
      clearTimeout(t);
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = "";
    };
  }, [open]);

  const submit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    if (!CONTACT.email) {
      setError("No contact address is configured yet. Set CONTACT.email in src/lib/contact.ts.");
      return;
    }
    const body = [
      String(f.get("message") ?? ""),
      "",
      `— ${f.get("name")}${f.get("org") ? `, ${f.get("org")}` : ""}`,
      String(f.get("email") ?? ""),
    ].join("\n");
    window.location.href = `mailto:${CONTACT.email}?subject=${encodeURIComponent(CONTACT.subject)}&body=${encodeURIComponent(body)}`;
    setSent(true);
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[80] flex items-center justify-center bg-void/80 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onMouseDown={(e) => e.target === e.currentTarget && setOpen(false)}
        >
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="contact-title"
            initial={{ y: 16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 8, opacity: 0 }}
            className="panel w-full max-w-lg border-signal/30 bg-night"
          >
            <div className="mb-5 flex items-start justify-between gap-4">
              <div>
                <p className="kicker mb-1.5 text-[10px]">Get in touch</p>
                <h2 id="contact-title" className="text-xl font-semibold">
                  Talk to the team
                </h2>
                <p className="mt-1 text-sm text-muted">Questions, a pilot on your fleet, or the full technical report.</p>
              </div>
              <button type="button" aria-label="Close" onClick={() => setOpen(false)} className="rounded-lg p-1.5 text-muted hover:text-ink">
                <X size={18} />
              </button>
            </div>

            {sent ? (
              <div className="rounded-xl border border-ok/30 bg-ok/5 p-5 text-sm text-soft">
                <p className="flex items-center gap-2 font-medium text-ok">
                  <Mail size={15} aria-hidden /> Your mail app should now be open with the message ready.
                </p>
                <p className="mt-2 text-muted">
                  If nothing opened, write to <span className="text-ink">{CONTACT.email}</span> directly.
                </p>
              </div>
            ) : (
              <form onSubmit={submit} className="space-y-4">
                <div className="grid gap-4 sm:grid-cols-2">
                  <label className="block text-xs text-muted">
                    Name
                    <input ref={first} name="name" required autoComplete="name" className={field} />
                  </label>
                  <label className="block text-xs text-muted">
                    Organisation <span className="text-faint">(optional)</span>
                    <input name="org" autoComplete="organization" className={field} />
                  </label>
                </div>
                <label className="block text-xs text-muted">
                  Email
                  <input name="email" type="email" required autoComplete="email" className={field} />
                </label>
                <label className="block text-xs text-muted">
                  Message
                  <textarea name="message" required rows={4} className={field} placeholder="What would you like to know?" />
                </label>
                {error && <p className="rounded-lg border border-warn/30 bg-warn/5 px-3 py-2 text-xs text-warn">{error}</p>}
                <div className="flex items-center justify-end gap-3 pt-1">
                  <button type="button" onClick={() => setOpen(false)} className="text-sm text-muted hover:text-ink">
                    Cancel
                  </button>
                  <button type="submit" className="btn btn-primary">
                    <Send size={14} aria-hidden /> Send message
                  </button>
                </div>
              </form>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
