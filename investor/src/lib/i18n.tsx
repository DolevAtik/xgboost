import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

/**
 * Two languages: English (default) and Hebrew. Strings live next to the component that
 * shows them, as t("English", "עברית"), so a sentence and its translation never drift
 * apart. Hebrew switches the document to right-to-left; charts, timelines and grids pin
 * themselves to left-to-right where geometry carries meaning (time runs left to right).
 */
export type Lang = "en" | "he";

const STORAGE_KEY = "lang";

interface LangContext {
  lang: Lang;
  setLang: (l: Lang) => void;
}

const Ctx = createContext<LangContext>({ lang: "en", setLang: () => {} });

function initialLang(): Lang {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "he" || v === "en") return v;
  } catch {
    /* storage can be unavailable; English it is */
  }
  return "en";
}

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);

  useEffect(() => {
    const root = document.documentElement;
    root.lang = lang;
    root.dir = lang === "he" ? "rtl" : "ltr";
    document.title = lang === "he" ? "Drive Foresight · חיזוי כשלים בכוננים" : "Drive Foresight";
  }, [lang]);

  const setLang = useCallback((l: Lang) => {
    setLangState(l);
    try {
      localStorage.setItem(STORAGE_KEY, l);
    } catch {
      /* not persisted; still switches for this visit */
    }
  }, []);

  const value = useMemo(() => ({ lang, setLang }), [lang, setLang]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useLang() {
  const { lang, setLang } = useContext(Ctx);
  const he = lang === "he";
  /** Pick the string (or node) for the current language. */
  const t = useCallback(<T,>(en: T, heb: T): T => (he ? heb : en), [he]);
  return { lang, he, t, setLang };
}

/** Bilingual text kept as data (arrays of steps, legends, team bios). */
export interface Bi {
  en: string;
  he: string;
}
