import { useEffect, useState } from "react";
import { useReducedMotion } from "framer-motion";

export const EASE = [0.22, 1, 0.36, 1] as const;

/** Fade-and-rise props for one element; static under reduced motion. */
export function useRise(delay = 0, y = 24) {
  const reduce = useReducedMotion();
  if (reduce) return {};
  return {
    initial: { opacity: 0, y },
    whileInView: { opacity: 1, y: 0 },
    viewport: { once: true, margin: "-10% 0px" },
    transition: { duration: 0.9, ease: EASE, delay },
  } as const;
}

/** True on narrow screens or coarse pointers: lighter 3D, fewer particles. */
export function useIsCompact(): boolean {
  const q = "(max-width: 768px), (pointer: coarse)";
  const [m, setM] = useState(() => typeof window !== "undefined" && window.matchMedia(q).matches);
  useEffect(() => {
    const mq = window.matchMedia(q);
    const on = () => setM(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return m;
}

/** WebGL can be absent (old GPUs, some VMs); the page must still read without it. */
export function hasWebGL(): boolean {
  try {
    const c = document.createElement("canvas");
    return !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    return false;
  }
}
