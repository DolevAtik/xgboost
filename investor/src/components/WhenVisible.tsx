import { useEffect, useRef, useState, type ReactNode } from "react";

/**
 * Mounts its children only once the placeholder is near the viewport, so heavy scenes
 * and charts never block the first paint. With `unmountWhenHidden` a WebGL scene is torn
 * down again when scrolled far away, freeing the GPU for the next one.
 */
export function WhenVisible({
  children,
  fallback = null,
  className = "",
  rootMargin = "400px 0px",
  unmountWhenHidden = false,
}: {
  children: ReactNode;
  fallback?: ReactNode;
  className?: string;
  rootMargin?: string;
  unmountWhenHidden?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [show, setShow] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) setShow(true);
        else if (unmountWhenHidden) setShow(false);
      },
      { rootMargin },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [rootMargin, unmountWhenHidden]);
  return (
    <div ref={ref} className={className}>
      {show ? children : fallback}
    </div>
  );
}
