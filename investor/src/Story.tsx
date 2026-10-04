import { lazy, Suspense, type ReactNode } from "react";
import { Footer, Nav } from "./components/Nav";
import { Loading } from "./components/States";
import { Hero } from "./sections/Hero";
import { ProblemComparison } from "./sections/ProblemComparison";

// Everything below the fold is split into its own chunk, fetched in parallel after the
// hero has painted. The 3D scenes inside are additionally mounted only when visible.
const Signal = lazy(() => import("./sections/Signal"));
const Intelligence = lazy(() => import("./sections/Intelligence"));
const PerformanceMetrics = lazy(() => import("./sections/PerformanceMetrics"));
const Product = lazy(() => import("./sections/Product"));
const ClosingScene = lazy(() => import("./sections/ClosingScene"));

/** A code-split chapter. The placeholder keeps the anchor id so links work while it loads. */
function Deferred({ id, children }: { id: string; children: ReactNode }) {
  const placeholder = (
    <section id={id} data-chapter={id} className="mx-auto flex min-h-screen max-w-6xl items-center px-4 sm:px-8">
      <Loading />
    </section>
  );
  return <Suspense fallback={placeholder}>{children}</Suspense>;
}

export function Story() {
  return (
    <div className="grain">
      <a href="#problem" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[70] focus:rounded focus:bg-ink focus:px-3 focus:py-2 focus:text-void">
        Skip to the story
      </a>
      <Nav />
      <main>
        <Hero />
        <ProblemComparison />
        <Deferred id="signal">
          <Signal />
        </Deferred>
        <Deferred id="intelligence">
          <Intelligence />
        </Deferred>
        <Deferred id="breakthrough">
          <PerformanceMetrics />
        </Deferred>
        <Deferred id="product">
          <Product />
        </Deferred>
        <Deferred id="vision">
          <ClosingScene />
        </Deferred>
      </main>
      <Footer />
    </div>
  );
}
