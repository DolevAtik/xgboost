import { lazy, Suspense, type ReactNode } from "react";
import { ContactDialog } from "./components/ContactDialog";
import { Footer, Nav } from "./components/Nav";
import { Loading } from "./components/States";
import { DemoTeaser } from "./components/DemoTeaser";
import { Hero } from "./sections/Hero";
import { ProblemComparison } from "./sections/ProblemComparison";

// Everything below the fold is split into its own chunk, fetched in parallel after the
// hero has painted. The 3D scenes inside are additionally mounted only when visible.
const Intelligence = lazy(() => import("./sections/Intelligence"));
const PerformanceMetrics = lazy(() => import("./sections/PerformanceMetrics"));
const Product = lazy(() => import("./sections/Product"));
const ValueCalculator = lazy(() => import("./sections/ValueCalculator"));
const ClosingScene = lazy(() => import("./sections/ClosingScene"));
const Close = lazy(() => import("./sections/ClosingScene").then((m) => ({ default: m.Close })));

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
        <Deferred id="intelligence">
          <Intelligence />
        </Deferred>
        <Deferred id="breakthrough">
          <PerformanceMetrics />
        </Deferred>
        <DemoTeaser />
        <Deferred id="value">
          <ValueCalculator />
        </Deferred>
        <Deferred id="vision">
          <ClosingScene />
        </Deferred>
        <Deferred id="product">
          <Product />
        </Deferred>
        <Suspense fallback={null}>
          <Close />
        </Suspense>
      </main>
      <Footer />
      <ContactDialog />
    </div>
  );
}
