import { lazy, Suspense, type ComponentType, type LazyExoticComponent } from "react";
import { ArrowLeft } from "lucide-react";
import { Loading } from "../components/States";

type Lazy = LazyExoticComponent<ComponentType>;
const pick = <M,>(load: () => Promise<M>, f: (m: M) => ComponentType): Lazy => lazy(() => load().then((m) => ({ default: f(m) })));

/** Artifacts A–H, each rendered alone, exactly as the story composes them. */
export const ARTIFACTS: { id: string; letter: string; title: string; component: string; full?: boolean; C: Lazy }[] = [
  { id: "a", letter: "A", title: "Cinematic Hero", component: "HeroScene + DriveModel3D", full: true, C: pick(() => import("../sections/Hero"), (m) => m.Hero) },
  { id: "b", letter: "B", title: "The Problem", component: "ProblemComparison", C: pick(() => import("../sections/ProblemComparison"), (m) => m.ProblemVisual) },
  { id: "c", letter: "C", title: "Data-to-Intelligence Pipeline", component: "DataPipeline", C: pick(() => import("../sections/DataPipeline"), (m) => m.DataPipeline) },
  {
    id: "d",
    letter: "D",
    title: "AI Intelligence Engine",
    component: "IntelligenceVisualization",
    C: pick(() => import("../sections/IntelligenceVisualization"), (m) => m.IntelligenceVisualization),
  },
  { id: "e", letter: "E", title: "Performance Proof", component: "PerformanceMetrics", C: pick(() => import("../sections/PerformanceMetrics"), (m) => m.PerformanceProof) },
  { id: "f", letter: "F", title: "Interactive Drive Explorer", component: "DriveExplorer", C: pick(() => import("../sections/DriveExplorer"), (m) => m.DriveExplorer) },
  { id: "g", letter: "G", title: "Model Comparison", component: "ModelComparison", C: pick(() => import("../sections/ModelComparison"), (m) => m.ModelComparisonView) },
  { id: "h", letter: "H", title: "The Future", component: "ClosingScene", C: pick(() => import("../sections/ClosingScene"), (m) => m.FleetVision) },
];

export function Lab({ id }: { id?: string }) {
  const art = ARTIFACTS.find((a) => a.id === id);

  if (art) {
    const C = art.C;
    return (
      <div className="min-h-screen">
        <header className="fixed inset-x-0 top-0 z-50 flex h-14 items-center justify-between border-b hairline bg-void/80 px-4 backdrop-blur sm:px-8">
          <a href="#/lab" className="flex items-center gap-2 text-sm text-muted hover:text-ink">
            <ArrowLeft size={14} aria-hidden /> Artifacts
          </a>
          <span className="font-mono text-xs text-soft">
            {art.letter} · {art.title} · <span className="text-faint">{art.component}</span>
          </span>
        </header>
        <Suspense fallback={<Loading className="p-20" />}>
          {art.full ? (
            <C />
          ) : (
            <div className="mx-auto max-w-6xl px-4 pb-24 pt-28 sm:px-8">
              <C />
            </div>
          )}
        </Suspense>
      </div>
    );
  }

  return (
    <div className="mx-auto min-h-screen max-w-6xl px-4 py-24 sm:px-8">
      <p className="kicker">Artifact lab</p>
      <h1 className="display mt-4 text-6xl">Eight artifacts, built and refined on their own.</h1>
      <p className="mt-6 max-w-2xl text-lg text-soft">
        Each one is a standalone component wired to real data. The story at <a href="#/" className="text-signal underline-offset-4 hover:underline">/</a>{" "}
        composes them in order.
      </p>
      <ul className="mt-16 grid gap-px overflow-hidden rounded-2xl border hairline bg-line sm:grid-cols-2">
        {ARTIFACTS.map((a) => (
          <li key={a.id}>
            <a href={`#/lab/${a.id}`} className="group flex h-full items-baseline gap-6 bg-night p-8 transition hover:bg-deep">
              <span className="num text-4xl text-faint transition group-hover:text-signal">{a.letter}</span>
              <span>
                <span className="block text-xl">{a.title}</span>
                <span className="mt-1 block font-mono text-xs text-muted">{a.component}</span>
              </span>
            </a>
          </li>
        ))}
      </ul>
    </div>
  );
}
