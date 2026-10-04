import { createContext, useContext, type ReactNode } from "react";
import type { Evaluation, ModelResponse } from "../../shared/types";
import { api } from "./api";
import { useApi, type Async } from "./useApi";

interface StoryData {
  evaluation: Async<Evaluation>;
  model: Async<ModelResponse>;
}

const Ctx = createContext<StoryData | null>(null);

/** The evaluation results and model metadata, fetched once and shared by every chapter. */
export function DataProvider({ children }: { children: ReactNode }) {
  const evaluation = useApi(api.evaluation);
  const model = useApi(api.model);
  return <Ctx.Provider value={{ evaluation, model }}>{children}</Ctx.Provider>;
}

export function useStoryData(): StoryData {
  const v = useContext(Ctx);
  if (!v) throw new Error("useStoryData outside DataProvider");
  return v;
}
