import { Bar, BarChart, CartesianGrid, LabelList, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { motion } from "framer-motion";
import type { Evaluation, RunInfo } from "../../shared/types";
import { ErrorState, Loading, Source } from "../components/States";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { useStoryData } from "../lib/data";
import { fmtInt } from "../lib/format";
import { useRise } from "../lib/motion";

const XGB = "var(--color-series-xgb)";
const CNN = "var(--color-series-cnn)";

function Ladder({ e }: { e: Evaluation }) {
  const data = e.precisionAtK.map((p) => ({
    k: `top ${fmtInt(p.k)}`,
    xgb: Math.round((p.xgbHits / p.k) * 1000) / 10,
    cnn: Math.round((p.cnnHits / p.k) * 1000) / 10,
  }));
  return (
    <figure className="min-w-0">
      <figcaption className="mb-3 text-sm text-soft">Share of the top-ranked drives that really failed</figcaption>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} barGap={2} margin={{ top: 22, right: 4, left: -16, bottom: 0 }}>
            <CartesianGrid vertical={false} />
            <XAxis dataKey="k" tickLine={false} axisLine={false} />
            <YAxis domain={[0, 100]} unit="%" tickLine={false} axisLine={false} width={50} />
            <Tooltip
              cursor={{ fill: "rgba(148,163,200,.06)" }}
              contentStyle={{ background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 }}
              formatter={(v, n) => [`${v}%`, n === "xgb" ? "XGBoost" : "CNN"]}
            />
            <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} formatter={(v) => (v === "xgb" ? "XGBoost" : "CNN")} />
            <Bar dataKey="xgb" fill={XGB} radius={[4, 4, 0, 0]} maxBarSize={26}>
              <LabelList dataKey="xgb" position="top" className="num" fill="var(--color-soft)" fontSize={10} formatter={(v) => `${v}`} />
            </Bar>
            <Bar dataKey="cnn" fill={CNN} radius={[4, 4, 0, 0]} maxBarSize={26} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </figure>
  );
}

function Row({ label, plain, xgb, cnn, better }: { label: string; plain: string; xgb: string; cnn: string; better: "xgb" | "cnn" | null }) {
  return (
    <tr className="border-t hairline align-top">
      <th scope="row" className="py-5 pr-6 text-left font-normal">
        <span className="block text-ink">{label}</span>
        <span className="mt-1 block text-xs text-muted">{plain}</span>
      </th>
      <td className={`num py-5 pr-6 text-sm ${better === "xgb" ? "text-ink" : "text-soft"}`}>{xgb}</td>
      <td className={`num py-5 text-sm ${better === "cnn" ? "text-ink" : "text-soft"}`}>{cnn}</td>
    </tr>
  );
}

/** Artifact G: two engines, trained on the identical windows, compared honestly. */
export function ModelComparisonView() {
  const { evaluation, model } = useStoryData();
  const rise = useRise();
  const e = evaluation.data;
  const run: RunInfo | undefined = model.data?.run;
  if (evaluation.loading || model.loading) return <Loading label="Loading the comparison" />;
  if (!e || !run) return <ErrorState message={evaluation.error ?? model.error ?? "no data"} onRetry={() => (evaluation.reload(), model.reload())} />;

  const x = e.models.find((m) => m.model === "XGBoost")!;
  const c = e.models.find((m) => m.model === "CNN")!;
  const xi = run.estimators.find((s) => s.kind === "xgb");
  const ci = run.estimators.find((s) => s.kind === "cnn");

  return (
    <motion.div {...rise} className="space-y-4">
      <div className="panel grid gap-10 lg:grid-cols-[1fr_1.2fr] lg:gap-16">
        <div className="space-y-4 text-[15px] leading-relaxed text-soft">
          <p>
            We built two very different engines and trained them on exactly the same data. <b className="font-medium text-ink">XGBoost</b>{" "}
            reads a summary of the month. The <b className="font-medium text-ink">CNN</b>, a neural network, reads the month day by day.
          </p>
          <p>
            XGBoost finds more of the failing drives at the top of the list, and it can say <i>why</i> it flagged each one. That is why it is
            the default. The CNN remains available as a second opinion.
          </p>
        </div>
        <Ladder e={e} />
      </div>

      <div className="panel overflow-x-auto">
        <table className="w-full min-w-[620px]">
          <thead>
            <tr className="kicker text-[10px]">
              <th className="pb-4 text-left font-normal" />
              <th className="pb-4 text-left font-normal">
                <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: XGB }} />
                XGBoost · default
              </th>
              <th className="pb-4 text-left font-normal">
                <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: CNN }} />
                CNN
              </th>
            </tr>
          </thead>
          <tbody>
            <Row label="Ranking quality" plain="PR-AUC on rare events; random = 0.0024" xgb={x.prAuc.toFixed(3)} cnn={c.prAuc.toFixed(3)} better="xgb" />
            <Row label="Overall separation" plain="ROC-AUC; random = 0.5" xgb={x.rocAuc.toFixed(3)} cnn={c.rocAuc.toFixed(3)} better="xgb" />
            <Row
              label="Finding the high-risk drives"
              plain="real failures among the top 100 / top 1,000"
              xgb={`${x.hitsAt100} / ${fmtInt(x.hitsAt1000)}`}
              cnn={`${c.hitsAt100} / ${fmtInt(c.hitsAt1000)}`}
              better="xgb"
            />
            <Row
              label="At its inspect threshold"
              plain="share of alerts that are real · share of failures caught"
              xgb={`${(x.precision * 100).toFixed(1)}% · ${(x.recall * 100).toFixed(1)}%`}
              cnn={`${(c.precision * 100).toFixed(1)}% · ${(c.recall * 100).toFixed(1)}%`}
              better="xgb"
            />
            <Row
              label="Explainability"
              plain="can it say why a drive was flagged?"
              xgb="Yes: exact contribution of every signal, per drive"
              cnn="No per-drive explanation in the product"
              better="xgb"
            />
            <Row
              label="What it reads"
              plain="input to the engine"
              xgb={`${run.n_channels * 3} summary numbers per drive`}
              cnn={`the full ${run.window_days}-day × ${run.n_channels}-signal sequence`}
              better={null}
            />
            <Row label="Size" plain="as served" xgb={xi?.size ?? "—"} cnn={ci?.size ?? "—"} better={null} />
          </tbody>
        </table>
      </div>

      <div className="pt-2" />
      <TechnicalDisclosure title="Do the two models agree?">
        <p>
          Both were evaluated on the identical {fmtInt(e.scale.testWindows)} held-out windows. They are different estimators, not
          two settings of one model: XGBoost is gradient-boosted trees over per-channel last value, mean and change; the CNN is a
          dilated 1-D convolutional network over the raw sequence, run in PyTorch. Their thresholds differ (
          {x.threshold.toFixed(3)} vs {c.threshold.toFixed(3)}) because each was tuned on validation separately, so their raw
          scores are not directly comparable. Compare rankings, not numbers.
        </p>
        <p>
          In the product, ticking “Also run the CNN” scores the same file with both and shows both trajectories, so the
          agreement on a real fleet can be checked directly.
        </p>
      </TechnicalDisclosure>
      <Source>results/w2022x_model_summary.csv · results/w2022x_cnn_vs_xgboost_precision_at_k.csv · /api/model</Source>
    </motion.div>
  );
}
