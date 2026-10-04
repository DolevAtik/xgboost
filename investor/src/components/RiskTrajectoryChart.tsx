import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DriveDetail, EstimatorKind } from "../../shared/types";

const SERIES: Record<EstimatorKind, { label: string; color: string }> = {
  xgb: { label: "XGBoost", color: "var(--color-series-xgb)" },
  cnn: { label: "CNN", color: "var(--color-series-cnn)" },
};

/**
 * The same model re-scored with the window slid back a day at a time. A drive climbing
 * across the threshold is deteriorating; one that sits high is merely unwell.
 */
export function RiskTrajectoryChart({
  trajectory,
  kinds,
  thresholds,
}: {
  trajectory: DriveDetail["trajectory"];
  kinds: EstimatorKind[];
  thresholds: Partial<Record<EstimatorKind, number>>;
}) {
  const data = trajectory.dates.map((d, i) => {
    const row: Record<string, string | number> = { date: d };
    for (const k of kinds) {
      const v = trajectory[k]?.[i];
      if (v !== undefined) row[k] = v;
    }
    return row;
  });
  const primary = kinds[0];
  const multi = kinds.length > 1;

  if (data.length < 2)
    return (
      <p className="rounded-xl border border-dashed border-line p-5 text-sm text-muted">
        This drive has no history in the file before its current window, so there is no earlier day to re-score and no trajectory
        to draw. Drives with a longer record show how their risk evolved.
      </p>
    );

  return (
    <div className="h-56">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
          <CartesianGrid vertical={false} />
          <XAxis dataKey="date" tickLine={false} axisLine={false} minTickGap={36} tickFormatter={(d: string) => d.slice(5)} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickLine={false} axisLine={false} width={36} />
          {thresholds[primary] !== undefined && (
            <ReferenceLine
              y={thresholds[primary]}
              stroke="var(--color-warn)"
              strokeDasharray="4 4"
              label={{ value: `${SERIES[primary].label} inspect threshold`, fill: "var(--color-muted)", fontSize: 10, position: "insideBottomRight", dy: 14 }}
            />
          )}
          <Tooltip
            contentStyle={{ background: "#0a1122", border: "1px solid rgba(148,163,200,.24)", borderRadius: 10, fontSize: 12 }}
            formatter={(v, name) => [Number(v).toFixed(3), SERIES[name as EstimatorKind]?.label ?? String(name)]}
          />
          {multi && <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} formatter={(v) => SERIES[v as EstimatorKind]?.label ?? v} />}
          {kinds.map((k) => (
            <Line key={k} type="monotone" dataKey={k} stroke={SERIES[k].color} strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
