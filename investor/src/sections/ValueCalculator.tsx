import { useState } from "react";
import { ArrowRight, Calculator } from "lucide-react";
import type { Evaluation } from "../../shared/types";
import { StorySection } from "../components/StorySection";
import { ErrorState, Loading, Source } from "../components/States";
import { TechnicalDisclosure } from "../components/TechnicalDisclosure";
import { openContact } from "../lib/contact";
import { useStoryData } from "../lib/data";
import { fmtInt, fmtPct } from "../lib/format";

/** Rates measured in this project, derived from the evaluation files — never typed in. */
function measuredRates(e: Evaluation) {
  const xgb = e.models.find((m) => m.model === "XGBoost");
  const failures = e.scale.splits.reduce((a, s) => a + s.drivesWithFailure, 0);
  const driveYears = e.scale.driveDays / 365.25;
  return {
    failures,
    driveYears,
    annualFailureRate: failures / driveYears,
    caughtShare: e.leadTime?.xgbFlaggedWithinHorizon ?? 0,
    precision: xgb?.precision ?? 0,
  };
}

const money = (n: number) =>
  `${n < 0 ? "−" : ""}$${Math.abs(n) >= 1e6 ? `${(Math.abs(n) / 1e6).toFixed(2)}M` : fmtInt(Math.abs(n))}`;

function NumberField({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step,
  prefix,
}: {
  label: string;
  hint: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step: number;
  prefix?: string;
}) {
  return (
    <label className="block">
      <span className="flex items-baseline justify-between gap-3">
        <span className="text-sm font-medium text-ink">{label}</span>
        <span className="rounded bg-violet/15 px-1.5 py-0.5 text-[10px] text-violet">your number</span>
      </span>
      <span className="mt-0.5 block text-xs text-muted">{hint}</span>
      <span className="mt-3 flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={Math.min(max, value)}
          onChange={(e) => onChange(Number(e.target.value))}
          className="h-1.5 flex-1 cursor-pointer accent-[#3fe0ff]"
          aria-label={label}
        />
        <span className="flex w-36 items-center rounded-lg border border-line bg-void/60 px-3 py-2 focus-within:border-signal">
          {prefix && <span className="mr-1 text-sm text-muted">{prefix}</span>}
          <input
            type="number"
            min={0}
            value={value}
            onChange={(e) => onChange(Math.max(0, Number(e.target.value) || 0))}
            className="w-full bg-transparent text-right text-sm tabular-nums text-ink outline-none"
            aria-label={`${label} (exact value)`}
          />
        </span>
      </span>
    </label>
  );
}

function Row({ label, value, tag, tone = "text-ink" }: { label: string; value: string; tag?: "measured" | "result"; tone?: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 border-b hairline py-3 last:border-0">
      <span className="text-sm text-soft">
        {label}
        {tag === "measured" && <span className="ml-2 rounded bg-signal/10 px-1.5 py-0.5 text-[10px] text-signal">from our data</span>}
      </span>
      <span className={`text-base font-semibold tabular-nums ${tone}`}>{value}</span>
    </div>
  );
}

export function ValueCalculatorView() {
  const { evaluation } = useStoryData();
  const [fleet, setFleet] = useState(10000);
  const [outageCost, setOutageCost] = useState(1000);
  const [inspectCost, setInspectCost] = useState(100);

  if (evaluation.loading) return <Loading label="Loading the measured rates" />;
  if (!evaluation.data) return <ErrorState message={evaluation.error ?? "no evaluation results"} onRetry={evaluation.reload} />;
  const r = measuredRates(evaluation.data);

  const failuresPerYear = fleet * r.annualFailureRate;
  const caught = failuresPerYear * r.caughtShare;
  const stillUnplanned = failuresPerYear - caught;
  // Each real alert comes with (1/precision − 1) false ones at the inspect threshold.
  const falseAlarms = r.precision > 0 ? caught * (1 / r.precision - 1) : 0;
  const avoided = caught * outageCost;
  const inspections = (caught + falseAlarms) * inspectCost;
  const net = avoided - inspections;

  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
      <div className="panel space-y-7">
        <div className="flex items-center gap-3">
          <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-signal/40 bg-signal/10 text-signal">
            <Calculator size={17} aria-hidden />
          </span>
          <div>
            <h3 className="panel-title">Your fleet</h3>
            <p className="panel-sub">Change these to match your company. The starting numbers are only examples.</p>
          </div>
        </div>
        <NumberField
          label="How many hard drives do you have?"
          hint="All the drives in your data centers"
          value={fleet}
          onChange={setFleet}
          min={1000}
          max={500000}
          step={1000}
        />
        <NumberField
          label="What does a surprise breakdown cost you?"
          hint="Downtime, urgent repair work and getting the data back (not counting the new drive)"
          value={outageCost}
          onChange={setOutageCost}
          min={0}
          max={20000}
          step={100}
          prefix="$"
        />
        <NumberField
          label="What does it cost to check or replace a drive in advance?"
          hint="Planned work during regular maintenance. Some checked drives will turn out fine; that is included below."
          value={inspectCost}
          onChange={setInspectCost}
          min={0}
          max={2000}
          step={10}
          prefix="$"
        />
      </div>

      <div className="panel flex flex-col border-signal/30">
        <p className="text-sm text-muted">Estimated savings per year</p>
        <p className={`mt-1 text-[clamp(2.5rem,5vw,3.5rem)] font-semibold leading-none tracking-tight tabular-nums ${net >= 0 ? "text-signal" : "text-risk"}`}>
          {money(net)}
        </p>
        <p className="mt-2 text-sm text-soft">
          Of the {fmtInt(failuresPerYear)} drives expected to break this year, {fmtInt(caught)} would be replaced before they break.
        </p>

        <div className="mt-6">
          <Row label="Drives expected to break each year" value={fmtInt(failuresPerYear)} />
          <Row label="Share of drives that break each year" value={fmtPct(r.annualFailureRate, 2)} tag="measured" />
          <Row label={`Caught early (${fmtPct(r.caughtShare, 0)} of breakdowns)`} value={fmtInt(caught)} tag="measured" tone="text-signal" />
          <Row label="Still break without warning" value={fmtInt(stillUnplanned)} tone="text-risk" />
          <Row label="Checks on drives that turn out fine" value={fmtInt(falseAlarms)} tag="measured" />
          <Row label="Money saved on breakdowns" value={money(avoided)} />
          <Row label="Cost of the checks and early replacements" value={money(-inspections)} />
        </div>

        <div className="mt-auto flex flex-wrap items-center justify-between gap-3 border-t hairline pt-5">
          <p className="text-sm text-soft">Want this calculated from your own drives?</p>
          <button type="button" onClick={openContact} className="btn btn-primary">
            Get in touch <ArrowRight size={15} aria-hidden />
          </button>
        </div>
      </div>

      <div className="lg:col-span-2">
        <TechnicalDisclosure title="How is this calculated?">
          <p>
            <b>How often drives break:</b> in the fleet we studied, {fmtInt(r.failures)} drives broke over {fmtInt(r.driveYears)} drive-years,
            so about {fmtPct(r.annualFailureRate, 2)} of drives break each year.
          </p>
          <p>
            <b>How many we catch early:</b> {fmtPct(r.caughtShare, 0)} of the drives that broke during our test were flagged in the 30 days
            before they broke. <b>Checks that turn out fine:</b> about half of the drives the system flags really are about to break, so for
            every drive caught there is roughly one extra check on a healthy drive.
          </p>
          <p>
            <b>Savings</b> = drives caught × the cost of a breakdown − every check × the cost of a check. The price of the new drive is left out,
            because a broken drive gets replaced either way. These rates come from one public fleet; yours may be different, and should be
            checked on your own data.
          </p>
        </TechnicalDisclosure>
        <Source>results/w2022x_summary.json · results/w2022x_window_coverage.csv · results/w2022x_lead_time_failing.csv</Source>
      </div>
    </div>
  );
}

/** The value: what the measured rates mean for a fleet of your size. */
export default function ValueCalculator() {
  return (
    <StorySection
      id="value"
      kicker="The value"
      band
      title={
        <>
          What it is <span className="text-signal">worth</span> to your fleet.
        </>
      }
      lede="Tell us how many drives you have and what a breakdown costs you. We do the math using the results from our own testing."
    >
      <ValueCalculatorView />
    </StorySection>
  );
}
