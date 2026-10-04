import { useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion, useInView, useReducedMotion } from "framer-motion";
import { Pause, Play, RotateCcw } from "lucide-react";

const COLS = 16;
const ROWS = 6;
const N = COLS * ROWS;
const TICK_MS = 120;
const SPAWN_EVERY_S = 0.7;
const PULSES_PER_TICK = 4;
const LOG_SIZE = 5;

// One wear event, seconds from its start: visible to the model from FLAG_AT, and the
// drive fails at FAIL_AT unless it was replaced first.
const FLAG_AT = 0.4;
const FAIL_AT = 2.6;
const FAILED_FOR = 1.0; // red flash
const OFFLINE_FOR = 3.4; // outage, until the drive is replaced
const SERVICED_FOR = 1.4; // cyan: replaced in a planned window
const LIFETIME = FAIL_AT + FAILED_FOR + OFFLINE_FOR;

type CellState = "ok" | "pulse" | "flagged" | "serviced" | "failed" | "offline";

const STYLE: Record<CellState, string> = {
  ok: "bg-[#1a2540]",
  pulse: "bg-[#26375f]",
  flagged: "bg-warn",
  serviced: "bg-signal",
  failed: "bg-risk shadow-[0_0_14px_rgba(255,77,106,0.7)]",
  offline: "bg-[#05070c] ring-1 ring-risk/40",
};

const LABEL: Record<CellState, string> = {
  ok: "healthy",
  pulse: "healthy, reporting telemetry",
  flagged: "flagged for inspection",
  serviced: "replaced in a planned window",
  failed: "failed",
  offline: "offline (outage)",
};

const LEGEND: CellState[] = ["ok", "flagged", "serviced", "failed", "offline"];

type Tone = "risk" | "warn" | "signal" | "muted";
interface LogLine {
  id: number;
  at: number;
  text: string;
  tone: Tone;
}

interface WearEvent {
  cell: number;
  t0: number;
  caught: boolean;
}

interface Sim {
  t: number;
  nextSpawn: number;
  rand: () => number;
  events: WearEvent[];
  busyUntil: number[];
  counts: { reactiveOutages: number; predictiveOutages: number; planned: number };
  logReactive: LogLine[];
  logPredictive: LogLine[];
  pulse: Set<number>;
  seq: number;
  spawned: number;
}

/** Deterministic, so both fleets — and every visitor — see the same story. */
function rng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const driveId = (cell: number) => `R${Math.floor(cell / COLS) + 1}-${String((cell % COLS) + 1).padStart(2, "0")}`;
const clock = (s: number) => `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

function newSim(): Sim {
  return {
    t: 0,
    nextSpawn: 0.3,
    rand: rng(7),
    events: [],
    busyUntil: new Array<number>(N).fill(-1),
    counts: { reactiveOutages: 0, predictiveOutages: 0, planned: 0 },
    logReactive: [],
    logPredictive: [],
    pulse: new Set(),
    seq: 0,
    spawned: 0,
  };
}

/** Advance the simulation by dt seconds, logging every boundary an event crosses. */
function step(sim: Sim, dt: number, catchRate: number) {
  const prev = sim.t;
  const now = prev + dt;
  sim.t = now;

  while (sim.nextSpawn <= now) {
    let cell = Math.floor(sim.rand() * N);
    for (let k = 0; k < N && sim.busyUntil[cell] > sim.nextSpawn; k++) cell = (cell + 7) % N;
    sim.busyUntil[cell] = sim.nextSpawn + LIFETIME + 0.5;
    // Spread catches evenly rather than randomly, so the share caught always tracks the
    // measured rate instead of drifting with luck.
    const n = sim.spawned++;
    const caught = Math.floor((n + 1) * catchRate) > Math.floor(n * catchRate);
    sim.events.push({ cell, t0: sim.nextSpawn, caught });
    sim.nextSpawn += SPAWN_EVERY_S * (0.6 + sim.rand() * 0.8);
  }

  const crossed = (e: WearEvent, at: number) => prev < e.t0 + at && e.t0 + at <= now;
  const log = (list: LogLine[], text: string, tone: Tone, at: number) => {
    list.unshift({ id: sim.seq++, at, text, tone });
    list.length = Math.min(list.length, LOG_SIZE);
  };

  for (const e of sim.events) {
    const id = driveId(e.cell);
    if (e.caught && crossed(e, FLAG_AT)) log(sim.logPredictive, `${id} flagged · inspection scheduled`, "warn", e.t0 + FLAG_AT);
    if (crossed(e, FAIL_AT)) {
      sim.counts.reactiveOutages++;
      log(sim.logReactive, `${id} failed · unplanned outage`, "risk", e.t0 + FAIL_AT);
      if (e.caught) {
        sim.counts.planned++;
        log(sim.logPredictive, `${id} replaced in planned window · no outage`, "signal", e.t0 + FAIL_AT);
      } else {
        sim.counts.predictiveOutages++;
        log(sim.logPredictive, `${id} failed · not flagged in time`, "risk", e.t0 + FAIL_AT);
      }
    }
    if (crossed(e, LIFETIME)) {
      log(sim.logReactive, `${id} back online after replacement`, "muted", e.t0 + LIFETIME);
      if (!e.caught) log(sim.logPredictive, `${id} back online after replacement`, "muted", e.t0 + LIFETIME);
    }
  }
  sim.events = sim.events.filter((e) => now - e.t0 < LIFETIME + 0.1);

  // Telemetry heartbeat: a few drives report in each tick, the same ones in both fleets.
  sim.pulse = new Set(Array.from({ length: PULSES_PER_TICK }, () => Math.floor(sim.rand() * N)));
}

function cellsOf(sim: Sim, predictive: boolean): CellState[] {
  const out = new Array<CellState>(N).fill("ok");
  for (const c of sim.pulse) out[c] = "pulse";
  for (const e of sim.events) {
    const a = sim.t - e.t0;
    let s: CellState | null = null;
    if (predictive && e.caught) {
      if (a >= FLAG_AT && a < FAIL_AT) s = "flagged";
      else if (a >= FAIL_AT && a < FAIL_AT + SERVICED_FOR) s = "serviced";
    } else if (a >= FAIL_AT && a < FAIL_AT + FAILED_FOR) s = "failed";
    else if (a >= FAIL_AT + FAILED_FOR && a < LIFETIME) s = "offline";
    if (s) out[e.cell] = s;
  }
  return out;
}

const TONE: Record<Tone, string> = { risk: "text-risk", warn: "text-warn", signal: "text-signal", muted: "text-muted" };

function Counter({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div>
      <dt className="text-[11px] text-muted">{label}</dt>
      <dd className="relative mt-0.5 h-8 overflow-hidden">
        <AnimatePresence initial={false} mode="popLayout">
          <motion.span
            key={value}
            initial={{ y: 14, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: -14, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className={`absolute left-0 text-2xl font-semibold tabular-nums ${value ? tone : "text-ink"}`}
          >
            {value}
          </motion.span>
        </AnimatePresence>
      </dd>
    </div>
  );
}

function Fleet({
  title,
  sub,
  badge,
  predictive,
  cells,
  outages,
  planned,
  log,
  running,
  sweep,
}: {
  title: string;
  sub: string;
  badge: string;
  predictive: boolean;
  cells: CellState[];
  outages: number;
  planned: number;
  log: LogLine[];
  running: boolean;
  sweep: boolean;
}) {
  const down = cells.filter((c) => c === "offline" || c === "failed").length;
  return (
    <div className={`panel flex flex-col ${predictive ? "border-signal/30" : ""}`}>
      <div className="mb-4 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h3 className="panel-title">{title}</h3>
          <p className="panel-sub">{sub}</p>
        </div>
        <span className={`rounded-full border px-2.5 py-0.5 text-[11px] ${predictive ? "border-signal/50 text-signal" : "border-risk/40 text-risk"}`}>
          {badge}
        </span>
      </div>

      <div className="relative overflow-hidden rounded-md">
        <div
          className="grid gap-1"
          style={{ gridTemplateColumns: `repeat(${COLS}, minmax(0, 1fr))` }}
          role="img"
          aria-label={`${title}: ${outages} unplanned outages${predictive ? `, ${planned} planned replacements` : ""} so far.`}
        >
          {cells.map((c, i) => (
            <span key={i} title={`${driveId(i)} · ${LABEL[c]}`} className={`aspect-[4/5] rounded-[3px] transition-colors duration-300 ${STYLE[c]}`} />
          ))}
        </div>
        {predictive && running && sweep && (
          <motion.div
            aria-hidden
            className="pointer-events-none absolute inset-y-0 w-16 bg-gradient-to-r from-transparent via-signal/15 to-transparent"
            initial={{ left: "-15%" }}
            animate={{ left: "105%" }}
            transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          />
        )}
      </div>

      <dl className="mt-5 grid grid-cols-3 gap-3 border-t hairline pt-4">
        <Counter label="Unplanned outages" value={outages} tone="text-risk" />
        <Counter label="Planned replacements" value={planned} tone="text-signal" />
        <Counter label="Drives down now" value={down} tone="text-ink" />
      </dl>

      <div className="mt-4 border-t hairline pt-3">
        <p className="mb-2 text-[11px] text-muted">Event feed</p>
        <ol className="h-[7.5rem] space-y-1 overflow-hidden font-mono text-[11.5px]">
          <AnimatePresence initial={false}>
            {log.map((l) => (
              <motion.li
                key={l.id}
                layout
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.25 }}
                className="flex gap-3 truncate"
              >
                <span className="text-faint">{clock(l.at)}</span>
                <span className={`truncate ${TONE[l.tone]}`}>{l.text}</span>
              </motion.li>
            ))}
          </AnimatePresence>
          {!log.length && <li className="text-faint">waiting for telemetry…</li>}
        </ol>
      </div>
    </div>
  );
}

/**
 * The same wear events, hitting two identical fleets in real time. One finds out when the
 * drive dies; the other is warned first for the share of failures the model actually catches.
 */
export function FleetComparison({ catchRate }: { catchRate: number }) {
  const reduce = useReducedMotion();
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { margin: "-10% 0px" });
  const sim = useRef<Sim>(newSim());
  const [, setFrame] = useState(0);
  // The simulation runs under reduced motion too: what changes is state (a drive's
  // colour, a counter, a log line), not decorative movement, and it can be paused.
  // Only the decorative motion (the scan sweep, sliding and rolling text) is dropped.
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    if (!playing || !inView) return;
    const id = setInterval(() => {
      step(sim.current, TICK_MS / 1000, catchRate);
      setFrame((f) => f + 1);
    }, TICK_MS);
    return () => clearInterval(id);
  }, [playing, inView, catchRate]);

  const restart = useCallback(() => {
    sim.current = newSim();
    setFrame((f) => f + 1);
  }, []);

  const s = sim.current;
  const running = playing && inView;

  return (
    <div ref={ref}>
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-xl border hairline bg-night/60 px-4 py-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.14em] text-ink">
            <span className="relative flex h-2.5 w-2.5">
              {running && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-risk opacity-70" />}
              <span className={`relative inline-flex h-2.5 w-2.5 rounded-full ${running ? "bg-risk" : "bg-faint"}`} />
            </span>
            {running ? "Live simulation" : "Simulation paused"}
          </span>
          <span className="num text-xs text-muted">
            running <span className="text-soft">{clock(s.t)}</span>
          </span>
          <span className="hidden text-xs text-muted sm:inline">2 fleets · {N} drives each · the same failures</span>
        </div>
        {(
          <div className="flex items-center gap-2">
            <button type="button" onClick={() => setPlaying((p) => !p)} className="chip hover:border-signal hover:text-ink">
              {playing ? <Pause size={12} aria-hidden /> : <Play size={12} aria-hidden />} {playing ? "Pause" : "Play"}
            </button>
            <button type="button" onClick={restart} className="chip hover:border-signal hover:text-ink">
              <RotateCcw size={12} aria-hidden /> Restart
            </button>
          </div>
        )}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Fleet
          title="Reactive maintenance"
          sub="The drive is replaced after it fails"
          badge="Today"
          predictive={false}
          cells={cellsOf(s, false)}
          outages={s.counts.reactiveOutages}
          planned={0}
          log={s.logReactive}
          running={running}
          sweep={!reduce}
        />
        <Fleet
          title="Predictive maintenance"
          sub="At-risk drives are flagged and replaced first"
          badge="With Drive Foresight"
          predictive
          cells={cellsOf(s, true)}
          outages={s.counts.predictiveOutages}
          planned={s.counts.planned}
          log={s.logPredictive}
          running={running}
          sweep={!reduce}
        />
      </div>

      <ul className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-xs text-muted" aria-label="Legend">
        {LEGEND.map((st) => (
          <li key={st} className="flex items-center gap-2">
            <span className={`inline-block h-3 w-2.5 rounded-[2px] ${STYLE[st]}`} />
            {LABEL[st].charAt(0).toUpperCase() + LABEL[st].slice(1)}
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs leading-relaxed text-faint">
        Simulation. Both fleets receive exactly the same failures at the same moments. In the predictive fleet, {Math.round(catchRate * 100)}% of
        failures are caught in advance — the share of failing test drives the model flagged within their final 30 days. The rest still fail. Timing
        and failure frequency are illustrative, not data.
      </p>
    </div>
  );
}
