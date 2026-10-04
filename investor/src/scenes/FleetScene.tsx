import { useEffect, useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

export type FleetMode = "reactive" | "predictive";

const IDLE = new THREE.Color("#1b2640");
const DARK = new THREE.Color("#07090f");
const RED = new THREE.Color("#ff4d6a");
const AMBER = new THREE.Color("#ffb547");
const CYAN = new THREE.Color("#3fe0ff");

/**
 * A fleet as a field of drives. The same random wear events happen in both modes; what
 * changes is when the operator learns about them.
 *   reactive   — a unit flashes red, goes dark (downtime), then is replaced.
 *   predictive — the unit turns amber first (flagged), is serviced (cyan), never goes dark.
 * An illustration of the operating model, not a simulation of measured rates.
 */
function Fleet({ mode, cols, rows, animate }: { mode: FleetMode; cols: number; rows: number; animate: boolean }) {
  const n = cols * rows;
  const mesh = useRef<THREE.InstancedMesh>(null);
  const age = useMemo(() => new Float32Array(n).fill(-1), [n]);
  const modeRef = useRef(mode);
  modeRef.current = mode;
  const tmp = useMemo(() => new THREE.Color(), []);

  useEffect(() => {
    const m = mesh.current;
    if (!m) return;
    const o = new THREE.Object3D();
    let i = 0;
    for (let r = 0; r < rows; r++)
      for (let c = 0; c < cols; c++) {
        o.position.set((c - (cols - 1) / 2) * 0.62, 0, (r - (rows - 1) / 2) * 0.9);
        o.updateMatrix();
        m.setMatrixAt(i, o.matrix);
        m.setColorAt(i, IDLE);
        i++;
      }
    m.instanceMatrix.needsUpdate = true;
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }, [cols, rows]);

  // A mode switch starts the field clean, so no unit is left frozen mid-event.
  useEffect(() => {
    age.fill(-1);
    const m = mesh.current;
    if (!m) return;
    for (let i = 0; i < n; i++) m.setColorAt(i, IDLE);
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
  }, [mode, age, n]);

  useFrame((state, dt) => {
    const m = mesh.current;
    if (!m) return;
    if (animate && Math.random() < dt * 2.2) {
      const i = Math.floor(Math.random() * n);
      if (age[i] < 0) age[i] = 0;
    }
    const predictive = modeRef.current === "predictive";
    for (let i = 0; i < n; i++) {
      if (age[i] < 0) continue;
      age[i] += dt;
      const a = age[i];
      if (predictive) {
        // flagged (amber) -> serviced (cyan pulse) -> idle; never offline
        if (a < 1.6) tmp.copy(IDLE).lerp(AMBER, Math.min(1, a / 0.5));
        else if (a < 2.4) tmp.copy(AMBER).lerp(CYAN, (a - 1.6) / 0.8);
        else if (a < 3.4) tmp.copy(CYAN).lerp(IDLE, (a - 2.4) / 1.0);
        else age[i] = -1;
      } else {
        // fails (red flash) -> offline (dark) -> replaced
        if (a < 0.5) tmp.copy(IDLE).lerp(RED, a / 0.15 > 1 ? 1 : a / 0.15);
        else if (a < 3.6) tmp.copy(RED).lerp(DARK, Math.min(1, (a - 0.5) / 0.4));
        else if (a < 4.4) tmp.copy(DARK).lerp(IDLE, (a - 3.6) / 0.8);
        else age[i] = -1;
      }
      if (age[i] < 0) tmp.copy(IDLE);
      m.setColorAt(i, tmp);
    }
    if (m.instanceColor) m.instanceColor.needsUpdate = true;
    m.rotation.y = animate ? Math.sin(state.clock.elapsedTime * 0.08) * 0.12 : 0;
  });

  return (
    <instancedMesh ref={mesh} args={[undefined, undefined, n]}>
      <boxGeometry args={[0.46, 0.08, 0.7]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
  );
}

export default function FleetScene({ mode, compact = false, animate = true }: { mode: FleetMode; compact?: boolean; animate?: boolean }) {
  return (
    <Canvas
      dpr={[1, 1.75]}
      frameloop={animate ? "always" : "demand"}
      camera={{ position: [0, 7.5, 7.5], fov: compact ? 52 : 38 }}
      gl={{ antialias: true, alpha: true }}
      aria-hidden
    >
      <fog attach="fog" args={["#04060b", 8, 17]} />
      <Fleet mode={mode} cols={compact ? 9 : 18} rows={compact ? 9 : 10} animate={animate} />
    </Canvas>
  );
}
