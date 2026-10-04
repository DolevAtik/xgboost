import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { RoundedBox } from "@react-three/drei";
import * as THREE from "three";

/** Concentric tracks for the platter, drawn once into a canvas. No image files. */
function usePlatterTexture() {
  return useMemo(() => {
    const size = 1024;
    const c = document.createElement("canvas");
    c.width = c.height = size;
    const g = c.getContext("2d")!;
    const r0 = size / 2;
    const grad = g.createRadialGradient(r0, r0, 0, r0, r0, r0);
    grad.addColorStop(0, "#c9d3e6");
    grad.addColorStop(0.5, "#7d8aa6");
    grad.addColorStop(1, "#aeb9d0");
    g.fillStyle = grad;
    g.fillRect(0, 0, size, size);
    for (let r = 120; r < r0; r += 3) {
      g.strokeStyle = `rgba(255,255,255,${0.025 + 0.04 * Math.random()})`;
      g.lineWidth = 1;
      g.beginPath();
      g.arc(r0, r0, r, 0, Math.PI * 2);
      g.stroke();
    }
    // A brushed highlight so rotation is visible.
    const sweep = g.createConicGradient?.(0, r0, r0);
    if (sweep) {
      sweep.addColorStop(0, "rgba(255,255,255,0.18)");
      sweep.addColorStop(0.12, "rgba(255,255,255,0)");
      sweep.addColorStop(0.5, "rgba(255,255,255,0.10)");
      sweep.addColorStop(0.62, "rgba(255,255,255,0)");
      sweep.addColorStop(1, "rgba(255,255,255,0.18)");
      g.fillStyle = sweep;
      g.fillRect(0, 0, size, size);
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    t.anisotropy = 8;
    return t;
  }, []);
}

const W = 1.6; // chassis width  (x)
const D = 2.3; // chassis depth  (z)
const H = 0.32; // chassis height (y)
const PLATTER = { x: 0, z: -0.28, r: 0.7 };
const PIVOT = { x: 0.56, z: 0.72 };
// The hidden defect sits on the platter, in the platter's own rotating frame.
const FAULT = { r: 0.5, angle: 0.9 };

interface Props {
  /** 0..1 how visible the hidden fault is. Animated by the scene. */
  faultRef?: React.RefObject<number>;
  animate?: boolean;
}

/** A stylised 3.5" drive with its cover off: chassis, platter stack, actuator, fault. */
export function DriveModel3D({ faultRef, animate = true }: Props) {
  const platter = useRef<THREE.Group>(null);
  const arm = useRef<THREE.Group>(null);
  const faultMat = useRef<THREE.MeshBasicMaterial>(null);
  const haloMat = useRef<THREE.MeshBasicMaterial>(null);
  const scan = useRef<THREE.Mesh>(null);
  const scanMat = useRef<THREE.MeshBasicMaterial>(null);
  const tex = usePlatterTexture();

  useFrame((state, dt) => {
    const t = state.clock.elapsedTime;
    if (animate && platter.current) platter.current.rotation.y -= dt * 1.6;
    if (animate && arm.current) arm.current.rotation.y = -0.55 + Math.sin(t * 0.9) * 0.18 + Math.sin(t * 2.7) * 0.04;

    // The scan ring sweeps outward from the spindle every 3.2s.
    const phase = (t % 3.2) / 3.2;
    const r = 0.12 + phase * (PLATTER.r - 0.1);
    if (scan.current) scan.current.scale.setScalar(r);
    if (scanMat.current) scanMat.current.opacity = animate ? 0.55 * (1 - phase) : 0;

    // The fault brightens as the ring crosses it, then lingers: it was there all along.
    const hit = Math.exp(-Math.pow((r - FAULT.r) / 0.06, 2));
    const base = faultRef?.current ?? 1;
    const glow = base * (0.35 + 0.65 * hit + 0.1 * Math.sin(t * 3));
    if (faultMat.current) faultMat.current.opacity = Math.min(1, 0.25 + glow);
    if (haloMat.current) haloMat.current.opacity = 0.45 * glow;
  });

  const screws = useMemo(() => {
    const p: [number, number][] = [];
    for (const sx of [-1, 1]) for (const sz of [-1, 0, 1]) p.push([sx * (W / 2 - 0.08), sz * (D / 2 - 0.1)]);
    return p;
  }, []);

  return (
    <group>
      {/* chassis */}
      <RoundedBox args={[W, H, D]} radius={0.05} smoothness={4} position={[0, -H / 2, 0]}>
        <meshStandardMaterial color="#2a3448" metalness={0.8} roughness={0.35} envMapIntensity={1.5} />
      </RoundedBox>
      {/* the recessed tray */}
      <mesh position={[0, 0.002, 0]} rotation={[-Math.PI / 2, 0, 0]}>
        <planeGeometry args={[W - 0.14, D - 0.14]} />
        <meshStandardMaterial color="#0b111d" metalness={0.6} roughness={0.7} />
      </mesh>
      {screws.map(([x, z], i) => (
        <mesh key={i} position={[x, 0.01, z]}>
          <cylinderGeometry args={[0.028, 0.028, 0.02, 16]} />
          <meshStandardMaterial color="#5b6780" metalness={1} roughness={0.25} />
        </mesh>
      ))}

      {/* platter + spindle */}
      <group position={[PLATTER.x, 0.06, PLATTER.z]}>
        <group ref={platter}>
          <mesh>
            <cylinderGeometry args={[PLATTER.r, PLATTER.r, 0.02, 128]} />
            <meshStandardMaterial map={tex} metalness={0.9} roughness={0.22} envMapIntensity={2.2} />
          </mesh>
          {/* fault point, in the platter's rotating frame */}
          <group position={[Math.cos(FAULT.angle) * FAULT.r, 0.013, Math.sin(FAULT.angle) * FAULT.r]}>
            <mesh rotation={[-Math.PI / 2, 0, 0]}>
              <circleGeometry args={[0.022, 24]} />
              <meshBasicMaterial ref={faultMat} color="#ff4d6a" transparent toneMapped={false} />
            </mesh>
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}>
              <ringGeometry args={[0.03, 0.09, 40]} />
              <meshBasicMaterial ref={haloMat} color="#ff4d6a" transparent depthWrite={false} toneMapped={false} blending={THREE.AdditiveBlending} />
            </mesh>
          </group>
        </group>
        {/* scan ring: the model reading the surface */}
        <mesh ref={scan} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.016, 0]}>
          <ringGeometry args={[0.985, 1, 128]} />
          <meshBasicMaterial ref={scanMat} color="#3fe0ff" transparent depthWrite={false} toneMapped={false} blending={THREE.AdditiveBlending} />
        </mesh>
        <mesh position={[0, 0.03, 0]}>
          <cylinderGeometry args={[0.12, 0.12, 0.05, 48]} />
          <meshStandardMaterial color="#9aa6bd" metalness={1} roughness={0.2} />
        </mesh>
        <mesh position={[0, 0.06, 0]}>
          <cylinderGeometry args={[0.05, 0.05, 0.02, 24]} />
          <meshStandardMaterial color="#2a3346" metalness={0.9} roughness={0.3} />
        </mesh>
      </group>

      {/* actuator: pivot, arm, head */}
      <group position={[PIVOT.x, 0.1, PIVOT.z]}>
        <mesh>
          <cylinderGeometry args={[0.13, 0.13, 0.12, 40]} />
          <meshStandardMaterial color="#323c52" metalness={0.9} roughness={0.3} />
        </mesh>
        <group ref={arm} rotation={[0, -0.55, 0]}>
          <mesh position={[-0.52, 0.02, 0]}>
            <boxGeometry args={[1.0, 0.025, 0.08]} />
            <meshStandardMaterial color="#b9c3d6" metalness={1} roughness={0.22} />
          </mesh>
          <mesh position={[-1.04, 0.0, 0]}>
            <boxGeometry args={[0.07, 0.025, 0.05]} />
            <meshStandardMaterial color="#3fe0ff" emissive="#3fe0ff" emissiveIntensity={1.4} toneMapped={false} />
          </mesh>
          {/* voice-coil tail */}
          <mesh position={[0.2, 0, 0]}>
            <boxGeometry args={[0.28, 0.04, 0.22]} />
            <meshStandardMaterial color="#232b3c" metalness={0.8} roughness={0.4} />
          </mesh>
        </group>
      </group>

      {/* connector edge */}
      <mesh position={[0, -H / 2, D / 2 + 0.005]}>
        <boxGeometry args={[0.9, 0.08, 0.01]} />
        <meshStandardMaterial color="#0f1522" metalness={0.5} roughness={0.6} />
      </mesh>
    </group>
  );
}
