import { useMemo, useRef } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Environment, Lightformer } from "@react-three/drei";
import * as THREE from "three";
import { DriveModel3D } from "./DriveModel3D";

/** Telemetry rising off the drive: a slow column of points that fades as it climbs. */
function TelemetryParticles({ count, animate }: { count: number; animate: boolean }) {
  const ref = useRef<THREE.Points>(null);
  const { positions, speeds } = useMemo(() => {
    const positions = new Float32Array(count * 3);
    const speeds = new Float32Array(count);
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = 0.15 + Math.random() * 1.6;
      positions[i * 3] = Math.cos(a) * r;
      positions[i * 3 + 1] = Math.random() * 2.6;
      positions[i * 3 + 2] = Math.sin(a) * r * 1.2;
      speeds[i] = 0.08 + Math.random() * 0.22;
    }
    return { positions, speeds };
  }, [count]);

  useFrame((_, dt) => {
    if (!animate || !ref.current) return;
    const p = ref.current.geometry.attributes.position as THREE.BufferAttribute;
    for (let i = 0; i < count; i++) {
      let y = p.getY(i) + speeds[i] * dt;
      if (y > 2.6) y = 0;
      p.setY(i, y);
    }
    p.needsUpdate = true;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.018}
        color="#3fe0ff"
        transparent
        opacity={0.55}
        depthWrite={false}
        blending={THREE.AdditiveBlending}
        sizeAttenuation
        toneMapped={false}
      />
    </points>
  );
}

/** Follows the pointer with a little lag, so the drive feels physical. */
function Rig({ animate, children }: { animate: boolean; children: React.ReactNode }) {
  const g = useRef<THREE.Group>(null);
  useFrame((state, dt) => {
    if (!g.current) return;
    const t = state.clock.elapsedTime;
    const tx = animate ? state.pointer.y * -0.12 + 0.55 : 0.55;
    const ty = animate ? state.pointer.x * 0.35 + Math.sin(t * 0.15) * 0.15 - 0.5 : -0.5;
    g.current.rotation.x = THREE.MathUtils.damp(g.current.rotation.x, tx, 3, dt);
    g.current.rotation.y = THREE.MathUtils.damp(g.current.rotation.y, ty, 3, dt);
    g.current.position.y = animate ? Math.sin(t * 0.6) * 0.04 : 0;
  });
  return <group ref={g}>{children}</group>;
}

export interface HeroSceneProps {
  animate?: boolean;
  compact?: boolean;
}

/** Artifact A: the cinematic opening. A dark room, one drive, and the defect inside it. */
export default function HeroScene({ animate = true, compact = false }: HeroSceneProps) {
  return (
    <Canvas
      dpr={compact ? [1, 1.5] : [1, 2]}
      frameloop={animate ? "always" : "demand"}
      camera={{ position: [0, 1.4, 5.2], fov: compact ? 40 : 34 }}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      aria-hidden
    >
      <fog attach="fog" args={["#04060b", 5, 11]} />
      <ambientLight intensity={0.6} />
      <spotLight position={[3, 5, 2]} angle={0.6} penumbra={1} intensity={120} color="#b8d6ff" />
      <directionalLight position={[-2, 4, 3]} intensity={2.2} color="#cfe0ff" />
      <pointLight position={[-3, 1, -2]} intensity={14} color="#9b7bff" />
      <pointLight position={[2, -1, 3]} intensity={6} color="#3fe0ff" />
      <Environment resolution={128} frames={1}>
        <Lightformer form="rect" intensity={2.5} color="#cfe6ff" position={[0, 4, 1]} scale={[6, 1.2, 1]} rotation-x={Math.PI / 2} />
        <Lightformer form="rect" intensity={1.2} color="#9b7bff" position={[-4, 1, -1]} scale={[2, 4, 1]} rotation-y={Math.PI / 2} />
        <Lightformer form="ring" intensity={1.5} color="#3fe0ff" position={[3, 2, 3]} scale={1.5} />
      </Environment>

      {/* The canvas is the drive's own column, so the drive is centred in it. The rig
          rotates it about its own centre. */}
      <group position={compact ? [0, 0.45, 0] : [-0.45, 0.0, 0]} scale={compact ? 0.82 : 1.08}>
        <Rig animate={animate}>
          <DriveModel3D animate={animate} />
          <TelemetryParticles count={compact ? 220 : 520} animate={animate} />
        </Rig>
      </group>
    </Canvas>
  );
}
