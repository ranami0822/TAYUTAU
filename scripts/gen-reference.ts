/**
 * Stage 5: reference data for the reservoir track.
 * Runs the physics layer (CPU reference solver) on a 2D slice (nz = 1) of the tank with a heater at
 * the bottom and a cooler near the top, so convection stays unsteady, and records snapshots.
 *
 * Usage: npx tsx scripts/gen-reference.ts [n=48] [snapshots=3000] [stride=4] [outDir=reservoir-py/data]
 * Output: <outDir>/convection2d_n<n>.f32 (float32, [snapshots, 3, n, n]: T, u_c, v_c at cell centres)
 *         <outDir>/convection2d_n<n>.json (metadata)
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { CpuSolver } from "../src/sim/cpuSolver";
import { defaultParams } from "../src/sim/params";
import { temperature } from "../src/sim/water";

const n = Number(process.argv[2] ?? 48);
const snapshots = Number(process.argv[3] ?? 3000);
const stride = Number(process.argv[4] ?? 4);
const outDir = process.argv[5] ?? "reservoir-py/data";
const spinup = Number(process.env.SPINUP ?? 600);

const p = { ...defaultParams(n), nz: 1 };
const L = n * p.h;
p.sources = [
  { x: L * 0.5, y: L * 0.08, z: 0.5 * p.h, r: L * 0.08, T: 90 },
  { x: L * 0.25, y: L * 0.88, z: 0.5 * p.h, r: L * 0.08, T: 5 },
  { x: L * 0.75, y: L * 0.88, z: 0.5 * p.h, r: L * 0.08, T: 5 },
];
const s = new CpuSolver(p, 40);
const t0 = performance.now();
for (let i = 0; i < spinup; i++) s.step();
const frame = 3 * n * n;
const out = new Float32Array(snapshots * frame);
for (let k = 0; k < snapshots; k++) {
  for (let q = 0; q < stride; q++) s.step();
  const o = k * frame;
  for (let j = 0; j < n; j++)
    for (let i = 0; i < n; i++) {
      const c = s.idx(i, j, 0);
      const [u, v] = s.cellVel(s.vel, i, j, 0);
      out[o + j * n + i] = temperature(s.H[c]);
      out[o + n * n + j * n + i] = u;
      out[o + 2 * n * n + j * n + i] = v;
    }
  if (k % 500 === 0) console.log(`snapshot ${k}/${snapshots}  t=${s.time.toFixed(2)} s  vmax=${s.maxSpeed().toFixed(4)}  ${((performance.now() - t0) / 1000).toFixed(0)} s`);
}
mkdirSync(outDir, { recursive: true });
const base = path.join(outDir, `convection2d_n${n}`);
writeFileSync(base + ".f32", Buffer.from(out.buffer));
writeFileSync(
  base + ".json",
  JSON.stringify(
    { n, snapshots, stride, dtSnapshot: stride * p.dt, spinupSteps: spinup, channels: ["T", "u", "v"], layout: "[snapshot][channel][j][i] float32 little-endian", params: { ...p } },
    null,
    2,
  ),
);
console.log("wrote", base + ".f32");
