import { CpuSolver } from "../src/sim/cpuSolver";
import { defaultParams } from "../src/sim/params";
const n = 12;
const p = defaultParams(n);
const L = n * p.h;
p.sources = [{ x: L / 2, y: L * 0.12, z: L / 2, r: L * 0.15, T: Number(process.argv[2] ?? 90) }];
const s = new CpuSolver(p);
for (let it = 1; it <= 240; it++) {
  s.step();
  if (it % 40 === 0) {
    const col = []; const vcol = [];
    for (let j = 0; j < n; j++) { col.push(s.temperatureAt(n/2, j, n/2).toFixed(1)); vcol.push((s.vel[s.vIdx(n/2, j, n/2)]*100).toFixed(2)); }
    console.log(`t=${s.time.toFixed(2)} T col: ${col.join(" ")}\n        v(cm/s): ${vcol.join(" ")}  vmax=${s.maxSpeed().toFixed(3)}`);
  }
}
