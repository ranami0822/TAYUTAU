import { CpuSolver } from "../src/sim/cpuSolver";
import { defaultParams } from "../src/sim/params";
import { temperature } from "../src/sim/water";
const n = 16;
const p = defaultParams(n);
const L = n * p.h;
p.sources = [{ x: L / 2, y: 0.1 * L, z: L / 2, r: 0.15 * L, T: 300 }];
const s = new CpuSolver(p);
const E0 = s.totalEnergy();
const t0 = performance.now();
for (let it = 1; it <= 600; it++) {
  s.step();
  if (it % 60 === 0) {
    let Tmax = -1e9, Tmean = 0;
    for (const H of s.H) { const T = temperature(H); Tmax = Math.max(Tmax, T); Tmean += T / s.H.length; }
    const ph = s.phaseMasses();
    console.log(`t=${s.time.toFixed(2)} Tmean=${Tmean.toFixed(2)} Tmax=${Tmax.toFixed(1)} vmax=${s.maxSpeed().toFixed(4)} div=${s.maxDivergence().toExponential(2)} dE-inj=${((s.totalEnergy()-E0-s.injectedEnergy())/(s.totalEnergy())).toExponential(2)} vap=${ph.vapor.toExponential(2)} ms/step=${((performance.now()-t0)/it).toFixed(1)}`);
  }
}
