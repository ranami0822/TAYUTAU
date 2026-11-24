import { openHarness } from "../tests/gpu/browser";
import { defaultParams } from "../src/sim/params";
import { temperature, vaporQuality } from "../src/sim/water";
const H = await openHarness();
const n = Number(process.argv[2] ?? 32);
for (const hc of (process.argv[3] ?? "3000,10000,30000").split(",").map(Number)) {
  const p = defaultParams(n);
  const L = n * p.h;
  p.hc = hc;
  p.sources = [{ x: L / 2, y: L * 0.12, z: L / 2, r: L * 0.1, T: 400 }];
  const readAt = [60, 120, 240, 360];
  const r: any = await H.page.evaluate((a) => window.tayutau.runSim(a), { params: p, steps: 360, readAt });
  const line = r.snaps.map((s: any) => {
    let maxT = 0, vap = 0, meanT = 0;
    for (const h of s.H) { maxT = Math.max(maxT, temperature(h)); vap += vaporQuality(h); meanT += temperature(h); }
    return `t=${(s.step * p.dt).toFixed(1)} maxT=${maxT.toFixed(0)} vapCells=${vap.toFixed(2)} meanT=${(meanT / s.H.length).toFixed(2)}`;
  });
  console.log(`hc=${hc}: ${line.join(" | ")}  (${(r.ms / 360).toFixed(0)} ms/step)`);
}
await H.close();
