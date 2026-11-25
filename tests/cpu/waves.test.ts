import { describe, expect, it } from "vitest";
import { CpuSolver } from "../../src/sim/cpuSolver";
import { waveSubsteps } from "../../src/sim/params";
import { enthalpyFromT } from "../../src/sim/water";
import { smallParams, sum, toF64 } from "./helpers";

const n = 16;
function bump(s: CpuSolver, amp: number) {
  // a Gaussian hump in the middle of the surface, with the mean removed so the mean level stays put
  const { nx, nz } = s.g;
  for (let k = 0; k < nz; k++)
    for (let i = 0; i < nx; i++) s.eta[i + nx * k] = amp * Math.exp(-((i - nx / 2 + 0.5) ** 2 + (k - nz / 2 + 0.5) ** 2) / 4);
  const m = sum(s.eta) / s.eta.length;
  for (let c = 0; c < s.eta.length; c++) s.eta[c] -= m;
}

describe("surface waves", () => {
  it("water and air volumes are conserved while the surface moves (closed tank)", () => {
    const p = smallParams(n, { waterLevel: 0.7 });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L * 0.12, z: L / 2, r: L * 0.15, T: 400 }];
    const s = toF64(new CpuSolver(p));
    (s as unknown as { eta: Float64Array }).eta = Float64Array.from(s.eta);
    (s as unknown as { q: Float64Array }).q = Float64Array.from(s.q);
    bump(s, 0.5 * p.h);
    const Vw0 = s.waterVolume();
    const Va0 = s.airVolume();
    let moved = 0;
    for (let it = 0; it < 60; it++) {
      const before = Array.from(s.eta);
      s.step();
      moved = Math.max(moved, ...s.eta.map((e, c) => Math.abs(e - before[c])));
    }
    expect(moved).toBeGreaterThan(1e-6); // the surface height really changes with time
    expect(Math.abs(s.waterVolume() - Vw0) / Vw0).toBeLessThan(1e-12);
    expect(Math.abs(s.airVolume() - Va0) / Va0).toBeLessThan(1e-12);
    expect(s.waterVolume() + s.airVolume()).toBeCloseTo(n * n * n * p.h ** 3, 15);
  });

  it("the same holds in f32 storage to rounding", () => {
    const p = smallParams(n, { waterLevel: 0.7 });
    const s = new CpuSolver(p);
    bump(s, p.h);
    const sum0 = sum(s.eta);
    for (let it = 0; it < 60; it++) s.step();
    expect(Math.abs(sum(s.eta) - sum0)).toBeLessThan(1e-6 * p.h * s.eta.length);
  });

  it("a hump spreads out as waves at the wave speed and is damped", () => {
    const p = smallParams(n, { waterLevel: 0.7, waveDamping: 0 });
    const s = new CpuSolver(p);
    bump(s, p.h);
    const centre = (n / 2) + n * (n / 2);
    const e0 = s.eta[centre];
    // after ~ (4 cells)/c the front has passed the ring at radius 4
    const steps = Math.ceil((4 * p.h) / p.waveSpeed / p.dt);
    let ringMax = 0;
    for (let it = 0; it < steps; it++) {
      s.step();
      ringMax = Math.max(ringMax, Math.abs(s.eta[n / 2 + 5 + n * (n / 2)]));
    }
    expect(s.eta[centre]).toBeLessThan(0.5 * e0); // the hump collapsed
    expect(ringMax).toBeGreaterThan(0.02 * p.h); // and the disturbance reached the ring
    // with damping the surface settles back to flat
    const d = new CpuSolver(smallParams(n, { waterLevel: 0.7 }));
    bump(d, p.h);
    for (let it = 0; it < 360; it++) d.step();
    expect(Math.max(...d.eta.map(Math.abs))).toBeLessThan(0.05 * p.h);
  });

  it("is stable on fine grids (wave sub-steps keep c dt / h <= 0.5)", () => {
    for (const m of [16, 48, 96, 160]) {
      const p = smallParams(m, { waterLevel: 0.7 });
      const ws = waveSubsteps(p);
      expect((p.waveSpeed * p.dt) / p.subcycles / ws / p.h).toBeLessThanOrEqual(0.5 + 1e-12);
    }
  });

  it("a heater's up-flow and boiling raise waves on an initially flat surface; ice stops them", () => {
    const p = smallParams(n, { waterLevel: 0.7, hc: 30000 });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L * 0.12, z: L / 2, r: L * 0.15, T: 400 }];
    const s = new CpuSolver(p);
    const E0 = s.totalEnergy();
    for (let it = 0; it < 150; it++) s.step();
    const amp = Math.max(...s.eta.map(Math.abs));
    expect(amp).toBeGreaterThan(0.01 * p.h);
    expect(amp).toBeLessThan(3 * p.h); // small-amplitude regime
    // energy bookkeeping is unaffected by the waves
    expect(Math.abs(s.totalEnergy() - E0 - s.injectedEnergy()) / E0).toBeLessThan(1e-5);
    // freeze the top row: no flux through frozen columns
    const f = new CpuSolver(smallParams(n, { waterLevel: 0.7 }));
    for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) f.H[f.idx(i, f.rows - 1, k)] = enthalpyFromT(-10);
    bump(f, p.h);
    const eta0 = Array.from(f.eta);
    for (let it = 0; it < 20; it++) f.step();
    expect(Array.from(f.eta)).toEqual(eta0);
  });
});
