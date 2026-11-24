/**
 * GPU kernels vs. the CPU reference, and conservation measured on the GPU itself.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { openHarness, type Harness } from "./browser";
import { CpuSolver } from "../../src/sim/cpuSolver";
import { defaultParams, type SimParams } from "../../src/sim/params";
import { enthalpyFromT, temperature, L_F } from "../../src/sim/water";
import { randomFlow, rng } from "../cpu/helpers";
import type { RunSimArgs } from "../../gpu-harness/harness";

let H: Harness;
beforeAll(async () => {
  H = await openHarness();
});
afterAll(async () => {
  await H?.close();
});

type Snap = { step: number; H: number[]; vel: number[]; srcAcc: number[] };
async function gpu(args: RunSimArgs) {
  return (await H.page.evaluate((a) => window.tayutau.runSim(a), args)) as { snaps: Snap[]; ms: number };
}

function setup(n: number, over: Partial<SimParams>, seed: number) {
  const p: SimParams = { ...defaultParams(n), ...over };
  const s = new CpuSolver(p);
  const r = rng(seed);
  // a lumpy temperature field (with a little ice and some boiling cells) and a random divergence-free flow
  for (let c = 0; c < s.g.nCells; c++) s.H[c] = enthalpyFromT(10 + 60 * r());
  s.H[s.idx(1, 1, 1)] = 0.2 * L_F;
  s.H[s.idx(2, 1, 1)] = -500;
  s.H[s.idx(n - 2, 1, n - 2)] = enthalpyFromT(100) + 3e5;
  randomFlow(s, 0.3 * p.vMax, seed + 1);
  return { p, s, H0: Array.from(s.H), vel0: Array.from(s.vel) };
}

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>) {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe("GPU kernels match the CPU reference", () => {
  it("one step with sources, buoyancy, vortex confinement and phase change", async () => {
    const n = 10;
    const L = n * defaultParams(n).h;
    const { p, s, H0, vel0 } = setup(n, { sources: [{ x: L / 2, y: L * 0.2, z: L / 2, r: L * 0.25, T: 250 }, { x: L * 0.8, y: L * 0.8, z: L * 0.3, r: L * 0.2, T: -30 }] }, 3);
    const g = await gpu({ params: p, H: H0, vel: vel0, steps: 1 });
    s.step();
    const snap = g.snaps[0];
    // enthalpy: values ~4e5 J/kg, f32 rounding ~0.05; allow a few J/kg
    expect(maxAbsDiff(snap.H, s.H)).toBeLessThan(5);
    expect(maxAbsDiff(snap.srcAcc, s.srcAcc)).toBeLessThan(5);
    // velocity: the pressure solve runs 40 SOR sweeps in f32 on both; agreement to ~1e-5 m/s
    expect(maxAbsDiff(snap.vel, s.vel)).toBeLessThan(1e-4 * p.vMax + 1e-6);
  });

  it("stays close over 10 steps", async () => {
    const n = 10;
    const L = n * defaultParams(n).h;
    const { p, s, H0, vel0 } = setup(n, { sources: [{ x: L / 2, y: L * 0.2, z: L / 2, r: L * 0.25, T: 200 }] }, 5);
    const g = await gpu({ params: p, H: H0, vel: vel0, steps: 10 });
    for (let k = 0; k < 10; k++) s.step();
    const snap = g.snaps[0];
    let maxT = 0;
    for (let c = 0; c < s.g.nCells; c++) maxT = Math.max(maxT, Math.abs(temperature(snap.H[c]) - temperature(s.H[c])));
    expect(maxT).toBeLessThan(0.05); // kelvin
    expect(maxAbsDiff(snap.vel, s.vel)).toBeLessThan(1e-3 * p.vMax);
  });
});

describe("conservation on the GPU", () => {
  it("energy: total enthalpy changes exactly by the heat-source exchange (closed adiabatic box)", async () => {
    const n = 12;
    const L = n * defaultParams(n).h;
    const { p, H0, vel0 } = setup(n, { sources: [{ x: L / 2, y: L * 0.15, z: L / 2, r: L * 0.2, T: 300 }, { x: L * 0.2, y: L * 0.8, z: L * 0.5, r: L * 0.2, T: -40 }] }, 9);
    const g = await gpu({ params: p, H: H0, vel: vel0, steps: 60, readAt: [20, 60] });
    const m = p.rho0 * p.h ** 3;
    const E0 = H0.reduce((a, b) => a + b, 0) * m;
    for (const snap of g.snaps) {
      const E = snap.H.reduce((a, b) => a + b, 0) * m;
      const inj = snap.srcAcc.reduce((a, b) => a + b, 0) * m;
      expect(Math.abs(inj)).toBeGreaterThan(10); // sources did work [J]
      expect(Math.abs(E - E0 - inj) / Math.abs(E0)).toBeLessThan(1e-5);
    }
  });

  it("mass: the projected velocity is divergence-free on the GPU", async () => {
    const n = 12;
    const L = n * defaultParams(n).h;
    const { p, H0, vel0 } = setup(n, { sources: [{ x: L / 2, y: L * 0.15, z: L / 2, r: L * 0.2, T: 300 }] }, 13);
    const g = await gpu({ params: p, H: H0, vel: vel0, steps: 30 });
    const s = new CpuSolver(p);
    s.H.set(g.snaps[0].H);
    s.vel.set(g.snaps[0].vel);
    const scale = s.maxSpeed() / p.h;
    expect(s.maxSpeed()).toBeGreaterThan(1e-4);
    expect(s.maxDivergence()).toBeLessThan(0.05 * scale);
  });
});
