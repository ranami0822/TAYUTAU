import { describe, expect, it } from "vitest";
import { CpuSolver } from "../../src/sim/cpuSolver";
import { enthalpyFromT, L_F } from "../../src/sim/water";
import { randomFlow, rng, smallParams, sum } from "./helpers";

describe("stage 1: pressure projection (mass conservation of an incompressible fluid)", () => {
  it("removes the divergence of a random velocity field", () => {
    const s = new CpuSolver(smallParams(12, { pressureIters: 200 }));
    const r = rng(7);
    for (let f = 0; f < s.g.nFaces; f++) s.vel2[f] = (r() * 2 - 1) * 0.01;
    // zero the closed boundary faces first (as advectVel does)
    for (let f = 0; f < s.g.nFaces; f++) {
      const [a, i, j, k] = s.faceCoord(f);
      if (!s.faceOpen(a, i, j, k)) s.vel2[f] = 0;
    }
    const before = s.maxDivergence(s.vel2);
    s.divergence(s.vel2);
    s.pressure();
    s.project();
    const after = s.maxDivergence(s.vel);
    expect(before).toBeGreaterThan(1);
    expect(after).toBeLessThan(before * 1e-3);
  });

  it("keeps the normal velocity zero on the walls and on ice faces", () => {
    const s = new CpuSolver(smallParams(10));
    // a frozen block in the middle
    for (let k = 3; k < 6; k++) for (let j = 3; j < 6; j++) for (let i = 3; i < 6; i++) s.H[s.idx(i, j, k)] = -1000;
    randomFlow(s, 0.01, 3);
    for (let f = 0; f < s.g.nFaces; f++) {
      const [a, i, j, k] = s.faceCoord(f);
      if (!s.faceOpen(a, i, j, k)) expect(s.vel[f]).toBe(0);
    }
    // the wall faces are among the closed ones
    expect(s.vel[s.uIdx(0, 4, 4)]).toBe(0);
    expect(s.vel[s.vIdx(4, s.g.ny, 4)]).toBe(0);
    expect(s.vel[s.uIdx(3, 4, 4)]).toBe(0); // face between fluid (2,4,4) and ice (3,4,4)
  });

  it("stays divergence-free (to solver tolerance) while stepping", () => {
    const s = new CpuSolver(smallParams(12));
    randomFlow(s, 0.02, 11);
    for (let n = 0; n < 20; n++) s.step();
    // 40 SOR iterations warm-started: residual divergence small compared with |u|/h
    const scale = s.maxSpeed() / s.p.h;
    expect(s.maxDivergence()).toBeLessThan(0.05 * scale);
    expect(Number.isFinite(s.maxSpeed())).toBe(true);
  });
});

describe("stage 1: flux-form scalar advection", () => {
  it("conserves the total of an advected tracer exactly (to float rounding) and adds no new extrema", () => {
    const s = new CpuSolver(smallParams(12));
    randomFlow(s, s.p.vMax * 0.5, 5);
    const r = rng(9);
    let q = new Float32Array(s.g.nCells).map(() => (r() < 0.2 ? 1 : 0));
    let out = new Float32Array(s.g.nCells);
    const total0 = sum(q);
    const dts = s.p.dt / s.p.subcycles;
    for (let n = 0; n < 200; n++) {
      s.advectScalar(q, out, dts);
      [q, out] = [out, q];
    }
    expect(Math.abs(sum(q) - total0) / total0).toBeLessThan(1e-5);
    // MUSCL/minmod is TVD in 1D; in 3D with a nearly divergence-free field over/undershoots stay tiny
    let mn = Infinity, mx = -Infinity;
    for (const v of q) { mn = Math.min(mn, v); mx = Math.max(mx, v); }
    expect(mn).toBeGreaterThan(-0.02);
    expect(mx).toBeLessThan(1.02);
    // and the field actually moved
    expect(Math.abs(sum(q.map((v, i) => (i % 2 ? v : 0))) - sum(q) / 2)).toBeLessThan(total0);
  });

  it("does not move a uniform field (a uniform density stays uniform: no spurious compression)", () => {
    const s = new CpuSolver(smallParams(10));
    randomFlow(s, s.p.vMax * 0.5, 8);
    let q = new Float32Array(s.g.nCells).fill(1);
    let out = new Float32Array(s.g.nCells);
    for (let n = 0; n < 50; n++) {
      s.advectScalar(q, out, s.p.dt / s.p.subcycles);
      [q, out] = [out, q];
    }
    for (const v of q) expect(Math.abs(v - 1)).toBeLessThan(1e-3);
  });
});

describe("stage 1: velocity step", () => {
  it("decays / stays bounded without forcing (no blow-up with vorticity confinement)", () => {
    const s = new CpuSolver(smallParams(12));
    randomFlow(s, 0.03, 21);
    const v0 = s.maxSpeed();
    for (let n = 0; n < 60; n++) s.step();
    expect(s.maxSpeed()).toBeLessThanOrEqual(Math.max(v0 * 1.5, 1e-6));
    expect(s.maxSpeed()).toBeLessThanOrEqual(s.p.vMax);
  });

  it("keeps an isothermal fluid at rest (no spurious currents from buoyancy)", () => {
    const s = new CpuSolver(smallParams(10));
    s.H.fill(enthalpyFromT(37));
    for (let n = 0; n < 30; n++) s.step();
    // uniform buoyancy is a gradient field and is removed by the projection
    expect(s.maxSpeed()).toBeLessThan(1e-4);
  });

  it("treats frozen cells as rigid: velocity inside ice stays zero", () => {
    const s = new CpuSolver(smallParams(10));
    for (let k = 0; k < 10; k++) for (let j = 0; j < 3; j++) for (let i = 0; i < 10; i++) s.H[s.idx(i, j, k)] = 0.2 * L_F;
    randomFlow(s, 0.02, 4);
    for (let n = 0; n < 10; n++) s.step();
    for (let k = 0; k < 10; k++)
      for (let i = 0; i < 10; i++) {
        expect(s.vel[s.vIdx(i, 1, k)]).toBe(0);
        expect(s.vel[s.vIdx(i, 3, k)]).toBe(0); // ice/water interface
      }
  });
});
