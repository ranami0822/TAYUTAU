import { describe, expect, it } from "vitest";
import { CpuSolver } from "../../src/sim/cpuSolver";
import { defaultParams, waterRows, APP_WATER_LEVEL } from "../../src/sim/params";
import { H_B, L_V, enthalpyFromT, vaporQuality } from "../../src/sim/water";
import { randomFlow, rng, smallParams, sum, toF64 } from "./helpers";

describe("water surface: position", () => {
  it("cells below the water level are water, cells above are air (no flow, no heat, not counted)", () => {
    const n = 10;
    const p = smallParams(n, { waterLevel: 0.7 });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L * 0.2, z: L / 2, r: L * 0.2, T: 300 }];
    const s = new CpuSolver(p);
    expect(s.rows).toBe(7);
    for (let j = 0; j < n; j++) {
      expect(s.isAir(j)).toBe(j >= 7);
      expect(s.solidAt(4, j, 4)).toBe(j >= 7);
    }
    randomFlow(s, 0.3 * p.vMax, 3);
    const air0 = Array.from(s.H).filter((_, c) => !s.isWaterCell(c));
    for (let it = 0; it < 40; it++) s.step();
    // the water moved and warmed, the air did not change at all
    expect(s.maxSpeed()).toBeGreaterThan(1e-4);
    expect(Array.from(s.H).filter((_, c) => !s.isWaterCell(c))).toEqual(air0);
    // no flow through the surface (v faces at j = 7) or anywhere in the air
    for (let k = 0; k < n; k++)
      for (let i = 0; i < n; i++) for (let j = 7; j <= n; j++) expect(s.vel[s.vIdx(i, j, k)]).toBe(0);
    // totals count only the water
    expect(s.phaseMasses().total).toBeCloseTo(0.7 * n * n * n * s.cellMass(), 12);
    expect(s.waterMass()).toBeCloseTo(0.7 * n * n * n * s.cellMass(), 12);
  });

  it("rounds the level to whole cell rows, keeps at least one row of water", () => {
    expect(waterRows({ ny: 48, waterLevel: APP_WATER_LEVEL })).toBe(34); // 33.6 -> 34
    expect(waterRows({ ny: 10, waterLevel: 1 })).toBe(10);
    expect(waterRows({ ny: 10, waterLevel: 0 })).toBe(1);
    expect(defaultParams(16).waterLevel).toBe(1); // the library default is a full tank
  });
});

describe("water surface: conservation at different water levels", () => {
  for (const level of [0.7, 0.4]) {
    it(`level ${level}: energy change equals the source heat (f32), and exactly in f64`, () => {
      const n = 10;
      const run = (f64: boolean) => {
        const p = smallParams(n, { waterLevel: level, hc: 30000 });
        const L = n * p.h;
        p.sources = [
          { x: L * 0.5, y: L * 0.12, z: L * 0.5, r: L * 0.15, T: 400 },
          { x: L * 0.2, y: L * 0.25, z: L * 0.3, r: L * 0.12, T: -50 },
        ];
        const s = new CpuSolver(p, 60);
        if (f64) toF64(s);
        const E0 = s.totalEnergy();
        for (let it = 0; it < 80; it++) s.step();
        return { s, E0 };
      };
      for (const f64 of [false, true]) {
        const { s, E0 } = run(f64);
        const ph = s.phaseMasses();
        expect(Math.abs(s.injectedEnergy())).toBeGreaterThan(1);
        expect(ph.vapor + ph.ice).toBeGreaterThan(0);
        expect(ph.ice + ph.liquid + ph.vapor).toBeCloseTo(s.waterMass(), 12); // water mass unchanged
        expect(Math.abs(s.totalEnergy() - E0 - s.injectedEnergy()) / E0).toBeLessThan(f64 ? 1e-12 : 1e-5);
      }
    });

    it(`level ${level}: projected flow is divergence-free in the water and a tracer is conserved`, () => {
      const n = 12;
      const s = new CpuSolver(smallParams(n, { waterLevel: level }));
      randomFlow(s, 0.5 * s.p.vMax, 5);
      expect(s.maxDivergence()).toBeLessThan(1e-3 * (s.maxSpeed() / s.p.h));
      const r = rng(2);
      let q = new Float32Array(s.g.nCells).map((_, c) => (s.isWaterCell(c) && r() < 0.3 ? 1 : 0));
      let out = new Float32Array(s.g.nCells);
      const total0 = sum(q);
      for (let it = 0; it < 100; it++) {
        s.advectScalar(q, out, s.p.dt / s.p.subcycles);
        [q, out] = [out, q];
      }
      expect(Math.abs(sum(q) - total0) / total0).toBeLessThan(1e-5);
      for (let c = 0; c < s.g.nCells; c++) if (!s.isWaterCell(c)) expect(q[c]).toBe(0); // nothing crossed the surface
    });
  }

  it("rising vapour collects just under the surface, not in the air", () => {
    const n = 10;
    const s = new CpuSolver(smallParams(n, { waterLevel: 0.6 }), 100);
    const rows = s.rows; // 6
    for (let c = 0; c < s.g.nCells; c++) {
      const j = Math.floor(c / n) % n;
      if (j < rows) s.H[c] = H_B + (j < 2 ? 0.02 * L_V : 0);
    }
    for (let it = 0; it < 120; it++) s.step();
    let top = 0, bottom = 0;
    for (let c = 0; c < s.g.nCells; c++) {
      const j = Math.floor(c / n) % n;
      if (j === rows - 1) top += vaporQuality(s.H[c]);
      if (j < 2) bottom += vaporQuality(s.H[c]);
      if (j >= rows) expect(s.H[c]).toBe(enthalpyFromT(100)); // air untouched (initial value)
    }
    expect(top).toBeGreaterThan(bottom);
  });
});
