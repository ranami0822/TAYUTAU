import { describe, expect, it } from "vitest";
import { CpuSolver } from "../../src/sim/cpuSolver";
import { H_B, H_VSAT, L_V, enthalpyFromT, temperature, vaporQuality } from "../../src/sim/water";
import { smallParams, toF64 } from "./helpers";

/** vapour mass [kg] in the cells with j in [j0, j1) */
function vaporIn(s: CpuSolver, j0: number, j1: number) {
  let v = 0;
  for (let c = 0; c < s.g.nCells; c++) {
    const j = Math.floor(c / s.g.nx) % s.g.ny;
    if (j >= j0 && j < j1) v += vaporQuality(s.H[c]);
  }
  return v * s.cellMass();
}

describe("stage 4: boiling", () => {
  it("water heated past 100 °C boils: temperature stays at 100 °C while vapour forms (latent plateau)", () => {
    const n = 10;
    const p = smallParams(n, { hc: 50000 });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L * 0.2, z: L / 2, r: L * 0.2, T: 300 }];
    const s = new CpuSolver(p, 95);
    for (let it = 0; it < 80; it++) s.step();
    const ph = s.phaseMasses();
    expect(ph.vapor).toBeGreaterThan(0);
    // every cell that holds vapour but is not fully vapour is exactly at the boiling point
    let mixed = 0;
    for (let c = 0; c < s.g.nCells; c++) {
      const chi = vaporQuality(s.H[c]);
      if (chi > 0 && chi < 1) {
        mixed++;
        expect(temperature(s.H[c])).toBe(100);
      }
    }
    expect(mixed).toBeGreaterThan(0);
    // liquid never exceeds 100 °C (no superheated liquid in the model)
    for (let c = 0; c < s.g.nCells; c++) if (s.H[c] < H_VSAT && s.H[c] > H_B) expect(temperature(s.H[c])).toBe(100);
  });

  it("vapour rises and collects at the top of the tank", () => {
    const n = 10;
    const p = smallParams(n);
    const s = new CpuSolver(p, 100);
    // saturated water with some vapour spread in the bottom half
    for (let c = 0; c < s.g.nCells; c++) {
      const j = Math.floor(c / n) % n;
      s.H[c] = H_B + (j < n / 2 ? 0.02 * L_V : 0);
    }
    const E0 = s.totalEnergy();
    const bottom0 = vaporIn(s, 0, n / 2);
    expect(vaporIn(s, n / 2, n)).toBe(0);
    for (let it = 0; it < 120; it++) s.step();
    const top = vaporIn(s, n / 2, n);
    const bottom = vaporIn(s, 0, n / 2);
    expect(top).toBeGreaterThan(bottom);
    expect(top + bottom).toBeCloseTo(bottom0, 6); // saturated surroundings: no condensation, vapour mass conserved
    // f32 storage rounding only (the scheme itself is exact, see the f64 test below)
    expect(Math.abs(s.totalEnergy() - E0) / E0).toBeLessThan(1e-5);
  });

  it("vapour rising into subcooled water condenses and releases its latent heat there", () => {
    const n = 10;
    const p = smallParams(n, { vortEps: 0 });
    const s = new CpuSolver(p, 30);
    // a layer of steam-water mixture at the bottom under 30 °C water
    for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) s.H[s.idx(i, 0, k)] = H_B + 0.05 * L_V;
    const E0 = s.totalEnergy();
    const v0 = s.phaseMasses().vapor;
    const T1before = s.temperatureAt(5, 1, 5);
    for (let it = 0; it < 60; it++) s.step();
    expect(s.phaseMasses().vapor).toBeLessThan(0.5 * v0); // condensed
    expect(s.temperatureAt(5, 1, 5)).toBeGreaterThan(T1before + 1); // the water above was heated
    expect(Math.abs(s.totalEnergy() - E0) / E0).toBeLessThan(1e-6); // latent heat went into the water, nothing lost
  });

  it("a heater boils water off; energy balance holds through all three regimes", () => {
    const n = 10;
    const p = smallParams(n, { hc: 50000 });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L * 0.15, z: L / 2, r: L * 0.18, T: 400 }];
    const s = new CpuSolver(p, 90);
    const E0 = s.totalEnergy();
    for (let it = 0; it < 150; it++) s.step();
    expect(s.phaseMasses().vapor).toBeGreaterThan(0);
    expect(Math.abs(s.totalEnergy() - E0 - s.injectedEnergy()) / E0).toBeLessThan(1e-5);
    // total H2O mass is unchanged by the phase change (closed tank)
    const ph = s.phaseMasses();
    expect(ph.ice + ph.liquid + ph.vapor).toBeCloseTo(ph.total, 12);
  });

  it("subcooled water does not boil even next to a hot source until it reaches 100 °C", () => {
    const n = 8;
    const p = smallParams(n, { hc: 100 });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L / 2, z: L / 2, r: L * 0.2, T: 300 }];
    const s = new CpuSolver(p, 20);
    for (let it = 0; it < 20; it++) s.step();
    expect(s.phaseMasses().vapor).toBe(0);
    let maxT = 0;
    for (const H of s.H) maxT = Math.max(maxT, temperature(H));
    expect(maxT).toBeGreaterThan(20);
    expect(maxT).toBeLessThan(100);
    expect(enthalpyFromT(100)).toBe(H_B);
  });
});

describe("conservation is exact in the scheme (f64 storage)", () => {
  it("all processes at once: sources, conduction, boiling, condensation, freezing, convection", () => {
    const n = 10;
    const p = smallParams(n, { hc: 30000 });
    const L = n * p.h;
    p.sources = [
      { x: L * 0.5, y: L * 0.15, z: L * 0.5, r: L * 0.18, T: 400 },
      { x: L * 0.2, y: L * 0.75, z: L * 0.3, r: L * 0.18, T: -50 },
    ];
    const s = toF64(new CpuSolver(p, 60));
    for (let k = 6; k < 9; k++) for (let j = 2; j < 5; j++) for (let i = 6; i < 9; i++) s.H[s.idx(i, j, k)] = enthalpyFromT(-20);
    const E0 = s.totalEnergy();
    for (let it = 0; it < 100; it++) s.step();
    const ph = s.phaseMasses();
    expect(ph.vapor).toBeGreaterThan(0);
    expect(ph.ice).toBeGreaterThan(0);
    expect(Math.abs(s.injectedEnergy())).toBeGreaterThan(1);
    expect(Math.abs(s.totalEnergy() - E0 - s.injectedEnergy()) / E0).toBeLessThan(1e-12);
  });
});
