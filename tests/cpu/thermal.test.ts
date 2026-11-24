import { beforeAll, describe, expect, it } from "vitest";
import { CpuSolver, buoyancy } from "../../src/sim/cpuSolver";
import { enthalpyFromT, rhoLiquid, temperature, voidFraction } from "../../src/sim/water";
import { rng, smallParams } from "./helpers";

function erf(x: number) {
  // Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7)
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

describe("stage 2: buoyancy from the equation of state", () => {
  it("hot water is lighter than the reference and rises", () => {
    const p = smallParams(8);
    expect(buoyancy(enthalpyFromT(60), p)).toBeGreaterThan(0);
    expect(buoyancy(enthalpyFromT(20), p)).toBeCloseTo(0, 6);
  });
  it("reproduces the 4 °C density anomaly: around 4 °C water is heaviest", () => {
    const p = smallParams(8, { rhoRef: rhoLiquid(4) });
    expect(buoyancy(enthalpyFromT(4), p)).toBeCloseTo(0, 4);
    expect(buoyancy(enthalpyFromT(1), p)).toBeGreaterThan(0); // colder than 4 °C rises
    expect(buoyancy(enthalpyFromT(8), p)).toBeGreaterThan(0);
  });
  it("vapour adds the (game-scaled) void buoyancy on top of the liquid's", () => {
    const p = smallParams(8);
    const liquid = buoyancy(enthalpyFromT(100), p);
    const chi = 0.01;
    const extra = buoyancy(enthalpyFromT(100) + chi * 2257e3, p) - liquid;
    const expected = (p.g * p.vaporBuoyancyScale * voidFraction(chi) * (rhoLiquid(100) - 0.598)) / p.rhoRef;
    expect(extra).toBeCloseTo(expected, 6);
    expect(extra).toBeGreaterThan(liquid); // even scaled down, a 1 % quality doubles the buoyancy
  });
});

describe("stage 2: heating from below drives convection", () => {
  // a 90 °C heater (below boiling, so this is pure single-phase convection) at the bottom centre
  const n = 12;
  const steps = 300; // 2.5 s
  const run = (g: number) => {
    const p = smallParams(n, { g });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L * 0.12, z: L / 2, r: L * 0.15, T: 90 }];
    const s = new CpuSolver(p);
    for (let it = 0; it < steps; it++) s.step();
    return s;
  };
  let withG: CpuSolver;
  let noG: CpuSolver;
  beforeAll(() => {
    withG = run(9.81);
    noG = run(0);
  });
  const heatCentre = (s: CpuSolver) => {
    let num = 0, den = 0;
    const H20 = enthalpyFromT(20);
    for (let c = 0; c < s.g.nCells; c++) {
      const d = Math.max(0, s.H[c] - H20);
      num += d * ((Math.floor(c / n) % n) + 0.5);
      den += d;
    }
    return num / den;
  };

  it("produces an upward plume above the heater", () => {
    let vUp = 0;
    for (let j = 3; j < n - 2; j++) vUp += withG.vel[withG.vIdx(n / 2, j, n / 2)];
    expect(vUp / (n - 5)).toBeGreaterThan(0.005); // > 5 mm/s on average up the column
    // and the water above the heater is warmer than at the same height far from it
    expect(withG.temperatureAt(n / 2, n / 2, n / 2)).toBeGreaterThan(withG.temperatureAt(0, n / 2, 0) + 2);
  });

  it("carries heat upward much faster than conduction alone (g = 0)", () => {
    expect(heatCentre(withG)).toBeGreaterThan(heatCentre(noG) + 1.5); // cells
    expect(noG.maxSpeed()).toBeLessThan(1e-6); // no gravity, no flow
  });
});

describe("stage 2: heat conduction (enthalpy method, single phase)", () => {
  it("matches the analytic erf solution of two half-spaces in contact", () => {
    const n = 24;
    const kScale = 1000; // only to make conduction measurable in a short test; discretisation is what is checked
    const p = { ...smallParams(n, { g: 0, kScale, vortEps: 0 }), ny: 3, nz: 3 };
    const s = new CpuSolver(p);
    for (let c = 0; c < s.g.nCells; c++) {
      const i = c % n;
      s.H[c] = enthalpyFromT(i < n / 2 ? 10 : 30);
    }
    const steps = 120 * p.subcycles;
    const dts = p.dt / p.subcycles;
    for (let it = 0; it < steps; it++) {
      s.thermal(dts);
      s.H.set(s.Htmp);
    }
    const t = steps * dts;
    const alpha = (0.6 * kScale) / (p.rho0 * 4186);
    let maxErr = 0;
    for (let i = 0; i < n; i++) {
      const x = (i + 0.5 - n / 2) * p.h;
      const exact = 20 + 10 * erf(x / (2 * Math.sqrt(alpha * t)));
      // only compare away from the adiabatic end walls
      if (Math.abs(x) < 0.25 * n * p.h) maxErr = Math.max(maxErr, Math.abs(s.temperatureAt(i, 1, 1) - exact));
    }
    expect(maxErr).toBeLessThan(0.3); // K, against a 20 K step
  });
});

describe("stage 2: energy conservation", () => {
  it("closed adiabatic box without sources conserves total enthalpy while convecting", () => {
    const n = 12;
    const p = smallParams(n);
    const s = new CpuSolver(p);
    const r = rng(42);
    for (let c = 0; c < s.g.nCells; c++) {
      const j = Math.floor(c / n) % n;
      s.H[c] = enthalpyFromT((j < 3 ? 70 : 15) + 0.5 * r()); // unstable stratification + small noise
    }
    const E0 = s.totalEnergy();
    for (let it = 0; it < 100; it++) s.step();
    expect(s.maxSpeed()).toBeGreaterThan(1e-3); // it does overturn
    expect(Math.abs(s.totalEnergy() - E0) / E0).toBeLessThan(1e-6);
  });

  it("with heat sources, the energy change equals the heat exchanged by the sources", () => {
    const n = 12;
    const p = smallParams(n);
    const L = n * p.h;
    p.sources = [
      { x: L * 0.3, y: L * 0.15, z: L * 0.5, r: L * 0.15, T: 350 },
      { x: L * 0.75, y: L * 0.7, z: L * 0.4, r: L * 0.15, T: -30 },
    ];
    const s = new CpuSolver(p);
    const E0 = s.totalEnergy();
    for (let it = 0; it < 100; it++) s.step();
    const dE = s.totalEnergy() - E0;
    const inj = s.injectedEnergy();
    expect(Math.abs(inj)).toBeGreaterThan(1);
    expect(Math.abs(dE - inj) / E0).toBeLessThan(1e-5);
  });

  it("a source never overshoots its own temperature", () => {
    const n = 8;
    const p = smallParams(n, { hc: 1e7 }); // absurdly strong coupling
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L / 2, z: L / 2, r: L * 0.3, T: 60 }];
    const s = new CpuSolver(p);
    for (let it = 0; it < 30; it++) s.step();
    for (let c = 0; c < s.g.nCells; c++) expect(temperature(s.H[c])).toBeLessThanOrEqual(60 + 1e-3);
  });
});
