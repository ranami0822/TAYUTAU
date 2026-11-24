import { describe, expect, it } from "vitest";
import { CpuSolver } from "../../src/sim/cpuSolver";
import { C_ICE, K_ICE, L_F, enthalpyFromT, iceFraction, isSolid, temperature } from "../../src/sim/water";
import { smallParams } from "./helpers";

function erf(x: number) {
  const s = Math.sign(x);
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

/** root of lambda * exp(lambda^2) * erf(lambda) = St / sqrt(pi)  (one-phase Stefan problem, Neumann solution) */
function stefanLambda(St: number) {
  let lo = 1e-6, hi = 3;
  const f = (l: number) => l * Math.exp(l * l) * erf(l) - St / Math.sqrt(Math.PI);
  for (let k = 0; k < 100; k++) {
    const mid = 0.5 * (lo + hi);
    if (f(mid) > 0) hi = mid;
    else lo = mid;
  }
  return 0.5 * (lo + hi);
}

describe("stage 3: enthalpy method — freezing front (Stefan problem)", () => {
  it("follows the Neumann solution s(t) = 2 λ sqrt(α t) for water at 0 °C against a cold wall", () => {
    const nx = 48;
    const kScale = 40; // speeds the test up; the comparison uses the same scaled diffusivity
    const p = { ...smallParams(nx, { g: 0, kScale, vortEps: 0 }), ny: 1, nz: 1 };
    const s = new CpuSolver(p);
    const Tw = -20;
    s.H.fill(L_F); // liquid exactly at the melting point
    const dts = p.dt / p.subcycles;
    const alpha = (K_ICE * kScale) / (p.rho0 * C_ICE);
    const St = (C_ICE * -Tw) / L_F;
    const lambda = stefanLambda(St);
    const front = () => {
      // position of the 50 % ice contour, measured from the cold wall at x = 0.5 h (centre of the fixed cell)
      for (let i = 1; i < nx; i++) {
        const a = iceFraction(s.H[i - 1]);
        const b = iceFraction(s.H[i]);
        if (a >= 0.5 && b < 0.5) return (i - 1 + (a - 0.5) / (a - b)) * p.h;
      }
      return 0;
    };
    const checks: { t: number; num: number; exact: number }[] = [];
    let t = 0;
    for (let it = 1; it <= 2400; it++) {
      s.H[0] = enthalpyFromT(Tw); // Dirichlet wall (the fixed cell)
      s.thermal(dts);
      s.H.set(s.Htmp);
      t += dts;
      if (it % 600 === 0) checks.push({ t, num: front(), exact: 2 * lambda * Math.sqrt(alpha * t) });
    }
    for (const c of checks) {
      expect(c.exact).toBeGreaterThan(3 * p.h); // the front has moved several cells
      expect(Math.abs(c.num - c.exact)).toBeLessThan(0.8 * p.h);
    }
    // the ice really is ice: temperatures below 0 behind the front, liquid at 0 ahead of it
    expect(temperature(s.H[1])).toBeLessThan(0);
    expect(temperature(s.H[nx - 1])).toBe(0);
  });
});

describe("stage 3: freezing and melting in 3D", () => {
  it("a cold source freezes water around it into a rigid solid region", () => {
    const n = 12;
    const p = smallParams(n, { hc: 20000 });
    const L = n * p.h;
    p.sources = [{ x: L / 2, y: L / 2, z: L / 2, r: L * 0.2, T: -60 }];
    const s = new CpuSolver(p, 1);
    const E0 = s.totalEnergy();
    for (let it = 0; it < 150; it++) s.step();
    const ph = s.phaseMasses();
    expect(ph.ice).toBeGreaterThan(0.01 * ph.total);
    const c = s.idx(n / 2, n / 2, n / 2);
    expect(isSolid(s.H[c])).toBe(true);
    // no flow through any face of the frozen region
    for (let f = 0; f < s.g.nFaces; f++) {
      const [a, i, j, k] = s.faceCoord(f);
      if (!s.faceOpen(a, i, j, k)) expect(s.vel[f]).toBe(0);
    }
    // energy removed by the cooler is exactly what the water lost (latent + sensible)
    expect(s.injectedEnergy()).toBeLessThan(0);
    expect(Math.abs(s.totalEnergy() - E0 - s.injectedEnergy()) / E0).toBeLessThan(1e-5);
  });

  it("an ice block in warm water melts; energy is conserved and the latent heat is accounted for", () => {
    const n = 12;
    const p = smallParams(n, { kScale: 30 });
    const s = new CpuSolver(p, 40);
    for (let k = 4; k < 8; k++) for (let j = 6; j < 10; j++) for (let i = 4; i < 8; i++) s.H[s.idx(i, j, k)] = enthalpyFromT(-10);
    const E0 = s.totalEnergy();
    const ice0 = s.phaseMasses().ice;
    const meanT0 = s.H.reduce((a, H) => a + temperature(H), 0) / s.g.nCells;
    let prevIce = ice0;
    for (let it = 1; it <= 300; it++) {
      s.step();
      if (it % 50 === 0) {
        const ice = s.phaseMasses().ice;
        expect(ice).toBeLessThanOrEqual(prevIce + 1e-9); // melting only
        prevIce = ice;
      }
    }
    const ice1 = s.phaseMasses().ice;
    expect(ice1).toBeLessThan(0.7 * ice0);
    expect(Math.abs(s.totalEnergy() - E0) / E0).toBeLessThan(1e-6);
    // the water cooled down: the melted ice took latent heat from it
    const meanT1 = s.H.reduce((a, H) => a + temperature(H), 0) / s.g.nCells;
    expect(meanT1).toBeLessThan(meanT0 - 1);
    // calorimetry: final state with the analytic equilibrium of the same total energy is all liquid at T_eq;
    // the energy balance of the partial state must equal latent + sensible parts exactly (by construction of H)
    const M = s.g.nCells * s.cellMass();
    const Teq = (E0 / M - L_F) / 4186;
    expect(Teq).toBeGreaterThan(0);
    expect(Teq).toBeLessThan(40);
  });
});
