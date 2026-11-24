import { describe, expect, it } from "vitest";
import {
  rhoLiquid,
  temperature,
  enthalpyFromT,
  iceFraction,
  vaporQuality,
  isSolid,
  voidFraction,
  H_B,
  H_VSAT,
  L_F,
  L_V,
  C_WATER,
  C_ICE,
} from "../../src/sim/water";

describe("equation of state (liquid density, Kell 1975)", () => {
  it("matches table values at 1 atm", () => {
    // reference table values [kg/m^3]
    expect(rhoLiquid(0)).toBeCloseTo(999.84, 1);
    expect(rhoLiquid(4)).toBeCloseTo(999.97, 1);
    expect(rhoLiquid(20)).toBeCloseTo(998.21, 1);
    expect(Math.abs(rhoLiquid(100) - 958.35)).toBeLessThan(0.1);
  });
  it("has its maximum near 4 °C (density anomaly)", () => {
    let best = 0;
    let bestT = 0;
    for (let T = 0; T <= 10; T += 0.01) {
      const r = rhoLiquid(T);
      if (r > best) {
        best = r;
        bestT = T;
      }
    }
    expect(bestT).toBeGreaterThan(3.8);
    expect(bestT).toBeLessThan(4.2);
  });
});

describe("enthalpy method: H <-> T and phase fractions", () => {
  it("round-trips temperature outside the plateaus", () => {
    for (const T of [-40, -5, 0.5, 20, 99, 101, 250]) {
      expect(temperature(enthalpyFromT(T))).toBeCloseTo(T, 6);
    }
  });
  it("is monotone (non-decreasing) in H", () => {
    let prev = -Infinity;
    for (let H = -2e5; H < H_VSAT + 5e5; H += 997) {
      const T = temperature(H);
      expect(T).toBeGreaterThanOrEqual(prev - 1e-9);
      prev = T;
    }
  });
  it("has latent plateaus at 0 and 100 °C of width L_F and L_V", () => {
    expect(temperature(0)).toBe(0);
    expect(temperature(L_F * 0.5)).toBe(0);
    expect(iceFraction(L_F * 0.25)).toBeCloseTo(0.75, 12);
    expect(temperature(H_B + 0.3 * L_V)).toBe(100);
    expect(vaporQuality(H_B + 0.3 * L_V)).toBeCloseTo(0.3, 12);
    expect(H_VSAT - H_B).toBe(L_V);
  });
  it("is continuous at the phase boundaries", () => {
    const eps = 1e-6;
    for (const Hb of [0, L_F, H_B, H_VSAT]) {
      expect(Math.abs(temperature(Hb - eps) - temperature(Hb + eps))).toBeLessThan(1e-6);
    }
  });
  it("uses the specific heats as slopes", () => {
    expect(temperature(enthalpyFromT(30) + C_WATER) - 30).toBeCloseTo(1, 9);
    expect(temperature(-C_ICE * 10)).toBeCloseTo(-10, 9);
  });
  it("marks a cell solid when more than half is frozen", () => {
    expect(isSolid(-1)).toBe(true);
    expect(isSolid(0.49 * L_F)).toBe(true);
    expect(isSolid(0.51 * L_F)).toBe(false);
    expect(isSolid(enthalpyFromT(20))).toBe(false);
  });
  it("gives a large void fraction for a small vapour quality (vapour is ~1600x lighter)", () => {
    expect(voidFraction(0)).toBe(0);
    expect(voidFraction(1)).toBeCloseTo(1, 12);
    const a = voidFraction(0.001);
    expect(a).toBeGreaterThan(0.5);
    expect(a).toBeLessThan(0.7);
  });
});
