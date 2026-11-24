/**
 * Thermodynamics of water (ice / liquid / vapour) used by the physics layer.
 *
 * Enthalpy method (Stefan problem on a fixed grid): every cell stores a specific
 * enthalpy H [J/kg]. Temperature and phase fractions are functions of H only:
 *
 *   H < 0                      ice,      T = H / C_ICE
 *   0 <= H < L_F               melting,  T = 0 °C,  liquid fraction = H / L_F
 *   L_F <= H < H_B             liquid,   T = (H - L_F) / C_WATER
 *   H_B <= H < H_B + L_V       boiling,  T = 100 °C, vapour quality = (H - H_B) / L_V
 *   H >= H_B + L_V             vapour,   T = 100 + (H - H_B - L_V) / C_VAPOR
 *
 * with H_B = L_F + 100 * C_WATER. Reference: H = 0 is ice at exactly 0 °C.
 * The pressure is assumed to stay at 1 atm (fixed melting/boiling points).
 *
 * All values are real physical values (rounded). Sources / confidence are noted next
 * to each constant; nothing here is scaled for the game (game scalings live in params.ts).
 */

/** Specific heat of liquid water [J/(kg K)] (≈4.18–4.22 kJ/(kg K) over 0–100 °C; 4186 is the common rounded value). */
export const C_WATER = 4186;
/** Specific heat of ice near 0 °C [J/(kg K)] (≈2.05–2.11 kJ/(kg K) near 0 °C). */
export const C_ICE = 2100;
/** Specific heat (cp) of steam near 100 °C, 1 atm [J/(kg K)]. Rounded; exact value unverified (未確認, ~2.0–2.1 kJ/(kg K)). */
export const C_VAPOR = 2080;
/** Latent heat of fusion [J/kg] (334 J/g, research/08 B). */
export const L_F = 334e3;
/** Latent heat of vaporisation at 100 °C [J/kg] (2257 J/g, research/08 B). */
export const L_V = 2257e3;
/** Enthalpy of saturated liquid at 100 °C [J/kg]. */
export const H_B = L_F + 100 * C_WATER;
/** Enthalpy of saturated vapour at 100 °C [J/kg]. */
export const H_VSAT = H_B + L_V;

/** Thermal conductivities [W/(m K)]. Water ≈0.6, ice ≈2.2, steam ≈0.025 (textbook rounded values). */
export const K_WATER = 0.6;
export const K_ICE = 2.2;
export const K_VAPOR = 0.025;

/** Densities [kg/m^3]. Ice ≈917; saturated steam at 100 °C, 1 atm ≈0.598. */
export const RHO_ICE = 917;
export const RHO_VAPOR = 0.598;

/** Freezing / boiling temperatures at 1 atm [°C]. */
export const T_MELT = 0;
export const T_BOIL = 100;

/**
 * Liquid water density [kg/m^3] at 1 atm as a function of temperature [°C].
 * Kell (1975) rational polynomial, stated range 0–150 °C. Checked in tests against
 * table values (0 °C 999.84, 4 °C 999.97, 20 °C 998.21, 100 °C 958.35).
 * Outside 0–150 °C the input is clamped (supercooled / superheated liquid is not modelled).
 */
export function rhoLiquid(T: number): number {
  const t = Math.min(150, Math.max(0, T));
  const num =
    999.83952 +
    16.945176 * t -
    7.9870401e-3 * t * t -
    46.170461e-6 * t * t * t +
    105.56302e-9 * t * t * t * t -
    280.54253e-12 * t * t * t * t * t;
  return num / (1 + 16.87985e-3 * t);
}

/** Temperature [°C] from specific enthalpy [J/kg]. */
export function temperature(H: number): number {
  if (H < 0) return H / C_ICE;
  if (H < L_F) return 0;
  if (H < H_B) return (H - L_F) / C_WATER;
  if (H < H_VSAT) return 100;
  return 100 + (H - H_VSAT) / C_VAPOR;
}

/** Mass fraction of ice in the cell (0..1). */
export function iceFraction(H: number): number {
  return 1 - Math.min(1, Math.max(0, H / L_F));
}

/** Vapour quality: mass fraction of vapour (0..1). */
export function vaporQuality(H: number): number {
  return Math.min(1, Math.max(0, (H - H_B) / L_V));
}

/** Specific enthalpy [J/kg] of water at temperature T. At exactly 0/100 °C the phase-change plateau is ambiguous:
 *  0 °C maps to liquid at 0 °C (H = L_F) and 100 °C maps to saturated liquid (H = H_B). */
export function enthalpyFromT(T: number): number {
  if (T < 0) return T * C_ICE;
  if (T <= 100) return L_F + T * C_WATER;
  return H_VSAT + (T - 100) * C_VAPOR;
}

/** A cell counts as solid (ice) when more than half of its mass is frozen. */
export const SOLID_ICE_FRACTION = 0.5;
export function isSolid(H: number): boolean {
  return H < (1 - SOLID_ICE_FRACTION) * L_F;
}

/** Mixture thermal conductivity, mass-fraction weighted (a simplification). */
export function conductivity(H: number): number {
  const fi = iceFraction(H);
  const fv = vaporQuality(H);
  return fi * K_ICE + fv * K_VAPOR + (1 - fi - fv) * K_WATER;
}

/**
 * Void fraction (vapour volume fraction) of a liquid/vapour mixture with quality chi at 100 °C,
 * from the homogeneous mixture relation alpha = chi/rho_v / (chi/rho_v + (1-chi)/rho_l).
 */
export function voidFraction(chi: number): number {
  if (chi <= 0) return 0;
  const vv = chi / RHO_VAPOR;
  const vl = (1 - chi) / rhoLiquid(100);
  return vv / (vv + vl);
}
