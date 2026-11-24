/**
 * Simulation parameters shared by the CPU reference solver and the WGSL kernels.
 *
 * GAME SCALINGS (values that are NOT the real physical value, and why) are marked [GAME].
 * Everything else is either a real physical value (see water.ts) or a numerical setting.
 */
import { rhoLiquid } from "./water";

export const MAX_SOURCES = 8;

export interface HeatSource {
  /** centre in metres, domain coordinates (0..nx*h etc.), y is up */
  x: number;
  y: number;
  z: number;
  /** radius [m] */
  r: number;
  /** surface temperature of the heater / cooler [°C] */
  T: number;
}

export interface SimParams {
  nx: number;
  ny: number;
  nz: number;
  /** cell size [m] */
  h: number;
  /** time step for velocity/pressure [s] */
  dt: number;
  /** number of sub-cycles of the thermal + scalar advection step per dt (flux-form advection needs CFL < 1) */
  subcycles: number;
  /** gravity [m/s^2] */
  g: number;
  /** Boussinesq reference density [kg/m^3] used for the buoyancy term */
  rhoRef: number;
  /** mass per cell is rho0 * h^3 (incompressible / Boussinesq) */
  rho0: number;
  /**
   * [GAME] heat-transfer coefficient between a heat source and the water cells inside it [W/(m^2 K)].
   * Real values span ~1e2–1e3 (natural convection) to ~1e4–1e5 (nucleate boiling) (order of magnitude,
   * textbook ranges; exact values 未確認). 1.5e4 lets a 400 °C heater boil water locally within a second
   * (checked on the GPU at 32^3: the default heater delivers ~1.6 kW into the 0.125 kg tank).
   */
  hc: number;
  /**
   * [GAME] multiplier on all thermal conductivities. 1 = real. Real conduction is so slow at this
   * scale that a freezing front would take minutes to grow by 1 cm, so the game uses > 1.
   */
  kScale: number;
  /**
   * [GAME] slip (rise) velocity of vapour relative to the liquid [m/s]. Real mm-sized bubbles rise at
   * roughly 0.2–0.3 m/s (未確認: order of magnitude only); reduced for CFL stability.
   * Applied as bubbleRise * g / 9.81 (the slip is driven by gravity).
   */
  bubbleRise: number;
  /**
   * [GAME] scale on the buoyancy caused by vapour. The real void fraction makes vapour cells nearly
   * weightless (buoyancy ~ g), which a Boussinesq solver on a coarse grid cannot handle; scaled down.
   */
  vaporBuoyancyScale: number;
  /** [GAME/numerical] velocity clamp per component [m/s], keeps flux-form advection within CFL */
  vMax: number;
  /** vorticity confinement strength (Fedkiw et al. 2001), dimensionless epsilon */
  vortEps: number;
  /** red-black Gauss–Seidel iterations per pressure solve */
  pressureIters: number;
  /** SOR over-relaxation factor (1 = plain Gauss–Seidel) */
  sorOmega: number;
  /** heat sources (at most MAX_SOURCES) */
  sources: HeatSource[];
}

export function defaultParams(n = 48): SimParams {
  const L = 0.05; // [GAME-ish] 5 cm tank: small so that heating to boiling takes seconds, not minutes
  const h = L / n;
  const dt = 1 / 120;
  const subcycles = 4;
  return {
    nx: n,
    ny: n,
    nz: n,
    h,
    dt,
    subcycles,
    g: 9.81,
    rhoRef: rhoLiquid(20),
    rho0: 1000,
    hc: 15000,
    kScale: 10,
    bubbleRise: 0.12,
    vaporBuoyancyScale: 0.05,
    // per-component clamp so that the sum of CFL numbers over 3 axes stays <= 0.5 in each sub-cycle
    vMax: (0.5 / 3) * h / (dt / subcycles),
    vortEps: 0.3,
    pressureIters: 40,
    sorOmega: 1.6,
    sources: [],
  };
}

/** Size of the packed parameter block (f32 words) shared with WGSL. */
export const PARAM_WORDS = 32 + MAX_SOURCES * 8;

/**
 * Pack params into a Float32Array matching the WGSL `Params` struct (see gpu/kernels.ts).
 * Integer values are stored as floats and converted in WGSL.
 */
export function packParams(p: SimParams): Float32Array {
  const a = new Float32Array(PARAM_WORDS);
  a[0] = p.nx;
  a[1] = p.ny;
  a[2] = p.nz;
  a[3] = p.h;
  a[4] = p.dt;
  a[5] = p.subcycles;
  a[6] = p.g;
  a[7] = p.rhoRef;
  a[8] = p.rho0;
  a[9] = p.hc;
  a[10] = p.kScale;
  a[11] = p.bubbleRise;
  a[12] = p.vaporBuoyancyScale;
  a[13] = p.vMax;
  a[14] = p.vortEps;
  a[15] = p.sorOmega;
  a[16] = Math.min(p.sources.length, MAX_SOURCES);
  for (let s = 0; s < Math.min(p.sources.length, MAX_SOURCES); s++) {
    const src = p.sources[s];
    const o = 32 + s * 8;
    a[o + 0] = src.x;
    a[o + 1] = src.y;
    a[o + 2] = src.z;
    a[o + 3] = src.r;
    a[o + 4] = src.T;
  }
  return a;
}
