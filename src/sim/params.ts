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
  /**
   * Water level as a fraction of the tank height (0..1]. Cells in rows j >= waterRows(p) are air:
   * no water, no flow, no heat. The water surface is a fixed horizontal boundary (rigid-lid
   * approximation, no waves yet): normal velocity 0, free slip, adiabatic. 1 = full tank.
   */
  waterLevel: number;
  /**
   * Surface waves: a small-amplitude height field eta(x, z) [m] above the mean water level, evolved
   * by a linear wave equation in flux form (d eta/dt = -div q, dq/dt = -c^2 grad(eta - etaForced) - gamma q).
   * Phase speed [m/s]: 0.23 m/s is the minimum phase speed of capillary-gravity waves on water (at a
   * wavelength of about 1.7 cm), the right order for cm-scale ripples (rounded textbook value; not dispersive here).
   */
  waveSpeed: number;
  /** [GAME] damping rate of the waves [1/s]; real viscous damping of cm ripples is weaker (未確認), chosen so the surface settles in ~1 s */
  waveDamping: number;
  /** [GAME] gain on the dynamic head w|w|/(2g) of the up-flow under the surface (real hump is ~0.1 mm, invisible) */
  waveGain: number;
  /**
   * [GAME] surface lift per unit vapour quality in the top water row [m] (bursting bubbles). Bursts are
   * intermittent: each column bursts in a step with probability burstProb (deterministic hash shared by
   * CPU and GPU), and a burst lifts by vaporLift * chi / burstProb, so the mean lift is vaporLift * chi.
   */
  vaporLift: number;
  /** [GAME] probability that a boiling surface column bursts in a given step */
  burstProb: number;
  /** [GAME/numerical] cap on |etaForced| [m] (keeps the linear, small-amplitude assumption) */
  waveCap: number;
  /** heat sources (at most MAX_SOURCES) */
  sources: HeatSource[];
}

/** Number of water rows (cells j < waterRows are water). At least one row. */
export function waterRows(p: Pick<SimParams, "ny" | "waterLevel">): number {
  return Math.min(p.ny, Math.max(1, Math.round(p.waterLevel * p.ny)));
}

/** 32-bit integer hash (same arithmetic as the WGSL `hash`), returns a float in [0, 1) */
export function hash01(x0: number): number {
  let x = x0 >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  x ^= x >>> 16;
  return (x >>> 8) / 16777216;
}
/** does surface column c burst in step `step`? */
export function bursts(c: number, step: number, prob: number): boolean {
  const seed = (Math.imul(c, 9781) + Math.imul(step, 6271) + 17) >>> 0;
  return hash01(seed) < prob;
}

/** wave sub-steps per thermal sub-cycle so that c * dt / h <= 0.5 (2D stability limit is 1/sqrt 2) */
export function waveSubsteps(p: Pick<SimParams, "waveSpeed" | "dt" | "subcycles" | "h">): number {
  return Math.max(1, Math.ceil((p.waveSpeed * (p.dt / p.subcycles)) / (0.5 * p.h)));
}

/** Water level used by the app: see README (why 70 %). */
export const APP_WATER_LEVEL = 0.7;

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
    waterLevel: 1,
    waveSpeed: 0.23,
    waveDamping: 1.5,
    waveGain: 40,
    vaporLift: 0.05,
    burstProb: 0.15,
    waveCap: 0.002, // 2 mm: about 2 cells at 48^3, independent of the grid
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
  a[17] = waterRows(p);
  a[18] = p.waveSpeed;
  a[19] = p.waveDamping;
  a[20] = p.waveGain;
  a[21] = p.vaporLift;
  a[22] = p.waveCap;
  a[23] = waveSubsteps(p);
  a[24] = p.burstProb;
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
