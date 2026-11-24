import { CpuSolver } from "../../src/sim/cpuSolver";
import { defaultParams, type SimParams } from "../../src/sim/params";

/** Small deterministic PRNG (mulberry32) so tests are reproducible. */
export function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function smallParams(n = 12, over: Partial<SimParams> = {}): SimParams {
  return { ...defaultParams(n), ...over };
}

/** Fill the face velocity with a random field and project it so it is (nearly) divergence free. */
export function randomFlow(s: CpuSolver, amp: number, seed = 1, iters = 1000) {
  const r = rng(seed);
  for (let f = 0; f < s.g.nFaces; f++) {
    const [a, i, j, k] = s.faceCoord(f);
    s.vel2[f] = s.faceOpen(a, i, j, k) ? (r() * 2 - 1) * amp : 0;
  }
  const saved = s.p.pressureIters;
  s.p.pressureIters = iters;
  s.divergence(s.vel2);
  s.pressure();
  s.project();
  s.p.pressureIters = saved;
}

export const sum = (a: ArrayLike<number>) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
};

export function domainLength(p: SimParams) {
  return { Lx: p.nx * p.h, Ly: p.ny * p.h, Lz: p.nz * p.h };
}

/** Switch a solver's storage to Float64Array (to separate scheme conservation from f32 rounding). */
export function toF64(s: CpuSolver) {
  const o = s as unknown as Record<string, ArrayLike<number>>;
  for (const k of ["H", "Htmp", "vel", "vel2", "phi", "div", "omega", "srcAcc"]) o[k] = Float64Array.from(o[k]);
  return s;
}
