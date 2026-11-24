/**
 * WGSL kernels of the physics layer. Each kernel is the per-cell / per-face twin of the
 * corresponding method in src/sim/cpuSolver.ts (the reference). Physical constants are
 * injected from src/sim/water.ts so CPU and GPU share one source of truth.
 *
 * Binding convention: binding 0 is always the parameter block P (see params.ts packParams).
 */
import * as W from "../sim/water";

export const WG = 64;

export interface KernelDef {
  name: string;
  /** buffers in binding order (after P), with access */
  bindings: { name: string; access: "read" | "read_write" }[];
  body: string;
  /** what the dispatch iterates over */
  domain: "cells" | "faces";
}

const f = (x: number) => {
  const s = x.toString();
  return s.includes(".") || s.includes("e") ? s : s + ".0";
};

const COMMON = /* wgsl */ `
const C_WATER : f32 = ${f(W.C_WATER)};
const C_ICE : f32 = ${f(W.C_ICE)};
const C_VAPOR : f32 = ${f(W.C_VAPOR)};
const L_F : f32 = ${f(W.L_F)};
const L_V : f32 = ${f(W.L_V)};
const H_B : f32 = ${f(W.H_B)};
const H_VSAT : f32 = ${f(W.H_VSAT)};
const K_WATER : f32 = ${f(W.K_WATER)};
const K_ICE : f32 = ${f(W.K_ICE)};
const K_VAPOR : f32 = ${f(W.K_VAPOR)};
const RHO_VAPOR : f32 = ${f(W.RHO_VAPOR)};
const SOLID_H : f32 = ${f((1 - W.SOLID_ICE_FRACTION) * W.L_F)};
const RHO_L100 : f32 = ${f(W.rhoLiquid(100))};

fn NX() -> i32 { return i32(P[0]); }
fn NY() -> i32 { return i32(P[1]); }
fn NZ() -> i32 { return i32(P[2]); }
fn HH() -> f32 { return P[3]; }
fn DT() -> f32 { return P[4]; }
fn SUBC() -> f32 { return P[5]; }
fn GRAV() -> f32 { return P[6]; }
fn RHOREF() -> f32 { return P[7]; }
fn RHO0() -> f32 { return P[8]; }
fn HC() -> f32 { return P[9]; }
fn KSCALE() -> f32 { return P[10]; }
fn BRISE() -> f32 { return P[11]; }
fn VBUOY() -> f32 { return P[12]; }
fn VMAX() -> f32 { return P[13]; }
fn VORT() -> f32 { return P[14]; }
fn SOR() -> f32 { return P[15]; }
fn NSRC() -> i32 { return i32(P[16]); }

fn ncells() -> i32 { return NX() * NY() * NZ(); }
fn nU() -> i32 { return (NX() + 1) * NY() * NZ(); }
fn nV() -> i32 { return NX() * (NY() + 1) * NZ(); }
fn nW() -> i32 { return NX() * NY() * (NZ() + 1); }
fn nFaces() -> i32 { return nU() + nV() + nW(); }
fn cidx(i : i32, j : i32, k : i32) -> i32 { return i + NX() * (j + NY() * k); }
fn cidxc(i : i32, j : i32, k : i32) -> i32 {
  return cidx(clamp(i, 0, NX() - 1), clamp(j, 0, NY() - 1), clamp(k, 0, NZ() - 1));
}
fn uIdx(i : i32, j : i32, k : i32) -> i32 { return i + (NX() + 1) * (j + NY() * k); }
fn vIdx(i : i32, j : i32, k : i32) -> i32 { return nU() + i + NX() * (j + (NY() + 1) * k); }
fn wIdx(i : i32, j : i32, k : i32) -> i32 { return nU() + nV() + i + NX() * (j + NY() * k); }
fn cellCoord(c : i32) -> vec3<i32> {
  let nx = NX(); let ny = NY();
  return vec3<i32>(c % nx, (c / nx) % ny, c / (nx * ny));
}
/** (axis, i, j, k) of a flat face index */
fn faceCoord(fi : i32) -> vec4<i32> {
  let nx = NX(); let ny = NY();
  if (fi < nU()) {
    let mx = nx + 1;
    return vec4<i32>(0, fi % mx, (fi / mx) % ny, fi / (mx * ny));
  }
  if (fi < nU() + nV()) {
    let g = fi - nU(); let my = ny + 1;
    return vec4<i32>(1, g % nx, (g / nx) % my, g / (nx * my));
  }
  let g = fi - nU() - nV();
  return vec4<i32>(2, g % nx, (g / nx) % ny, g / (nx * ny));
}
fn inDomain(i : i32, j : i32, k : i32) -> bool {
  return i >= 0 && j >= 0 && k >= 0 && i < NX() && j < NY() && k < NZ();
}

// ---- water thermodynamics (twin of src/sim/water.ts)
fn temperature(H : f32) -> f32 {
  if (H < 0.0) { return H / C_ICE; }
  if (H < L_F) { return 0.0; }
  if (H < H_B) { return (H - L_F) / C_WATER; }
  if (H < H_VSAT) { return 100.0; }
  return 100.0 + (H - H_VSAT) / C_VAPOR;
}
fn enthalpyFromT(T : f32) -> f32 {
  if (T < 0.0) { return T * C_ICE; }
  if (T <= 100.0) { return L_F + T * C_WATER; }
  return H_VSAT + (T - 100.0) * C_VAPOR;
}
fn iceFraction(H : f32) -> f32 { return 1.0 - clamp(H / L_F, 0.0, 1.0); }
fn vaporQuality(H : f32) -> f32 { return clamp((H - H_B) / L_V, 0.0, 1.0); }
fn isSolidH(H : f32) -> bool { return H < SOLID_H; }
fn conductivity(H : f32) -> f32 {
  let fi = iceFraction(H); let fv = vaporQuality(H);
  return fi * K_ICE + fv * K_VAPOR + (1.0 - fi - fv) * K_WATER;
}
fn rhoLiquid(T0 : f32) -> f32 {
  let t = clamp(T0, 0.0, 150.0);
  let num = 999.83952 + t * (16.945176 + t * (-7.9870401e-3 + t * (-46.170461e-6 + t * (105.56302e-9 + t * (-280.54253e-12)))));
  return num / (1.0 + 16.87985e-3 * t);
}
fn voidFraction(chi : f32) -> f32 {
  if (chi <= 0.0) { return 0.0; }
  let vv = chi / RHO_VAPOR;
  let vl = (1.0 - chi) / RHO_L100;
  return vv / (vv + vl);
}
fn buoyancyOf(H : f32) -> f32 {
  let T = temperature(H);
  let rl = rhoLiquid(T);
  let a = voidFraction(vaporQuality(H));
  return GRAV() * ((RHOREF() - rl) + VBUOY() * a * (rl - RHO_VAPOR)) / RHOREF();
}
fn minmod(a : f32, b : f32) -> f32 {
  if (a * b <= 0.0) { return 0.0; }
  if (abs(a) < abs(b)) { return a; }
  return b;
}
`;

/** helpers that need the state enthalpy buffer `Hs` */
const SOLID = /* wgsl */ `
fn solidAt(i : i32, j : i32, k : i32) -> bool {
  if (!inDomain(i, j, k)) { return true; }
  return isSolidH(Hs[cidx(i, j, k)]);
}
fn faceOpen(a : i32, i : i32, j : i32, k : i32) -> bool {
  if (a == 0) { return !solidAt(i - 1, j, k) && !solidAt(i, j, k); }
  if (a == 1) { return !solidAt(i, j - 1, k) && !solidAt(i, j, k); }
  return !solidAt(i, j, k - 1) && !solidAt(i, j, k);
}
`;

/** trilinear sampling of one velocity component from buffer `vin` (twin of CpuSolver.sampleComp) */
const SAMPLE = /* wgsl */ `
fn sampleComp(comp : i32, x : f32, y : f32, z : f32) -> f32 {
  let nx = NX(); let ny = NY(); let nz = NZ();
  var s = vec3<f32>(x, y, z);
  var m = vec3<i32>(nx, ny, nz);
  var off = 0;
  if (comp == 0) { s = vec3<f32>(x, y - 0.5, z - 0.5); m = vec3<i32>(nx + 1, ny, nz); off = 0; }
  else if (comp == 1) { s = vec3<f32>(x - 0.5, y, z - 0.5); m = vec3<i32>(nx, ny + 1, nz); off = nU(); }
  else { s = vec3<f32>(x - 0.5, y - 0.5, z); m = vec3<i32>(nx, ny, nz + 1); off = nU() + nV(); }
  s = clamp(s, vec3<f32>(0.0), vec3<f32>(m - vec3<i32>(1)));
  let i0 = min(i32(floor(s.x)), max(m.x - 2, 0));
  let j0 = min(i32(floor(s.y)), max(m.y - 2, 0));
  let k0 = min(i32(floor(s.z)), max(m.z - 2, 0));
  let i1 = min(i0 + 1, m.x - 1); let j1 = min(j0 + 1, m.y - 1); let k1 = min(k0 + 1, m.z - 1);
  let fx = s.x - f32(i0); let fy = s.y - f32(j0); let fz = s.z - f32(k0);
  let c00 = vin[off + i0 + m.x * (j0 + m.y * k0)] * (1.0 - fx) + vin[off + i1 + m.x * (j0 + m.y * k0)] * fx;
  let c10 = vin[off + i0 + m.x * (j1 + m.y * k0)] * (1.0 - fx) + vin[off + i1 + m.x * (j1 + m.y * k0)] * fx;
  let c01 = vin[off + i0 + m.x * (j0 + m.y * k1)] * (1.0 - fx) + vin[off + i1 + m.x * (j0 + m.y * k1)] * fx;
  let c11 = vin[off + i0 + m.x * (j1 + m.y * k1)] * (1.0 - fx) + vin[off + i1 + m.x * (j1 + m.y * k1)] * fx;
  let c0 = c00 * (1.0 - fy) + c10 * fy;
  let c1 = c01 * (1.0 - fy) + c11 * fy;
  return c0 * (1.0 - fz) + c1 * fz;
}
`;

const CELL_HEAD = /* wgsl */ `
  let c = i32(gid.x) + i32(gid.y) * ${WG} * 65535;
  if (c >= ncells()) { return; }
  let ijk = cellCoord(c); let i = ijk.x; let j = ijk.y; let k = ijk.z;
`;
const FACE_HEAD = /* wgsl */ `
  let fi = i32(gid.x) + i32(gid.y) * ${WG} * 65535;
  if (fi >= nFaces()) { return; }
  let fc = faceCoord(fi); let a = fc.x; let i = fc.y; let j = fc.z; let k = fc.w;
`;

export const KERNELS: Record<string, KernelDef> = {
  thermal: {
    name: "thermal",
    domain: "cells",
    bindings: [
      { name: "Hs", access: "read" },
      { name: "Hout", access: "read_write" },
      { name: "srcAcc", access: "read_write" },
    ],
    body: SOLID + /* wgsl */ `
fn vaporEnthalpy(H : f32) -> f32 { return max(H, H_VSAT); }
fn liquidEnthalpy(H : f32) -> f32 { return min(max(H, L_F), H_B); }
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${CELL_HEAD}
  let h = HH();
  let dts = DT() / SUBC();
  let G = HC() / (RHO0() * h);
  let condK = dts * KSCALE() / (RHO0() * h * h);
  let cD = BRISE() * dts / h;
  let Hc = Hs[c];
  let Tc = temperature(Hc);
  // heat sources
  var Hsrc = Hc;
  let x = (f32(i) + 0.5) * h; let y = (f32(j) + 0.5) * h; let z = (f32(k) + 0.5) * h;
  for (var s = 0; s < NSRC(); s++) {
    let o = 32 + s * 8;
    let d = vec3<f32>(x - P[o], y - P[o + 1], z - P[o + 2]);
    let r = P[o + 3];
    if (dot(d, d) < r * r) {
      let Tsrc = P[o + 4];
      let T = temperature(Hsrc);
      let Ht = enthalpyFromT(Tsrc);
      let dH = G * dts * (Tsrc - T);
      if (dH > 0.0) { Hsrc = min(Hsrc + dH, max(Ht, Hsrc)); }
      else { Hsrc = max(Hsrc + dH, min(Ht, Hsrc)); }
    }
  }
  let dSrc = Hsrc - Hc;
  // conduction
  let kc = conductivity(Hc);
  var dCond = 0.0;
  for (var n = 0; n < 6; n++) {
    var o = vec3<i32>(0);
    if (n == 0) { o = vec3<i32>(-1, 0, 0); } else if (n == 1) { o = vec3<i32>(1, 0, 0); }
    else if (n == 2) { o = vec3<i32>(0, -1, 0); } else if (n == 3) { o = vec3<i32>(0, 1, 0); }
    else if (n == 4) { o = vec3<i32>(0, 0, -1); } else { o = vec3<i32>(0, 0, 1); }
    let a = i + o.x; let b = j + o.y; let e = k + o.z;
    if (!inDomain(a, b, e)) { continue; }
    let Hn = Hs[cidx(a, b, e)];
    let kn = conductivity(Hn);
    let kf = 2.0 * kc * kn / (kc + kn);
    dCond += kf * (temperature(Hn) - Tc);
  }
  dCond *= condK;
  // vapour drift
  var dDrift = 0.0;
  let solidC = isSolidH(Hc);
  if (!solidC && j > 0) {
    let Hb = Hs[cidx(i, j - 1, k)];
    if (!isSolidH(Hb)) {
      let x0 = cD * vaporQuality(Hb) * (1.0 - vaporQuality(Hc));
      dDrift += x0 * (vaporEnthalpy(Hb) - liquidEnthalpy(Hc));
    }
  }
  if (!solidC && j < NY() - 1) {
    let Ha = Hs[cidx(i, j + 1, k)];
    if (!isSolidH(Ha)) {
      let x0 = cD * vaporQuality(Hc) * (1.0 - vaporQuality(Ha));
      dDrift -= x0 * (vaporEnthalpy(Hc) - liquidEnthalpy(Ha));
    }
  }
  Hout[c] = Hc + dSrc + dCond + dDrift;
  srcAcc[c] = srcAcc[c] + dSrc;
}`,
  },

  advectScalar: {
    name: "advectScalar",
    domain: "cells",
    bindings: [
      { name: "q", access: "read" },
      { name: "vel", access: "read" },
      { name: "qout", access: "read_write" },
    ],
    body: /* wgsl */ `
fn qa(i : i32, j : i32, k : i32) -> f32 { return q[cidxc(i, j, k)]; }
fn faceFlux(uf : f32, r : f32, qLL : f32, qL : f32, qR : f32, qRR : f32) -> f32 {
  if (uf >= 0.0) { return uf * (qL + 0.5 * (1.0 - uf * r) * minmod(qL - qLL, qR - qL)); }
  return uf * (qR - 0.5 * (1.0 + uf * r) * minmod(qR - qL, qRR - qR));
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${CELL_HEAD}
  let r = DT() / SUBC() / HH();
  let qc = q[c];
  let fxl = faceFlux(vel[uIdx(i, j, k)], r, qa(i - 2, j, k), qa(i - 1, j, k), qc, qa(i + 1, j, k));
  let fxr = faceFlux(vel[uIdx(i + 1, j, k)], r, qa(i - 1, j, k), qc, qa(i + 1, j, k), qa(i + 2, j, k));
  let fyl = faceFlux(vel[vIdx(i, j, k)], r, qa(i, j - 2, k), qa(i, j - 1, k), qc, qa(i, j + 1, k));
  let fyr = faceFlux(vel[vIdx(i, j + 1, k)], r, qa(i, j - 1, k), qc, qa(i, j + 1, k), qa(i, j + 2, k));
  let fzl = faceFlux(vel[wIdx(i, j, k)], r, qa(i, j, k - 2), qa(i, j, k - 1), qc, qa(i, j, k + 1));
  let fzr = faceFlux(vel[wIdx(i, j, k + 1)], r, qa(i, j, k - 1), qc, qa(i, j, k + 1), qa(i, j, k + 2));
  qout[c] = qc - r * (fxr - fxl + fyr - fyl + fzr - fzl);
}`,
  },

  advectVel: {
    name: "advectVel",
    domain: "faces",
    bindings: [
      { name: "vin", access: "read" },
      { name: "Hs", access: "read" },
      { name: "vout", access: "read_write" },
    ],
    body: SOLID + SAMPLE + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${FACE_HEAD}
  if (!faceOpen(a, i, j, k)) { vout[fi] = 0.0; return; }
  let s = DT() / HH();
  var x = f32(i) + 0.5; var y = f32(j) + 0.5; var z = f32(k) + 0.5;
  if (a == 0) { x = f32(i); } else if (a == 1) { y = f32(j); } else { z = f32(k); }
  let vx = sampleComp(0, x, y, z); let vy = sampleComp(1, x, y, z); let vz = sampleComp(2, x, y, z);
  let mx = x - 0.5 * s * vx; let my = y - 0.5 * s * vy; let mz = z - 0.5 * s * vz;
  let ux = sampleComp(0, mx, my, mz); let uy = sampleComp(1, mx, my, mz); let uz = sampleComp(2, mx, my, mz);
  var val = sampleComp(a, x - s * ux, y - s * uy, z - s * uz);
  if (a == 1) {
    let bA = buoyancyOf(Hs[cidx(i, j - 1, k)]);
    let bB = buoyancyOf(Hs[cidx(i, j, k)]);
    val += DT() * 0.5 * (bA + bB);
  }
  vout[fi] = clamp(val, -VMAX(), VMAX());
}`,
  },

  curl: {
    name: "curl",
    domain: "cells",
    bindings: [
      { name: "vin", access: "read" },
      { name: "omega", access: "read_write" },
    ],
    body: /* wgsl */ `
fn cv(i0 : i32, j0 : i32, k0 : i32) -> vec3<f32> {
  let i = clamp(i0, 0, NX() - 1); let j = clamp(j0, 0, NY() - 1); let k = clamp(k0, 0, NZ() - 1);
  return vec3<f32>(0.5 * (vin[uIdx(i, j, k)] + vin[uIdx(i + 1, j, k)]),
                   0.5 * (vin[vIdx(i, j, k)] + vin[vIdx(i, j + 1, k)]),
                   0.5 * (vin[wIdx(i, j, k)] + vin[wIdx(i, j, k + 1)]));
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${CELL_HEAD}
  let xp = cv(i + 1, j, k); let xm = cv(i - 1, j, k);
  let yp = cv(i, j + 1, k); let ym = cv(i, j - 1, k);
  let zp = cv(i, j, k + 1); let zm = cv(i, j, k - 1);
  let inv = 1.0 / (2.0 * HH());
  let wx = (yp.z - ym.z - (zp.y - zm.y)) * inv;
  let wy = (zp.x - zm.x - (xp.z - xm.z)) * inv;
  let wz = (xp.y - xm.y - (yp.x - ym.x)) * inv;
  omega[c * 4] = wx; omega[c * 4 + 1] = wy; omega[c * 4 + 2] = wz;
  omega[c * 4 + 3] = length(vec3<f32>(wx, wy, wz));
}`,
  },

  confine: {
    name: "confine",
    domain: "faces",
    bindings: [
      { name: "omega", access: "read" },
      { name: "Hs", access: "read" },
      { name: "vout", access: "read_write" },
    ],
    body: SOLID + /* wgsl */ `
fn mag(a : i32, b : i32, e : i32) -> f32 { return omega[cidxc(a, b, e) * 4 + 3]; }
fn confForce(i : i32, j : i32, k : i32) -> vec3<f32> {
  let h = HH();
  let g = vec3<f32>(mag(i + 1, j, k) - mag(i - 1, j, k), mag(i, j + 1, k) - mag(i, j - 1, k), mag(i, j, k + 1) - mag(i, j, k - 1)) / (2.0 * h);
  let len = length(g) + 1e-12;
  let N = g / len;
  let o = cidx(i, j, k) * 4;
  let w = vec3<f32>(omega[o], omega[o + 1], omega[o + 2]);
  return VORT() * h * cross(N, w);
}
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${FACE_HEAD}
  if (!faceOpen(a, i, j, k)) { return; }
  var fa = vec3<f32>(0.0); let fb = confForce(i, j, k);
  if (a == 0) { fa = confForce(i - 1, j, k); } else if (a == 1) { fa = confForce(i, j - 1, k); } else { fa = confForce(i, j, k - 1); }
  vout[fi] = vout[fi] + DT() * 0.5 * (fa[a] + fb[a]);
}`,
  },

  divergence: {
    name: "divergence",
    domain: "cells",
    bindings: [
      { name: "vin", access: "read" },
      { name: "div", access: "read_write" },
    ],
    body: /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${CELL_HEAD}
  div[c] = (vin[uIdx(i + 1, j, k)] - vin[uIdx(i, j, k)] + vin[vIdx(i, j + 1, k)] - vin[vIdx(i, j, k)]
          + vin[wIdx(i, j, k + 1)] - vin[wIdx(i, j, k)]) / HH();
}`,
  },

  pressureRed: pressureKernel(0),
  pressureBlack: pressureKernel(1),

  project: {
    name: "project",
    domain: "faces",
    bindings: [
      { name: "vin", access: "read" },
      { name: "phi", access: "read" },
      { name: "Hs", access: "read" },
      { name: "vout", access: "read_write" },
    ],
    body: SOLID + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${FACE_HEAD}
  if (!faceOpen(a, i, j, k)) { vout[fi] = 0.0; return; }
  let pb = phi[cidx(i, j, k)];
  var pa = 0.0;
  if (a == 0) { pa = phi[cidx(i - 1, j, k)]; } else if (a == 1) { pa = phi[cidx(i, j - 1, k)]; } else { pa = phi[cidx(i, j, k - 1)]; }
  vout[fi] = clamp(vin[fi] - (pb - pa) / HH(), -VMAX(), VMAX());
}`,
  },
};

function pressureKernel(color: number): KernelDef {
  return {
    name: color === 0 ? "pressureRed" : "pressureBlack",
    domain: "cells",
    bindings: [
      { name: "div", access: "read" },
      { name: "Hs", access: "read" },
      { name: "phi", access: "read_write" },
    ],
    body: SOLID + /* wgsl */ `
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
${CELL_HEAD}
  if (((i + j + k) & 1) != ${color}) { return; }
  if (solidAt(i, j, k)) { phi[c] = 0.0; return; }
  var sum = 0.0; var n = 0;
  if (!solidAt(i - 1, j, k)) { sum += phi[cidx(i - 1, j, k)]; n++; }
  if (!solidAt(i + 1, j, k)) { sum += phi[cidx(i + 1, j, k)]; n++; }
  if (!solidAt(i, j - 1, k)) { sum += phi[cidx(i, j - 1, k)]; n++; }
  if (!solidAt(i, j + 1, k)) { sum += phi[cidx(i, j + 1, k)]; n++; }
  if (!solidAt(i, j, k - 1)) { sum += phi[cidx(i, j, k - 1)]; n++; }
  if (!solidAt(i, j, k + 1)) { sum += phi[cidx(i, j, k + 1)]; n++; }
  if (n == 0) { phi[c] = 0.0; return; }
  let h2 = HH() * HH();
  let gs = (sum - h2 * div[c]) / f32(n);
  phi[c] = (1.0 - SOR()) * phi[c] + SOR() * gs;
}`,
  };
}

/** Full WGSL source of a kernel. */
export function kernelSource(k: KernelDef): string {
  const decl = [`@group(0) @binding(0) var<storage, read> P : array<f32>;`];
  k.bindings.forEach((b, n) => decl.push(`@group(0) @binding(${n + 1}) var<storage, ${b.access}> ${b.name} : array<f32>;`));
  return decl.join("\n") + "\n" + COMMON + k.body;
}

export function bindingsMapping(k: KernelDef) {
  const m: Record<string, { group: number; binding: number }> = { P: { group: 0, binding: 0 } };
  k.bindings.forEach((b, n) => (m[b.name] = { group: 0, binding: n + 1 }));
  return m;
}

/** workgroup counts for n items (2D when n exceeds the 1D limit) */
export function dispatchSize(n: number): [number, number] {
  const groups = Math.ceil(n / WG);
  if (groups <= 65535) return [groups, 1];
  return [65535, Math.ceil(groups / 65535)];
}
