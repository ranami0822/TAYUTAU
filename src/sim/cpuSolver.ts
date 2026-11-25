/**
 * CPU reference implementation of the physics layer.
 *
 * This is the specification of the algorithm: every function here has a WGSL twin in
 * src/gpu/kernels.ts that performs the same arithmetic per cell / per face. Tests check
 * the physics (conservation, divergence, phase change) on this implementation in Node, and
 * check that the GPU kernels agree with it in a browser.
 *
 * Grid: MAC (staggered) grid, nx*ny*nz cells of size h, y is up. Closed box (no-slip normal,
 * adiabatic walls). Velocity faces are stored in one array: [u | v | w].
 *
 * One step(dt):
 *   repeat `subcycles` times with dts = dt / subcycles:
 *     thermal:      H -> Htmp   (heat sources, conduction, vapour drift; all flux form)
 *     advectScalar: Htmp -> H   (MUSCL/minmod flux-form advection with the projected velocity)
 *   advectVel:   vel -> vel2  (semi-Lagrangian RK2, + buoyancy on v, solid faces = 0, clamp)
 *   curl:        vel2 -> omega
 *   confine:     vel2 += dt * vorticity-confinement force
 *   divergence:  vel2 -> div
 *   pressure:    red-black SOR on phi (warm-started)
 *   project:     vel = vel2 - grad phi (open faces only), clamp
 */
import { type SimParams, MAX_SOURCES, waterRows, waveSubsteps, bursts } from "./params";
import {
  temperature,
  enthalpyFromT,
  vaporQuality,
  isSolid,
  conductivity,
  rhoLiquid,
  voidFraction,
  H_B,
  H_VSAT,
  L_F,
  RHO_VAPOR,
} from "./water";

export interface Grid {
  nx: number;
  ny: number;
  nz: number;
  nCells: number;
  nU: number;
  nV: number;
  nW: number;
  oV: number;
  oW: number;
  nFaces: number;
}

export function makeGrid(nx: number, ny: number, nz: number): Grid {
  const nU = (nx + 1) * ny * nz;
  const nV = nx * (ny + 1) * nz;
  const nW = nx * ny * (nz + 1);
  return { nx, ny, nz, nCells: nx * ny * nz, nU, nV, nW, oV: nU, oW: nU + nV, nFaces: nU + nV + nW };
}

const minmod = (a: number, b: number) => (a * b <= 0 ? 0 : Math.abs(a) < Math.abs(b) ? a : b);

/** Buoyant acceleration [m/s^2] of a cell from its enthalpy (Boussinesq). */
export function buoyancy(H: number, p: SimParams): number {
  const T = temperature(H);
  const rl = rhoLiquid(T);
  const alpha = voidFraction(vaporQuality(H));
  return (p.g * ((p.rhoRef - rl) + p.vaporBuoyancyScale * alpha * (rl - RHO_VAPOR))) / p.rhoRef;
}

/** Enthalpy carried by rising vapour leaving a cell (saturated, or the cell's own if superheated). */
const vaporEnthalpy = (H: number) => Math.max(H, H_VSAT);
/** Enthalpy of the liquid that sinks to replace vapour (clamped to the liquid range). */
const liquidEnthalpy = (H: number) => Math.min(Math.max(H, L_F), H_B);

export class CpuSolver {
  readonly g: Grid;
  p: SimParams;
  H: Float32Array;
  Htmp: Float32Array;
  vel: Float32Array;
  vel2: Float32Array;
  phi: Float32Array;
  div: Float32Array;
  omega: Float32Array; // 4 per cell: wx, wy, wz, |w|
  /** accumulated enthalpy added by heat sources per cell [J/kg] (exact energy bookkeeping) */
  srcAcc: Float32Array;
  /** surface height above the mean level [m], nx*nz (index i + nx*k) */
  eta: Float32Array;
  /** surface volume flux per unit width [m^2/s]: x faces (nx+1)*nz then z faces nx*(nz+1) */
  q: Float32Array;
  time = 0;
  /** steps taken (drives the intermittent bubble bursts; mirrored by the GPU clock buffer) */
  stepCount = 0;

  constructor(p: SimParams, initialT = 20) {
    this.p = p;
    this.g = makeGrid(p.nx, p.ny, p.nz);
    const n = this.g.nCells;
    this.H = new Float32Array(n).fill(enthalpyFromT(initialT));
    this.Htmp = new Float32Array(n);
    this.vel = new Float32Array(this.g.nFaces);
    this.vel2 = new Float32Array(this.g.nFaces);
    this.phi = new Float32Array(n);
    this.div = new Float32Array(n);
    this.omega = new Float32Array(n * 4);
    this.srcAcc = new Float32Array(n);
    this.eta = new Float32Array(p.nx * p.nz);
    this.q = new Float32Array((p.nx + 1) * p.nz + p.nx * (p.nz + 1));
  }

  idx(i: number, j: number, k: number) {
    return i + this.g.nx * (j + this.g.ny * k);
  }
  uIdx(i: number, j: number, k: number) {
    return i + (this.g.nx + 1) * (j + this.g.ny * k);
  }
  vIdx(i: number, j: number, k: number) {
    return this.g.oV + i + this.g.nx * (j + (this.g.ny + 1) * k);
  }
  wIdx(i: number, j: number, k: number) {
    return this.g.oW + i + this.g.nx * (j + this.g.ny * k);
  }
  /** number of water rows: cells with j >= rows are air */
  get rows() {
    return waterRows(this.p);
  }
  /** air cell (above the water surface) */
  isAir(j: number) {
    return j >= this.rows;
  }
  /** blocked for flow: ice, air (the surface acts as a rigid lid) or out of the domain */
  solidAt(i: number, j: number, k: number, H = this.H) {
    const { nx, ny, nz } = this.g;
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return true;
    if (this.isAir(j)) return true;
    return isSolid(H[this.idx(i, j, k)]);
  }

  step() {
    const p = this.p;
    const dts = p.dt / p.subcycles;
    for (let s = 0; s < p.subcycles; s++) {
      this.thermal(dts);
      this.advectScalar(this.Htmp, this.H, dts);
      const ws = waveSubsteps(p);
      for (let w = 0; w < ws; w++) {
        this.waveFlux(dts / ws);
        this.waveHeight(dts / ws);
      }
    }
    this.advectVel();
    this.curl();
    this.confine();
    this.divergence(this.vel2);
    this.pressure();
    this.project();
    this.time += p.dt;
    this.stepCount++;
  }

  // ---------------------------------------------------------------- thermal
  /** Heat sources + conduction + vapour drift. Reads H, writes Htmp, accumulates srcAcc. */
  thermal(dts: number) {
    const { nx, ny, nz } = this.g;
    const p = this.p;
    const H = this.H;
    const out = this.Htmp;
    const h = p.h;
    const G = p.hc / (p.rho0 * h); // [J/(kg K s)]
    const condK = (dts * p.kScale) / (p.rho0 * h * h);
    // bubbles rise because of gravity: slip velocity scales with g (no drift in zero gravity)
    const cD = (p.bubbleRise * Math.max(0, p.g / 9.81) * dts) / h;
    const nSrc = Math.min(p.sources.length, MAX_SOURCES);
    const rows = this.rows;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const c = this.idx(i, j, k);
          const Hc = H[c];
          if (j >= rows) {
            out[c] = Hc; // air: holds no water, exchanges nothing
            continue;
          }
          const Tc = temperature(Hc);
          // heat sources (applied sequentially, each limited so it cannot overshoot its own temperature)
          let Hs = Hc;
          const x = (i + 0.5) * h,
            y = (j + 0.5) * h,
            z = (k + 0.5) * h;
          for (let s = 0; s < nSrc; s++) {
            const src = p.sources[s];
            const dx = x - src.x,
              dy = y - src.y,
              dz = z - src.z;
            if (dx * dx + dy * dy + dz * dz < src.r * src.r) {
              const T = temperature(Hs);
              const Ht = enthalpyFromT(src.T);
              const dH = G * dts * (src.T - T);
              Hs = dH > 0 ? Math.min(Hs + dH, Math.max(Ht, Hs)) : Math.max(Hs + dH, Math.min(Ht, Hs));
            }
          }
          const dSrc = Hs - Hc;
          // conduction (adiabatic walls)
          const kc = conductivity(Hc);
          let dCond = 0;
          const nbs: [number, number, number][] = [
            [i - 1, j, k],
            [i + 1, j, k],
            [i, j - 1, k],
            [i, j + 1, k],
            [i, j, k - 1],
            [i, j, k + 1],
          ];
          for (const [a, b, d] of nbs) {
            if (a < 0 || b < 0 || d < 0 || a >= nx || b >= rows || d >= nz) continue; // walls and surface adiabatic
            const Hn = H[this.idx(a, b, d)];
            const kn = conductivity(Hn);
            const kf = (2 * kc * kn) / (kc + kn);
            dCond += kf * (temperature(Hn) - Tc);
          }
          dCond *= condK;
          // vapour drift through the bottom (in) and top (out) faces
          let dDrift = 0;
          const solidC = isSolid(Hc);
          if (!solidC && j > 0) {
            const Hb = H[this.idx(i, j - 1, k)];
            if (!isSolid(Hb)) {
              const x0 = cD * vaporQuality(Hb) * (1 - vaporQuality(Hc));
              dDrift += x0 * (vaporEnthalpy(Hb) - liquidEnthalpy(Hc));
            }
          }
          if (!solidC && j < rows - 1) {
            const Ha = H[this.idx(i, j + 1, k)];
            if (!isSolid(Ha)) {
              const x0 = cD * vaporQuality(Hc) * (1 - vaporQuality(Ha));
              dDrift -= x0 * (vaporEnthalpy(Hc) - liquidEnthalpy(Ha));
            }
          }
          out[c] = Hc + dSrc + dCond + dDrift;
          this.srcAcc[c] += dSrc;
        }
  }

  // ---------------------------------------------------------------- surface waves
  /** surface column (i,k) is frozen (top water cell is ice): no waves there */
  surfaceFrozen(i: number, k: number) {
    return isSolid(this.H[this.idx(i, this.rows - 1, k)]);
  }
  /** height the flow below pushes the surface to [m]: dynamic head of the up-flow + bursting vapour, capped */
  etaForced(i: number, k: number) {
    const p = this.p;
    const j = this.rows - 1;
    const w = j > 0 ? this.vel[this.vIdx(i, j, k)] : 0; // vertical velocity just under the top water cell
    const chi = vaporQuality(this.H[this.idx(i, j, k)]);
    const burst = p.burstProb > 0 ? (bursts(i + this.g.nx * k, this.stepCount, p.burstProb) ? 1 / p.burstProb : 0) : 1;
    const e = (p.waveGain * w * Math.abs(w)) / (2 * p.g || 1) + p.vaporLift * chi * burst;
    return Math.min(p.waveCap, Math.max(-p.waveCap, e));
  }
  /** dq/dt = -c^2 grad(eta - etaForced) - gamma q on interior surface faces; closed walls and ice: q = 0 */
  waveFlux(dts: number) {
    const { nx, nz } = this.g;
    const p = this.p;
    const c2 = p.waveSpeed * p.waveSpeed;
    const damp = 1 / (1 + p.waveDamping * dts); // implicit damping (unconditionally stable)
    const oz = (nx + 1) * nz;
    const d = (i: number, k: number) => this.eta[i + nx * k] - this.etaForced(i, k);
    for (let k = 0; k < nz; k++)
      for (let i = 0; i <= nx; i++) {
        const f = i + (nx + 1) * k;
        if (i === 0 || i === nx || this.surfaceFrozen(i - 1, k) || this.surfaceFrozen(i, k)) { this.q[f] = 0; continue; }
        this.q[f] = (this.q[f] - (dts * c2 * (d(i, k) - d(i - 1, k))) / p.h) * damp;
      }
    for (let k = 0; k <= nz; k++)
      for (let i = 0; i < nx; i++) {
        const f = oz + i + nx * k;
        if (k === 0 || k === nz || this.surfaceFrozen(i, k - 1) || this.surfaceFrozen(i, k)) { this.q[f] = 0; continue; }
        this.q[f] = (this.q[f] - (dts * c2 * (d(i, k) - d(i, k - 1))) / p.h) * damp;
      }
  }
  /** d eta/dt = -div q (flux form: the sum of eta, i.e. the water volume, is conserved exactly) */
  /** vertical velocity just under the top water cell (proxy for the surface velocity under the rigid lid) */
  surfaceW(i: number, k: number) {
    const j = this.rows - 1;
    return j > 0 ? this.vel[this.vIdx(i, j, k)] : 0;
  }
  /** mean of surfaceW over the unfrozen columns: [mean, count] (twin of the surfaceMean kernel) */
  surfaceMean(): [number, number] {
    const { nx, nz } = this.g;
    let sum = 0, n = 0;
    for (let k = 0; k < nz; k++)
      for (let i = 0; i < nx; i++)
        if (!this.surfaceFrozen(i, k)) {
          sum += this.surfaceW(i, k);
          n++;
        }
    return [n > 0 ? sum / n : 0, n];
  }
  /**
   * d eta/dt = -div q + (w - mean w): wave propagation plus the kinematic surface condition (the surface rises
   * where the water below moves up). Subtracting the mean keeps the volume exact; for an incompressible flow
   * under a closed lid the mean is zero anyway (up to the pressure-solver residual).
   */
  waveHeight(dts: number) {
    const { nx, nz } = this.g;
    const oz = (nx + 1) * nz;
    const r = dts / this.p.h;
    const [wm] = this.surfaceMean();
    for (let k = 0; k < nz; k++)
      for (let i = 0; i < nx; i++) {
        const qx = this.q[i + 1 + (nx + 1) * k] - this.q[i + (nx + 1) * k];
        const qz = this.q[oz + i + nx * (k + 1)] - this.q[oz + i + nx * k];
        let e = this.eta[i + nx * k] - r * (qx + qz);
        if (!this.surfaceFrozen(i, k)) e += dts * (this.surfaceW(i, k) - wm);
        this.eta[i + nx * k] = e;
      }
  }
  /** water volume [m^3]: fixed water rows + surface displacement */
  waterVolume() {
    let s = 0;
    for (const e of this.eta) s += e;
    return this.waterCells() * this.p.h ** 3 + s * this.p.h * this.p.h;
  }
  /** air volume [m^3]: the rest of the closed tank */
  airVolume() {
    return this.g.nCells * this.p.h ** 3 - this.waterVolume();
  }

  // ---------------------------------------------------------------- scalar advection
  /** Flux-form MUSCL (minmod) advection of a cell-centred scalar with the current face velocity. */
  advectScalar(q: Float32Array, out: Float32Array, dts: number) {
    const { nx, ny, nz } = this.g;
    const h = this.p.h;
    const vel = this.vel;
    const r = dts / h;
    const qa = (i: number, j: number, k: number) =>
      q[this.idx(Math.min(nx - 1, Math.max(0, i)), Math.min(ny - 1, Math.max(0, j)), Math.min(nz - 1, Math.max(0, k)))];
    // face value for a face between cells L (lower) and R (upper) along an axis, given the 4 cells LL,L,R,RR
    const faceFlux = (uf: number, qLL: number, qL: number, qR: number, qRR: number) => {
      if (uf >= 0) return uf * (qL + 0.5 * (1 - uf * r) * minmod(qL - qLL, qR - qL));
      return uf * (qR - 0.5 * (1 + uf * r) * minmod(qR - qL, qRR - qR));
    };
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const c = this.idx(i, j, k);
          const qc = q[c];
          const fxl = faceFlux(vel[this.uIdx(i, j, k)], qa(i - 2, j, k), qa(i - 1, j, k), qc, qa(i + 1, j, k));
          const fxr = faceFlux(vel[this.uIdx(i + 1, j, k)], qa(i - 1, j, k), qc, qa(i + 1, j, k), qa(i + 2, j, k));
          const fyl = faceFlux(vel[this.vIdx(i, j, k)], qa(i, j - 2, k), qa(i, j - 1, k), qc, qa(i, j + 1, k));
          const fyr = faceFlux(vel[this.vIdx(i, j + 1, k)], qa(i, j - 1, k), qc, qa(i, j + 1, k), qa(i, j + 2, k));
          const fzl = faceFlux(vel[this.wIdx(i, j, k)], qa(i, j, k - 2), qa(i, j, k - 1), qc, qa(i, j, k + 1));
          const fzr = faceFlux(vel[this.wIdx(i, j, k + 1)], qa(i, j, k - 1), qc, qa(i, j, k + 1), qa(i, j, k + 2));
          out[c] = qc - r * (fxr - fxl + fyr - fyl + fzr - fzl);
        }
  }

  // ---------------------------------------------------------------- velocity
  /** Sample one velocity component (0=u,1=v,2=w) at a position in cell units (cell centres at i+0.5). */
  sampleComp(vel: Float32Array, comp: number, x: number, y: number, z: number) {
    const { nx, ny, nz } = this.g;
    let sx: number, sy: number, sz: number, mx: number, my: number, mz: number, off: number;
    if (comp === 0) {
      sx = x; sy = y - 0.5; sz = z - 0.5; mx = nx + 1; my = ny; mz = nz; off = 0;
    } else if (comp === 1) {
      sx = x - 0.5; sy = y; sz = z - 0.5; mx = nx; my = ny + 1; mz = nz; off = this.g.oV;
    } else {
      sx = x - 0.5; sy = y - 0.5; sz = z; mx = nx; my = ny; mz = nz + 1; off = this.g.oW;
    }
    sx = Math.min(Math.max(sx, 0), mx - 1);
    sy = Math.min(Math.max(sy, 0), my - 1);
    sz = Math.min(Math.max(sz, 0), mz - 1);
    const i0 = Math.min(Math.floor(sx), mx - 2 < 0 ? 0 : mx - 2);
    const j0 = Math.min(Math.floor(sy), my - 2 < 0 ? 0 : my - 2);
    const k0 = Math.min(Math.floor(sz), mz - 2 < 0 ? 0 : mz - 2);
    const i1 = Math.min(i0 + 1, mx - 1),
      j1 = Math.min(j0 + 1, my - 1),
      k1 = Math.min(k0 + 1, mz - 1);
    const fx = sx - i0,
      fy = sy - j0,
      fz = sz - k0;
    const at = (i: number, j: number, k: number) => vel[off + i + mx * (j + my * k)];
    const c00 = at(i0, j0, k0) * (1 - fx) + at(i1, j0, k0) * fx;
    const c10 = at(i0, j1, k0) * (1 - fx) + at(i1, j1, k0) * fx;
    const c01 = at(i0, j0, k1) * (1 - fx) + at(i1, j0, k1) * fx;
    const c11 = at(i0, j1, k1) * (1 - fx) + at(i1, j1, k1) * fx;
    const c0 = c00 * (1 - fy) + c10 * fy;
    const c1 = c01 * (1 - fy) + c11 * fy;
    return c0 * (1 - fz) + c1 * fz;
  }

  /** Face open = both neighbouring cells exist and are not solid. axis 0/1/2, face index (i,j,k). */
  faceOpen(axis: number, i: number, j: number, k: number) {
    if (axis === 0) return !this.solidAt(i - 1, j, k) && !this.solidAt(i, j, k);
    if (axis === 1) return !this.solidAt(i, j - 1, k) && !this.solidAt(i, j, k);
    return !this.solidAt(i, j, k - 1) && !this.solidAt(i, j, k);
  }

  /** decode a flat face index into (axis, i, j, k) */
  faceCoord(f: number): [number, number, number, number] {
    const { nx, ny, nU, nV } = this.g;
    if (f < nU) {
      const mx = nx + 1;
      return [0, f % mx, Math.floor(f / mx) % ny, Math.floor(f / (mx * ny))];
    }
    if (f < nU + nV) {
      const g = f - nU;
      const my = ny + 1;
      return [1, g % nx, Math.floor(g / nx) % my, Math.floor(g / (nx * my))];
    }
    const g = f - nU - nV;
    return [2, g % nx, Math.floor(g / nx) % ny, Math.floor(g / (nx * ny))];
  }

  advectVel() {
    const p = this.p;
    const s = p.dt / p.h; // velocity [m/s] -> displacement in cells
    for (let f = 0; f < this.g.nFaces; f++) {
      const [axis, i, j, k] = this.faceCoord(f);
      if (!this.faceOpen(axis, i, j, k)) {
        this.vel2[f] = 0;
        continue;
      }
      const x = axis === 0 ? i : i + 0.5;
      const y = axis === 1 ? j : j + 0.5;
      const z = axis === 2 ? k : k + 0.5;
      const vx = this.sampleComp(this.vel, 0, x, y, z);
      const vy = this.sampleComp(this.vel, 1, x, y, z);
      const vz = this.sampleComp(this.vel, 2, x, y, z);
      const mx = x - 0.5 * s * vx,
        my = y - 0.5 * s * vy,
        mz = z - 0.5 * s * vz;
      const ux = this.sampleComp(this.vel, 0, mx, my, mz);
      const uy = this.sampleComp(this.vel, 1, mx, my, mz);
      const uz = this.sampleComp(this.vel, 2, mx, my, mz);
      const bx = x - s * ux,
        by = y - s * uy,
        bz = z - s * uz;
      let val = this.sampleComp(this.vel, axis, bx, by, bz);
      if (axis === 1) {
        const bA = buoyancy(this.H[this.idx(i, j - 1, k)], p);
        const bB = buoyancy(this.H[this.idx(i, j, k)], p);
        val += p.dt * 0.5 * (bA + bB);
      }
      // a source that is being moved pushes the water: inside its sphere the velocity follows the source
      for (let q = 0; q < Math.min(p.sources.length, MAX_SOURCES); q++) {
        const src = p.sources[q];
        const sv = [src.vx ?? 0, src.vy ?? 0, src.vz ?? 0];
        if (sv[0] === 0 && sv[1] === 0 && sv[2] === 0) continue;
        const dx = x * p.h - src.x, dy = y * p.h - src.y, dz = z * p.h - src.z;
        if (dx * dx + dy * dy + dz * dz < src.r * src.r) val = sv[axis];
      }
      this.vel2[f] = Math.min(p.vMax, Math.max(-p.vMax, val));
    }
  }

  cellVel(vel: Float32Array, i: number, j: number, k: number): [number, number, number] {
    return [
      0.5 * (vel[this.uIdx(i, j, k)] + vel[this.uIdx(i + 1, j, k)]),
      0.5 * (vel[this.vIdx(i, j, k)] + vel[this.vIdx(i, j + 1, k)]),
      0.5 * (vel[this.wIdx(i, j, k)] + vel[this.wIdx(i, j, k + 1)]),
    ];
  }

  curl() {
    const { nx, ny, nz } = this.g;
    const h = this.p.h;
    const cv = (i: number, j: number, k: number) =>
      this.cellVel(this.vel2, Math.min(nx - 1, Math.max(0, i)), Math.min(ny - 1, Math.max(0, j)), Math.min(nz - 1, Math.max(0, k)));
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const xp = cv(i + 1, j, k), xm = cv(i - 1, j, k);
          const yp = cv(i, j + 1, k), ym = cv(i, j - 1, k);
          const zp = cv(i, j, k + 1), zm = cv(i, j, k - 1);
          const inv = 1 / (2 * h);
          const wx = (yp[2] - ym[2] - (zp[1] - zm[1])) * inv;
          const wy = (zp[0] - zm[0] - (xp[2] - xm[2])) * inv;
          const wz = (xp[1] - xm[1] - (yp[0] - ym[0])) * inv;
          const c = this.idx(i, j, k) * 4;
          this.omega[c] = wx;
          this.omega[c + 1] = wy;
          this.omega[c + 2] = wz;
          this.omega[c + 3] = Math.hypot(wx, wy, wz);
        }
  }

  /** Vorticity-confinement force at a cell centre [m/s^2]. */
  confForce(i: number, j: number, k: number): [number, number, number] {
    const { nx, ny, nz } = this.g;
    const h = this.p.h;
    const mag = (a: number, b: number, c: number) =>
      this.omega[this.idx(Math.min(nx - 1, Math.max(0, a)), Math.min(ny - 1, Math.max(0, b)), Math.min(nz - 1, Math.max(0, c))) * 4 + 3];
    const gx = (mag(i + 1, j, k) - mag(i - 1, j, k)) / (2 * h);
    const gy = (mag(i, j + 1, k) - mag(i, j - 1, k)) / (2 * h);
    const gz = (mag(i, j, k + 1) - mag(i, j, k - 1)) / (2 * h);
    const len = Math.hypot(gx, gy, gz) + 1e-12;
    const Nx = gx / len, Ny = gy / len, Nz = gz / len;
    const o = this.idx(i, j, k) * 4;
    const wx = this.omega[o], wy = this.omega[o + 1], wz = this.omega[o + 2];
    const e = this.p.vortEps * h;
    return [e * (Ny * wz - Nz * wy), e * (Nz * wx - Nx * wz), e * (Nx * wy - Ny * wx)];
  }

  confine() {
    const dt = this.p.dt;
    for (let f = 0; f < this.g.nFaces; f++) {
      const [axis, i, j, k] = this.faceCoord(f);
      if (!this.faceOpen(axis, i, j, k)) continue;
      let a: [number, number, number], b: [number, number, number];
      if (axis === 0) { a = this.confForce(i - 1, j, k); b = this.confForce(i, j, k); }
      else if (axis === 1) { a = this.confForce(i, j - 1, k); b = this.confForce(i, j, k); }
      else { a = this.confForce(i, j, k - 1); b = this.confForce(i, j, k); }
      this.vel2[f] += dt * 0.5 * (a[axis] + b[axis]);
    }
  }

  // ---------------------------------------------------------------- pressure projection
  divergence(vel: Float32Array) {
    const { nx, ny, nz } = this.g;
    const h = this.p.h;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          const c = this.idx(i, j, k);
          this.div[c] =
            (vel[this.uIdx(i + 1, j, k)] - vel[this.uIdx(i, j, k)] +
              vel[this.vIdx(i, j + 1, k)] - vel[this.vIdx(i, j, k)] +
              vel[this.wIdx(i, j, k + 1)] - vel[this.wIdx(i, j, k)]) / h;
        }
  }

  /** One red (color 0) or black (color 1) SOR sweep. */
  pressureSweep(color: number) {
    const { nx, ny, nz } = this.g;
    const h2 = this.p.h * this.p.h;
    const w = this.p.sorOmega;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) {
          if (((i + j + k) & 1) !== color) continue;
          const c = this.idx(i, j, k);
          if (this.solidAt(i, j, k)) {
            this.phi[c] = 0;
            continue;
          }
          let sum = 0, n = 0;
          const nbs: [number, number, number][] = [
            [i - 1, j, k], [i + 1, j, k], [i, j - 1, k], [i, j + 1, k], [i, j, k - 1], [i, j, k + 1],
          ];
          for (const [a, b, d] of nbs) {
            if (this.solidAt(a, b, d)) continue;
            sum += this.phi[this.idx(a, b, d)];
            n++;
          }
          if (n === 0) {
            this.phi[c] = 0;
            continue;
          }
          const gs = (sum - h2 * this.div[c]) / n;
          this.phi[c] = (1 - w) * this.phi[c] + w * gs;
        }
  }

  pressure() {
    for (let it = 0; it < this.p.pressureIters; it++) {
      this.pressureSweep(0);
      this.pressureSweep(1);
    }
  }

  project() {
    const p = this.p;
    for (let f = 0; f < this.g.nFaces; f++) {
      const [axis, i, j, k] = this.faceCoord(f);
      if (!this.faceOpen(axis, i, j, k)) {
        this.vel[f] = 0;
        continue;
      }
      const b = this.phi[this.idx(i, j, k)];
      const a = axis === 0 ? this.phi[this.idx(i - 1, j, k)] : axis === 1 ? this.phi[this.idx(i, j - 1, k)] : this.phi[this.idx(i, j, k - 1)];
      const v = this.vel2[f] - (b - a) / p.h;
      this.vel[f] = Math.min(p.vMax, Math.max(-p.vMax, v));
    }
  }

  // ---------------------------------------------------------------- diagnostics
  cellMass() {
    return this.p.rho0 * this.p.h ** 3;
  }
  /** is cell index c a water cell (below the surface)? */
  isWaterCell(c: number) {
    return Math.floor(c / this.g.nx) % this.g.ny < this.rows;
  }
  /** number of water cells */
  waterCells() {
    return this.g.nx * this.rows * this.g.nz;
  }
  /** total mass of water (all phases) [kg]: constant, the grid is fixed and incompressible */
  waterMass() {
    return this.waterCells() * this.cellMass();
  }
  /** total enthalpy (energy) of the water [J], relative to ice at 0 °C. Air cells hold no water. */
  totalEnergy(H = this.H) {
    let s = 0;
    for (let c = 0; c < H.length; c++) if (this.isWaterCell(c)) s += H[c];
    return s * this.cellMass();
  }
  /** total energy injected (or removed, if negative) by heat sources [J] */
  injectedEnergy() {
    let s = 0;
    for (let c = 0; c < this.srcAcc.length; c++) s += this.srcAcc[c];
    return s * this.cellMass();
  }
  /** masses [kg] of ice, liquid and vapour (water cells only) */
  phaseMasses(H = this.H) {
    let ice = 0, vap = 0, n = 0;
    for (let c = 0; c < H.length; c++) {
      if (!this.isWaterCell(c)) continue;
      ice += 1 - Math.min(1, Math.max(0, H[c] / L_F));
      vap += vaporQuality(H[c]);
      n++;
    }
    const m = this.cellMass();
    return { ice: ice * m, liquid: (n - ice - vap) * m, vapor: vap * m, total: n * m };
  }
  /** max |div u| over fluid cells [1/s] for the given velocity */
  maxDivergence(vel = this.vel) {
    this.divergence(vel);
    let m = 0;
    const { nx, ny, nz } = this.g;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let i = 0; i < nx; i++) if (!this.solidAt(i, j, k)) m = Math.max(m, Math.abs(this.div[this.idx(i, j, k)]));
    return m;
  }
  maxSpeed(vel = this.vel) {
    let m = 0;
    for (let f = 0; f < vel.length; f++) m = Math.max(m, Math.abs(vel[f]));
    return m;
  }
  temperatureAt(i: number, j: number, k: number) {
    return temperature(this.H[this.idx(i, j, k)]);
  }
}
