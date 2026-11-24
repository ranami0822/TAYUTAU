/**
 * GPU twin of CpuSolver: owns the storage buffers and dispatches the WGSL kernels.
 * Field data stays on the GPU; the renderer binds `H` and `vel` directly.
 */
import { ComputeShader, Constants, StorageBuffer, type WebGPUEngine } from "@babylonjs/core";
import { KERNELS, kernelSource, bindingsMapping, dispatchSize, type KernelDef } from "./kernels";
import { packParams, PARAM_WORDS, type SimParams } from "../sim/params";
import { makeGrid, type Grid } from "../sim/cpuSolver";
import { enthalpyFromT } from "../sim/water";

const RW = Constants.BUFFER_CREATIONFLAG_READWRITE;

export class GpuSolver {
  readonly engine: WebGPUEngine;
  readonly g: Grid;
  p: SimParams;
  readonly P: StorageBuffer;
  readonly H: StorageBuffer;
  readonly Htmp: StorageBuffer;
  readonly vel: StorageBuffer;
  readonly vel2: StorageBuffer;
  readonly phi: StorageBuffer;
  readonly div: StorageBuffer;
  readonly omega: StorageBuffer;
  readonly srcAcc: StorageBuffer;
  private shaders: Record<string, ComputeShader> = {};
  time = 0;

  constructor(engine: WebGPUEngine, p: SimParams, initialT = 20) {
    this.engine = engine;
    this.p = p;
    this.g = makeGrid(p.nx, p.ny, p.nz);
    const nC = this.g.nCells;
    const nF = this.g.nFaces;
    const buf = (n: number, label: string, extraFlags = 0) => new StorageBuffer(engine, n * 4, RW | extraFlags, label);
    this.P = buf(PARAM_WORDS, "params");
    this.H = buf(nC, "H");
    this.Htmp = buf(nC, "Htmp");
    this.vel = buf(nF, "vel");
    this.vel2 = buf(nF, "vel2");
    this.phi = buf(nC, "phi");
    this.div = buf(nC, "div");
    this.omega = buf(nC * 4, "omega");
    this.srcAcc = buf(nC, "srcAcc");
    this.setParams(p);
    this.H.update(new Float32Array(nC).fill(enthalpyFromT(initialT)));
    this.reset(initialT);

    const B: Record<string, StorageBuffer> = {
      Hs: this.H,
      Hout: this.Htmp,
      srcAcc: this.srcAcc,
      omega: this.omega,
      div: this.div,
      phi: this.phi,
    };
    // explicit per-kernel binding tables (ping-pong is expressed by distinct kernels, never by rebinding)
    this.make(KERNELS.thermal, B);
    this.make(KERNELS.advectScalar, { q: this.Htmp, vel: this.vel, qout: this.H });
    this.make(KERNELS.advectVel, { vin: this.vel, Hs: this.H, vout: this.vel2 });
    this.make(KERNELS.curl, { vin: this.vel2, omega: this.omega });
    this.make(KERNELS.confine, { omega: this.omega, Hs: this.H, vout: this.vel2 });
    this.make(KERNELS.divergence, { vin: this.vel2, div: this.div });
    this.make(KERNELS.pressureRed, B);
    this.make(KERNELS.pressureBlack, B);
    this.make(KERNELS.project, { vin: this.vel2, phi: this.phi, Hs: this.H, vout: this.vel });
  }

  /** Create an extra instance of a kernel with a custom binding table (used by tests). */
  makeKernel(k: KernelDef, bufs: Record<string, StorageBuffer>, key = k.name) {
    return this.make(k, bufs, key);
  }

  private make(k: KernelDef, bufs: Record<string, StorageBuffer>, key = k.name) {
    const cs = new ComputeShader(k.name, this.engine, { computeSource: kernelSource(k) }, { bindingsMapping: bindingsMapping(k) });
    cs.setStorageBuffer("P", this.P);
    for (const b of k.bindings) {
      const sb = bufs[b.name];
      if (!sb) throw new Error(`kernel ${k.name}: no buffer for binding ${b.name}`);
      cs.setStorageBuffer(b.name, sb);
    }
    this.shaders[key] = cs;
    return cs;
  }

  /** Resolves once every kernel has compiled. */
  async ready(timeoutMs = 60000) {
    const t0 = performance.now();
    while (!Object.values(this.shaders).every((s) => s.isReady())) {
      if (performance.now() - t0 > timeoutMs) throw new Error("compute shaders did not compile in time");
      await new Promise((r) => setTimeout(r, 16));
    }
    for (const s of Object.values(this.shaders)) s.fastMode = true;
  }

  /** Must be called at most once per frame: the write is applied before the frame's dispatches. */
  setParams(p: SimParams) {
    this.p = p;
    this.P.update(packParams(p));
  }

  reset(initialT = 20) {
    const nC = this.g.nCells;
    this.H.update(new Float32Array(nC).fill(enthalpyFromT(initialT)));
    for (const b of [this.Htmp, this.phi, this.div, this.srcAcc]) b.update(new Float32Array(nC));
    this.omega.update(new Float32Array(nC * 4));
    this.vel.update(new Float32Array(this.g.nFaces));
    this.vel2.update(new Float32Array(this.g.nFaces));
    this.time = 0;
  }

  dispatch(name: string, n: number) {
    const [x, y] = dispatchSize(n);
    if (!this.shaders[name].dispatch(x, y, 1)) throw new Error(`dispatch failed: ${name}`);
  }

  step() {
    const { nCells, nFaces } = this.g;
    for (let s = 0; s < this.p.subcycles; s++) {
      this.dispatch("thermal", nCells);
      this.dispatch("advectScalar", nCells);
    }
    this.dispatch("advectVel", nFaces);
    if (this.p.vortEps > 0) {
      this.dispatch("curl", nCells);
      this.dispatch("confine", nFaces);
    }
    this.dispatch("divergence", nCells);
    for (let it = 0; it < this.p.pressureIters; it++) {
      this.dispatch("pressureRed", nCells);
      this.dispatch("pressureBlack", nCells);
    }
    this.dispatch("project", nFaces);
    this.time += this.p.dt;
  }

  async read(b: StorageBuffer, n: number): Promise<Float32Array> {
    const v = (await b.read(0, n * 4)) as Uint8Array;
    return new Float32Array(v.buffer.slice(v.byteOffset, v.byteOffset + n * 4));
  }

  dispose() {
    for (const b of [this.P, this.H, this.Htmp, this.vel, this.vel2, this.phi, this.div, this.omega, this.srcAcc]) b.dispose();
  }
}
