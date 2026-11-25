/**
 * Browser-side test harness. Loaded by tests/gpu/*.test.ts through a Vite dev server in
 * headless Chromium (WebGPU via SwiftShader). Exposes window.tayutau with test entry points.
 */
import { ComputeShader, StorageBuffer, Constants, type WebGPUEngine } from "@babylonjs/core";
import { checkWebGPU, createEngine } from "../src/gpu/engine";
import { GpuSolver } from "../src/gpu/gpuSolver";
import type { SimParams } from "../src/sim/params";

export interface RunSimArgs {
  params: SimParams;
  H?: number[];
  vel?: number[];
  eta?: number[];
  steps: number;
  /** read back after these step counts (default: only at the end) */
  readAt?: number[];
}

declare global {
  interface Window {
    tayutau: Record<string, (...args: any[]) => Promise<unknown>>;
    tayutauReady: Promise<string>;
  }
}

let enginePromise: Promise<WebGPUEngine> | null = null;
export function engine() {
  if (!enginePromise)
    enginePromise = createEngine(document.getElementById("c") as HTMLCanvasElement).then((e) => {
      // Babylon submits the recorded compute passes at the end of each frame.
      e.runRenderLoop(() => {});
      return e;
    });
  return enginePromise;
}

async function runSim(args: RunSimArgs) {
  const eng = await engine();
  const sim = new GpuSolver(eng, args.params);
  if (args.H) sim.H.update(new Float32Array(args.H));
  if (args.vel) sim.vel.update(new Float32Array(args.vel));
  if (args.eta) sim.eta.update(new Float32Array(args.eta));
  await sim.ready();
  const { nCells, nFaces } = sim.g;
  const snaps: { step: number; H: number[]; vel: number[]; srcAcc: number[]; eta: number[] }[] = [];
  const readAt = args.readAt ?? [args.steps];
  const t0 = performance.now();
  for (let n = 1; n <= args.steps; n++) {
    sim.step();
    if (readAt.includes(n)) {
      snaps.push({
        step: n,
        H: Array.from(await sim.read(sim.H, nCells)),
        vel: Array.from(await sim.read(sim.vel, nFaces)),
        srcAcc: Array.from(await sim.read(sim.srcAcc, nCells)),
        eta: Array.from(await sim.read(sim.eta, sim.g.nx * sim.g.nz)),
      });
    }
  }
  const ms = performance.now() - t0;
  sim.dispose();
  return { snaps, ms };
}

window.tayutau = {
  runSim,
  /** stage 0: a compute shader writes out[i] = in[i] * 2 + 1 and the result is read back */
  async smoke(n: number) {
    const eng = await engine();
    const src = new Float32Array(n).map((_, i) => i);
    const a = new StorageBuffer(eng, n * 4, Constants.BUFFER_CREATIONFLAG_READWRITE);
    const b = new StorageBuffer(eng, n * 4, Constants.BUFFER_CREATIONFLAG_READWRITE);
    a.update(src);
    const cs = new ComputeShader(
      "smoke",
      eng,
      {
        computeSource: /* wgsl */ `
          @group(0) @binding(0) var<storage, read> inp : array<f32>;
          @group(0) @binding(1) var<storage, read_write> outp : array<f32>;
          @compute @workgroup_size(64)
          fn main(@builtin(global_invocation_id) id : vec3<u32>) {
            if (id.x >= arrayLength(&inp)) { return; }
            outp[id.x] = inp[id.x] * 2.0 + 1.0;
          }`,
      },
      { bindingsMapping: { inp: { group: 0, binding: 0 }, outp: { group: 0, binding: 1 } } },
    );
    cs.setStorageBuffer("inp", a);
    cs.setStorageBuffer("outp", b);
    await cs.dispatchWhenReady(Math.ceil(n / 64));
    const out = (await b.read()) as Uint8Array;
    const f = new Float32Array(out.buffer, out.byteOffset, n);
    a.dispose();
    b.dispose();
    return Array.from(f);
  },
};

window.tayutauReady = checkWebGPU().then((r) => (r.ok ? "ok" : r.reason));
