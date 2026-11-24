/**
 * Browser-side test harness. Loaded by tests/gpu/*.test.ts through a Vite dev server in
 * headless Chromium (WebGPU via SwiftShader). Exposes window.tayutau with test entry points.
 */
import { ComputeShader, StorageBuffer, Constants, type WebGPUEngine } from "@babylonjs/core";
import { checkWebGPU, createEngine } from "../src/gpu/engine";

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

window.tayutau = {
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
