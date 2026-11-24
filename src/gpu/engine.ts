import { WebGPUEngine } from "@babylonjs/core";

export type WebGPUCheck = { ok: true } | { ok: false; reason: string };

/** Decide before creating any scene whether WebGPU (and compute shaders) can be used. No WebGL fallback. */
export async function checkWebGPU(): Promise<WebGPUCheck> {
  if (typeof navigator === "undefined" || !("gpu" in navigator) || !navigator.gpu) {
    return { ok: false, reason: "このブラウザは WebGPU に対応していません（navigator.gpu がありません）。" };
  }
  try {
    const supported = await WebGPUEngine.IsSupportedAsync;
    if (!supported) return { ok: false, reason: "WebGPU のアダプタを取得できませんでした（GPU が無効、または非対応）。" };
  } catch (e) {
    return { ok: false, reason: `WebGPU の初期化に失敗しました: ${String(e)}` };
  }
  return { ok: true };
}

export async function createEngine(canvas: HTMLCanvasElement): Promise<WebGPUEngine> {
  const engine = new WebGPUEngine(canvas, { antialias: true, adaptToDeviceRatio: true });
  await engine.initAsync();
  if (!engine.getCaps().supportComputeShaders) {
    engine.dispose();
    throw new Error("この環境の WebGPU ではコンピュートシェーダーが使えません。");
  }
  return engine;
}
