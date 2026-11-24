import { checkWebGPU, createEngine } from "./gpu/engine";

function showUnsupported(reason: string) {
  const box = document.getElementById("unsupported")!;
  document.getElementById("unsupported-reason")!.textContent = reason;
  box.hidden = false;
  document.getElementById("view")!.style.display = "none";
  document.body.dataset.state = "unsupported";
}

async function main() {
  const check = await checkWebGPU();
  if (!check.ok) return showUnsupported(check.reason);
  const canvas = document.getElementById("view") as HTMLCanvasElement;
  let engine;
  try {
    engine = await createEngine(canvas);
  } catch (e) {
    return showUnsupported(String(e instanceof Error ? e.message : e));
  }
  const { startApp } = await import("./app");
  await startApp(engine, canvas);
  document.body.dataset.state = "running";
}

main().catch((e) => {
  console.error(e);
  showUnsupported(`起動に失敗しました: ${String(e)}`);
});
