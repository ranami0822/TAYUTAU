import { ArcRotateCamera, Color4, HemisphericLight, MeshBuilder, Scene, StandardMaterial, Color3, Vector3, type WebGPUEngine } from "@babylonjs/core";

export async function startApp(engine: WebGPUEngine, canvas: HTMLCanvasElement) {
  const scene = new Scene(engine);
  scene.clearColor = new Color4(0.04, 0.06, 0.08, 1);
  const cam = new ArcRotateCamera("cam", -Math.PI / 3, Math.PI / 3, 3, Vector3.Zero(), scene);
  cam.attachControl(canvas, true);
  new HemisphericLight("light", new Vector3(0.3, 1, 0.2), scene);
  const tank = MeshBuilder.CreateBox("tank", { size: 1 }, scene);
  const mat = new StandardMaterial("glass", scene);
  mat.diffuseColor = new Color3(0.4, 0.6, 0.9);
  mat.alpha = 0.3;
  tank.material = mat;
  engine.runRenderLoop(() => scene.render());
  window.addEventListener("resize", () => engine.resize());
}
