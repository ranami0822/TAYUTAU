import {
  ArcRotateCamera,
  Color3,
  Color4,
  Matrix,
  MeshBuilder,
  Scene,
  Vector3,
  type WebGPUEngine,
} from "@babylonjs/core";
import { GpuSolver, type SimStats } from "./gpu/gpuSolver";
import { defaultParams, MAX_SOURCES, type HeatSource, type SimParams } from "./sim/params";
import { VolumeRenderer } from "./render/volume";
import { BubbleRenderer } from "./render/bubbles";
import { Interaction, type Tool } from "./ui/interaction";
import { buildUI } from "./ui/controls";
import { enthalpyFromT } from "./sim/water";

export interface AppState {
  sim: GpuSolver;
  params: SimParams;
  sources: HeatSource[];
  selected: number;
  tool: Tool;
  paused: boolean;
  stepsPerFrame: number;
  viewMode: number;
  stats: SimStats | null;
  dirty: boolean;
  /** step counter, used by tests and the HUD */
  steps: number;
  /** energy at the last reset [J], to show the energy balance */
  stats0?: number;
}

export interface App {
  state: AppState;
  addSource(s: HeatSource): number;
  removeSource(i: number): void;
  updateSource(i: number, s: Partial<HeatSource>): void;
  reset(preset?: Preset): void;
  volume: VolumeRenderer;
  scene: Scene;
  camera: ArcRotateCamera;
  /** number of live bubble particles (reads the particle buffer back; for tests / debugging) */
  countBubbles(): Promise<number>;
  /** CSS-pixel position of a domain point on the canvas (for tests and overlays) */
  screenOf(x: number, y: number, z: number): { x: number; y: number };
}

export type Preset = "water" | "ice-block" | "cold-water";

declare global {
  interface Window {
    tayutauApp?: App;
  }
}

export async function startApp(engine: WebGPUEngine, canvas: HTMLCanvasElement): Promise<App> {
  const url = new URLSearchParams(location.search);
  // phones / tablets get a coarser default grid (the whole step runs on their GPU every frame)
  const coarse = typeof matchMedia === "function" && matchMedia("(pointer: coarse)").matches;
  const n = Math.min(96, Math.max(16, Number(url.get("n")) || (coarse ? 32 : 48)));
  const params = defaultParams(n);

  const scene = new Scene(engine);
  scene.clearColor = new Color4(0.035, 0.05, 0.07, 1);
  const camera = new ArcRotateCamera("cam", -Math.PI * 0.62, Math.PI * 0.36, 2.3, Vector3.Zero(), scene);
  camera.lowerRadiusLimit = 0.7;
  camera.upperRadiusLimit = 6;
  camera.wheelDeltaPercentage = 0.01;
  camera.pinchDeltaPercentage = 0.004;
  camera.minZ = 0.01;
  camera.attachControl(canvas, true);

  const sim = new GpuSolver(engine, params);
  const volume = new VolumeRenderer(scene, params, sim.H, sim.P);
  const bubbles = new BubbleRenderer(scene, sim, volume);

  // tank outline and a floor for orientation
    const a = volume.boxMin, b = volume.boxMax;
  const V = (x: number, y: number, z: number) => new Vector3(x, y, z);
  const corners = [V(a.x, a.y, a.z), V(b.x, a.y, a.z), V(b.x, a.y, b.z), V(a.x, a.y, b.z), V(a.x, b.y, a.z), V(b.x, b.y, a.z), V(b.x, b.y, b.z), V(a.x, b.y, b.z)];
  const E = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  const edges = MeshBuilder.CreateLineSystem("tankEdges", { lines: E.map(([p, q]) => [corners[p], corners[q]]) }, scene);
  edges.color = new Color3(0.55, 0.7, 0.85);
  edges.alpha = 0.6;
  edges.isPickable = false;
  const floor = MeshBuilder.CreateDisc("floor", { radius: 2.2, tessellation: 64 }, scene);
  floor.rotation.x = Math.PI / 2;
  floor.position.y = volume.boxMin.y - 0.002;
  floor.isPickable = false;
  const { StandardMaterial } = await import("@babylonjs/core");
  const fm = new StandardMaterial("floorMat", scene);
  fm.diffuseColor = new Color3(0.08, 0.1, 0.13);
  fm.specularColor = new Color3(0, 0, 0);
  fm.emissiveColor = new Color3(0.05, 0.065, 0.085);
  floor.material = fm;

  const L = { x: params.nx * params.h, y: params.ny * params.h, z: params.nz * params.h };
  const state: AppState = {
    sim,
    params,
    sources: [{ x: L.x * 0.5, y: L.y * 0.12, z: L.z * 0.5, r: L.x * 0.1, T: 400 }],
    selected: 0,
    tool: "hot",
    paused: false,
    stepsPerFrame: 2,
    viewMode: 0,
    stats: null,
    dirty: true,
    steps: 0,
  };

  let statsGen = 0;
  const app: App = {
    state,
    volume,
    scene,
    camera,
    addSource(s) {
      if (state.sources.length >= MAX_SOURCES) return -1;
      state.sources.push(s);
      state.dirty = true;
      return state.sources.length - 1;
    },
    removeSource(i) {
      if (i < 0 || i >= state.sources.length) return;
      state.sources.splice(i, 1);
      if (state.selected === i) state.selected = -1;
      else if (state.selected > i) state.selected--;
      state.dirty = true;
    },
    updateSource(i, s) {
      if (i < 0 || i >= state.sources.length) return;
      Object.assign(state.sources[i], s);
      state.dirty = true;
    },
    async countBubbles() {
      const v = (await bubbles.parts.read()) as Uint8Array;
      const a = new Float32Array(v.buffer, v.byteOffset, bubbles.count * 8);
      let n = 0;
      for (let i = 0; i < bubbles.count; i++) if (a[i * 8 + 3] > 0.5) n++;
      return n;
    },
    screenOf(x, y, z) {
      const eng = scene.getEngine();
      const w = volume.toWorld(x, y, z);
      const vp = camera.viewport.toGlobal(eng.getRenderWidth(), eng.getRenderHeight());
      const sp = Vector3.Project(w, Matrix.Identity(), scene.getTransformMatrix(), vp);
      const r = canvas.getBoundingClientRect();
      return { x: r.left + (sp.x / eng.getRenderWidth()) * r.width, y: r.top + (sp.y / eng.getRenderHeight()) * r.height };
    },
    reset(preset: Preset = "water") {
      sim.reset(preset === "cold-water" ? 2 : 20);
      if (preset === "ice-block") {
        // a block of ice at -15 °C in 20 °C water
        const H = new Float32Array(sim.g.nCells).fill(enthalpyFromT(20));
        const { nx, ny, nz } = sim.g;
        for (let k = 0; k < nz; k++)
          for (let j = 0; j < ny; j++)
            for (let i = 0; i < nx; i++) {
              const inBlock = Math.abs(i - nx / 2) < nx * 0.18 && Math.abs(k - nz / 2) < nz * 0.18 && j > ny * 0.45 && j < ny * 0.8;
              if (inBlock) H[i + nx * (j + ny * k)] = enthalpyFromT(-15);
            }
        sim.H.update(H);
      }
      bubbles.reset();
      state.steps = 0;
      state.stats = null;
      state.stats0 = undefined;
      statsGen++;
    },
  };
  window.tayutauApp = app;

  const ui = buildUI(app);
  new Interaction(app, canvas, ui.refresh);

  await sim.ready();
  await bubbles.ready();

  let statsBusy = false;
  let lastStats = 0;
  scene.onBeforeRenderObservable.add(() => {
    if (state.dirty) {
      sim.setParams({ ...params, sources: state.sources });
      state.dirty = false;
    }
    volume.viewMode = state.viewMode;
    volume.selected = state.selected;
    if (!state.paused) {
      for (let s = 0; s < state.stepsPerFrame; s++) sim.step();
      state.steps += state.stepsPerFrame;
      bubbles.step(state.stepsPerFrame);
    }
    const now = performance.now();
    if (!statsBusy && now - lastStats > 400) {
      statsBusy = true;
      lastStats = now;
      const gen = statsGen;
      sim
        .stats()
        .then((s) => {
          if (gen !== statsGen) return; // a reset happened while this was in flight
          if (state.stats0 === undefined) state.stats0 = s.energy - s.injected;
          state.stats = s;
          ui.showStats(s);
        })
        .finally(() => (statsBusy = false));
    }
  });

  engine.onContextLostObservable.add(() => {
    // simulation state lives only on the GPU; it cannot be restored after a device loss
    engine.stopRenderLoop();
    const box = document.getElementById("unsupported");
    if (box) {
      document.getElementById("unsupported-reason")!.textContent = "GPU がリセットされたため、シミュレーションを続けられません。ページを再読み込みしてください。";
      box.querySelector("h1")!.textContent = "GPU が停止しました";
      box.hidden = false;
      document.body.dataset.state = "lost";
    }
  });
  engine.runRenderLoop(() => scene.render());
  window.addEventListener("resize", () => engine.resize());
  return app;
}
