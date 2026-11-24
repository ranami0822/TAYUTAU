/**
 * Bubble display: GPU particles advanced by a compute kernel (kernels.ts "bubbles") and drawn as
 * camera-facing instanced quads that read the particle buffer directly in the vertex shader.
 * Particles are visual only; the vapour itself is part of the conserved enthalpy field.
 */
import { ComputeShader, Constants, MeshBuilder, ShaderLanguage, ShaderMaterial, StorageBuffer, Vector3, type Mesh, type Scene } from "@babylonjs/core";
import { KERNELS, kernelSource, bindingsMapping, dispatchSize } from "../gpu/kernels";
import type { GpuSolver } from "../gpu/gpuSolver";
import type { VolumeRenderer } from "./volume";

const VERT = /* wgsl */ `
attribute position : vec3f;
uniform viewProjection : mat4x4f;
uniform camRight : vec3f;
uniform camUp : vec3f;
uniform boxMin : vec3f;
uniform metersToWorld : f32;
var<storage, read> parts : array<f32>;
varying vUV : vec2f;
varying vAlive : f32;
@vertex
fn main(input : VertexInputs) -> FragmentInputs {
  let o = i32(vertexInputs.instanceIndex) * 8;
  let alive = parts[o + 3];
  let size = parts[o + 4] * uniforms.metersToWorld;
  let c = uniforms.boxMin + vec3f(parts[o], parts[o + 1], parts[o + 2]) * uniforms.metersToWorld;
  let corner = vertexInputs.position.xy * 2.0;
  let wp = c + (uniforms.camRight * corner.x + uniforms.camUp * corner.y) * size;
  var pos = uniforms.viewProjection * vec4f(wp, 1.0);
  if (alive < 0.5) { pos = vec4f(0.0, 0.0, -2.0, 1.0); }
  vertexOutputs.position = pos;
  vertexOutputs.vUV = corner;
  vertexOutputs.vAlive = alive;
}
`;

const FRAG = /* wgsl */ `
varying vUV : vec2f;
varying vAlive : f32;
@fragment
fn main(input : FragmentInputs) -> FragmentOutputs {
  let r = length(fragmentInputs.vUV);
  if (r > 1.0 || fragmentInputs.vAlive < 0.5) { discard; }
  // thin bright rim, faint body, small highlight: reads as a gas bubble in liquid
  let rim = smoothstep(0.65, 0.95, r) * (1.0 - smoothstep(0.95, 1.0, r));
  let hl = 1.0 - smoothstep(0.0, 0.28, length(fragmentInputs.vUV - vec2f(-0.35, 0.35)));
  let a = clamp(0.12 + 0.75 * rim + 0.8 * hl, 0.0, 1.0);
  let c = vec3f(0.92, 0.97, 1.0);
  fragmentOutputs.color = vec4f(c * a, a);
}
`;

export class BubbleRenderer {
  readonly count: number;
  readonly parts: StorageBuffer;
  readonly BP: StorageBuffer;
  readonly mesh: Mesh;
  private cs: ComputeShader;
  private frame = 0;
  private sim: GpuSolver;

  constructor(scene: Scene, sim: GpuSolver, volume: VolumeRenderer, count = 6144) {
    this.sim = sim;
    this.count = count;
    const eng = sim.engine;
    this.parts = new StorageBuffer(eng, count * 8 * 4, Constants.BUFFER_CREATIONFLAG_READWRITE, "bubbles");
    this.BP = new StorageBuffer(eng, 4 * 4, Constants.BUFFER_CREATIONFLAG_READWRITE, "bubbleParams");
    this.reset();
    const k = KERNELS.bubbles;
    this.cs = new ComputeShader("bubbles", eng, { computeSource: kernelSource(k) }, { bindingsMapping: bindingsMapping(k) });
    this.cs.setStorageBuffer("P", sim.P);
    this.cs.setStorageBuffer("Hs", sim.H);
    this.cs.setStorageBuffer("vin", sim.vel);
    this.cs.setStorageBuffer("BP", this.BP);
    this.cs.setStorageBuffer("parts", this.parts);

    this.mesh = MeshBuilder.CreatePlane("bubbleQuad", { size: 1 }, scene);
    this.mesh.isPickable = false;
    this.mesh.alwaysSelectAsActiveMesh = true;
    this.mesh.forcedInstanceCount = count;
    const m = new ShaderMaterial(
      "bubbleMat",
      scene,
      { vertexSource: VERT, fragmentSource: FRAG },
      {
        attributes: ["position"],
        uniforms: ["viewProjection", "camRight", "camUp", "boxMin", "metersToWorld"],
        storageBuffers: ["parts"],
        shaderLanguage: ShaderLanguage.WGSL,
        needAlphaBlending: true,
      },
    );
    m.backFaceCulling = false;
    m.alphaMode = Constants.ALPHA_PREMULTIPLIED;
    m.disableDepthWrite = true;
    m.setStorageBuffer("parts", this.parts);
    m.setVector3("boxMin", volume.boxMin);
    m.setFloat("metersToWorld", volume.metersToWorld);
    this.mesh.material = m;
    this.mesh.alphaIndex = 20;
    scene.onBeforeRenderObservable.add(() => {
      const cam = scene.activeCamera;
      if (!cam) return;
      const v = cam.getViewMatrix();
      const mm = v.m;
      // rows of the view matrix are the camera axes in world space (left-handed Babylon convention)
      m.setVector3("camRight", new Vector3(mm[0], mm[4], mm[8]));
      m.setVector3("camUp", new Vector3(mm[1], mm[5], mm[9]));
    });
  }

  async ready() {
    while (!this.cs.isReady()) await new Promise((r) => setTimeout(r, 16));
    this.cs.fastMode = true;
  }

  reset() {
    this.parts.update(new Float32Array(this.count * 8));
  }

  /** advance after `steps` simulation steps have been dispatched this frame */
  step(steps: number) {
    this.frame++;
    const p = this.sim.p;
    this.BP.update(new Float32Array([this.frame, steps * p.dt, 3, 0.35 * p.h * Math.max(p.nx, p.ny, p.nz) / 48 * 1.6]));
    const [x, y] = dispatchSize(this.count);
    this.cs.dispatch(x, y, 1);
  }
}
