/**
 * Volume renderer: ray-marches the enthalpy field H straight from the simulation's storage
 * buffer (no CPU round trip). Water is tinted by temperature, ice is shaded as a lit surface
 * (iso-surface of ice fraction 0.5), vapour is a white fog, and heat sources are drawn as
 * analytic spheres inside the march so they are correctly occluded.
 */
import { Constants, MeshBuilder, ShaderLanguage, ShaderMaterial, Vector3, type Mesh, type Scene, type StorageBuffer } from "@babylonjs/core";
import * as W from "../sim/water";
import type { SimParams } from "../sim/params";

const f = (x: number) => {
  const s = x.toString();
  return s.includes(".") || s.includes("e") ? s : s + ".0";
};

const VERTEX = /* wgsl */ `
attribute position : vec3f;
uniform world : mat4x4f;
uniform worldViewProjection : mat4x4f;
varying vWorld : vec3f;
@vertex
fn main(input : VertexInputs) -> FragmentInputs {
  let wp = uniforms.world * vec4f(vertexInputs.position, 1.0);
  vertexOutputs.position = uniforms.worldViewProjection * vec4f(vertexInputs.position, 1.0);
  vertexOutputs.vWorld = wp.xyz;
}
`;

const FRAGMENT = /* wgsl */ `
uniform camPos : vec3f;
uniform boxMin : vec3f;
uniform boxMax : vec3f;
uniform dims : vec3f;
uniform metersToWorld : f32;
uniform viewMode : f32;
uniform selected : f32;
uniform lightDir : vec3f;
varying vWorld : vec3f;
var<storage, read> Hs : array<f32>;
var<storage, read> P : array<f32>;
var<storage, read> eta : array<f32>;

const C_WATER : f32 = ${f(W.C_WATER)};
const C_ICE : f32 = ${f(W.C_ICE)};
const C_VAPOR : f32 = ${f(W.C_VAPOR)};
const L_F : f32 = ${f(W.L_F)};
const L_V : f32 = ${f(W.L_V)};
const H_B : f32 = ${f(W.H_B)};
const H_VSAT : f32 = ${f(W.H_VSAT)};

fn temperature(H : f32) -> f32 {
  if (H < 0.0) { return H / C_ICE; }
  if (H < L_F) { return 0.0; }
  if (H < H_B) { return (H - L_F) / C_WATER; }
  if (H < H_VSAT) { return 100.0; }
  return 100.0 + (H - H_VSAT) / C_VAPOR;
}
fn iceFraction(H : f32) -> f32 { return 1.0 - clamp(H / L_F, 0.0, 1.0); }
fn vaporQuality(H : f32) -> f32 { return clamp((H - H_B) / L_V, 0.0, 1.0); }

fn cellH(i : i32, j : i32, k : i32) -> f32 {
  let n = vec3<i32>(uniforms.dims);
  let c = clamp(vec3<i32>(i, j, k), vec3<i32>(0), n - vec3<i32>(1));
  return Hs[c.x + n.x * (c.y + n.y * c.z)];
}
/** trilinear H at grid coordinate g (cell centres at i + 0.5) */
fn sampleH(g : vec3f) -> f32 {
  let s = g - vec3f(0.5);
  let b = vec3<i32>(floor(s));
  let t = s - floor(s);
  let c00 = mix(cellH(b.x, b.y, b.z), cellH(b.x + 1, b.y, b.z), t.x);
  let c10 = mix(cellH(b.x, b.y + 1, b.z), cellH(b.x + 1, b.y + 1, b.z), t.x);
  let c01 = mix(cellH(b.x, b.y, b.z + 1), cellH(b.x + 1, b.y, b.z + 1), t.x);
  let c11 = mix(cellH(b.x, b.y + 1, b.z + 1), cellH(b.x + 1, b.y + 1, b.z + 1), t.x);
  return mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
}

/** normalised, clamped enthalpy used for ice normals: grows from ice to water */
fn hn(g : vec3f) -> f32 { return clamp(sampleH(g), -0.6 * L_F, 1.6 * L_F) / L_F; }

/** temperature colour map: deep blue (<=0) - cyan - clear (20) - yellow - orange - red (100) - white-hot */
fn tempColor(T : f32) -> vec3f {
  if (T < 0.0) { return vec3f(0.35, 0.55, 1.0); }
  if (T < 20.0) { return mix(vec3f(0.15, 0.35, 1.0), vec3f(0.3, 0.75, 0.95), T / 20.0); }
  if (T < 50.0) { return mix(vec3f(0.3, 0.75, 0.95), vec3f(1.0, 0.85, 0.25), (T - 20.0) / 30.0); }
  if (T < 100.0) { return mix(vec3f(1.0, 0.85, 0.25), vec3f(1.0, 0.2, 0.08), (T - 50.0) / 50.0); }
  return mix(vec3f(1.0, 0.2, 0.08), vec3f(1.0, 0.95, 0.9), clamp((T - 100.0) / 200.0, 0.0, 1.0));
}

/** surface height (metres above the mean level), bilinear between column centres; g = grid coords x,z */
fn etaAt(gx : f32, gz : f32) -> f32 {
  let nx = i32(uniforms.dims.x); let nz = i32(uniforms.dims.z);
  let sx = clamp(gx - 0.5, 0.0, f32(nx - 1)); let sz = clamp(gz - 0.5, 0.0, f32(nz - 1));
  let i0 = min(i32(floor(sx)), max(nx - 2, 0)); let k0 = min(i32(floor(sz)), max(nz - 2, 0));
  let i1 = min(i0 + 1, nx - 1); let k1 = min(k0 + 1, nz - 1);
  let fx = sx - f32(i0); let fz = sz - f32(k0);
  let a = mix(eta[i0 + nx * k0], eta[i1 + nx * k0], fx);
  let b = mix(eta[i0 + nx * k1], eta[i1 + nx * k1], fx);
  return mix(a, b, fz);
}

/** water surface (normal nUp): Fresnel reflection of a simple sky, sun highlight and a faint tint.
    Returns premultiplied colour and alpha. */
fn surfaceColor(rd : vec3f, nUp : vec3f, fromAbove : bool) -> vec4f {
  var n = nUp;
  if (!fromAbove) { n = -nUp; }
  let cosi = max(dot(n, -rd), 0.0);
  var F = 0.02 + 0.98 * pow(1.0 - cosi, 5.0);
  if (!fromAbove && cosi < 0.661) { F = 1.0; } // critical angle of water (n = 1.333)
  let r = reflect(rd, n);
  // sky: dark horizon, bright zenith and a soft light band along the light direction, so tilted
  // facets of the waves reflect visibly different brightness
  let L = normalize(uniforms.lightDir);
  var sky = mix(vec3f(0.08, 0.11, 0.16), vec3f(0.55, 0.68, 0.85), clamp(r.y * 0.5 + 0.5, 0.0, 1.0));
  sky += vec3f(0.5, 0.55, 0.6) * pow(max(dot(normalize(vec3f(r.x, 0.0, r.z) + vec3f(1e-5)), normalize(vec3f(L.x, 0.0, L.z))), 0.0), 8.0) * clamp(r.y * 2.0, 0.0, 1.0);
  var sun = 0.0;
  if (fromAbove) { sun = pow(max(dot(r, L), 0.0), 60.0) * 0.8; }
  let tint = vec3f(0.45, 0.7, 0.9);
  let a = clamp(0.2 + 0.8 * F + sun, 0.0, 1.0);
  let c = sky * F * 0.9 + vec3f(1.0) * sun + tint * 0.2;
  return vec4f(c, a);
}

fn raySphere(ro : vec3f, rd : vec3f, c : vec3f, r : f32) -> f32 {
  let oc = ro - c;
  let b = dot(oc, rd);
  let cc = dot(oc, oc) - r * r;
  let disc = b * b - cc;
  if (disc < 0.0) { return 1e9; }
  let t = -b - sqrt(disc);
  if (t < 0.0) { return 1e9; }
  return t;
}

@fragment
fn main(input : FragmentInputs) -> FragmentOutputs {
  let ro = uniforms.camPos;
  let rd = normalize(fragmentInputs.vWorld - ro);
  // ray / box
  let inv = 1.0 / rd;
  let t0 = (uniforms.boxMin - ro) * inv;
  let t1 = (uniforms.boxMax - ro) * inv;
  let tmin = min(t0, t1);
  let tmax = max(t0, t1);
  let tNear = max(max(max(tmin.x, tmin.y), tmin.z), 0.0);
  var tFar = min(min(tmax.x, tmax.y), tmax.z);
  if (tFar <= tNear) { discard; }
  let tBox = tFar;

  var col = vec3f(0.0);
  var alpha = 0.0;
  // nearest heat source along the ray
  var srcT = 1e9;
  var srcIdx = -1;
  let nSrc = i32(P[16]);
  for (var s = 0; s < nSrc; s++) {
    let o = 32 + s * 8;
    let c = uniforms.boxMin + vec3f(P[o], P[o + 1], P[o + 2]) * uniforms.metersToWorld;
    let t = raySphere(ro, rd, c, P[o + 3] * uniforms.metersToWorld);
    if (t < srcT) { srcT = t; srcIdx = s; }
  }
  tFar = min(tFar, srcT);

  let size = uniforms.boxMax - uniforms.boxMin;
  let toGrid = uniforms.dims / size;
  let cellW = size.x / uniforms.dims.x;

  // water surface: height field ys0 + eta(x, z) on top of the water rows (P[17]); above it is air
  let ys0 = uniforms.boxMin.y + P[17] * (size.y / uniforms.dims.y);
  var wA = tNear;
  var wB = tFar;
  var tSurf = -1.0;
  var nSurf = vec3f(0.0, 1.0, 0.0);
  if (abs(rd.y) > 1e-7) {
    // intersect the plane, then refine on the height field by fixed-point iteration (small slopes)
    var tp = (ys0 - ro.y) / rd.y;
    for (var it = 0; it < 4; it++) {
      let q = (ro + rd * tp - uniforms.boxMin) * toGrid;
      tp = (ys0 + etaAt(q.x, q.z) * uniforms.metersToWorld - ro.y) / rd.y;
    }
    if (rd.y < 0.0) { wA = max(wA, tp); } else { wB = min(wB, tp); }
    if (tp > tNear && tp < tBox && tp < srcT) {
      tSurf = tp;
      let q = (ro + rd * tp - uniforms.boxMin) * toGrid;
      let s = uniforms.metersToWorld / cellW; // eta metres -> grid cells
      let dx = (etaAt(q.x + 0.5, q.z) - etaAt(q.x - 0.5, q.z)) * s;
      let dz = (etaAt(q.x, q.z + 0.5) - etaAt(q.x, q.z - 0.5)) * s;
      // display-only exaggeration of the slopes (x3) so millimetre ripples read on screen
      nSurf = normalize(vec3f(-3.0 * dx, 1.0, -3.0 * dz));
    }
  } else if (ro.y >= ys0) { wB = wA; }
  if (rd.y < 0.0 && tSurf > 0.0) {
    let sc = surfaceColor(rd, nSurf, true);
    col += (1.0 - alpha) * sc.rgb;
    alpha += (1.0 - alpha) * sc.a;
  }
  let stepLen = cellW * 0.75;
  // per-pixel jitter of the first sample hides step banding
  let fc = input.position.xy;
  let jit = fract(sin(dot(fc, vec2f(12.9898, 78.233))) * 43758.5453);
  var t = wA + stepLen * jit;
  tFar = min(wB, tFar);
  let tempMode = uniforms.viewMode > 0.5;
  var hitIce = false;
  loop {
    if (t >= tFar || alpha > 0.985) { break; }
    let p = ro + rd * t;
    let g = (p - uniforms.boxMin) * toGrid;
    let H = sampleH(g);
    let fi = iceFraction(H);
    if (fi > 0.5) {
      // refine the iso-surface (ice fraction 0.5) between the previous and this sample
      var lo = max(t - stepLen, wA);
      var hi = t;
      for (var r = 0; r < 5; r++) {
        let mid = 0.5 * (lo + hi);
        if (iceFraction(sampleH((ro + rd * mid - uniforms.boxMin) * toGrid)) > 0.5) { hi = mid; } else { lo = mid; }
      }
      let g = (ro + rd * hi - uniforms.boxMin) * toGrid;
      let H = sampleH(g);
      // ice surface: normal from the gradient of a clamped enthalpy (smoother than the ice fraction itself)
      let e = 1.0;
      let gx = hn(g + vec3f(e, 0.0, 0.0)) - hn(g - vec3f(e, 0.0, 0.0));
      let gy = hn(g + vec3f(0.0, e, 0.0)) - hn(g - vec3f(0.0, e, 0.0));
      let gz = hn(g + vec3f(0.0, 0.0, e)) - hn(g - vec3f(0.0, 0.0, e));
      var n = vec3f(gx, gy, gz);
      if (length(n) < 1e-4) { n = -rd; }
      n = normalize(n);
      let L = normalize(uniforms.lightDir);
      let diff = max(dot(n, L), 0.0);
      let hv = normalize(L - rd);
      let spec = pow(max(dot(n, hv), 0.0), 40.0);
      let fres = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
      let Tice = temperature(H);
      var base = vec3f(0.78, 0.9, 1.0) * (0.35 + 0.6 * diff) + vec3f(0.15, 0.3, 0.45) * clamp(-Tice / 30.0, 0.0, 1.0);
      if (tempMode) { base = mix(base, tempColor(Tice), 0.5); }
      let c = base + vec3f(1.0) * (0.6 * spec + 0.35 * fres);
      let a = 0.92;
      col += (1.0 - alpha) * a * c;
      alpha += (1.0 - alpha) * a;
      hitIce = true;
      break;
    }
    let T = temperature(H);
    let chi = vaporQuality(H);
    // water: faint absorption everywhere, stronger where it is hot or near freezing so plumes are visible
    let dev = clamp(abs(T - 20.0) / 80.0, 0.0, 1.0);
    var sigma = 0.25 + 9.0 * dev * dev;
    var c = tempColor(T);
    if (tempMode) { sigma = 6.0 + 6.0 * dev; }
    // vapour: white fog (quality -> density)
    let fog = clamp(chi * 40.0, 0.0, 1.0);
    sigma += 30.0 * fog;
    c = mix(c, vec3f(0.95, 0.97, 1.0), fog);
    // mushy zone (partly frozen): whitish slush
    let slush = clamp(fi * 2.0, 0.0, 1.0);
    sigma += 10.0 * slush;
    c = mix(c, vec3f(0.85, 0.93, 1.0), slush);
    let a = 1.0 - exp(-sigma * stepLen);
    col += (1.0 - alpha) * a * c;
    alpha += (1.0 - alpha) * a;
    t += stepLen;
  }

  if (!hitIce && rd.y > 0.0 && tSurf > 0.0 && alpha < 0.99) {
    // the surface seen from below (total internal reflection beyond the critical angle)
    let sc = surfaceColor(rd, nSurf, false);
    col += (1.0 - alpha) * sc.rgb;
    alpha += (1.0 - alpha) * sc.a;
  }

  if (!hitIce && srcIdx >= 0 && srcT < 1e8 && srcT >= wA && srcT <= wB + 1e-6 && alpha < 0.99) {
    let o = 32 + srcIdx * 8;
    let c = uniforms.boxMin + vec3f(P[o], P[o + 1], P[o + 2]) * uniforms.metersToWorld;
    let p = ro + rd * srcT;
    let n = normalize(p - c);
    let L = normalize(uniforms.lightDir);
    let diff = max(dot(n, L), 0.0);
    let Ts = P[o + 4];
    var sc = vec3f(0.0);
    if (Ts >= 20.0) {
      let glow = clamp((Ts - 20.0) / 380.0, 0.0, 1.0);
      sc = mix(vec3f(0.35, 0.3, 0.28), vec3f(1.0, 0.35, 0.05), glow) * (0.4 + 0.6 * diff) + vec3f(1.0, 0.5, 0.1) * glow * 0.8;
    } else {
      let cold = clamp((20.0 - Ts) / 60.0, 0.0, 1.0);
      sc = mix(vec3f(0.35, 0.35, 0.38), vec3f(0.25, 0.55, 1.0), cold) * (0.35 + 0.65 * diff);
    }
    let rim = pow(1.0 - max(dot(n, -rd), 0.0), 2.0);
    sc += vec3f(0.3) * rim;
    if (f32(srcIdx) == uniforms.selected) { sc += vec3f(0.9, 0.9, 0.3) * rim * 1.5; }
    col += (1.0 - alpha) * sc;
    alpha = 1.0;
  }
  fragmentOutputs.color = vec4f(col, alpha);
}
`;

export class VolumeRenderer {
  readonly mesh: Mesh;
  readonly material: ShaderMaterial;
  readonly boxMin: Vector3;
  readonly boxMax: Vector3;
  readonly metersToWorld: number;
  viewMode = 0;
  selected = -1;

  constructor(scene: Scene, p: SimParams, H: StorageBuffer, P: StorageBuffer, eta: StorageBuffer) {
    const maxN = Math.max(p.nx, p.ny, p.nz);
    this.metersToWorld = 1 / (maxN * p.h);
    const ext = new Vector3(p.nx / maxN, p.ny / maxN, p.nz / maxN);
    this.boxMin = ext.scale(-0.5);
    this.boxMax = ext.scale(0.5);
    this.mesh = MeshBuilder.CreateBox("volume", { width: ext.x, height: ext.y, depth: ext.z }, scene);
    this.mesh.isPickable = false;
    this.material = new ShaderMaterial(
      "volumeMat",
      scene,
      { vertexSource: VERTEX, fragmentSource: FRAGMENT },
      {
        attributes: ["position"],
        uniforms: ["world", "worldViewProjection", "camPos", "boxMin", "boxMax", "dims", "metersToWorld", "viewMode", "selected", "lightDir"],
        storageBuffers: ["Hs", "P", "eta"],
        shaderLanguage: ShaderLanguage.WGSL,
        needAlphaBlending: true,
      },
    );
    const m = this.material;
    m.backFaceCulling = true;
    m.cullBackFaces = false; // draw back faces so the march also works with the camera inside the box
    m.alphaMode = Constants.ALPHA_PREMULTIPLIED;
    m.disableDepthWrite = true;
    m.setStorageBuffer("Hs", H);
    m.setStorageBuffer("P", P);
    m.setStorageBuffer("eta", eta);
    m.setVector3("boxMin", this.boxMin);
    m.setVector3("boxMax", this.boxMax);
    m.setVector3("dims", new Vector3(p.nx, p.ny, p.nz));
    m.setFloat("metersToWorld", this.metersToWorld);
    m.setVector3("lightDir", new Vector3(0.4, 1, 0.3));
    this.mesh.material = m;
    this.mesh.alphaIndex = 10;
    scene.onBeforeRenderObservable.add(() => {
      const cam = scene.activeCamera;
      if (cam) m.setVector3("camPos", cam.globalPosition);
      m.setFloat("viewMode", this.viewMode);
      m.setFloat("selected", this.selected);
    });
  }

  /** domain metres -> world */
  toWorld(x: number, y: number, z: number) {
    return new Vector3(this.boxMin.x + x * this.metersToWorld, this.boxMin.y + y * this.metersToWorld, this.boxMin.z + z * this.metersToWorld);
  }
  /** world -> domain metres */
  toDomain(v: Vector3) {
    return { x: (v.x - this.boxMin.x) / this.metersToWorld, y: (v.y - this.boxMin.y) / this.metersToWorld, z: (v.z - this.boxMin.z) / this.metersToWorld };
  }
}
