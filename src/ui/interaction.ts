/**
 * Mouse / touch editing of heat sources:
 *  - tap/click on the tank with the 加熱 / 冷却 tool: place a source on the inner wall you see
 *    through the glass (the far side of the box along the ray, e.g. the bottom when looking down)
 *  - drag a source: move it in the plane facing the camera (camera orbit is suspended meanwhile)
 *  - tap a source with the 消す tool, double-click / double-tap it, or press Delete: remove it
 *  - drag on empty space: orbit the camera; wheel / pinch: zoom
 */
import { Vector3, type Ray } from "@babylonjs/core";
import type { App } from "../app";

export type Tool = "hot" | "cold" | "move" | "erase";

export class Interaction {
  private app: App;
  private canvas: HTMLCanvasElement;
  private onChange: () => void;
  private drag: { idx: number; pointerId: number; planeN: Vector3; planeD: number; offset: Vector3 } | null = null;
  private down: { x: number; y: number; t: number; id: number } | null = null;
  private lastTap = { t: 0, idx: -1 };

  constructor(app: App, canvas: HTMLCanvasElement, onChange: () => void) {
    this.app = app;
    this.canvas = canvas;
    this.onChange = onChange;
    canvas.addEventListener("pointerdown", this.onDown, true);
    window.addEventListener("pointermove", this.onMove, true);
    window.addEventListener("pointerup", this.onUp, true);
    window.addEventListener("pointercancel", this.onUp, true);
    window.addEventListener("keydown", (e) => {
      if ((e.key === "Delete" || e.key === "Backspace") && app.state.selected >= 0 && !(e.target instanceof HTMLInputElement)) {
        app.removeSource(app.state.selected);
        this.onChange();
      }
    });
  }

  private ray(e: PointerEvent): Ray {
    const r = this.canvas.getBoundingClientRect();
    const scene = this.app.scene;
    // scene.pointerX/Y are in render pixels; derive from the event to be robust to DPR scaling
    const engine = scene.getEngine();
    const sx = ((e.clientX - r.left) / r.width) * engine.getRenderWidth();
    const sy = ((e.clientY - r.top) / r.height) * engine.getRenderHeight();
    return scene.createPickingRay(sx, sy, null, this.app.camera);
  }

  /** index of the nearest source hit by the ray, or -1 */
  pickSource(ray: Ray): number {
    const v = this.app.volume;
    let best = -1;
    let bestT = Infinity;
    this.app.state.sources.forEach((s, i) => {
      const c = v.toWorld(s.x, s.y, s.z);
      const r = s.r * v.metersToWorld * 1.15;
      const oc = ray.origin.subtract(c);
      const b = Vector3.Dot(oc, ray.direction);
      const cc = oc.lengthSquared() - r * r;
      const disc = b * b - cc;
      if (disc < 0) return;
      const t = -b - Math.sqrt(disc);
      if (t > 0 && t < bestT) {
        bestT = t;
        best = i;
      }
    });
    return best;
  }

  /** far intersection of the ray with the tank (the inner wall seen through the glass) */
  wallPoint(ray: Ray): Vector3 | null {
    const v = this.app.volume;
    let tNear = -Infinity;
    let tFar = Infinity;
    for (const ax of ["x", "y", "z"] as const) {
      const o = ray.origin[ax];
      const d = ray.direction[ax];
      if (Math.abs(d) < 1e-9) {
        if (o < v.boxMin[ax] || o > v.boxMax[ax]) return null;
        continue;
      }
      const t0 = (v.boxMin[ax] - o) / d;
      const t1 = (v.boxMax[ax] - o) / d;
      tNear = Math.max(tNear, Math.min(t0, t1));
      tFar = Math.min(tFar, Math.max(t0, t1));
    }
    if (tFar < Math.max(tNear, 0)) return null;
    return ray.origin.add(ray.direction.scale(tFar));
  }

  /** clamp a source centre (domain metres) so the sphere stays inside the tank */
  clampInside(x: number, y: number, z: number, r: number) {
    const p = this.app.state.params;
    const L = [p.nx * p.h, p.ny * p.h, p.nz * p.h];
    const m = Math.min(r * 0.6, Math.min(...L) * 0.45);
    return {
      x: Math.min(L[0] - m, Math.max(m, x)),
      y: Math.min(L[1] - m, Math.max(m, y)),
      z: Math.min(L[2] - m, Math.max(m, z)),
    };
  }

  private onDown = (e: PointerEvent) => {
    if (e.button !== 0) return;
    const ray = this.ray(e);
    const idx = this.pickSource(ray);
    const st = this.app.state;
    if (idx >= 0) {
      const now = performance.now();
      const dbl = this.lastTap.idx === idx && now - this.lastTap.t < 350;
      this.lastTap = { t: now, idx };
      if (st.tool === "erase" || dbl) {
        this.app.removeSource(idx);
        this.onChange();
        e.stopPropagation();
        e.preventDefault();
        return;
      }
      st.selected = idx;
      const s = st.sources[idx];
      const c = this.app.volume.toWorld(s.x, s.y, s.z);
      const n = this.app.camera.getForwardRay().direction.normalize();
      const hit = this.planeHit(ray, n, -Vector3.Dot(n, c)) ?? c;
      this.drag = { idx, pointerId: e.pointerId, planeN: n, planeD: -Vector3.Dot(n, c), offset: c.subtract(hit) };
      this.app.camera.detachControl();
      this.canvas.setPointerCapture?.(e.pointerId);
      e.stopPropagation();
      e.preventDefault();
      this.onChange();
      return;
    }
    this.down = { x: e.clientX, y: e.clientY, t: performance.now(), id: e.pointerId };
  };

  private planeHit(ray: Ray, n: Vector3, d: number): Vector3 | null {
    const denom = Vector3.Dot(n, ray.direction);
    if (Math.abs(denom) < 1e-6) return null;
    const t = -(Vector3.Dot(n, ray.origin) + d) / denom;
    if (t < 0) return null;
    return ray.origin.add(ray.direction.scale(t));
  }

  private onMove = (e: PointerEvent) => {
    if (!this.drag || e.pointerId !== this.drag.pointerId) return;
    const hit = this.planeHit(this.ray(e), this.drag.planeN, this.drag.planeD);
    if (!hit) return;
    const w = hit.add(this.drag.offset);
    const s = this.app.state.sources[this.drag.idx];
    if (!s) return;
    const d = this.app.volume.toDomain(w);
    this.app.updateSource(this.drag.idx, this.clampInside(d.x, d.y, d.z, s.r));
    e.preventDefault();
  };

  private onUp = (e: PointerEvent) => {
    if (this.drag && e.pointerId === this.drag.pointerId) {
      this.drag = null;
      this.app.camera.attachControl(this.canvas, true);
      this.onChange();
      return;
    }
    const d = this.down;
    this.down = null;
    if (!d || d.id !== e.pointerId) return;
    const moved = Math.hypot(e.clientX - d.x, e.clientY - d.y);
    if (moved > 6 || performance.now() - d.t > 500) return;
    // a click / tap on empty space
    const st = this.app.state;
    if (st.tool === "hot" || st.tool === "cold") {
      const wp = this.wallPoint(this.ray(e));
      if (!wp) return;
      const p = st.params;
      const r = Math.min(p.nx, p.ny, p.nz) * p.h * 0.1;
      const dom = this.app.volume.toDomain(wp);
      const c = this.clampInside(dom.x, dom.y, dom.z, r);
      const idx = this.app.addSource({ ...c, r, T: st.tool === "hot" ? 400 : -40 });
      if (idx >= 0) st.selected = idx;
    } else {
      st.selected = -1;
    }
    this.onChange();
  };
}
