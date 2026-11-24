/**
 * End-to-end: the real app in Chromium/WebGPU. Rendering, HUD, and mouse editing of heat sources.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { openHarness, type Harness } from "./browser";

let H: Harness;
beforeAll(async () => {
  H = await openHarness("/index.html?n=20");
  await H.page.waitForFunction(() => document.body.dataset.state === "running" && !!window.tayutauApp?.state.stats, null, { timeout: 90_000 });
});
afterAll(async () => {
  await H?.close();
});

const st = () => H.page.evaluate(() => {
  const s = window.tayutauApp!.state;
  return { n: s.sources.length, sources: s.sources.map((q) => ({ ...q })), selected: s.selected, steps: s.steps, stats: s.stats };
});

describe("app", () => {
  it("starts, renders a non-empty frame and shows the HUD", async () => {
    const png = await H.page.screenshot();
    expect(png.length).toBeGreaterThan(10_000);
    const hud = await H.page.textContent("#hud");
    expect(hud).toContain("平均");
    expect(hud).toContain("エネルギー収支");
    expect(H.errors.filter((e) => !e.includes("favicon"))).toEqual([]);
  });

  it("places a heat source where the user clicks on the tank", async () => {
    const before = await st();
    await H.page.click('button[data-tool="hot"]');
    // click on the visible bottom of the tank: project a point on the floor of the domain
    const p = await H.page.evaluate(() => {
      const a = window.tayutauApp!;
      const q = a.state.params;
      return a.screenOf(q.nx * q.h * 0.3, 0, q.nz * q.h * 0.7);
    });
    await H.page.mouse.click(p.x, p.y);
    const after = await st();
    expect(after.n).toBe(before.n + 1);
    const s = after.sources[after.n - 1];
    expect(s.T).toBeGreaterThan(100);
    expect(after.selected).toBe(after.n - 1);
    // it sits near the clicked floor point
    const q = await H.page.evaluate(() => window.tayutauApp!.state.params);
    expect(Math.abs(s.x - q.nx * q.h * 0.3)).toBeLessThan(q.nx * q.h * 0.1);
    expect(s.y).toBeLessThan(q.ny * q.h * 0.2);
  });

  it("moves a source by dragging it, without orbiting the camera", async () => {
    const before = await st();
    const i = before.n - 1;
    const cam0 = await H.page.evaluate(() => [window.tayutauApp!.camera.alpha, window.tayutauApp!.camera.beta]);
    const p = await H.page.evaluate((k) => {
      const a = window.tayutauApp!;
      const s = a.state.sources[k];
      return a.screenOf(s.x, s.y, s.z);
    }, i);
    await H.page.mouse.move(p.x, p.y);
    await H.page.mouse.down();
    for (let k = 1; k <= 8; k++) await H.page.mouse.move(p.x + 8 * k, p.y - 6 * k);
    await H.page.mouse.up();
    const after = await st();
    const a = before.sources[i], b = after.sources[i];
    expect(Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z)).toBeGreaterThan(0.002);
    const cam1 = await H.page.evaluate(() => [window.tayutauApp!.camera.alpha, window.tayutauApp!.camera.beta]);
    expect(cam1).toEqual(cam0);
  });

  it("orbits the camera when dragging empty space", async () => {
    const cam0 = await H.page.evaluate(() => window.tayutauApp!.camera.alpha);
    await H.page.mouse.move(700, 500);
    await H.page.mouse.down();
    for (let k = 1; k <= 8; k++) await H.page.mouse.move(700 - 10 * k, 500);
    await H.page.mouse.up();
    const cam1 = await H.page.evaluate(() => window.tayutauApp!.camera.alpha);
    expect(cam1).not.toBe(cam0);
  });

  it("removes a source with the erase tool", async () => {
    const before = await st();
    await H.page.click('button[data-tool="erase"]');
    const p = await H.page.evaluate(() => {
      const a = window.tayutauApp!;
      const s = a.state.sources[0];
      return a.screenOf(s.x, s.y, s.z);
    });
    await H.page.mouse.click(p.x, p.y);
    const after = await st();
    expect(after.n).toBe(before.n - 1);
  });

  it("keeps the energy balance (HUD stats from the GPU reduction) near zero while running", async () => {
    await H.page.waitForFunction((s0) => window.tayutauApp!.state.steps > s0 + 20, (await st()).steps, { timeout: 120_000 });
    await H.page.waitForTimeout(1000);
    const s = (await st()).stats!;
    const s0 = await H.page.evaluate(() => window.tayutauApp!.state.stats0!);
    const drift = s.energy - s0 - s.injected;
    expect(Math.abs(s.injected)).toBeGreaterThan(0);
    expect(Math.abs(drift) / s.energy).toBeLessThan(1e-5);
  });
});
