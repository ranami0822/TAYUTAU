import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { openHarness, type Harness } from "./browser";

let H: Harness;
beforeAll(async () => {
  H = await openHarness();
});
afterAll(async () => {
  await H?.close();
});

describe("stage 0: Babylon.js ComputeShader round trip", () => {
  it("runs a WGSL kernel on the GPU and reads the result back", async () => {
    const out = (await H.page.evaluate((n) => window.tayutau.smoke(n), 1000)) as number[];
    expect(out.length).toBe(1000);
    for (let i = 0; i < 1000; i++) expect(out[i]).toBe(i * 2 + 1);
  });
});

describe("stage 8 (early): no WebGPU -> explicit message, no WebGL fallback", () => {
  it("shows the unsupported message when no adapter is available", async () => {
    const { chromium } = await import("playwright");
    // headless Chromium without --enable-unsafe-webgpu exposes navigator.gpu but returns no adapter here;
    // additionally remove navigator.gpu to cover browsers that do not implement WebGPU at all.
    for (const removeGpu of [false, true]) {
      const browser = await chromium.launch({ headless: true, args: ["--disable-features=WebGPU"] });
      const page = await browser.newPage();
      if (removeGpu) await page.addInitScript(() => Object.defineProperty(Navigator.prototype, "gpu", { get: () => undefined }));
      await page.goto(H.url + "/index.html");
      await page.waitForFunction(() => document.body.dataset.state === "unsupported", null, { timeout: 30_000 });
      const visible = await page.isVisible("#unsupported");
      const text = await page.textContent("#unsupported");
      expect(visible).toBe(true);
      expect(text).toContain("WebGPU");
      expect(await page.isVisible("#view")).toBe(false);
      await browser.close();
    }
  });
});
