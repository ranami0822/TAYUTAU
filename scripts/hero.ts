/** README screenshot: heater boiling + cooler freezing at 48^3. Usage: tsx scripts/hero.ts out.png seconds */
import { openHarness } from "../tests/gpu/browser";
const out = process.argv[2] ?? "docs/screenshot.png";
const secs = Number(process.argv[3] ?? 150);
const H = await openHarness("/index.html?n=48");
await H.page.setViewportSize({ width: 1100, height: 680 });
await H.page.waitForFunction(() => document.body.dataset.state === "running", null, { timeout: 120000 });
await H.page.evaluate(() => {
  const a = window.tayutauApp!;
  const q = a.state.params;
  const L = q.nx * q.h;
  a.updateSource(0, { x: L * 0.38, y: L * 0.1, z: L * 0.55, r: L * 0.09, T: 450 });
  a.addSource({ x: L * 0.72, y: L * 0.72, z: L * 0.35, r: L * 0.08, T: -60 });
  a.state.selected = -1;
  a.state.stepsPerFrame = 6;
  a.scene.getEngine().setHardwareScalingLevel(6);
  a.camera.radius = 2.05;
});
const t0 = Date.now();
while (Date.now() - t0 < secs * 1000) {
  await H.page.waitForTimeout(10000);
  const s = await H.page.evaluate(() => window.tayutauApp!.state.stats);
  console.log(Math.round((Date.now() - t0) / 1000), JSON.stringify(s));
}
await H.page.evaluate(() => {
  window.tayutauApp!.state.stepsPerFrame = 1;
  window.tayutauApp!.scene.getEngine().setHardwareScalingLevel(1);
});
await H.page.waitForTimeout(8000);
await H.page.screenshot({ path: out, timeout: 600_000 });
await H.close();
