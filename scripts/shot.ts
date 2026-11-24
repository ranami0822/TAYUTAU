/** Open the app in Chromium (WebGPU), let it run, and take screenshots. Usage: tsx scripts/shot.ts out.png [seconds] [query] [js] */
import { openHarness } from "../tests/gpu/browser";
const out = process.argv[2] ?? "shot.png";
const secs = Number(process.argv[3] ?? 3);
const query = process.argv[4] ?? "";
const js = process.argv[5] ?? "";
const H = await openHarness("/index.html" + query);
H.page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") console.log("[page]", m.type(), m.text().slice(0, 400)); });
await H.page.waitForFunction(() => document.body.dataset.state === "running" || document.body.dataset.state === "unsupported", null, { timeout: 60000 });
console.log("state", await H.page.evaluate(() => document.body.dataset.state));
if (js) await H.page.evaluate(js);
const t0 = Date.now();
await H.page.waitForTimeout(secs * 1000);
const info = await H.page.evaluate(() => { const a = window.tayutauApp!; return { steps: a.state.steps, stats: a.state.stats }; });
console.log("wall", (Date.now() - t0) / 1000, JSON.stringify(info));
await H.page.screenshot({ path: out });
console.log("errors", H.errors.slice(0, 5));
await H.close();
