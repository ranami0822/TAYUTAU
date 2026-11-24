import { openHarness } from "../tests/gpu/browser";
const H = await openHarness();
H.page.on("console", (m) => console.log("[page]", m.type(), m.text()));
const fn = process.argv[2] ?? "smoke";
const args = JSON.parse(process.argv[3] ?? "[64]");
try {
  const r = await H.page.evaluate(([f, a]) => (window.tayutau as any)[f](...a), [fn, args] as const);
  console.log(JSON.stringify(r).slice(0, 2000));
} catch (e) { console.log("ERR", String(e).slice(0, 2000)); }
await H.close();
