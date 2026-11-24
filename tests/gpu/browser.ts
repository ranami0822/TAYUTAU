/**
 * Starts a Vite dev server and a Chromium with WebGPU enabled, and opens the GPU harness page.
 *
 * Linux without a display (CI / containers): WebGPU canvas presentation crashes in headless
 * Chromium with SwiftShader, so an Xvfb server is started and Chromium runs headed with
 * Vulkan on SwiftShader. Elsewhere Chromium runs headless with the real GPU.
 * Env: TAYUTAU_CHROMIUM = browser binary, TAYUTAU_HEADLESS=1 to force headless.
 */
import { createServer, type ViteDevServer } from "vite";
import { chromium, type Browser, type Page } from "playwright";
import { fileURLToPath } from "node:url";
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import path from "node:path";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");

export interface Harness {
  page: Page;
  url: string;
  errors: string[];
  close(): Promise<void>;
}

async function startXvfbIfNeeded(): Promise<ChildProcess | null> {
  if (process.platform !== "linux" || process.env.DISPLAY || process.env.TAYUTAU_HEADLESS) return null;
  if (spawnSync("which", ["Xvfb"]).status !== 0) return null;
  const display = `:${90 + Math.floor(Math.random() * 400)}`;
  const xvfb = spawn("Xvfb", [display, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"], { stdio: "ignore" });
  process.env.DISPLAY = display;
  await new Promise((r) => setTimeout(r, 500));
  return xvfb;
}

export async function openHarness(pagePath = "/gpu-harness/index.html"): Promise<Harness> {
  const server: ViteDevServer = await createServer({
    root,
    configFile: path.join(root, "vite.config.ts"),
    server: { port: 0, host: "127.0.0.1", hmr: false },
    optimizeDeps: { include: ["@babylonjs/core"] },
    logLevel: "error",
  });
  await server.listen();
  const addr = server.httpServer!.address();
  const port = typeof addr === "object" && addr ? addr.port : 5173;
  const xvfb = await startXvfbIfNeeded();
  const headed = xvfb !== null || (process.platform === "linux" && !!process.env.DISPLAY && !process.env.TAYUTAU_HEADLESS);
  const args = ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist"];
  if (process.platform === "linux") args.push("--enable-features=Vulkan", "--use-vulkan=swiftshader", "--enable-unsafe-swiftshader");
  const browser: Browser = await chromium.launch({
    executablePath: process.env.TAYUTAU_CHROMIUM || undefined,
    headless: !headed,
    args,
  });
  const page = await browser.newPage({ viewport: { width: 800, height: 600 } });
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  const url = `http://127.0.0.1:${port}`;
  await page.goto(url + pagePath);
  if (pagePath.startsWith("/gpu-harness")) {
    const ready = await page.evaluate(() => window.tayutauReady);
    if (ready !== "ok") throw new Error(`WebGPU unavailable in Chromium: ${ready}\n${errors.join("\n")}`);
  }
  return {
    page,
    url,
    errors,
    async close() {
      await browser.close();
      await server.close();
      xvfb?.kill();
    },
  };
}
