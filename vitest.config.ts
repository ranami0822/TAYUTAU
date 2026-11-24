import { defineConfig } from "vitest/config";

// Two projects:
//  - cpu: pure TypeScript physics (reference solver, equation of state, enthalpy), runs in Node.
//  - gpu: runs the WGSL kernels in headless Chromium (WebGPU via SwiftShader) and compares them with the CPU reference.
export default defineConfig({
  test: {
    projects: [
      { test: { name: "cpu", include: ["tests/cpu/**/*.test.ts"], environment: "node" } },
      {
        test: {
          name: "gpu",
          include: ["tests/gpu/**/*.test.ts"],
          environment: "node",
          testTimeout: 180_000,
          hookTimeout: 180_000,
          fileParallelism: false,
        },
      },
    ],
  },
});
