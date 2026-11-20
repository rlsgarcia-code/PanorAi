import { defineConfig } from "vite";
import { resolve } from "node:path";

export default defineConfig({
  server: {
    proxy: {
      "/api": process.env.PANORAI_WORKBENCH_API_URL ?? "http://127.0.0.1:8000",
    },
  },
  build: {
    rollupOptions: {
      input: {
        laboratory: resolve(import.meta.dirname, "index.html"),
        catalog: resolve(import.meta.dirname, "catalog.html"),
        applications: resolve(import.meta.dirname, "applications.html"),
        canvas: resolve(import.meta.dirname, "canvas.html"),
        singleView: resolve(import.meta.dirname, "single-view.html"),
        twoView: resolve(import.meta.dirname, "two-view.html"),
        multiView: resolve(import.meta.dirname, "multi-view.html"),
        academy: resolve(import.meta.dirname, "academy.html"),
        segmentation: resolve(import.meta.dirname, "segmentation.html"),
        geometry: resolve(import.meta.dirname, "geometry.html"),
        runner: resolve(import.meta.dirname, "runner.html"),
        knownHeightPlanar: resolve(
          import.meta.dirname,
          "known-height-planar.html",
        ),
      },
    },
  },
});
