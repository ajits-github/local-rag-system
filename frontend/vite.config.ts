/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev-server proxy keeps the frontend same-origin with the backend so the
// FastAPI app never needs CORS middleware added to it (see CLAUDE.md's
// "Web UI" section for the reasoning). The Docker build mirrors this with
// an nginx reverse proxy instead (frontend/nginx.conf).
const BACKEND_ORIGIN = process.env.VITE_DEV_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/query": BACKEND_ORIGIN,
      "/agent": BACKEND_ORIGIN,
      "/health": BACKEND_ORIGIN,
      "/metrics": BACKEND_ORIGIN,
      "/docs": BACKEND_ORIGIN,
      "/openapi.json": BACKEND_ORIGIN,
      // The bare "/" is both the SPA's own document root and the
      // backend's GET / (FeatureFlagsBar's info endpoint) -- the only
      // path where the frontend and the API contract collide. A real
      // browser navigation sends `Accept: text/html`; FeatureFlagsBar's
      // own `fetch("/")` doesn't, so that header is what disambiguates
      // which one a given request to "/" actually wants.
      "^/$": {
        target: BACKEND_ORIGIN,
        changeOrigin: true,
        bypass(req) {
          const accept = req.headers.accept ?? "";
          return accept.includes("text/html") ? req.url : undefined;
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/unit/**/*.test.ts?(x)", "tests/integration/**/*.test.ts?(x)"],
  },
});
