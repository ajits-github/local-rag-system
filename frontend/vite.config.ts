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
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/unit/**/*.test.ts?(x)", "tests/integration/**/*.test.ts?(x)"],
  },
});
