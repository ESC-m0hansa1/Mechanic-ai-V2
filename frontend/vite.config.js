import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies /api to uvicorn, so the browser only ever talks to one
// origin (:5173) and CORS never enters the picture during development. In
// production there is no proxy and no second origin either: FastAPI serves the
// built files from dist/ and the API from /api on the same host.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: {
    // FastAPI mounts this directory (app/main.py: FRONTEND_DIST).
    outDir: "dist",
    sourcemap: false,
  },
});
