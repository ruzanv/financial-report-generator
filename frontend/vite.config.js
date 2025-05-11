import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    strictPort: true,
    proxy: {
      "/upload": "http://backend:8000",
      "/status": "http://backend:8000",
      "/download": "http://backend:8000"
    }
  }
});