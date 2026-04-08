import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to the Python backend
      '/api': 'http://127.0.0.1:8084',
      '/health': 'http://127.0.0.1:8084',
      '/ws': {
        target: 'ws://127.0.0.1:8084',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    // Output goes to CueCatcher/ui/dist/ — the FastAPI server serves this
  },
})
