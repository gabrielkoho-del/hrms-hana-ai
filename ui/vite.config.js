import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  base: '/ui/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy API calls to the FastAPI backend during development
      '/v1': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/charts': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
      '/exports': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
