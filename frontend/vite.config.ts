import react from '@vitejs/plugin-react'
import fs from 'fs'
import path from 'path'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-local-data-media',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          if (req.url && req.url.startsWith('/data/')) {
            const relPath = req.url.slice(1).split('?')[0]
            const filePath = path.resolve(process.cwd(), '..', relPath)
            if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
              const ext = path.extname(filePath).toLowerCase()
              const mimeTypes: Record<string, string> = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.json': 'application/json',
              }
              res.setHeader('Content-Type', mimeTypes[ext] || 'application/octet-stream')
              fs.createReadStream(filePath).pipe(res)
              return
            }
          }
          next()
        })
      },
    },
  ],
  server: {
    port: 5173,
    proxy: {
      '/query': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/evidence': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/memory': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
