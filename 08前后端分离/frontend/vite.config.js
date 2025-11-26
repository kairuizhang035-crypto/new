import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    open: false,
    host: true,
    allowedHosts: ['216ef8b.r7.cpolar.top', '3eab0cd0.r12.vip.cpolar.cn'],
    proxy: {
      '/api': {
        target: 'http://localhost:5000',
        changeOrigin: true,
        secure: false
      }
    }
  }
})