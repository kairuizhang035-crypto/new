import { createApp } from 'vue'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './应用主组件.vue'
import router from './路由配置'
import './全局样式.css'

const app = createApp(App)

app.use(createPinia())
app.use(router)
app.use(ElementPlus)

// 暴露 router 供简单页面使用（例如无 <script setup> 的跳转）
window.$vueRouter = router
// 挂载静态资源基路径，供页面计算视频地址
window.__BASE_URL__ = import.meta.env && import.meta.env.BASE_URL ? import.meta.env.BASE_URL : '/'

app.mount('#app')
