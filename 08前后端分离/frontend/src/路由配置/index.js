import { createRouter, createWebHistory } from 'vue-router'
import 上传入口页面 from '../页面视图/上传入口页面.vue'
import 知识图谱主页面 from '../页面视图/知识图谱主页面.vue'
import 用户登录页面 from '../页面视图/用户登录页面.vue'

const routes = [
  {
    path: '/',
    redirect: '/login'
  },
  {
    path: '/login',
    name: '登录',
    component: 用户登录页面
  },
  {
    path: '/upload',
    name: '上传入口页面',
    component: 上传入口页面,
    meta: { requiresAuth: true }
  },
  {
    path: '/graph',
    name: '知识图谱主页面',
    component: 知识图谱主页面,
    meta: { requiresAuth: true }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach(async (to, from, next) => {
  if (to.meta?.requiresAuth) {
    try {
      const res = await fetch('/api/auth/me')
      const json = await res.json()
      if (json?.authenticated) return next()
      const origin = encodeURIComponent(window.location.href)
      return next({ path: '/login', query: { redirect: to.fullPath, origin } })
    } catch (e) {
      const origin = encodeURIComponent(window.location.href)
      return next({ path: '/login', query: { redirect: to.fullPath, origin } })
    }
  }
  next()
})

export default router
