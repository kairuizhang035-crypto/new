<template>
  <div class="login-wrap">
    <video
      class="bg-video"
      :src="bgSrc"
      autoplay
      muted
      loop
      playsinline
      preload="metadata"
      @loadeddata="onVideoReady"
      @error="onVideoError"
      ref="bgVideoEl"
    ></video>
    <div class="bg-mask"></div>
    <div class="login-card">
      <div class="login-header">
        <h1 class="login-title">用户登录</h1>
        <p class="login-desc">访问增强知识图谱需先登录</p>
      </div>

      <form class="login-form" @submit.prevent="onSubmit">
        <div class="field">
          <div class="field-control" :class="{ invalid: usernameInvalid }">
            <input
              id="username"
              v-model.trim="username"
              type="text"
              class="input"
              placeholder=" "
              autocomplete="username"
              @blur="onBlur('username')"
              :aria-invalid="usernameInvalid ? 'true' : 'false'"
            />
            <label class="floating-label" for="username">账号</label>
          </div>
          <div v-if="usernameInvalid" class="field-feedback">请输入账号</div>
        </div>

        <div class="field">
          <div class="field-control" :class="{ invalid: passwordInvalid }">
            <input
              id="password"
              v-model.trim="password"
              type="password"
              class="input"
              placeholder=" "
              autocomplete="current-password"
              @blur="onBlur('password')"
              :aria-invalid="passwordInvalid ? 'true' : 'false'"
            />
            <label class="floating-label" for="password">密码</label>
          </div>
          <div v-if="passwordInvalid" class="field-feedback">请输入密码</div>
        </div>

        <div class="form-actions">
          <button type="submit" class="btn primary" :disabled="loading">{{ loading ? '登录中...' : '登录' }}</button>
        </div>
        <div v-if="error" class="error" role="alert">{{ error }}</div>
      </form>
    </div>
  </div>
</template>

<script>
import { ref, onMounted, computed } from 'vue'

export default {
  name: '用户登录页面',
  setup() {
    const username = ref('')
    const password = ref('')
    const loading = ref(false)
    const error = ref('')
    const touched = ref({ username: false, password: false })
    const onBlur = (field) => { touched.value[field] = true }
    const usernameInvalid = computed(() => touched.value.username && !username.value)
    const passwordInvalid = computed(() => touched.value.password && !password.value)
    const bgSrc = ref('/background.mp4')
    const bgVideoEl = ref(null)
    const onVideoReady = () => {
      try {
        const el = bgVideoEl.value
        if (!el) return
        el.muted = true
        el.play().catch(() => {})
      } catch (_) {}
    }
    const onVideoError = () => {
      try { bgSrc.value = '/07分离/背景.mp4' } catch (_) {}
    }

    const onSubmit = async () => {
      error.value = ''
      if (!username.value || !password.value) {
        error.value = '请输入账号与密码'
        return
      }
      loading.value = true
      try {
        const res = await fetch('/api/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ username: username.value, password: password.value })
        })
        const json = await res.json()
        if (!json?.success) {
          throw new Error(json?.error || '登录失败')
        }
        try {
          const meRes = await fetch('/api/auth/me', { credentials: 'include' })
          const meJson = await meRes.json()
          const u = meJson?.user || {}
          const payload = {
            name: u.name || u.username || '',
            username: u.username || u.name || '',
            email: u.email || '',
            avatar_url: u.avatar_url || ''
          }
          sessionStorage.setItem('current_user', JSON.stringify(payload))
        } catch (_) {}
        const ensureAuthenticated = async () => {
          for (let i = 0; i < 5; i++) {
            try {
              const r = await fetch('/api/auth/me', { credentials: 'include' })
              const j = await r.json()
              if (j && j.authenticated) return true
            } catch (e) {}
            await new Promise(resolve => setTimeout(resolve, 200))
          }
          return false
        }
        await ensureAuthenticated()
        // 跳过首次登录强制改密提示：直接进入后续界面
        // 原逻辑为弹窗提示后再继续，这里按需求绕过
        const qp = new URLSearchParams(window.location.search)
        const redirect = qp.get('redirect') || '/upload'
        const origin = qp.get('origin')
        if (window?.$vueRouter) {
          window.$vueRouter.replace(redirect)
        } else {
          window.location.href = redirect
        }
      } catch (e) {
        error.value = e.message || '登录失败'
      } finally {
        loading.value = false
      }
    }

    onMounted(() => {
      try { document.getElementById('username')?.focus() } catch (e) {}
    })

    return { username, password, loading, error, onSubmit, onBlur, usernameInvalid, passwordInvalid, bgSrc, bgVideoEl, onVideoReady, onVideoError }
  }
}
</script>

<style scoped>
.login-wrap {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}
.bg-video {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  z-index: 0;
  pointer-events: none;
}
.bg-mask {
  position: absolute;
  inset: 0;
  background: rgba(17,24,39,0.35);
  backdrop-filter: blur(2px);
  z-index: 1;
}
.login-card {
  width: min(92vw, 440px);
  border-radius: 16px;
  background: #ffffffee;
  backdrop-filter: saturate(180%) blur(6px);
  box-shadow: 0 12px 36px rgba(17,24,39,0.18);
  border: 1px solid #eef2f7;
  z-index: 2;
}
.login-header { padding: 24px 24px 8px 24px; text-align: center; border-bottom: 1px solid #f1f3f5; }
.login-title { margin: 0; font-size: 22px; font-weight: 700; color: #111827; letter-spacing: .2px; }
.login-desc { margin: 8px 0 0 0; font-size: 13px; color: #6b7280; }

.login-form { padding: 16px 24px 24px 24px; display: grid; grid-template-columns: 1fr; gap: 16px; }
.field { display: grid; gap: 8px; }
.field-control {
  position: relative;
  height: 48px;
}
.input {
  width: 100%;
  height: 48px;
  padding: 8px 16px 0 16px;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  outline: none;
  font-size: 14px;
  background: #fff;
  transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}
.input:focus {
  border-color: #6366f1;
  box-shadow: 0 0 0 3px rgba(99,102,241,.2);
}
.floating-label {
  position: absolute;
  left: 16px;
  top: 50%;
  transform: translateY(-50%);
  background: transparent;
  color: #6b7280;
  font-size: 14px;
  pointer-events: none;
  transition: all .18s ease;
}
.input:focus + .floating-label,
.input:not(:placeholder-shown) + .floating-label {
  top: 8px;
  font-size: 12px;
  color: #4f46e5;
}
.field-control.invalid .input { border-color: #fca5a5; box-shadow: 0 0 0 3px rgba(239,68,68,.15); }
.field-feedback { color: #b91c1c; font-size: 12px; padding: 0 8px; }

.form-actions { display: grid; grid-auto-flow: column; gap: 8px; justify-content: center; }
.btn {
  padding: 16px 16px;
  border-radius: 16px;
  border: 1px solid #e5e7eb;
  background: #fff;
  color: #111827;
  cursor: pointer;
  transition: background-color .18s ease, transform .1s ease, box-shadow .18s ease, border-color .18s ease;
  will-change: transform;
}
.btn.primary { background: #4f46e5; color: #fff; border-color: #4f46e5; box-shadow: 0 6px 16px rgba(79,70,229,.25); }
.btn.primary:hover { background: #4338ca; border-color: #4338ca; transform: translateY(-1px); box-shadow: 0 8px 18px rgba(67,56,202,.28); }
.btn.primary:active { transform: translateY(0); box-shadow: 0 4px 12px rgba(67,56,202,.22); }
.btn[disabled] { opacity: .6; cursor: not-allowed; box-shadow: none; transform: none; }

.error { margin-top: -8px; color: #b91c1c; font-size: 12px; text-align: center; }

@media (max-width: 420px) {
  .login-card { width: min(96vw, 420px); border-radius: 14px; }
  .login-header { padding: 20px 20px 8px 20px; }
  .login-form { padding: 16px 20px 20px 20px; gap: 16px; }
}
@media (prefers-reduced-motion: reduce) {
  * { transition: none !important; }
}
</style>
    const bgSrc = computed(() => {
      try {
        const base = (import.meta && import.meta.env && import.meta.env.BASE_URL) || '/'
        return base.replace(/\/$/, '') + '/07分离/背景.mp4'
      } catch (_) { return '/07分离/背景.mp4' }
    })
