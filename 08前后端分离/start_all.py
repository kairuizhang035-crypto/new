#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = BASE_DIR / 'backend'
FRONTEND_DIR = BASE_DIR / 'frontend'

COLOR_Y = '\033[1;33m'
COLOR_G = '\033[1;32m'
COLOR_R = '\033[1;31m'
COLOR_N = '\033[0m'


def log(msg: str):
    print(f"{COLOR_Y}[dev]{COLOR_N} {msg}")


def ok(msg: str):
    print(f"{COLOR_G}[ok]{COLOR_N} {msg}")


def warn(msg: str):
    print(f"{COLOR_R}[warn]{COLOR_N} {msg}")


def is_port_in_use(port: int) -> bool:
    # Prefer lsof, fallback to ss
    try:
        res = subprocess.run(["bash", "-lc", f"lsof -i :{port} -sTCP:LISTEN"],
                             capture_output=True, text=True)
        return res.returncode == 0 and res.stdout.strip() != ""
    except Exception:
        pass
    try:
        res = subprocess.run(["bash", "-lc", "ss -ltn | awk '{print $4}'"],
                             capture_output=True, text=True)
        return any(line.endswith(f":{port}") for line in res.stdout.splitlines())
    except Exception:
        return False


def run_cmd(cmd: str, cwd: Path | None = None, blocking: bool = False):
    if cwd is None:
        cwd = BASE_DIR
    log(f"运行命令: {cmd} (cwd={cwd})")
    if blocking:
        return subprocess.run(cmd, shell=True, cwd=str(cwd))
    else:
        # Use nohup for background
        return subprocess.Popen(f"nohup {cmd} > /dev/null 2>&1 &", shell=True, cwd=str(cwd))


def start_backend():
    log("后端：检查虚拟环境与依赖...")
    conda_prefix = os.environ.get('CONDA_PREFIX', '')
    use_conda = os.path.exists(os.path.join(conda_prefix, 'bin', 'python'))
    venv_dir = BACKEND_DIR / 'venv'
    if use_conda:
        python_bin = Path(conda_prefix) / 'bin' / 'python'
        pip_bin = Path(conda_prefix) / 'bin' / 'pip'
    else:
        if not venv_dir.exists():
            log("后端：创建Python虚拟环境...")
            subprocess.run([sys.executable, '-m', 'venv', str(venv_dir)], check=True)
        python_bin = venv_dir / 'bin' / 'python'
        pip_bin = venv_dir / 'bin' / 'pip'

    log("后端：安装依赖（requirements.txt）...")
    subprocess.run([str(pip_bin), 'install', '-r', 'requirements.txt'], cwd=str(BACKEND_DIR), check=True)

    if is_port_in_use(5000):
        ok("后端已在运行：http://localhost:5000")
        return

    env = os.environ.copy()
    env['FLASK_APP'] = 'app.py'
    env['FLASK_ENV'] = 'development'
    env['FLASK_DEBUG'] = '1'

    log("后端：启动 Flask (http://localhost:5000)...")
    # Start app.py using venv python, redirect logs
    log_file = BACKEND_DIR / 'server.log'
    with open(log_file, 'ab') as f:
        subprocess.Popen([str(python_bin), 'app.py'], cwd=str(BACKEND_DIR), env=env, stdout=f, stderr=f)
    # wait a bit for server to bind
    for _ in range(20):
        if is_port_in_use(5000):
            ok(f"后端启动成功；日志：{log_file}")
            return
        time.sleep(0.2)
    warn("后端启动可能失败，请检查 backend/server.log")


def pick_free_port(ranges: list[tuple[int, int]]) -> int:
    for start, end in ranges:
        for p in range(start, end + 1):
            if not is_port_in_use(p):
                return p
    return -1


def start_frontend() -> int:
    log("前端：检查依赖...")
    if not (FRONTEND_DIR / 'node_modules').exists():
        log("前端：安装Node.js依赖...")
        subprocess.run(['bash', '-lc', 'npm install --no-audit --no-fund'], cwd=str(FRONTEND_DIR), check=True)

    port = pick_free_port([(8080, 8090), (8000, 8010)])
    if port == -1:
        warn("8080-8090 与 8000-8010 均占用，改用 3000。")
        port = 3000

    log(f"前端：启动 Vite 开发服务器 (http://localhost:{port})...")
    log_file = FRONTEND_DIR / 'server.log'
    with open(log_file, 'ab') as f:
        subprocess.Popen(['bash', '-lc', f'npm run dev -- --port {port}'], cwd=str(FRONTEND_DIR), stdout=f, stderr=f)

    # wait for server to be ready
    for _ in range(40):
        if is_port_in_use(port):
            ok(f"前端启动成功；地址：http://localhost:{port}；日志：{log_file}")
            return port
        time.sleep(0.25)
    warn("前端启动可能失败，请检查 frontend/server.log")
    return port


def main():
    print("\n=== 一键启动：后端 + 前端 ===\n")
    # sanity checks
    if not BACKEND_DIR.exists() or not (BACKEND_DIR / 'app.py').exists():
        warn("后端目录或 app.py 不存在，请检查 08前后端分离/backend")
        sys.exit(1)
    if not FRONTEND_DIR.exists() or not (FRONTEND_DIR / 'package.json').exists():
        warn("前端目录或 package.json 不存在，请检查 08前后端分离/frontend")
        sys.exit(1)

    start_backend()
    port = start_frontend()
    print("\n=== 启动完成 ===")
    print(f"前端：http://localhost:{port}")
    print("后端：http://localhost:5000")
    print("日志：backend/server.log, frontend/server.log")
    print("(脚本退出不影响服务运行；如需停止，可手动结束对应进程或使用端口工具。)")


if __name__ == '__main__':
    main()
