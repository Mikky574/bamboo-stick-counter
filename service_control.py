"""Own, start and stop this checkout's Windows web/model process group.

The manager uses a named Windows Job Object, an unpredictable instance token,
and PID creation-time/command-line checks. It never kills by image name or port.
Only the standard library is required by the controller itself.
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
STATE = OUTPUTS / "service.json"
LOCK = OUTPUTS / "service.start.lock"
TOKEN = re.compile(r"^[a-f0-9]{32}$")


def _write_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_state():
    try:
        value = json.loads(STATE.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        return value
    except (OSError, ValueError):
        return None


def _process(pid):
    """Obtain OS-owned identity; state-file PID alone is never sufficient."""
    if os.name != "nt":
        raise RuntimeError("服务控制脚本仅支持 Windows。")
    if not isinstance(pid, int) or pid <= 0:
        return None
    command = (
        "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); "
        f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
        "if($null -ne $p){[ordered]@{pid=[int]$p.ProcessId; "
        "created=$p.CreationDate.ToUniversalTime().ToString('o'); "
        "executable=$p.ExecutablePath; command_line=$p.CommandLine}|ConvertTo-Json -Compress}"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=subprocess.CREATE_NO_WINDOW, timeout=12,
    )
    if result.returncode:
        raise RuntimeError("无法验证进程归属：" + result.stderr.strip()[:300])
    return json.loads(result.stdout) if result.stdout.strip() else None


def _owned(state):
    if not state or not TOKEN.fullmatch(str(state.get("token", ""))):
        return False
    try:
        if Path(state["root"]).resolve() != ROOT:
            return False
        current = _process(state["pid"])
        return bool(current and current["created"] == state["created"]
                    and current["executable"] == state["executable"]
                    and state["token"] in current["command_line"]
                    and str(Path(__file__).resolve()).lower() in current["command_line"].lower()
                    and "_serve" in current["command_line"])
    except (KeyError, TypeError, OSError):
        return False


class WindowsJob:
    """Kernel cleanup remains effective during a blocked model preload."""

    def __init__(self, name, create=False):
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.OpenJobObjectW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
        self.kernel.OpenJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.handle = (self.kernel.CreateJobObjectW(None, name) if create else
                       self.kernel.OpenJobObjectW(0x0008 | 0x0004, False, name))
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if create:
            class BASIC(ctypes.Structure):
                _fields_ = [("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
                            ("Flags", wintypes.DWORD), ("MinWorkingSet", ctypes.c_size_t),
                            ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("Priority", wintypes.DWORD),
                            ("Scheduling", wintypes.DWORD)]

            class IO(ctypes.Structure):
                _fields_ = [(name, ctypes.c_uint64) for name in
                            ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "OtherBytes")]

            class EXTENDED(ctypes.Structure):
                _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemory", ctypes.c_size_t),
                            ("JobMemory", ctypes.c_size_t), ("PeakProcess", ctypes.c_size_t),
                            ("PeakJob", ctypes.c_size_t)]

            info = EXTENDED()
            info.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.AssignProcessToJobObject(self.handle, self.kernel.GetCurrentProcess()):
                raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self):
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _alive(pid):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
    finally:
        kernel.CloseHandle(handle)


def _probe(state):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"http://127.0.0.1:{state['port']}/api/service-status", timeout=1) as response:
            payload = json.load(response)
        return payload if payload.get("token") == state["token"] and payload.get("pid") == state["pid"] else None
    except (OSError, ValueError):
        return None


def _wait_ready(token, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = _read_state()
        if state and state.get("token") == token:
            if state.get("state") in {"failed", "stopped"}:
                raise RuntimeError(state.get("error") or "服务在就绪前已停止。")
            if not _alive(state["pid"]):
                raise RuntimeError("服务进程在就绪前退出；请检查 outputs/service.stderr.log。")
            if state.get("state") == "ready":
                probe = _probe(state)
                if probe and probe.get("ready") is True:
                    return state
        time.sleep(.2)
    raise TimeoutError(f"等待服务和模型预热超过 {timeout:g} 秒。")


def stop(timeout=15, expected_token=None):
    state = _read_state()
    if expected_token and (not state or state.get("token") != expected_token):
        return {"state": "stopped", "message": "该实例已结束。"}
    if not _owned(state):
        return {"state": "stopped", "message": "没有可验证归属的本项目服务；没有终止任何进程。"}
    signal = OUTPUTS / f"service.{state['token']}.stop"
    signal.touch()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and _alive(state["pid"]):
        time.sleep(.15)
    forced = False
    if _alive(state["pid"]):
        # Recheck creation time/command line immediately before the destructive call.
        if not _owned(state):
            raise RuntimeError("服务身份已变化，拒绝终止。")
        job = WindowsJob("Local\\BambooCounter_" + state["token"])
        try:
            job.terminate()
        finally:
            job.close()
        forced = True
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(state["pid"]):
            time.sleep(.1)
        if _alive(state["pid"]):
            raise RuntimeError("服务进程未退出，请查看日志；未触碰其他项目。")
    state.update(state="stopped", stopped_unix=time.time(), forced=forced)
    _write_json(STATE, state)
    with contextlib.suppress(OSError):
        signal.unlink()
    return {"state": "stopped", "forced": forced, "pid": state["pid"],
            "message": "网页服务与该实例所属模型子进程已停止。"}


def status():
    state = _read_state()
    if not _owned(state):
        return {"state": "stopped", "ready": False}
    probe = _probe(state) if state.get("state") == "ready" else None
    return {**state, "ready": bool(probe and probe.get("ready"))}


def start(port=5080, timeout=90):
    OUTPUTS.mkdir(exist_ok=True)
    existing = _read_state()
    if _owned(existing):
        if existing["port"] != port:
            raise RuntimeError(f"本项目已在端口 {existing['port']} 启动。")
        return {**_wait_ready(existing["token"], timeout), "already_running": True}
    try:
        lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            lock_state = json.loads(LOCK.read_text(encoding="utf-8"))
            current = _process(lock_state["pid"])
            live_lock = current and current["created"] == lock_state["created"]
        except (OSError, ValueError, KeyError, TypeError):
            live_lock = True  # A concurrent writer may not have flushed yet.
        if live_lock:
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                candidate = _read_state()
                if candidate and candidate.get("state") in {"preparing", "ready"} and _owned(candidate):
                    return {**_wait_ready(candidate["token"], max(1, deadline-time.monotonic())),
                            "already_running": True}
                if not LOCK.exists():
                    return start(port, max(1, deadline-time.monotonic()))
                time.sleep(.2)
            raise TimeoutError("另一个启动操作尚未完成，请查看 status.ps1。")
        LOCK.unlink()
        return start(port, timeout)
    with os.fdopen(lock_fd, "w", encoding="utf-8") as handle:
        json.dump(_process(os.getpid()), handle)
    token = uuid.uuid4().hex
    try:
        with socket.socket() as listener:
            # Windows SO_REUSEADDR would permit stealing a live listener; leave it off.
            listener.bind(("127.0.0.1", port))
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "_serve",
                   "--token", token, "--port", str(port)]
        with (OUTPUTS / "service.stdout.log").open("ab") as stdout, \
                (OUTPUTS / "service.stderr.log").open("ab") as stderr:
            subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                             creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
                             close_fds=True)
        try:
            return {**_wait_ready(token, timeout), "already_running": False}
        except Exception:
            stop(timeout=2, expected_token=token)
            raise
    finally:
        with contextlib.suppress(OSError):
            LOCK.unlink()


def serve(token, port):
    if not TOKEN.fullmatch(token):
        raise ValueError("服务实例 token 无效。")
    OUTPUTS.mkdir(exist_ok=True)
    job = WindowsJob("Local\\BambooCounter_" + token, create=True)
    identity = _process(os.getpid())
    state = {**identity, "root": str(ROOT), "token": token, "port": port,
             "state": "preparing", "started_unix": time.time()}
    _write_json(STATE, state)
    stop_event = threading.Event()
    signal = OUTPUTS / f"service.{token}.stop"
    httpd = None
    serving = threading.Event()

    def watch():
        while not stop_event.wait(.2):
            if signal.exists():
                stop_event.set()
                if serving.is_set():
                    httpd.shutdown()
                return

    threading.Thread(target=watch, name="service-stop", daemon=True).start()
    shutdown_models = None
    try:
        from app import app
        from model_startup import prepare_models, shutdown_models
        from werkzeug.serving import make_server
        # Bind first: model initialization must not consume GPU for a port conflict.
        httpd = make_server("127.0.0.1", port, app, threaded=True)
        app.config["LOCAL_SERVICE_STATUS"] = lambda: {
            "token": token, "pid": os.getpid(), "root": str(ROOT),
            "ready": state["state"] == "ready", "state": state["state"],
            "models": state.get("models", {}),
        }
        print("正在预加载 GPU 计数模型及 CPU 分类模型……", flush=True)
        prepared = prepare_models()
        _write_json(OUTPUTS / "startup_models.json", prepared)
        state["models"] = prepared
        if stop_event.is_set():
            return
        if prepared.get("state") != "ready":
            raise RuntimeError("模型预热未完整成功：" + json.dumps(prepared, ensure_ascii=False))
        state.update(state="ready", ready_unix=time.time())
        _write_json(STATE, state)
        print(f"服务及模型就绪：http://127.0.0.1:{port}/", flush=True)
        serving.set()
        if not stop_event.is_set():
            httpd.serve_forever(poll_interval=.2)
    except BaseException as error:
        state.update(state="failed", error=str(error)[:2500])
        _write_json(STATE, state)
        raise
    finally:
        stop_event.set()
        if httpd is not None:
            httpd.server_close()
        if shutdown_models is not None:
            for error in shutdown_models():
                print("模型释放未完成：" + error, flush=True)
        if state.get("state") != "failed":
            state.update(state="stopped", stopped_unix=time.time())
            _write_json(STATE, state)
        with contextlib.suppress(OSError):
            signal.unlink()
        # Do not close the last kill-on-close handle while still executing Python;
        # Windows closes it on process exit and also reaps any remaining workers.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status", "_serve"])
    parser.add_argument("--port", type=int, default=5080)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("服务控制脚本仅支持 Windows。")
    if not 1 <= args.port <= 65535 or args.timeout is not None and args.timeout <= 0:
        parser.error("端口须为 1–65535，超时须大于 0。")
    try:
        if args.action == "_serve":
            serve(args.token or "", args.port)
            return 0
        result = start(args.port, args.timeout or 90) if args.action == "start" else \
                 stop(args.timeout or 15) if args.action == "stop" else status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print("操作失败：" + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
