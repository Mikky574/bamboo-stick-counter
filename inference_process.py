"""Owned JSON-lines inference subprocess; None disables idle expiry."""
import atexit
import json
import os
import queue
import subprocess
import threading


class ResidentWorker:
    def __init__(self, idle_timeout_seconds=120):
        self.idle_timeout_seconds = idle_timeout_seconds
        self._lock = threading.RLock()
        self._process = None
        self._responses = None
        self._log = None
        self._timer = None
        self._busy = False
        self._ready = False
        atexit.register(self.close)

    def snapshot(self):
        with self._lock:
            alive = self._process is not None and self._process.poll() is None
            return dict(model_resident=alive and self._ready,
                        cache_device='cuda' if alive and self._ready else None,
                        idle_timeout_seconds=self.idle_timeout_seconds,
                        residency_policy=('until_release_or_shutdown' if self.idle_timeout_seconds is None
                                          else 'idle_expiry'))

    @staticmethod
    def _read_stdout(stream, responses):
        try:
            for line in stream:
                responses.put(('line', line))
        except (OSError, ValueError) as error:
            responses.put(('error', str(error)))
        finally:
            responses.put(('eof', None))

    def _cancel_timer_locked(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _stop_locked(self):
        self._cancel_timer_locked()
        process = self._process
        self._ready = False
        if process is not None:
            if process.poll() is None:
                if os.name == 'nt':
                    # A Windows venv python.exe may be a launcher whose child owns
                    # CUDA. Kill only our still-running root PID and its children.
                    taskkill = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'),
                                            'System32', 'taskkill.exe')
                    stopped = subprocess.run(
                        [taskkill, '/PID', str(process.pid), '/T', '/F'],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
                    if stopped.returncode and process.poll() is None:
                        raise RuntimeError('无法释放本项目推理进程树，请重试释放模型。')
                else:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        # Retain ownership so a failed cleanup can be retried;
                        # killing only the launcher could leave CUDA alive.
                        raise RuntimeError('本项目推理进程树尚未退出，请重试释放模型。')
                    else:
                        process.kill()
                        process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass
        self._process = None
        self._responses = None
        if self._log is not None:
            self._log.close()
            self._log = None
        return process is not None

    def _expire(self, process):
        with self._lock:
            if self._process is process and not self._busy:
                self._stop_locked()

    def close(self):
        """Terminate only this manager's process, including during app shutdown."""
        with self._lock:
            return self._stop_locked()

    def release(self):
        with self._lock:
            if self._busy:
                raise RuntimeError('识别任务正在运行，完成后才能释放模型。')
            return self._stop_locked()

    def request(self, command, payload, *, cwd, env, log_path, timeout=600,
                creationflags=0):
        with self._lock:
            if self._busy:
                raise RuntimeError('本项目已有识别任务正在运行，请等待完成。')
            self._cancel_timer_locked()
            reused = (self._process is not None and
                      self._process.poll() is None and self._ready)
            if not reused:
                self._stop_locked()
                self._log = open(log_path, 'a', encoding='utf-8')
                try:
                    self._process = subprocess.Popen(
                        command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, stderr=self._log, text=True,
                        encoding='utf-8', errors='replace', bufsize=1,
                        creationflags=creationflags)
                except Exception:
                    self._stop_locked()
                    raise
                self._responses = queue.Queue()
                threading.Thread(target=self._read_stdout,
                                 args=(self._process.stdout, self._responses),
                                 daemon=True).start()
            process, responses = self._process, self._responses
            self._busy = True
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
            process.stdin.flush()
            try:
                kind, line = responses.get(timeout=timeout)
            except queue.Empty as error:
                raise RuntimeError(
                    f'识别超过 {timeout:g} 秒，子进程已结束，模型资源已释放。') from error
            if kind != 'line':
                raise RuntimeError(f'常驻推理进程意外结束；详情见 {log_path}。')
            try:
                response = json.loads(line)
            except (ValueError, TypeError) as error:
                raise RuntimeError('常驻推理进程返回了无效结果，模型资源已释放。') from error
            if not isinstance(response, dict) or response.get('ok') is not True:
                detail = response.get('error', 'unknown error') if isinstance(response, dict) else 'unknown error'
                raise RuntimeError('推理失败：' + str(detail)[:350])
            with self._lock:
                self._ready = True
            return reused
        except Exception:
            with self._lock:
                self._stop_locked()
            raise
        finally:
            with self._lock:
                self._busy = False
                if (self.idle_timeout_seconds is not None and
                        self._process is process and process.poll() is None):
                    self._timer = threading.Timer(self.idle_timeout_seconds,
                                                  self._expire, args=(process,))
                    self._timer.daemon = True
                    self._timer.start()
