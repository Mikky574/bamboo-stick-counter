"""Exercise owned inference subprocesses without loading any model or CUDA.

All processes created here are disposable Python helpers. A separate sentinel
stands in for another project and must survive every worker cleanup.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from inference_process import ResidentWorker


HELPER = r'''
import json, os, subprocess, sys, time
from pathlib import Path
for line in sys.stdin:
    job = json.loads(line)
    mode = job.get('mode', 'ok')
    if mode == 'hang':
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        Path(job['pids']).write_text(json.dumps([os.getpid(), child.pid]))
        time.sleep(120)
    elif mode == 'exit':
        sys.exit(3)
    elif mode == 'invalid':
        print('not JSON', flush=True)
    elif mode == 'failed':
        print(json.dumps({'ok': False, 'error': 'controlled failure'}), flush=True)
    else:
        time.sleep(job.get('delay', 0))
        print(json.dumps({'ok': True}), flush=True)
'''


def pid_running(pid):
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class WorkerTimeoutChecks(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='owned_worker_test_')
        self.root = Path(self.folder.name)
        self.script = self.root/'helper.py'
        self.script.write_text(HELPER, encoding='utf-8')
        self.log = self.root/'actual_worker.log'
        self.worker = ResidentWorker(idle_timeout_seconds=5)
        self.sentinel = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(120)'],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def tearDown(self):
        try:
            self.worker.close()
            self.assertIsNone(self.sentinel.poll(), 'Worker cleanup touched an unrelated process')
        finally:
            if self.sentinel.poll() is None:
                self.sentinel.terminate()
            self.sentinel.wait(timeout=5)
            self.folder.cleanup()

    def request(self, payload, timeout=2):
        return self.worker.request(
            [sys.executable, str(self.script)], payload,
            cwd=self.root, env=os.environ.copy(), log_path=self.log,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def assert_clean_and_recover(self):
        self.assertIsNone(self.worker._process)
        self.assertIsNone(self.worker._responses)
        self.assertIsNone(self.worker._log)
        self.assertFalse(self.worker.snapshot()['model_resident'])
        self.assertFalse(self.request({'mode': 'ok'}), 'Recovery must start a fresh worker')
        self.assertTrue(self.request({'mode': 'ok'}), 'A healthy worker should be reusable')

    @unittest.skipUnless(os.name == 'nt', 'Windows owned-process-tree cleanup regression')
    def test_timeout_kills_only_owned_process_tree_and_recovers(self):
        pids_file = self.root/'pids.json'
        started = time.perf_counter()
        with self.assertRaisesRegex(RuntimeError, r'超过 1 秒'):
            self.request({'mode': 'hang', 'pids': str(pids_file)}, timeout=1)
        self.assertLess(time.perf_counter()-started, 8)
        pids = json.loads(pids_file.read_text())
        self.assertEqual(len(pids), 2)
        self.assertTrue(all(not pid_running(pid) for pid in pids), pids)
        self.assertIsNone(self.sentinel.poll())
        self.assert_clean_and_recover()

    def test_unexpected_exit_uses_actual_log_and_recovers(self):
        with self.assertRaises(RuntimeError) as raised:
            self.request({'mode': 'exit'})
        self.assertIn(str(self.log), str(raised.exception))
        self.assertNotIn('countgdpp_server.log', str(raised.exception))
        self.assert_clean_and_recover()

    def test_invalid_json_releases_worker_and_recovers(self):
        with self.assertRaisesRegex(RuntimeError, '无效结果'):
            self.request({'mode': 'invalid'})
        self.assert_clean_and_recover()

    def test_worker_reported_failure_releases_worker_and_recovers(self):
        with self.assertRaisesRegex(RuntimeError, 'controlled failure'):
            self.request({'mode': 'failed'})
        self.assert_clean_and_recover()

    def test_busy_rejects_second_request_and_release_without_waiting(self):
        result = []
        started = threading.Event()
        original = self.worker._read_stdout

        def read_stdout(stream, responses):
            started.set()
            original(stream, responses)

        def run():
            try:
                result.append(self.request({'delay': .4}))
            except Exception as error:
                result.append(error)

        with patch.object(self.worker, '_read_stdout', side_effect=read_stdout):
            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(started.wait(2))
            before = time.perf_counter()
            with self.assertRaisesRegex(RuntimeError, '已有识别任务'):
                self.request({})
            with self.assertRaisesRegex(RuntimeError, '识别任务正在运行'):
                self.worker.release()
            self.assertLess(time.perf_counter()-before, .25)
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [False])
        self.assertTrue(self.request({}))

    def test_idle_expiry_does_not_interrupt_a_busy_request(self):
        self.worker.idle_timeout_seconds = .1
        self.assertFalse(self.request({'delay': .3}))
        self.assertTrue(self.request({'delay': .3}))
        deadline = time.monotonic()+3
        while self.worker._process is not None and time.monotonic() < deadline:
            time.sleep(.025)
        self.assertIsNone(self.worker._process)
        self.assertFalse(self.worker.snapshot()['model_resident'])
        self.assertFalse(self.request({}))


if __name__ == '__main__':
    unittest.main()
