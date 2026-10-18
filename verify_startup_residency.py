"""Real helper processes verify permanent residency without models or CUDA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from inference_process import ResidentWorker
from verify_worker_timeout import HELPER, pid_running


class PermanentWorkerResidency(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='resident_worker_test_')
        self.root = Path(self.folder.name)
        self.script = self.root / 'helper.py'
        self.script.write_text(HELPER, encoding='utf-8')
        self.log = self.root / 'worker.log'
        self.worker = ResidentWorker(idle_timeout_seconds=None)
        self.sentinel = subprocess.Popen(
            [sys.executable, '-c', 'import time; time.sleep(120)'],
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def tearDown(self):
        try:
            self.worker.close()
            self.assertIsNone(self.sentinel.poll(),
                              'Residency cleanup touched another project')
        finally:
            if self.sentinel.poll() is None:
                self.sentinel.terminate()
            self.sentinel.wait(timeout=5)
            self.folder.cleanup()

    def request(self, payload=None, timeout=2):
        return self.worker.request(
            [sys.executable, str(self.script)], payload or {},
            cwd=self.root, env=os.environ.copy(), log_path=self.log,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def test_disabled_idle_expiry_never_schedules_timer_and_reuses_process(self):
        # A real child and response thread run here. Patching Timer catches even
        # a scheduled timer that has not expired by the time assertions run.
        with patch('inference_process.threading.Timer',
                   side_effect=AssertionError('Permanent workers must not expire')) as timer:
            self.assertFalse(self.request())
            owned = self.worker._process
            self.assertIsNotNone(owned)
            self.assertIsNone(self.worker._timer)
            self.assertTrue(self.worker.snapshot()['model_resident'])
            self.assertIsNone(self.worker.snapshot()['idle_timeout_seconds'])
            time.sleep(.2)
            self.assertIsNone(owned.poll())
            self.assertTrue(self.request())
            self.assertIs(self.worker._process, owned)
            self.assertIsNone(self.worker._timer)
            timer.assert_not_called()

    def test_explicit_release_still_releases_permanent_process_and_can_reload(self):
        self.assertFalse(self.request())
        owned = self.worker._process
        self.assertTrue(self.worker.release())
        self.assertIsNotNone(owned.poll())
        self.assertIsNone(self.worker._process)
        self.assertIsNone(self.worker._responses)
        self.assertIsNone(self.worker._log)
        self.assertIsNone(self.worker._timer)
        self.assertFalse(self.worker.snapshot()['model_resident'])
        self.assertIsNone(self.sentinel.poll())
        self.assertFalse(self.worker.release())
        self.assertFalse(self.request())
        self.assertIsNot(self.worker._process, owned)
        self.assertTrue(self.request())

    @unittest.skipUnless(os.name == 'nt', 'Windows owned-process-tree regression')
    def test_timeout_still_releases_owned_tree_without_touching_other_project(self):
        pids_file = self.root / 'pids.json'
        started = time.perf_counter()
        with self.assertRaisesRegex(RuntimeError, r'超过 1 秒'):
            self.request({'mode': 'hang', 'pids': str(pids_file)}, timeout=1)
        self.assertLess(time.perf_counter() - started, 8)
        owned_pids = json.loads(pids_file.read_text())
        self.assertEqual(len(owned_pids), 2)
        self.assertTrue(all(not pid_running(pid) for pid in owned_pids), owned_pids)
        self.assertIsNone(self.sentinel.poll())
        self.assertIsNone(self.worker._process)
        self.assertIsNone(self.worker._responses)
        self.assertIsNone(self.worker._log)
        self.assertIsNone(self.worker._timer)
        self.assertFalse(self.worker.snapshot()['model_resident'])
        self.assertFalse(self.request())
        self.assertTrue(self.request())


if __name__ == '__main__':
    unittest.main()
