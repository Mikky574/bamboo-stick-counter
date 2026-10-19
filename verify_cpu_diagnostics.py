"""CPU diagnostic behavior using tiny owned processes, no model weights/GPU."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.diagnose_cpu import cpu_environment, inference_status, run_logged


class CPUDiagnosticContracts(unittest.TestCase):
    def test_no_initial_exemplars_means_forward_completed(self):
        result = dict(count=0, points=[], meta=dict(device='cpu', prediction_valid=False,
                                                   diagnosis='no_initial_exemplars'))
        self.assertEqual(inference_status(result), 'executed_no_initial_exemplars')

    def test_wrong_device_cannot_pass_cpu_probe(self):
        with self.assertRaises(RuntimeError):
            inference_status(dict(meta=dict(device='cuda', prediction_valid=True)))

    def test_cpu_environment_is_child_only(self):
        original = os.environ.get('CUDA_VISIBLE_DEVICES')
        self.assertEqual(cpu_environment()['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), original)

    def test_missing_interpreter_is_launch_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = run_logged([str(root/'missing_python.exe')], root, root/'launch.log', 2)
            self.assertFalse(result['launched'])
            self.assertIsNone(result['returncode'])

    def test_import_failure_preserves_actual_stderr(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = run_logged([sys.executable, '-c', 'import missing_cpu_probe_package'],
                                root, root/'import.log', 5)
            self.assertTrue(result['launched'])
            self.assertNotEqual(result['returncode'], 0)
            self.assertFalse(result['timed_out'])
            self.assertIn('ModuleNotFoundError', (root/'import.log').read_text(encoding='utf-8'))

    def test_timeout_closes_only_owned_probe(self):
        flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        sentinel = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], creationflags=flags)
        try:
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                result = run_logged([sys.executable, '-c', 'import time;time.sleep(30)'],
                                    root, root/'timeout.log', .2)
                self.assertTrue(result['timed_out'])
                self.assertIsNotNone(result['returncode'])
                self.assertIsNone(sentinel.poll())
        finally:
            sentinel.terminate()
            sentinel.wait(timeout=5)

    def test_old_success_files_cannot_hide_failed_current_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root/'reports'
            output.mkdir()
            (output/'environment_result.json').write_text(json.dumps(dict(ok=True)), encoding='utf-8')
            result = subprocess.run([sys.executable, str(Path(__file__).parent/'tools/diagnose_cpu.py'),
                '--project-root', str(root), '--out-dir', str(output), '--python', str(root/'missing.exe')],
                capture_output=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            reports = list(output.glob('*/diagnostics.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text(encoding='utf-8'))
            self.assertEqual(report['status'], 'environment_failed')


if __name__ == '__main__':
    unittest.main()
