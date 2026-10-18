"""Windows process-ownership acceptance tests; no model weights or GPU used.

Each test installs the real controller in a temporary checkout alongside a
minimal HTTP app and a model worker fixture. The unrelated sentinel runs
outside the controller's Job Object and must survive every stop/failure.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest


APP = '''from flask import Flask,jsonify
app=Flask(__name__)
@app.get('/api/service-status')
def service_status():
    return jsonify(app.config['LOCAL_SERVICE_STATUS']())
'''
MODELS = '''import os,subprocess,sys,time
from pathlib import Path
worker=None
def prepare_models():
    global worker
    worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'])
    Path('outputs/fixture_worker.pid').write_text(str(worker.pid))
    time.sleep(float(os.environ.get('FIXTURE_DELAY','0.1')))
    return dict(state='degraded' if os.environ.get('FIXTURE_FAIL') else 'ready',models={})
def shutdown_models():
    if worker and worker.poll() is None:
        worker.terminate()
        worker.wait(timeout=5)
    return []
'''


@unittest.skipUnless(os.name == 'nt', 'Windows Job Object integration tests')
class ServiceControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='bamboo_service_')
        self.root = Path(self.directory.name)
        shutil.copyfile(Path(__file__).with_name('service_control.py'), self.root/'service_control.py')
        (self.root/'app.py').write_text(APP, encoding='utf-8')
        (self.root/'model_startup.py').write_text(MODELS, encoding='utf-8')
        self.env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            self.port = listener.getsockname()[1]
        self.sentinel = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'],
                                          creationflags=subprocess.CREATE_NO_WINDOW)
        self.background = []

    def tearDown(self):
        try:
            self.call('stop', timeout=1)
            for process in self.background:
                process.wait(timeout=20)
            self.assertIsNone(self.sentinel.poll(), 'Unrelated process must never be terminated')
        finally:
            self.sentinel.terminate()
            self.sentinel.wait(timeout=5)
            self.directory.cleanup()

    def command(self, action, timeout=15):
        return [sys.executable, str(self.root/'service_control.py'), action,
                '--port', str(self.port), '--timeout', str(timeout)]

    def call(self, action, timeout=15):
        return subprocess.run(self.command(action, timeout), cwd=self.root, env=self.env,
                              capture_output=True, text=True, encoding='utf-8', timeout=40,
                              creationflags=subprocess.CREATE_NO_WINDOW)

    def state(self):
        try:
            return json.loads((self.root/'outputs/service.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return None

    def worker_alive(self):
        pid = int((self.root/'outputs/fixture_worker.pid').read_text())
        # Import the real low-level liveness helper without touching production state.
        from service_control import _alive
        return _alive(pid)

    def test_start_ready_repeat_and_complete_stop(self):
        first = self.call('start')
        self.assertEqual(first.returncode, 0, first.stderr)
        first_state = json.loads(first.stdout)
        self.assertEqual(first_state['state'], 'ready')
        self.assertTrue(self.worker_alive())
        second = self.call('start')
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(json.loads(second.stdout)['pid'], first_state['pid'])
        self.assertTrue(json.loads(second.stdout)['already_running'])
        stopped = self.call('stop')
        self.assertEqual(stopped.returncode, 0, stopped.stderr)
        self.assertFalse(self.worker_alive())
        self.assertFalse(json.loads(self.call('status').stdout)['ready'])
        self.assertEqual(self.call('stop').returncode, 0)

    def test_foreign_listener_is_not_touched(self):
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', self.port))
            listener.listen()
            result = self.call('start')
            self.assertEqual(result.returncode, 1)
            self.assertIsNone(self.state())
            self.assertEqual(self.call('stop').returncode, 0)
            self.assertGreater(listener.fileno(), 0)

    def test_stale_pid_cannot_target_other_process(self):
        from service_control import _process
        identity = _process(self.sentinel.pid)
        (self.root/'outputs').mkdir()
        forged = {**identity, 'root': str(self.root), 'token': 'f'*32,
                  'port': self.port, 'state': 'ready'}
        (self.root/'outputs/service.json').write_text(json.dumps(forged), encoding='utf-8')
        result = self.call('stop', timeout=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('没有终止任何进程', json.loads(result.stdout)['message'])
        self.assertIsNone(self.sentinel.poll())

    def test_preload_timeout_reaps_owned_worker(self):
        self.env['FIXTURE_DELAY'] = '120'
        result = self.call('start', timeout=4)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(self.state()['state'], 'stopped')
        self.assertFalse(self.worker_alive())
        self.assertIsNone(self.sentinel.poll())

    def test_stop_during_preload(self):
        self.env['FIXTURE_DELAY'] = '120'
        output = (self.root/'controller.log').open('w', encoding='utf-8')
        try:
            process = subprocess.Popen(self.command('start', timeout=25), cwd=self.root, env=self.env,
                                       stdout=output, stderr=output, creationflags=subprocess.CREATE_NO_WINDOW)
            self.background.append(process)
            deadline = time.monotonic()+15
            while time.monotonic() < deadline and not (self.root/'outputs/fixture_worker.pid').exists():
                time.sleep(.1)
            self.assertTrue((self.root/'outputs/fixture_worker.pid').exists())
            stopped = self.call('stop', timeout=1)
            self.assertEqual(stopped.returncode, 0, stopped.stderr)
            self.assertFalse(self.worker_alive())
            process.wait(timeout=15)
            self.assertEqual(process.returncode, 1)
        finally:
            output.close()

    def test_incomplete_model_warmup_is_not_reported_ready(self):
        self.env['FIXTURE_FAIL'] = '1'
        result = self.call('start')
        self.assertEqual(result.returncode, 1)
        self.assertIn(self.state()['state'], {'failed', 'stopped'})
        self.assertFalse(self.worker_alive())


if __name__ == '__main__':
    unittest.main()
