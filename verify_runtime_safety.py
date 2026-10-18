"""Interactive resource/error safeguards; no neural model or GPU allocation."""
import base64
import io
import json
import logging
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import auto_counter
import smart_counter
from app import app, _request_log


IMAGE = np.indices((48, 64)).sum(axis=0).astype(np.uint8)
IMAGE = np.repeat((IMAGE * 2)[:, :, None], 3, axis=2)


def image_request():
    encoded = cv2.imencode('.png', IMAGE)[1]
    return {'image': 'data:image/png;base64,' + base64.b64encode(encoded).decode()}


class InteractiveRuntimeSafety(unittest.TestCase):
    def test_gpu_threshold_and_failed_query_never_silently_select_cpu(self):
        with patch('auto_counter.readiness', return_value={'model_resident': False}):
            with patch('smart_counter.shutil.which', return_value='nvidia-smi'):
                for memory, device in ((4095, None), (4096, 'cuda'), (7826, 'cuda')):
                    with self.subTest(memory=memory):
                        completed = subprocess.CompletedProcess([], 0, str(memory)+'\n', '')
                        with patch('smart_counter.subprocess.run', return_value=completed):
                            self.assertEqual(smart_counter.inference_device(), device)
                with patch('smart_counter.subprocess.run', side_effect=OSError('GPU query failed')):
                    self.assertIsNone(smart_counter.inference_device())

    def test_already_loaded_owned_gpu_worker_can_be_reused(self):
        with patch('auto_counter.readiness', return_value={'model_resident': True}):
            with patch('smart_counter.subprocess.run') as probe:
                self.assertEqual(smart_counter.inference_device(), 'cuda')
                probe.assert_not_called()

    def test_worker_and_router_share_the_same_four_gib_admission_budget(self):
        from types import SimpleNamespace
        import countgdpp_worker
        from runtime_capacity import COUNTGDPP_MIN_FREE_BYTES
        self.assertEqual(COUNTGDPP_MIN_FREE_BYTES, 4096 * 1024**2)
        self.assertEqual(smart_counter.MIN_GPU_FREE_MIB * 1024**2,
                         countgdpp_worker.COUNTGDPP_MIN_FREE_BYTES)
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            mem_get_info=lambda: (COUNTGDPP_MIN_FREE_BYTES-1, 24*1024**3)))
        with patch.object(countgdpp_worker, '_RUNTIME', None):
            with self.assertRaisesRegex(RuntimeError, '4096 MiB'):
                countgdpp_worker.load_runtime(fake_torch, 'cuda')

    def test_dedicated_service_sessions_do_not_expire_while_idle(self):
        import scene_classifier
        self.assertIsNone(auto_counter._warm_worker.idle_timeout_seconds)
        self.assertIsNone(scene_classifier._WORKER.idle_timeout_seconds)

    def test_home_and_api_results_cannot_be_reused_from_browser_cache(self):
        with app.test_client() as client:
            page = client.get('/')
            self.assertIn('no-store', page.headers['Cache-Control'])
            self.assertTrue(page.headers['X-Counter-UI-Version'])
            with patch('smart_counter.detect_smart', return_value={
                    'count': None, 'points': [], 'meta': {'prediction_valid': False}}):
                response = client.post('/api/count', json=image_request())
            self.assertIn('no-store', response.headers['Cache-Control'])
            self.assertEqual(response.headers['X-Counter-UI-Version'],
                             page.headers['X-Counter-UI-Version'])

    def test_gpu_pressure_returns_retryable_result_without_starting_cpu_counter(self):
        with patch('particle_gate.detect_separated', return_value=None):
            with patch('smart_counter.scene_hint', return_value={'accepted': True, 'scene': 'cotton'}):
                with patch('smart_counter.inference_device', return_value=None):
                    with patch('auto_counter.detect_auto', side_effect=AssertionError('No CPU fallback')) as model:
                        with app.test_client() as client:
                            response = client.post('/api/count', json=image_request())
        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIsNone(result['count'])
        self.assertEqual(result['points'], [])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertTrue(result['meta']['retryable'])
        self.assertEqual(result['meta']['diagnosis'], 'gpu_resources_unavailable')
        model.assert_not_called()

    def test_failed_scene_worker_does_not_trigger_more_expensive_category_probes(self):
        with patch('particle_gate.detect_separated', return_value=None):
            with patch('scene_classifier.readiness', return_value={'installed': True}):
                with patch('scene_classifier.classify_scene', side_effect=RuntimeError('scene timed out')):
                    with patch('auto_counter.detect_auto') as model:
                        with app.test_client() as client:
                            response = client.post('/api/count', json=image_request())
        self.assertEqual(response.status_code, 503)
        self.assertIn('scene timed out', response.get_json()['error'])
        model.assert_not_called()

    def test_missing_optional_scene_model_still_allows_normal_category_selection(self):
        with patch('scene_classifier.readiness', return_value={'installed': False}):
            with patch('scene_classifier.classify_scene') as classify:
                self.assertIsNone(smart_counter.scene_hint(IMAGE))
                classify.assert_not_called()

    def test_warm_counter_has_a_short_interactive_deadline(self):
        def respond(command, payload, **settings):
            self.assertEqual(settings['timeout'], 60)
            Path(payload['output']).write_text(json.dumps({'count': 1,
                'points': [{'x': 10, 'y': 10}], 'meta': {'device': 'cuda'}}), encoding='utf-8')
            return False

        with patch.object(auto_counter._warm_worker, 'request', side_effect=respond):
            result = auto_counter._execute(IMAGE, 'countgdpp', 'cuda',
                                          {'category': 'cotton'}, keep_warm=True)
        self.assertEqual(result['count'], 1)
        self.assertFalse(auto_counter._job_lock.locked())

    def test_request_diagnostics_capture_errors_without_image_payload(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        _request_log.addHandler(handler)
        body = image_request()
        try:
            with patch('smart_counter.detect_smart', side_effect=RuntimeError('worker timeout')):
                with app.test_client() as client:
                    response = client.post('/api/count', json=body)
        finally:
            _request_log.removeHandler(handler)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([r['event'] for r in records], ['started', 'finished'])
        self.assertEqual(records[0]['request_id'], records[1]['request_id'])
        self.assertEqual(records[1]['request_id'], response.headers['X-Count-Request-ID'])
        self.assertEqual(records[1]['status'], 503)
        self.assertIn('worker timeout', records[1]['error'])
        self.assertNotIn(body['image'], stream.getvalue())
        self.assertNotIn('base64', stream.getvalue())


if __name__ == '__main__':
    unittest.main(verbosity=2)
