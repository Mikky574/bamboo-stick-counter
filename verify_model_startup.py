"""Startup contract checks without weights or GPU allocation."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import auto_counter
import model_startup
import scene_classifier


class StartupContracts(unittest.TestCase):
    def test_both_preparations_finish_and_are_recorded_before_server_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'outputs').mkdir()
            import app as web
            def serve(**settings):
                status = json.loads((root/'outputs/startup_models.json').read_text(encoding='utf-8'))
                self.assertEqual(status['state'], 'ready')
                self.assertTrue(status['models']['counting']['ready'])
                self.assertTrue(status['models']['scene']['ready'])
                self.assertFalse(settings['debug'])
            with patch('auto_counter.prewarm_model', return_value={'ready': True}) as counting, \
                 patch('scene_classifier.prewarm_scene_model', return_value={'ready': True}) as scene, \
                 patch('auto_counter.release_model') as release_gpu, \
                 patch('scene_classifier.release_scene_model') as release_cpu, \
                 patch.object(web, 'ROOT', root), patch.object(web.app, 'run', side_effect=serve), \
                 patch('sys.stdout', new=io.StringIO()):
                web.start_server()
            counting.assert_called_once()
            scene.assert_called_once()
            release_gpu.assert_called_once()
            release_cpu.assert_called_once()

    def test_gpu_unavailable_is_recorded_without_falsely_claiming_ready(self):
        with patch('auto_counter.prewarm_model', side_effect=RuntimeError('GPU unavailable')), \
             patch('scene_classifier.prewarm_scene_model', return_value={'ready': True}):
            report = model_startup.prepare_models()
        self.assertEqual(report['state'], 'degraded')
        self.assertFalse(report['models']['counting']['ready'])
        self.assertIn('GPU unavailable', report['models']['counting']['error'])
        self.assertTrue(report['models']['scene']['ready'])

    def test_gpu_warmup_protocol_has_no_image_and_reuses_owned_session(self):
        def respond(command, payload, **settings):
            self.assertEqual(payload['action'], 'warmup')
            self.assertNotIn('image', payload)
            self.assertEqual(command[-3:], ['--serve', '--device', 'cuda'])
            self.assertEqual(settings['timeout'], 60)
            Path(payload['output']).write_text(json.dumps({'ready': True,
                'meta': {'device': 'cuda', 'model_load_seconds': 0}}), encoding='utf-8')
            return True
        with patch('auto_counter.readiness', return_value={'installed': True}), \
             patch.object(auto_counter._warm_worker, 'request', side_effect=respond):
            result = auto_counter.prewarm_model()
        self.assertTrue(result['ready'])
        self.assertTrue(result['meta']['cache_reused'])
        self.assertFalse(auto_counter._job_lock.locked())

    def test_invalid_gpu_warmup_releases_session_and_unlocks(self):
        def respond(command, payload, **settings):
            Path(payload['output']).write_text(json.dumps({'ready': False}), encoding='utf-8')
            return False
        with patch('auto_counter.readiness', return_value={'installed': True}), \
             patch.object(auto_counter._warm_worker, 'request', side_effect=respond), \
             patch.object(auto_counter._warm_worker, 'release') as release:
            with self.assertRaisesRegex(RuntimeError, '预热'):
                auto_counter.prewarm_model()
        release.assert_called_once()
        self.assertFalse(auto_counter._job_lock.locked())

    def test_cpu_warmup_protocol_has_no_user_image_or_count(self):
        def respond(command, payload, **settings):
            self.assertEqual(payload['action'], 'warmup')
            self.assertNotIn('image', payload)
            self.assertEqual(settings['env']['CUDA_VISIBLE_DEVICES'], '-1')
            self.assertEqual(settings['timeout'], 30)
            Path(payload['output']).write_text(json.dumps({'ready': True,
                'device': 'cpu', 'model_load_seconds': 0}), encoding='utf-8')
            return True
        with patch('scene_classifier.readiness', return_value={'installed': True}), \
             patch.object(scene_classifier._WORKER, 'request', side_effect=respond):
            result = scene_classifier.prewarm_scene_model()
        self.assertTrue(result['ready'])
        self.assertTrue(result['meta']['cache_reused'])
        self.assertNotIn('count', result)
        self.assertFalse(scene_classifier._LOCK.locked())

    def test_cpu_synthetic_model_warmup_runs_both_paths_without_returning_a_count(self):
        import torch
        import countgdpp_worker
        calls = []
        def model(first, second, boxes, targets, labels, captions):
            self.assertIs(first, second)
            self.assertEqual(captions, ['rice grain .'])
            calls.append(len(boxes[0]))
            return {}
        runtime = dict(model=model, transform=lambda image, target: (torch.zeros(3,800,800),target),
                       nested_tensor=lambda tensor: tensor)
        with patch.object(countgdpp_worker, 'load_runtime', return_value=(runtime, 0)), \
             patch.object(torch, 'load', side_effect=AssertionError('No weights')), \
             patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('No CUDA')):
            result = countgdpp_worker.warmup_runtime('cpu')
        self.assertEqual(calls, [0, 1])
        self.assertTrue(result['ready'])
        self.assertEqual(result['meta']['warmup_forward_passes'], 2)
        self.assertNotIn('count', result)
        self.assertNotIn('points', result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
