"""CPU-only mocked scene interface regressions; no model or GPU is loaded."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import scene_classifier as scene


class SceneInterfaceRegression(unittest.TestCase):
    def setUp(self):
        self.image = np.full((40, 64, 3), 125, np.uint8)
        self.enterContext(patch.object(scene, 'readiness', return_value={'installed': True}))
        self.model = self.enterContext(patch.object(
            scene, '_classify_local', side_effect=AssertionError('No real model is allowed.')))
        self.enterContext(patch.object(scene._WORKER, 'release', return_value=True))

    def response(self, accepted=True):
        return dict(scene='chopsticks' if accepted else 'unknown', accepted=accepted,
                    suggested_scene='chopsticks', family='laid_rods' if accepted else None,
                    candidates=[{'scene': 'chopsticks', 'score': .43}], margin=.04 if accepted else .01,
                    meta=dict(device='cpu', framework='TinyCLIP', routing_hint_only=True,
                              open_set_rejection_calibrated=False))

    def test_cpu_worker_contract_and_metadata(self):
        predicted = self.response()
        reuse = iter((False, True))

        def worker(command, payload, **settings):
            self.assertEqual(command, [str(scene.PYTHON), str(Path(scene.__file__).resolve()), '--serve'])
            self.assertEqual(set(payload), {'image', 'output'})
            self.assertTrue(Path(payload['image']).is_absolute())
            self.assertTrue(Path(payload['image']).is_file())
            self.assertEqual(settings['env']['CUDA_VISIBLE_DEVICES'], '-1')
            self.assertEqual(settings['env']['PYTORCH_NVML_BASED_CUDA_CHECK'], '0')
            self.assertEqual(settings['env']['PYTHONIOENCODING'], 'utf-8')
            self.assertEqual(settings['timeout'], 30)
            self.assertEqual(settings['log_path'].name, 'scene_classifier.log')
            Path(payload['output']).write_text(json.dumps(predicted), encoding='utf-8')
            return next(reuse)

        with patch.object(scene._WORKER, 'request', side_effect=worker) as request:
            first = scene.classify_scene(self.image)
            second = scene.classify_scene(self.image)
        self.assertEqual(request.call_count, 2)
        self.assertFalse(first['meta']['cache_reused'])
        self.assertTrue(second['meta']['cache_reused'])
        self.assertFalse(second['meta']['worker_exited'])
        self.assertEqual(second['meta']['device'], 'cpu')
        self.assertGreaterEqual(second['meta']['total_elapsed_seconds'], 0)
        self.assertEqual(second['scene'], 'chopsticks')
        self.assertNotIn('count', second)
        self.model.assert_not_called()

    def test_unknown_is_preserved_and_suggestion_is_not_promoted(self):
        predicted = self.response(False)

        def worker(command, payload, **settings):
            Path(payload['output']).write_text(json.dumps(predicted), encoding='utf-8')
            return False

        with patch.object(scene._WORKER, 'request', side_effect=worker):
            result = scene.classify_scene(self.image)
        self.assertEqual(result['scene'], 'unknown')
        self.assertFalse(result['accepted'])
        self.assertEqual(result['suggested_scene'], 'chopsticks')
        self.assertIsNone(result['family'])
        self.assertNotIn('count', result)

    def test_bad_inputs_never_dispatch(self):
        with patch.object(scene._WORKER, 'request') as request:
            for image in (None, np.zeros((12, 12, 3), np.uint8),
                          np.zeros((32, 32), np.uint8), np.zeros((32, 32, 3), np.float32)):
                with self.subTest(shape=getattr(image, 'shape', None)):
                    with self.assertRaises(ValueError):
                        scene.classify_scene(image)
            request.assert_not_called()

    def test_broken_worker_result_releases_owned_process(self):
        def worker(command, payload, **settings):
            Path(payload['output']).write_text('{}', encoding='utf-8')
            return False

        with patch.object(scene._WORKER, 'request', side_effect=worker):
            with self.assertRaises(RuntimeError):
                scene.classify_scene(self.image)
        scene._WORKER.release.assert_called_once_with()

    def test_cpu_snapshot_and_busy_release_guard(self):
        state = dict(model_resident=True, cache_device='cuda', idle_timeout_seconds=120)
        with patch.object(scene.ResidentWorker, 'snapshot', return_value=state):
            self.assertEqual(scene._WORKER.snapshot()['cache_device'], 'cpu')
        scene._LOCK.acquire()
        try:
            with self.assertRaises(RuntimeError):
                scene.release_scene_model()
        finally:
            scene._LOCK.release()
        scene._WORKER.release.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
