"""Positive/negative references survive API and isolated-worker transport.

Uses synthetic pixels and mocked dispatch; never loads a model or touches CUDA.
"""
import base64
import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import auto_counter
import smart_counter

POSITIVE = [dict(x=25., y=25., diameter=16.)]
NEGATIVE = [dict(x=72., y=52., diameter=22.)]


def prediction():
    return dict(count=1, points=[dict(x=25, y=25)], width=100, height=80,
                meta=dict(prediction_valid=True, review_required=True, count_is_total=False))


class NegativeAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()
        self.image = np.full((80, 100, 3), 180, np.uint8)
        ok, encoded = cv2.imencode('.png', self.image)
        self.assertTrue(ok)
        self.data_url = 'data:image/png;base64,'+base64.b64encode(encoded).decode('ascii')

    def post(self, **options):
        return self.client.post('/api/count', json=dict(image=self.data_url, **options))

    def test_smart_and_auto_forward_both_roles(self):
        for engine in ('smart', 'auto'):
            with self.subTest(engine=engine), \
                 patch.object(smart_counter, 'detect_smart', return_value=prediction()) as smart:
                response = self.post(engine=engine, exemplar_points=POSITIVE,
                                     negative_exemplar_points=NEGATIVE)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(smart.call_args.kwargs,
                             dict(exemplar_points=POSITIVE, negative_exemplar_points=NEGATIVE))

    def test_explicit_cpu_and_classified_auto_keep_both_roles(self):
        for engine in ('countgdpp', 'auto'):
            with self.subTest(engine=engine), \
                 patch.object(auto_counter, 'detect_auto', return_value=prediction()) as neural, \
                 patch.object(smart_counter, 'detect_smart') as smart:
                response = self.post(engine=engine, category='rice', device='cpu',
                                     exemplar_points=POSITIVE, negative_exemplar_points=NEGATIVE)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(neural.call_args.args[1:], ('rice', 'cpu'))
            self.assertEqual(neural.call_args.kwargs, dict(keep_warm=False,
                allow_category_recovery=False, exemplar_points=POSITIVE,
                negative_exemplar_points=NEGATIVE))
            smart.assert_not_called()

    def test_invalid_or_conflicting_negative_references_fail_before_dispatch(self):
        invalid = [None, [], {}, NEGATIVE*4, [dict(x=True, y=30, diameter=20)],
                   [dict(x=20, y=float('nan'), diameter=20)],
                   [dict(x=100, y=30, diameter=20)],
                   [dict(x=20, y=80, diameter=20)],
                   [dict(x=20, y=30, diameter=401)], NEGATIVE*2,
                   POSITIVE, [dict(x=26, y=25, diameter=30)]]
        with patch.object(smart_counter, 'detect_smart') as smart, \
             patch.object(auto_counter, 'detect_auto') as neural:
            for negative in invalid:
                with self.subTest(negative=negative):
                    response = self.post(exemplar_points=POSITIVE,
                                         negative_exemplar_points=negative)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn('error', response.get_json())
        smart.assert_not_called()
        neural.assert_not_called()

    def test_negative_only_is_rejected_instead_of_silently_running_automatic(self):
        with patch.object(smart_counter, 'detect_smart') as smart, \
             patch.object(auto_counter, 'detect_auto') as neural:
            for engine in ('smart', 'auto', 'countgdpp', 'grains'):
                with self.subTest(engine=engine):
                    response = self.post(engine=engine, negative_exemplar_points=NEGATIVE)
                    self.assertEqual(response.status_code, 400)
        smart.assert_not_called()
        neural.assert_not_called()

    def test_unsupported_engine_cannot_ignore_negative_references(self):
        with patch.object(smart_counter, 'detect_smart') as smart, \
             patch.object(auto_counter, 'detect_auto') as neural:
            for engine in ('opencv', 'beans', 'grains', 'geco2'):
                response = self.post(engine=engine, exemplar_points=POSITIVE,
                                     negative_exemplar_points=NEGATIVE)
                self.assertEqual(response.status_code, 400)
        smart.assert_not_called()
        neural.assert_not_called()


class NegativeRouting(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 100, 3), 180, np.uint8)

    def test_manual_roles_bypass_fast_paths_and_share_gpu_worker(self):
        with patch.object(smart_counter, 'inference_device', return_value='cuda'), \
             patch.object(auto_counter, 'detect_auto', return_value=prediction()) as neural, \
             patch.object(smart_counter, 'scene_hint', side_effect=AssertionError('No classification')), \
             patch('particle_gate.detect_separated', side_effect=AssertionError('No particle route')):
            result = smart_counter.detect_smart(self.image, exemplar_points=POSITIVE,
                                               negative_exemplar_points=NEGATIVE)
        neural.assert_called_once_with(self.image, 'automatic', 'cuda', keep_warm=True,
                                       exemplar_points=POSITIVE, negative_exemplar_points=NEGATIVE)
        self.assertEqual(result['count'], 1)

    def test_resource_shortage_preserves_both_roles(self):
        with patch.object(smart_counter, 'inference_device', return_value=None), \
             patch.object(auto_counter, 'detect_auto') as neural:
            result = smart_counter.detect_smart(self.image, exemplar_points=POSITIVE,
                                               negative_exemplar_points=NEGATIVE)
        neural.assert_not_called()
        self.assertIsNone(result['count'])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['exemplar_points'], POSITIVE)
        self.assertEqual(result['meta']['negative_exemplar_points'], NEGATIVE)

    def test_conflicts_fail_before_gpu_probe(self):
        with patch.object(smart_counter, 'inference_device') as device, \
             patch.object(auto_counter, 'detect_auto') as neural:
            for positive in (None, POSITIVE):
                with self.subTest(positive=positive), self.assertRaises(ValueError):
                    smart_counter.detect_smart(self.image, exemplar_points=positive,
                                               negative_exemplar_points=POSITIVE)
        device.assert_not_called()
        neural.assert_not_called()

    def test_auto_validates_and_forwards_both_roles(self):
        with patch.object(auto_counter, 'readiness', return_value=dict(installed=True)), \
             patch.object(auto_counter, '_execute', return_value=prediction()) as execute:
            auto_counter.detect_auto(self.image, device='cpu', exemplar_points=POSITIVE,
                                     negative_exemplar_points=NEGATIVE)
        execute.assert_called_once_with(self.image, 'countgdpp', 'cpu',
            dict(category='bamboo', exemplar_points=POSITIVE, negative_exemplar_points=NEGATIVE),
            keep_warm=False)

    def test_oneoff_cpu_serializes_both_arrays_and_hides_cuda(self):
        seen = {}

        def run(command, **kwargs):
            for flag, role in (('--exemplar-points-file', 'positive'),
                               ('--negative-exemplar-points-file', 'negative')):
                source = Path(command[command.index(flag)+1])
                seen[role] = json.loads(source.read_text(encoding='utf-8'))
            seen['env'] = kwargs['env']
            Path(command[command.index('--output')+1]).write_text(json.dumps(prediction()), encoding='utf-8')
            return subprocess.CompletedProcess(command, 0, stdout=b'', stderr=b'')

        with tempfile.TemporaryDirectory(prefix='negative_cpu_') as folder, \
             patch.object(auto_counter, 'ROOT', Path(folder)), \
             patch.object(auto_counter._warm_worker, 'release'), \
             patch.object(auto_counter._warm_worker, 'request') as resident, \
             patch.object(auto_counter.subprocess, 'run', side_effect=run):
            auto_counter._execute(self.image, 'countgdpp', 'cpu',
                dict(category='automatic', exemplar_points=POSITIVE, negative_exemplar_points=NEGATIVE))
        resident.assert_not_called()
        self.assertEqual(seen['positive'], POSITIVE)
        self.assertEqual(seen['negative'], NEGATIVE)
        self.assertEqual(seen['env']['CUDA_VISIBLE_DEVICES'], '-1')

    def test_resident_job_uses_same_protocol_without_new_model_process(self):
        seen = {}

        def request(command, job, **kwargs):
            seen.update(command=command, job=copy.deepcopy(job))
            Path(job['output']).write_text(json.dumps(prediction()), encoding='utf-8')
            return True

        with tempfile.TemporaryDirectory(prefix='negative_gpu_') as folder, \
             patch.object(auto_counter, 'ROOT', Path(folder)), \
             patch.object(auto_counter._warm_worker, 'release') as release, \
             patch.object(auto_counter._warm_worker, 'request', side_effect=request) as resident, \
             patch.object(auto_counter.subprocess, 'run') as process:
            result = auto_counter._execute(self.image, 'countgdpp', 'cuda',
                dict(category='automatic', exemplar_points=POSITIVE, negative_exemplar_points=NEGATIVE),
                keep_warm=True)
        release.assert_not_called()
        process.assert_not_called()
        resident.assert_called_once()
        self.assertEqual(seen['job']['negative_exemplar_points'], NEGATIVE)
        self.assertEqual(seen['job']['exemplar_points'], POSITIVE)
        self.assertTrue(result['meta']['cache_reused'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
