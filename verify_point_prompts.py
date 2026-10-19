"""Visual point-example contracts without model weights, a server or CUDA.

Scripted detections verify coordinate transport, one visual forward and output
semantics. They are synthetic plumbing fixtures, not counting-accuracy claims.
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
from PIL import Image
import torch

import auto_counter
import countgdpp_worker as worker
from point_prompts import exemplar_boxes, validate_exemplar_points
import smart_counter


REFERENCES = [dict(x=0., y=0., diameter=16.),
              dict(x=30., y=40., diameter=20.),
              dict(x=99., y=79., diameter=16.)]


def synthetic_result():
    return dict(count=1, points=[dict(x=70., y=20.)], width=100, height=80,
                meta=dict(prediction_valid=True, manual_exemplars=True,
                          count_is_total=False, review_required=True))


class PointValidation(unittest.TestCase):
    def test_valid_one_to_three_points_normalize_without_mutating_request(self):
        for number in range(1, 4):
            with self.subTest(number=number):
                points = copy.deepcopy(REFERENCES[:number])
                before = copy.deepcopy(points)
                normalized = validate_exemplar_points(points, 100, 80)
                self.assertEqual(normalized, before)
                self.assertEqual(points, before)
                self.assertIsNot(normalized, points)
                self.assertTrue(all(type(value) is float
                                    for point in normalized for value in point.values()))

    def test_container_and_cardinality_are_strict(self):
        for value in (None, [], {}, 'one point', (REFERENCES[0],),
                      REFERENCES + [dict(x=50, y=60, diameter=16)]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_exemplar_points(value, 100, 80)

    def test_fields_reject_boolean_nonfinite_and_nonnumeric_values(self):
        for key in ('x', 'y', 'diameter'):
            for value in (True, False, float('nan'), float('inf'),
                          -float('inf'), 10**1000, None, '12', [], {}):
                with self.subTest(key=key, value=value):
                    point = dict(x=30, y=40, diameter=20)
                    point[key] = value
                    with self.assertRaises(ValueError):
                        validate_exemplar_points([point], 100, 80)

    def test_malformed_or_missing_fields_are_rejected(self):
        for value in (None, [], [20, 30], 'point', 5,
                      dict(x=20, y=30), dict(y=30, diameter=20),
                      dict(x=20, diameter=20)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_exemplar_points([value], 100, 80)

    def test_coordinates_use_half_open_original_image_bounds(self):
        for x, y in ((-0.01, 20), (100, 20), (20, -0.01), (20, 80)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                validate_exemplar_points([dict(x=x, y=y, diameter=20)], 100, 80)
        self.assertEqual(len(validate_exemplar_points(
            [dict(x=99.999, y=79.999, diameter=20)], 100, 80)), 1)

    def test_diameter_bounds_include_eight_and_four_hundred(self):
        for diameter in (7.999, 400.001, 0, -8):
            with self.subTest(diameter=diameter), self.assertRaises(ValueError):
                validate_exemplar_points([dict(x=30, y=40, diameter=diameter)], 100, 80)
        for diameter in (8, 400):
            with self.subTest(diameter=diameter):
                self.assertEqual(validate_exemplar_points(
                    [dict(x=30, y=40, diameter=diameter)], 100, 80)[0]['diameter'],
                    float(diameter))

    def test_repeated_or_nearly_repeated_centers_are_rejected(self):
        for point in (dict(x=30, y=40, diameter=10),
                      dict(x=31, y=41, diameter=40)):
            with self.subTest(point=point), self.assertRaises(ValueError):
                validate_exemplar_points([REFERENCES[1], point], 100, 80)
        self.assertEqual(len(validate_exemplar_points(
            [REFERENCES[1], dict(x=32, y=40, diameter=20)], 100, 80)), 2)

    def test_edge_examples_clip_to_original_image_without_moving_center(self):
        self.assertEqual(exemplar_boxes(REFERENCES, 100, 80),
                         [[0., 0., 8., 8.], [20., 30., 40., 50.],
                          [91., 71., 100., 80.]])
        self.assertEqual(exemplar_boxes([dict(x=30, y=40, diameter=400)], 100, 80),
                         [[0., 0., 100., 80.]])


class VisualForward(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='point_example_test_')
        self.addCleanup(self.folder.cleanup)
        self.image_path = Path(self.folder.name) / 'synthetic.png'
        Image.new('RGB', (100, 80), (220, 190, 140)).save(self.image_path)

    def run_visual(self, probabilities=(.9, .8, .7, .1)):
        calls = []

        def model(nested, exemplars_image, boxes, targets, labels, captions):
            self.assertIs(nested, exemplars_image)
            self.assertEqual(nested.shape, (1, 3, 30, 50))
            self.assertEqual(captions, ['object .'])
            self.assertEqual(targets, [])
            self.assertEqual(labels, [])
            self.assertEqual(len(boxes), 1)
            self.assertEqual(boxes[0].device.type, 'cpu')
            expected = torch.tensor([[0., 0., 4., 3.],
                                     [10., 11.25, 20., 18.75],
                                     [45.5, 26.625, 50., 30.]])
            self.assertTrue(torch.allclose(boxes[0], expected))
            calls.append(boxes[0].tolist())
            # Two exact duplicate boxes and one separate target. A low-score
            # fourth box must not be rescued by the number of reference points.
            positions = [[.7, .25, .12, .16], [.7, .25, .12, .16],
                         [.5, .6, .12, .16], [.2, .8, .12, .16]]
            logits = torch.logit(torch.tensor(probabilities)).reshape(1, -1, 1)
            return dict(pred_logits=logits,
                        pred_boxes=torch.tensor(positions).reshape(1, 4, 4))

        runtime = dict(model=model, device='cpu',
                       transform=lambda image, target: (torch.zeros(3, 30, 50), target),
                       nested_tensor=lambda tensor: tensor)
        with patch.object(worker, 'load_runtime', return_value=(runtime, 0.)) as loader, \
             patch.object(worker, '_infer_once', side_effect=AssertionError('No text-only pass')), \
             patch.object(worker, 'probe_category', side_effect=AssertionError('No category probe')), \
             patch.object(torch, 'load', side_effect=AssertionError('Weights must not load')), \
             patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU test')), \
             patch.object(torch.cuda, 'mem_get_info', side_effect=AssertionError('CPU test')), \
             patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CPU test')):
            result = worker.infer(self.image_path, 'rice', 'cpu',
                                  allow_category_recovery=True,
                                  exemplar_points=REFERENCES)
        loader.assert_called_once_with(torch, 'cpu')
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['width'], 100)
        self.assertEqual(result['height'], 80)
        self.assertEqual(result['exemplars'], exemplar_boxes(REFERENCES, 100, 80))
        return result

    def test_points_become_scaled_visual_boxes_in_one_cpu_forward(self):
        result = self.run_visual()
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['count'], len(result['points']))
        self.assertEqual({(p['x'], p['y']) for p in result['points']},
                         {(70., 20.), (50., 48.)})
        meta = result['meta']
        self.assertEqual(meta['raw_prediction_count'], 3)
        self.assertEqual(meta['user_exemplar_count'], 3)
        self.assertEqual(meta['pseudo_exemplar_count'], 0)
        self.assertEqual(meta['category'], 'user_examples')
        self.assertEqual(meta['reference_source'], 'current_request_user_points')
        self.assertTrue(meta['manual_exemplars'])
        self.assertTrue(meta['prediction_valid'])
        self.assertFalse(meta['automatic_category'])
        self.assertFalse(meta['count_is_total'])
        self.assertTrue(meta['review_required'])
        self.assertTrue(meta['final_nms'])
        self.assertEqual(meta['final_nms_iou'], .7)

    def test_zero_detections_never_turn_reference_points_into_counted_objects(self):
        result = self.run_visual((.1, .1, .1, .1))
        self.assertIsNone(result['count'])
        self.assertEqual(result['points'], [])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['diagnosis'], 'no_final_detections')
        self.assertEqual(result['meta']['user_exemplar_count'], 3)

    def test_text_override_cannot_compete_with_user_visual_examples(self):
        with patch.object(worker, 'load_runtime') as loader, \
             self.assertRaises(ValueError):
            worker.infer(self.image_path, 'rice', 'cpu', text='rice grain',
                         exemplar_points=REFERENCES)
        loader.assert_not_called()

    def test_invalid_points_fail_before_loading_a_model(self):
        with patch.object(worker, 'load_runtime') as loader:
            for points in ([], [dict(x=100, y=20, diameter=20)],
                           [dict(x=float('nan'), y=20, diameter=20)]):
                with self.subTest(points=points), self.assertRaises(ValueError):
                    worker.infer(self.image_path, 'rice', 'cpu', exemplar_points=points)
        loader.assert_not_called()

    def test_omitted_or_none_reference_argument_preserves_original_dispatch(self):
        result = synthetic_result()
        for options in ({}, dict(exemplar_points=None)):
            with self.subTest(options=options), \
                 patch.object(worker, '_infer_once', return_value=result) as original, \
                 patch.object(worker, 'infer_point_examples') as visual:
                self.assertIs(worker.infer(self.image_path, 'rice', 'cpu', **options), result)
                original.assert_called_once_with(self.image_path, 'rice', 'cpu', .23, None)
                visual.assert_not_called()


class SmartPointRoute(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 100, 3), 220, np.uint8)
        self.scene = self.enterContext(patch.object(smart_counter, 'scene_hint',
                                      side_effect=AssertionError('No category classification')))
        self.geometry = self.enterContext(patch.object(smart_counter, '_strict_rod_geometry',
                                         side_effect=AssertionError('No rod geometry')))
        self.particles = self.enterContext(patch('particle_gate.detect_separated',
                                          side_effect=AssertionError('No particle gate')))

    def test_valid_references_bypass_classification_and_geometry(self):
        with patch.object(smart_counter, 'inference_device', return_value='cuda'), \
             patch.object(auto_counter, 'detect_auto', return_value=synthetic_result()) as detect:
            result = smart_counter.detect_smart(self.image, exemplar_points=REFERENCES)
        detect.assert_called_once_with(self.image, 'automatic', 'cuda', keep_warm=True,
                                       exemplar_points=REFERENCES)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['meta']['routing'], 'user_points_then_visual_exemplars')
        self.scene.assert_not_called()
        self.geometry.assert_not_called()
        self.particles.assert_not_called()

    def test_unavailable_gpu_preserves_references_but_does_not_start_a_model(self):
        with patch.object(smart_counter, 'inference_device', return_value=None), \
             patch.object(auto_counter, 'detect_auto') as detect:
            result = smart_counter.detect_smart(self.image, exemplar_points=REFERENCES)
        detect.assert_not_called()
        self.assertIsNone(result['count'])
        self.assertEqual(result['points'], [])
        self.assertEqual(result['meta']['exemplar_points'], REFERENCES)
        self.assertTrue(result['meta']['manual_exemplars'])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['routing'], 'user_examples_runtime_unavailable')
        self.scene.assert_not_called()
        self.particles.assert_not_called()

    def test_invalid_references_do_not_even_probe_gpu_capacity(self):
        with patch.object(smart_counter, 'inference_device') as device, \
             patch.object(auto_counter, 'detect_auto') as detect, self.assertRaises(ValueError):
            smart_counter.detect_smart(self.image, exemplar_points=[])
        device.assert_not_called()
        detect.assert_not_called()


class PointTransport(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 100, 3), 220, np.uint8)

    def test_detect_auto_validates_and_forwards_original_pixel_references(self):
        with patch.object(auto_counter, 'readiness', return_value=dict(installed=True)), \
             patch.object(auto_counter, '_execute', return_value=synthetic_result()) as execute:
            auto_counter.detect_auto(self.image, 'bamboo', 'cpu', exemplar_points=REFERENCES)
        execute.assert_called_once_with(self.image, 'countgdpp', 'cpu',
            dict(category='bamboo', exemplar_points=REFERENCES), keep_warm=False)

    def test_detect_auto_invalid_references_never_start_a_worker(self):
        with patch.object(auto_counter, 'readiness', return_value=dict(installed=True)), \
             patch.object(auto_counter, '_execute') as execute, self.assertRaises(ValueError):
            auto_counter.detect_auto(self.image, exemplar_points=[])
        execute.assert_not_called()

    def test_oneoff_cpu_uses_json_file_and_hides_cuda(self):
        seen = {}

        def run(command, **kwargs):
            seen.update(command=command, env=kwargs['env'])
            reference_path = Path(command[command.index('--exemplar-points-file')+1])
            seen['references'] = json.loads(reference_path.read_text(encoding='utf-8'))
            result_path = Path(command[command.index('--output')+1])
            result_path.write_text(json.dumps(synthetic_result()), encoding='utf-8')
            return subprocess.CompletedProcess(command, 0, stdout=b'', stderr=b'')

        with tempfile.TemporaryDirectory(prefix='point_transport_') as folder, \
             patch.object(auto_counter, 'ROOT', Path(folder)), \
             patch.object(auto_counter._warm_worker, 'release') as release, \
             patch.object(auto_counter._warm_worker, 'request') as resident, \
             patch.object(auto_counter.subprocess, 'run', side_effect=run):
            result = auto_counter._execute(self.image, 'countgdpp', 'cpu',
                dict(category='bamboo', exemplar_points=REFERENCES))
        release.assert_called_once()
        resident.assert_not_called()
        self.assertEqual(seen['references'], REFERENCES)
        self.assertEqual(seen['env']['CUDA_VISIBLE_DEVICES'], '-1')
        self.assertNotIn('--serve', seen['command'])
        self.assertTrue(result['meta']['worker_exited'])
        self.assertEqual(result['count'], 1)

    def test_resident_gpu_job_carries_current_references_without_extra_process(self):
        seen = {}

        def request(command, job, **kwargs):
            seen.update(command=command, job=copy.deepcopy(job))
            Path(job['output']).write_text(json.dumps(synthetic_result()), encoding='utf-8')
            return True

        with tempfile.TemporaryDirectory(prefix='point_transport_') as folder, \
             patch.object(auto_counter, 'ROOT', Path(folder)), \
             patch.object(auto_counter._warm_worker, 'release') as release, \
             patch.object(auto_counter._warm_worker, 'request', side_effect=request) as resident, \
             patch.object(auto_counter.subprocess, 'run') as process:
            result = auto_counter._execute(self.image, 'countgdpp', 'cuda',
                dict(category='bamboo', exemplar_points=REFERENCES), keep_warm=True)
        release.assert_not_called()
        process.assert_not_called()
        resident.assert_called_once()
        self.assertEqual(seen['job']['exemplar_points'], REFERENCES)
        self.assertIn('--serve', seen['command'])
        self.assertFalse(result['meta']['worker_exited'])
        self.assertTrue(result['meta']['cache_reused'])
        self.assertEqual(result['count'], 1)


class PointAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()
        image = np.full((80, 100, 3), 220, np.uint8)
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        self.image = 'data:image/png;base64,'+base64.b64encode(encoded).decode('ascii')

    def post(self, **options):
        return self.client.post('/api/count', json=dict(image=self.image, **options))

    def test_image_only_api_keeps_the_existing_smart_call_signature(self):
        with patch.object(smart_counter, 'detect_smart', return_value=synthetic_result()) as smart, \
             patch.object(auto_counter, 'detect_auto') as neural:
            response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(smart.call_count, 1)
        self.assertEqual(smart.call_args.kwargs, {})
        neural.assert_not_called()

    def test_smart_and_unclassified_auto_forward_references(self):
        for options in ({}, dict(engine='smart'), dict(engine='auto')):
            with self.subTest(options=options), \
                 patch.object(smart_counter, 'detect_smart', return_value=synthetic_result()) as smart:
                response = self.post(exemplar_points=REFERENCES, **options)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(smart.call_args.kwargs, dict(exemplar_points=REFERENCES))
            self.assertEqual(response.get_json()['count'], 1)

    def test_explicit_auto_or_countgdpp_points_cannot_fall_into_particle_counting(self):
        for engine in ('auto', 'countgdpp'):
            with self.subTest(engine=engine), \
                 patch.object(auto_counter, 'detect_auto', return_value=synthetic_result()) as neural, \
                 patch.object(smart_counter, 'detect_smart') as smart, \
                 patch('grain_counter.detect_grains', side_effect=AssertionError('No legacy grain route')):
                response = self.post(engine=engine, category='rice', device='cpu',
                                     exemplar_points=REFERENCES)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(neural.call_count, 1)
            self.assertEqual(neural.call_args.args[1:], ('rice', 'cpu'))
            self.assertEqual(neural.call_args.kwargs,
                             dict(keep_warm=False, allow_category_recovery=False,
                                  exemplar_points=REFERENCES))
            smart.assert_not_called()

    def test_invalid_reference_inputs_return_400_before_model_dispatch(self):
        invalid = [None, [], {}, REFERENCES+[dict(x=50, y=60, diameter=16)],
                   [dict(x=True, y=20, diameter=16)],
                   [dict(x=float('nan'), y=20, diameter=16)],
                   [dict(x=20, y=float('inf'), diameter=16)],
                   [dict(x=100, y=20, diameter=16)],
                   [dict(x=20, y=80, diameter=16)],
                   [dict(x=20, y=20, diameter=401)],
                   [REFERENCES[0], REFERENCES[0]]]
        with patch.object(smart_counter, 'detect_smart') as smart, \
             patch.object(auto_counter, 'detect_auto') as neural:
            for points in invalid:
                with self.subTest(points=points):
                    response = self.post(exemplar_points=points)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn('error', response.get_json())
        smart.assert_not_called()
        neural.assert_not_called()

    def test_unsupported_engines_reject_points_before_dispatch(self):
        with patch.object(smart_counter, 'detect_smart') as smart, \
             patch.object(auto_counter, 'detect_auto') as neural:
            for engine in ('opencv', 'beans', 'grains', 'geco2', 'unknown'):
                with self.subTest(engine=engine):
                    response = self.post(engine=engine, exemplar_points=REFERENCES)
                    self.assertEqual(response.status_code, 400)
        smart.assert_not_called()
        neural.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
