"""CPU-only image-only API regressions; all model/device calls are mocked.

The 30/17/8 counts are fixed sample regression expectations, not an independent
accuracy benchmark. The touching-rice fixture is an existing recorded prediction
used to verify rejection; this test never reruns its GPU model.
"""
import base64
import copy
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import app
from particle_gate import detect_separated


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / 'samples' / 'counting_eval'


def read_image(name):
    path = SAMPLES / name
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError('Cannot decode regression fixture: ' + str(path))
    return image


class Controls(HTMLParser):
    def __init__(self):
        super().__init__()
        self.controls = []

    def handle_starttag(self, tag, attributes):
        if tag in {'input', 'button', 'select', 'textarea'}:
            self.controls.append((tag, dict(attributes)))


class SmartCounterRegression(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        # Any accidental unmocked model/device dispatch immediately fails.
        self.model = self.enterContext(patch(
            'auto_counter.detect_auto',
            side_effect=AssertionError('A real model must not run in this CPU regression.')))
        self.device = self.enterContext(patch(
            'smart_counter.inference_device',
            side_effect=AssertionError('GPU probing is not needed on a CPU fast path.')))
        self.scene = self.enterContext(patch('smart_counter.scene_hint',return_value=None))
        self.enterContext(patch(
            'auto_counter.detect_geco2',
            side_effect=AssertionError('GeCo2 must not run in this CPU regression.')))

    def post_image(self, image):
        ok, encoded = cv2.imencode('.png', image)
        self.assertTrue(ok)
        # This is the complete public request: no category, engine or device.
        body = {'image': 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')}
        response = self.client.post('/api/count', json=body)
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def assert_image_points(self, result, image):
        self.assertEqual((result['height'], result['width']), image.shape[:2])
        self.assertEqual(result['count'], len(result['points']))
        for point in result['points']:
            self.assertTrue(0 <= point['x'] < image.shape[1])
            self.assertTrue(0 <= point['y'] < image.shape[0])

    def test_real_separated_samples_are_cpu_only(self):
        for filename, expected in (
                ('rice_gray.png', 30),
                ('app_mung_dark.jpeg', 17),
                ('app_coins_white.jpeg', 8)):
            with self.subTest(filename=filename):
                image = read_image(filename)
                result = self.post_image(image)
                self.assertEqual(result['count'], expected)
                self.assert_image_points(result, image)
                self.assertEqual(result['meta']['framework'], 'AutomaticParticles')
                self.assertEqual(result['meta']['device'], 'cpu')
                self.assertTrue(result['meta']['prediction_valid'])
                self.assertTrue(result['meta']['automatic_category'])
                self.assertGreaterEqual(result['meta']['total_elapsed_seconds'], 0)
        self.model.assert_not_called()
        self.device.assert_not_called()

    def test_blank_is_invalid_without_a_zero_count(self):
        result = self.post_image(np.full((120, 160, 3), 128, np.uint8))
        self.assertIsNone(result['count'])
        self.assertEqual(result['points'], [])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['diagnosis'], 'no_visible_targets')
        self.model.assert_not_called()
        self.device.assert_not_called()

    def test_classifier_selects_candidate_route_without_supplying_count(self):
        image = read_image('dense_cotton_overhead.jpg')
        self.scene.return_value = dict(accepted=True,scene='cotton',candidates=[])
        self.use_mock_model(dict(count=1,points=[dict(x=40,y=50)],width=image.shape[1],
            height=image.shape[0],meta=dict(framework='CountGD++',category='cotton',
            prediction_valid=True,diagnosis='detections_returned')))
        result = self.post_image(image)
        self.assertEqual(self.model.call_args.args[1:],('cotton','cuda'))
        self.assertEqual(result['meta']['routing'],'cpu_scene_hint_then_counting')
        self.assertEqual(result['count'],1)

    def test_touching_long_pairs_never_use_particle_count_as_pair_count(self):
        folder=ROOT/'samples/flat_rods_eval_20261018'
        rows=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        row=next(r for r in rows if r['id']=='chopsticks_daiso_octagonal')
        image=cv2.imdecode(np.fromfile(folder/row['file'],np.uint8),cv2.IMREAD_COLOR)
        self.assertIsNone(detect_separated(image))

    def test_scene_routed_chopsticks_count_rods_not_pairs_without_deep_counting(self):
        folder=ROOT/'samples/flat_rods_eval_20261018'
        rows=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        self.scene.return_value=dict(accepted=True,scene='chopsticks',candidates=[])
        for identity in ('chopsticks_daiso_octagonal','chopsticks_striped'):
            row=next(r for r in rows if r['id']==identity)
            image=cv2.imdecode(np.fromfile(folder/row['file'],np.uint8),cv2.IMREAD_COLOR)
            with self.subTest(identity=identity):
                result=self.post_image(image)
                self.assertEqual(result['count'],10)
                self.assertIn(result['meta']['framework'],{'LaidRodSections','RodSilhouetteConsensus'})
                self.assertTrue(all(len(p['segment'])==2 for p in result['points']))
        self.model.assert_not_called()
        self.device.assert_not_called()

    def test_unresolved_whole_rods_do_not_fall_back_to_known_bad_instance_count(self):
        folder=ROOT/'samples/flat_rods_eval_20261018'
        rows=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        row=next(r for r in rows if r['id']=='rods_walmart_six')
        image=cv2.imdecode(np.fromfile(folder/row['file'],np.uint8),cv2.IMREAD_COLOR)
        self.scene.return_value=dict(accepted=True,scene='laid_rods',candidates=[])
        result=self.post_image(image)
        self.assertIsNone(result['count'])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['diagnosis'],'unresolved_individual_rods')
        self.model.assert_not_called()

    def test_ambiguous_side_view_does_not_get_counted_as_bundled_tips(self):
        image=read_image('dense_flat_craft.jpg')
        self.scene.return_value=dict(accepted=False,scene='unknown',suggested_scene='chopsticks')
        result=self.post_image(image)
        self.assertIsNone(result['count'])
        self.assertEqual(result['meta']['diagnosis'],'uncertain_whole_rod_view')
        self.model.assert_not_called()
        self.device.assert_not_called()

    def use_mock_model(self, result):
        self.model.side_effect = lambda *args, **kwargs: copy.deepcopy(result)
        self.device.side_effect = None
        self.device.return_value = 'cuda'

    def assert_automatic_cuda_call(self, image):
        self.model.assert_called_once()
        args, kwargs = self.model.call_args
        np.testing.assert_array_equal(args[0], image)
        self.assertEqual(args[1:], ('automatic', 'cuda'))
        self.assertEqual(kwargs, {'keep_warm': True})
        self.device.assert_called_once_with()

    def test_dense_images_abstain_then_use_automatic_model_category(self):
        for filename, category in (
                ('dense_flat_craft.jpg', 'bamboo'),
                ('dense_cotton_overhead.jpg', 'cotton')):
            with self.subTest(filename=filename):
                self.model.reset_mock()
                self.device.reset_mock()
                image = read_image(filename)
                self.assertIsNone(detect_separated(image))
                predicted = dict(count=1, points=[{'x': 40., 'y': 50., 'r': 5.}],
                                 width=image.shape[1], height=image.shape[0],
                                 meta=dict(framework='CountGD++', category=category,
                                           detected_category=category, prediction_valid=True,
                                           diagnosis='detections_returned', device='cuda'))
                self.use_mock_model(predicted)
                result = self.post_image(image)
                self.assert_automatic_cuda_call(image)
                self.assertEqual(result['count'], 1)
                self.assertEqual(result['meta']['detected_category'], category)
                self.assertEqual(result['meta']['routing'], 'automatic_category_after_geometry_gate')

    def test_ambiguous_model_response_stays_invalid_and_never_reports_zero(self):
        image = read_image('dense_flat_craft.jpg')
        selection = dict(accepted=False, category=None,
                         candidates=[{'category': 'bamboo', 'evidence': .24},
                                     {'category': 'cotton', 'evidence': .23}])
        # A legacy zero sentinel must also be converted into "no reliable count".
        for count in (None, 0):
            with self.subTest(model_count=count):
                self.model.reset_mock()
                self.device.reset_mock()
                self.use_mock_model(dict(
                    count=count, points=[], width=image.shape[1], height=image.shape[0],
                    meta=dict(framework='CountGD++', prediction_valid=False,
                              diagnosis='category_ambiguous', category=None,
                              category_selection=selection, user_message='类别不明确，请换图。')))
                result = self.post_image(image)
                self.assert_automatic_cuda_call(image)
                self.assertIsNone(result['count'])
                self.assertEqual(result['points'], [])
                self.assertFalse(result['meta']['prediction_valid'])
                self.assertEqual(result['meta']['diagnosis'], 'category_ambiguous')
                self.assertEqual(result['meta']['category_selection'], selection)
                self.assertEqual(result['meta']['user_message'], '类别不明确，请换图。')

    def test_touching_rice_recorded_bean_prediction_is_rejected_by_shape(self):
        image = read_image('rice_extra.jpg')
        self.assertIsNone(detect_separated(image))
        fixture = ROOT / 'outputs' / 'smart_validation' / 'rice_touching_probe.json'
        predicted = json.loads(fixture.read_text(encoding='utf-8'))
        self.assertGreater(predicted['count'], 0)
        self.assertEqual(predicted['meta']['category'], 'mung')
        original_count = predicted['count']
        self.use_mock_model(predicted)
        result = self.post_image(image)
        self.assert_automatic_cuda_call(image)
        self.assertIsNone(result['count'])
        self.assertEqual(result['points'], [])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['diagnosis'], 'target_shape_mismatch')
        self.assertEqual(result['meta']['rejected_prediction_count'], original_count)
        # The saved probe remains an unchanged historical prediction.
        self.assertEqual(json.loads(fixture.read_text(encoding='utf-8')), predicted)

    def test_page_has_only_upload_and_run_and_sends_image_only(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        parser = Controls()
        parser.feed(html)
        self.assertEqual([(tag, attributes.get('id')) for tag, attributes in parser.controls],
                         [('input', 'file'), ('button', 'run')])
        self.assertEqual(parser.controls[0][1].get('type'), 'file')
        self.assertNotRegex(html, r'<\s*select\b')
        bodies = re.findall(r'body\s*:\s*JSON\.stringify\((\{[^{}]*\})\)', html)
        self.assertEqual(len(bodies), 1)
        self.assertRegex(bodies[0], r'^\{\s*image\s*:\s*dataURL\s*\}$')
        self.model.assert_not_called()
        self.device.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
