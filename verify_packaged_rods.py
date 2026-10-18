"""CPU pixel-only regressions for visible packaged chopstick candidates.

The three user photographs are development cases, not a held-out accuracy
benchmark. Counts below are independently stated test expectations. Predictors
receive decoded pixels only, never filenames, count labels or audit JSON.
"""
import base64
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import app
from boxed_rod_counter import detect_boxed_rods
from closeup_rod_counter import detect_closeup_rods
from laid_rod_counter import detect_laid_rods


FIXTURES = (Path(__file__).resolve().parent / 'samples' /
            'packaged_rods_regression_20261018')


def read_image(filename):
    image = cv2.imdecode(np.frombuffer((FIXTURES / filename).read_bytes(),
                                     np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError('Cannot decode fixture: ' + filename)
    return image


class PackagedRodChecks(unittest.TestCase):
    def assert_bounded_points(self, points, image):
        height, width = image.shape[:2]
        for point in points:
            self.assertTrue(0 <= point['x'] < width, point)
            self.assertTrue(0 <= point['y'] < height, point)
            self.assertEqual(len(point['segment']), 2)
            for x, y in point['segment']:
                self.assertTrue(0 <= x < width and 0 <= y < height, point)

    def require_candidates(self, result, expected, image, framework):
        self.assertIsNotNone(result)
        points, meta = result
        self.assertEqual(len(points), expected)
        self.assertEqual(meta['framework'], framework)
        self.assertEqual(meta['device'], 'cpu')
        self.assertEqual(meta['count_kind'], 'visible_rod_candidates')
        self.assertFalse(meta['count_is_total'])
        self.assertTrue(meta['review_required'])
        self.assertTrue(meta['prediction_valid'])
        self.assertFalse(meta['manual_exemplars'])
        self.assert_bounded_points(points, image)
        return points, meta

    def test_user_boxes_count_ten_individual_tracks_not_five_pairs(self):
        for filename in ('user_box_light.jpg', 'user_box_dark.jpg'):
            with self.subTest(filename=filename):
                image = read_image(filename)
                result = detect_boxed_rods(image)
                _, meta = self.require_candidates(
                    result, 10, image, 'BoxedRodTerminalTracks')
                self.assertFalse(meta['evidence']['pair_multiplier_used'])
                self.assertEqual(
                    meta['evidence']['directly_observed_individual_strokes'], 10)
                self.assertGreaterEqual(
                    len(meta['evidence']['independently_agreeing_saturation_thresholds']), 2)

    def test_mirrored_recompressed_box_cannot_silently_lose_two_rods(self):
        image = read_image('counterfactual_flip_horizontal.jpg')
        # Abstention is allowed under changed pixels; a confident subset of 8
        # tracks in this ten-rod image must not be returned as the whole result.
        for detector in (detect_boxed_rods, detect_laid_rods):
            with self.subTest(detector=detector.__name__):
                result = detector(image)
                if result is not None:
                    self.require_candidates(
                        result, 10, image, 'BoxedRodTerminalTracks')

    def test_longitudinal_paint_inside_one_slab_does_not_prove_ten_rods(self):
        image = read_image('counterfactual_one_painted_slab_inside_box.jpg')
        self.assertIsNone(detect_boxed_rods(image))
        self.assertIsNone(detect_laid_rods(image))

    def test_user_closeup_reports_four_visible_candidates_not_hidden_total(self):
        image = read_image('user_closeup.jpg')
        _, meta = self.require_candidates(
            detect_closeup_rods(image), 4, image, 'CloseupRodTerminalFaces')
        self.assertEqual(meta['diagnosis'], 'visible_end_candidates')
        self.assertIn('未完整入镜', meta['user_message'])
        self.assertIn('总数量', meta['user_message'])

    def test_closeup_count_varies_with_visible_terminal_faces(self):
        for expected in (1, 3, 5):
            with self.subTest(expected=expected):
                image = read_image('closeup_synthetic_%d.jpg' % expected)
                result = detect_closeup_rods(image)
                # This JPEG's second terminal has unstable distance-core
                # counts [1, 1, 2, 1, 1]. Rejecting that region must not leave
                # a silently accepted count of four for this five-face image.
                if expected != 5 or result is not None:
                    self.require_candidates(result, expected, image,
                                            'CloseupRodTerminalFaces')
        continuing = read_image('closeup_one_shaft_three_paint_patches.jpg')
        self.assertIsNone(detect_closeup_rods(continuing))
        self.assertIsNone(detect_laid_rods(continuing))

    def test_rotated_inputs_keep_counts_and_original_coordinate_bounds(self):
        for filename, detector, expected, framework in (
                ('user_box_light.jpg', detect_boxed_rods, 10, 'BoxedRodTerminalTracks'),
                ('user_box_dark.jpg', detect_boxed_rods, 10, 'BoxedRodTerminalTracks'),
                ('user_closeup.jpg', detect_closeup_rods, 4, 'CloseupRodTerminalFaces')):
            image = read_image(filename)
            for turns in (1, 2, 3):
                with self.subTest(filename=filename, quarter_turns=turns):
                    turned = np.ascontiguousarray(np.rot90(image, turns))
                    result = detector(turned)
                    # Packaging may become unsupported at another orientation;
                    # an accepted result must retain the actual visible count.
                    if result is not None or detector is detect_closeup_rods:
                        self.require_candidates(result, expected, turned, framework)

    def test_image_only_api_routes_without_neural_counting_and_keeps_scope_message(self):
        with patch('smart_counter.scene_hint', return_value=dict(
                accepted=True, scene='laid_rods', candidates=[])) as scene:
            with patch('auto_counter.detect_auto', side_effect=AssertionError(
                    'Packaged rod geometry must not dispatch a neural counter')) as neural:
                with patch('smart_counter.inference_device', side_effect=AssertionError(
                        'Packaged rod geometry must not probe a GPU')) as device:
                    with app.test_client() as client:
                        for filename, expected, framework in (
                                ('user_box_light.jpg', 10, 'BoxedRodTerminalTracks'),
                                ('user_box_dark.jpg', 10, 'BoxedRodTerminalTracks'),
                                ('user_closeup.jpg', 4, 'CloseupRodTerminalFaces')):
                            with self.subTest(filename=filename):
                                image = read_image(filename)
                                direct = detect_laid_rods(image)
                                _, direct_meta = self.require_candidates(
                                    direct, expected, image, framework)
                                encoded = cv2.imencode('.png', image)[1]
                                # Complete public request: no category, mode,
                                # exemplar, model, or device is supplied.
                                response = client.post('/api/count', json={
                                    'image': 'data:image/png;base64,' +
                                    base64.b64encode(encoded).decode('ascii')})
                                self.assertEqual(response.status_code, 200,
                                                 response.get_json())
                                result = response.get_json()
                                self.assertEqual(result['count'], expected)
                                self.assertEqual((result['height'], result['width']),
                                                 image.shape[:2])
                                self.require_candidates((result['points'], result['meta']),
                                                        expected, image, framework)
                                self.assertEqual(result['meta']['user_message'],
                                                 direct_meta['user_message'])
                    neural.assert_not_called()
                    device.assert_not_called()
                    self.assertEqual(scene.call_count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
