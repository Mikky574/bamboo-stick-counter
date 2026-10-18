"""Self-contained pixel regressions for the orange display-card photograph.

The source is one development photo. Rotated/recompressed variants are the same
scene, and synthetic counterfactuals are rejection/count-change checks. None is
a held-out accuracy benchmark or a training input. Predictors receive only BGR
pixels; expected counts, filenames, annotation JSON and scene hints stay in tests.
The API tests mock semantic hints and runtime dispatch, never geometric outputs.
"""
import base64
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import app
from carded_rod_counter import detect_carded_rods


FIXTURES = (Path(__file__).resolve().parent / 'samples' /
            'carded_rods_regression_20261018')
PHOTOGRAPHIC_VARIANTS = (
    'rotate_0.jpg', 'rotate_90.jpg', 'rotate_180.jpg', 'rotate_270.jpg',
    'flip_horizontal.jpg', 'flip_vertical.jpg',
    'jpeg_60.jpg', 'jpeg_75.jpg', 'jpeg_95.jpg',
    'rotate_free_-27.jpg', 'rotate_free_-12.jpg',
    'rotate_free_17.jpg', 'rotate_free_38.jpg',
    'background_35_40_45.jpg', 'background_220_235_240.jpg',
    'background_185_205_180.jpg',
)


def read_image(filename):
    image = cv2.imdecode(np.frombuffer((FIXTURES / filename).read_bytes(),
                                      np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError('Cannot decode regression fixture: ' + filename)
    return image


class CardedRodChecks(unittest.TestCase):
    def require_candidates(self, result, expected, image):
        self.assertIsNotNone(result)
        points, meta = result
        self.assertEqual(len(points), expected)
        self.assertEqual(meta['framework'], 'CardedRodTerminalTracks')
        self.assertEqual(meta['device'], 'cpu')
        self.assertEqual(meta['routing_evidence'], 'complete_individual_terminal_tracks')
        self.assertEqual(meta['count_kind'], 'visible_rod_candidates')
        self.assertTrue(meta['prediction_valid'])
        self.assertTrue(meta['review_required'])
        self.assertFalse(meta['count_is_total'])
        self.assertFalse(meta['manual_exemplars'])
        self.assertFalse(meta['evidence']['pair_multiplier_used'])
        self.assertEqual(meta['evidence']['directly_observed_individual_strokes'], expected)
        self.assertEqual(meta['evidence']['independent_terminal_caps'], expected)
        height, width = image.shape[:2]
        self.assertEqual(len({(p['x'], p['y']) for p in points}), expected)
        for point in points:
            self.assertTrue(0 <= point['x'] < width and 0 <= point['y'] < height, point)
            self.assertEqual(len(point['segment']), 2)
            for x, y in point['segment']:
                self.assertTrue(0 <= x < width and 0 <= y < height, point)

    def test_user_photo_counts_ten_individual_visible_tracks(self):
        image = read_image('user_orange_card.jpg')
        self.require_candidates(detect_carded_rods(image), 10, image)

    def test_photographic_variants_may_abstain_but_must_not_change_count(self):
        for filename in PHOTOGRAPHIC_VARIANTS:
            with self.subTest(filename=filename):
                image = read_image(filename)
                result = detect_carded_rods(image)
                if result is not None:
                    self.require_candidates(result, 10, image)

    def test_other_known_synthetic_counts_cannot_default_to_ten(self):
        for expected in (3, 5, 7, 9, 10, 11):
            with self.subTest(expected=expected):
                image = read_image('synthetic_%d_individual_rods.jpg' % expected)
                result = detect_carded_rods(image)
                if result is not None:
                    self.require_candidates(result, expected, image)

    def test_removing_one_complete_rod_cannot_leave_ten_predictions(self):
        # The independent audit removes both visible fragments of one physical
        # synthetic rod. A missing terminal must not borrow its neighbour's cap.
        image = read_image('synthetic_remove_one_of_ten.jpg')
        result = detect_carded_rods(image)
        if result is not None:
            self.require_candidates(result, 9, image)

    def test_real_photo_crops_with_complete_rods_removed_count_nine_and_eight(self):
        # Crop boundaries follow actual inter-rod gaps, removing the complete
        # upper and lower fragments of the leftmost rod or pair together. The
        # audit manually checked all remaining trajectories. These positive
        # cases must work; abstaining would not establish variable counting.
        for filename, expected in (
                ('real_remove_complete_leftmost_rod_9.jpg', 9),
                ('real_remove_complete_leftmost_pair_8.jpg', 8)):
            with self.subTest(filename=filename):
                image = read_image(filename)
                self.require_candidates(detect_carded_rods(image), expected, image)

    def test_drawn_lengthwise_lines_on_a_card_or_single_board_are_not_rods(self):
        for filename in ('counterfactual_flat_card_ten_painted_lines.jpg',
                         'counterfactual_one_board_ten_painted_lines.jpg'):
            with self.subTest(filename=filename):
                self.assertIsNone(detect_carded_rods(read_image(filename)))

    def test_incomplete_left_edge_cannot_be_silently_dropped_from_the_count(self):
        # Nine visible shafts remain, but one no longer has a complete highlight
        # edge. Returning the other eight was a real development regression.
        image = read_image('counterfactual_clipped_edge_nine_must_abstain.jpg')
        self.assertIsNone(detect_carded_rods(image))

    def test_image_only_api_uses_real_geometry_and_preserves_semantic_hint(self):
        recorded = json.loads((FIXTURES / 'recorded_scene_hint.json').read_text(encoding='utf-8'))
        self.assertFalse(recorded['accepted'])
        self.assertEqual(recorded['scene'], 'unknown')
        self.assertEqual([row['scene'] for row in recorded['candidates'][:2]],
                         ['flat_end', 'laid_rods'])
        hints = [dict(accepted=True, scene=scene, candidates=[])
                 for scene in ('flat_end', 'sharp', 'laid_rods', 'chopsticks')]
        hints.append(recorded)
        for first, second in [('sharp', 'laid_rods'), ('laid_rods', 'sharp')]:
            hints.append(dict(accepted=False, scene='unknown', suggested_scene=first,
                candidates=[dict(scene=first, score=.53), dict(scene=second, score=.46)]))
        image = read_image('user_orange_card.jpg')
        encoded = cv2.imencode('.png', image)[1]
        body = {'image': 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')}
        with patch('auto_counter.detect_auto', side_effect=AssertionError(
                'Strong terminal tracks must not dispatch neural counting.')) as neural:
            with patch('smart_counter.inference_device', side_effect=AssertionError(
                    'Strong terminal tracks must not need a GPU.')) as device:
                with app.test_client() as client:
                    for hint in hints:
                        with self.subTest(scene=hint['scene'], accepted=hint['accepted']):
                            with patch('smart_counter.scene_hint',
                                       return_value=copy.deepcopy(hint)) as scene:
                                response = client.post('/api/count', json=body)
                                self.assertEqual(response.status_code, 200, response.get_json())
                                result = response.get_json()
                                self.assertEqual(result['count'], 10)
                                self.assertEqual((result['height'], result['width']), image.shape[:2])
                                self.require_candidates((result['points'], result['meta']), 10, image)
                                self.assertEqual(result['meta']['routing'], 'scene_then_strict_terminal_tracks')
                                self.assertEqual(result['meta']['scene_classification'], hint)
                                scene.assert_called_once()
                neural.assert_not_called()
                device.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
