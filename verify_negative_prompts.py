"""Native positive/negative visual prompting contracts, without weights or CUDA.

Scripted logits are plumbing fixtures: they demonstrate label separation and
coordinate transport, never a real-photo counting accuracy measurement.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import countgdpp_worker as worker
from point_prompts import (exemplar_boxes, validate_negative_exemplar_points)


POSITIVE = [dict(x=0., y=0., diameter=16.),
            dict(x=30., y=40., diameter=20.),
            dict(x=99., y=79., diameter=16.)]
NEGATIVE = [dict(x=20., y=10., diameter=16.),
            dict(x=90., y=10., diameter=16.)]


class NegativeValidation(unittest.TestCase):
    def test_independent_groups_normalize_without_mutating_inputs(self):
        original = copy.deepcopy((POSITIVE, NEGATIVE))
        result = validate_negative_exemplar_points(NEGATIVE, POSITIVE, 100, 80)
        self.assertEqual(result, NEGATIVE)
        self.assertIsNot(result, NEGATIVE)
        self.assertEqual((POSITIVE, NEGATIVE), original)
        self.assertEqual(validate_negative_exemplar_points(
            [dict(x=2, y=0, diameter=8)], POSITIVE, 100, 80),
            [dict(x=2., y=0., diameter=8.)])

    def test_explicit_invalid_negative_groups_are_rejected(self):
        values = [None, [], {}, 'point', NEGATIVE*2,
                  [dict(x=20, y=10)], [dict(x=True, y=10, diameter=16)],
                  [dict(x=float('nan'), y=10, diameter=16)],
                  [dict(x=20, y=float('inf'), diameter=16)],
                  [dict(x=10**1000, y=10, diameter=16)],
                  [dict(x=100, y=10, diameter=16)],
                  [dict(x=20, y=80, diameter=16)],
                  [dict(x=20, y=10, diameter=7.9)],
                  [dict(x=20, y=10, diameter=401)], [NEGATIVE[0]]*2]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_negative_exemplar_points(value, POSITIVE, 100, 80)

    def test_negative_examples_require_a_valid_positive_group(self):
        for value in (None, [], {}, [dict(x=100, y=10, diameter=16)]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_negative_exemplar_points(NEGATIVE, value, 100, 80)

    def test_conflicting_positive_and_negative_centers_are_rejected(self):
        for value in (POSITIVE[:1], [dict(x=1, y=1, diameter=8)],
                      [dict(x=31, y=40, diameter=20)]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_negative_exemplar_points(value, POSITIVE, 100, 80)


class NativeNegativeForward(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='negative_example_test_')
        self.addCleanup(self.folder.cleanup)
        self.image_path = Path(self.folder.name)/'synthetic.png'
        Image.new('RGB', (100, 80), (220, 190, 140)).save(self.image_path)

    def run_visual(self, positive_scores=None, negative_scores=None):
        if positive_scores is None: positive_scores = [.8, .6, .1, .6, .75, .7]
        if negative_scores is None: negative_scores = [.1, .95, .99, .6, .1, .1]
        calls = []

        def model(nested, exemplar_image, boxes, negative_images, negative_boxes, captions):
            self.assertIs(nested, exemplar_image)
            self.assertEqual(nested.shape, (1, 3, 30, 50))
            self.assertEqual(captions, ['object . .'])
            self.assertEqual(len(boxes), 1)
            self.assertTrue(torch.allclose(boxes[0], torch.tensor(
                [[0., 0., 4., 3.], [10., 11.25, 20., 18.75],
                 [45.5, 26.625, 50., 30.]])))
            self.assertEqual(len(negative_images), 1)
            self.assertIs(negative_images[0], nested)
            self.assertEqual(len(negative_boxes), 1)
            self.assertEqual(len(negative_boxes[0]), 1)
            self.assertTrue(torch.allclose(negative_boxes[0][0], torch.tensor(
                [[6., .75, 14., 6.75], [41., .75, 49., 6.75]])))
            calls.append(True)
            # CLS/object/three positive ROI stars/period/two negative ROI
            # stars/period/SEP, followed by padded output slots.
            token_ids = [101, 4874, 1008, 1008, 1008, 1012,
                         1008, 1008, 1012, 102]
            probabilities = torch.full((1, 6, 14), .01)
            probabilities[0, :, 2] = torch.tensor(positive_scores)
            probabilities[0, :, 6] = torch.tensor(negative_scores)
            # Padding must never become positive or negative evidence.
            probabilities[0, :, 13] = .999
            positions = [[.7, .25, .12, .16], [.3, .2, .12, .16],
                         [.2, .8, .12, .16], [.4, .6, .12, .16],
                         [.7, .25, .12, .16], [.5, .6, .12, .16]]
            return dict(pred_logits=torch.logit(probabilities),
                        pred_boxes=torch.tensor(positions).reshape(1, 6, 4),
                        input_ids=torch.tensor([token_ids]))

        runtime = dict(model=model, device='cpu',
                       transform=lambda image, target: (torch.zeros(3, 30, 50), target),
                       nested_tensor=lambda tensor: tensor)
        with patch.object(worker, 'load_runtime', return_value=(runtime, 0.)) as loader, \
             patch.object(worker, '_infer_once', side_effect=AssertionError('No text-only pass')), \
             patch.object(worker, 'probe_category', side_effect=AssertionError('No category probe')), \
             patch.object(torch, 'load', side_effect=AssertionError('Weights must not load')), \
             patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU fixture')), \
             patch.object(torch.cuda, 'mem_get_info', side_effect=AssertionError('CPU fixture')):
            result = worker.infer(self.image_path, 'rice', 'cpu',
                                  allow_category_recovery=True,
                                  exemplar_points=POSITIVE,
                                  negative_exemplar_points=NEGATIVE)
        loader.assert_called_once_with(torch, 'cpu')
        self.assertEqual(len(calls), 1)
        return result

    def test_native_same_forward_scales_both_groups_and_only_counts_positive_winners(self):
        result = self.run_visual()
        self.assertEqual(result['count'], 2)
        self.assertEqual({(p['x'], p['y']) for p in result['points']},
                         {(70., 20.), (50., 48.)})
        self.assertEqual(sorted(p['score'] for p in result['points']), [.7, .8])
        self.assertEqual(result['exemplars'], exemplar_boxes(POSITIVE, 100, 80))
        self.assertEqual(result['negative_exemplars'], exemplar_boxes(NEGATIVE, 100, 80))
        meta = result['meta']
        self.assertEqual(meta['raw_prediction_count'], 3)
        self.assertEqual(meta['negative_rejected_prediction_count'], 2)
        self.assertEqual(meta['exemplar_points'], POSITIVE)
        self.assertEqual(meta['negative_exemplar_points'], NEGATIVE)
        self.assertEqual(meta['user_exemplar_count'], 3)
        self.assertEqual(meta['user_negative_exemplar_count'], 2)
        self.assertEqual(meta['pseudo_exemplar_count'], 0)
        self.assertEqual(meta['negative_exemplar_method'], 'native_negative_visual_tokens')
        self.assertTrue(meta['prediction_valid'])
        self.assertFalse(meta['count_is_total'])
        self.assertTrue(meta['review_required'])
        self.assertTrue(meta['model_reused'])

    def test_all_negative_winners_return_null_not_reference_subtraction(self):
        result = self.run_visual([.6]*6, [.9]*6)
        self.assertIsNone(result['count'])
        self.assertEqual(result['points'], [])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['diagnosis'], 'no_final_detections')
        self.assertEqual(result['meta']['negative_rejected_prediction_count'], 6)

    def test_invalid_or_conflicting_negatives_fail_before_model_loading(self):
        with patch.object(worker, 'load_runtime') as loader:
            for value in ([], POSITIVE[:1], [dict(x=100, y=10, diameter=16)]):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    worker.infer(self.image_path, 'rice', 'cpu',
                                 exemplar_points=POSITIVE, negative_exemplar_points=value)
            with self.assertRaises(ValueError):
                worker.infer(self.image_path, 'rice', 'cpu',
                             negative_exemplar_points=NEGATIVE)
        loader.assert_not_called()

    def test_omitted_negatives_keep_the_original_visual_dispatch(self):
        for options in ({}, dict(negative_exemplar_points=None)):
            with self.subTest(options=options), \
                 patch.object(worker, 'infer_point_examples', return_value={'old': True}) as visual:
                self.assertEqual(worker.infer(self.image_path, 'rice', 'cpu',
                    exemplar_points=POSITIVE, **options), {'old': True})
            visual.assert_called_once_with(self.image_path, POSITIVE, 'cpu', .23)

    def test_missing_separator_or_injected_visual_tokens_is_rejected(self):
        cases = [[101, 4874, 1008, 1008, 102],
                 [101, 4874, 1012, 1008, 1012, 102],
                 [101, 4874, 1008, 1012, 1012, 102]]
        for token_ids in cases:
            with self.subTest(token_ids=token_ids), self.assertRaises(RuntimeError):
                worker.positive_negative_scores(dict(
                    pred_logits=torch.zeros(1, 1, len(token_ids)),
                    input_ids=torch.tensor([token_ids])))


if __name__ == '__main__':
    unittest.main(verbosity=2)
