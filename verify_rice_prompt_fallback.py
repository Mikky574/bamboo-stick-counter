"""CPU-only inference-path checks with scripted logits, never model weights.

These fixtures verify fallback decisions and pseudo-exemplar plumbing. They do
not measure counting accuracy. Real torchvision NMS is retained so duplicate
seed boxes cannot pass the distinct-object requirement merely by raw count.
"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import countgdpp_worker as worker


DISTINCT = [[.20, .25, .12, .16], [.50, .50, .12, .16],
            [.80, .75, .12, .16], [.20, .75, .12, .16]]


def response(probabilities, boxes=None):
    """One token per query is enough to exercise token-restricted scores."""
    boxes = DISTINCT[:len(probabilities)] if boxes is None else boxes
    return dict(probabilities=probabilities, boxes=boxes)


class ScriptedModel:
    def __init__(self, test, steps):
        self.test = test
        self.steps = list(steps)
        self.calls = []

    def __call__(self, nested, exemplars_image, boxes, targets, labels, captions):
        self.test.assertTrue(self.steps, 'Unexpected extra model invocation')
        expected_caption, expects_pseudo, output = self.steps.pop(0)
        self.test.assertEqual(captions, [expected_caption])
        self.test.assertEqual(len(boxes), 1)
        self.test.assertEqual(boxes[0].device.type, 'cpu')
        self.test.assertEqual(boxes[0].ndim, 2)
        self.test.assertEqual(boxes[0].shape[1], 4)
        self.test.assertEqual(bool(len(boxes[0])), expects_pseudo)
        self.test.assertIs(nested, exemplars_image)
        self.test.assertEqual(nested.device.type, 'cpu')
        self.calls.append(dict(caption=captions[0], pseudo=boxes[0].tolist()))
        probabilities = torch.tensor(output['probabilities'], dtype=torch.float32)
        logits = torch.logit(probabilities).reshape(1, -1, 1)
        coordinates = torch.tensor(output['boxes'], dtype=torch.float32).reshape(1, -1, 4)

        class Token:
            @staticmethod
            def char_to_token(offset):
                return 0

        return dict(pred_logits=logits, pred_boxes=coordinates, token=[Token()])


class RicePromptFallbackRegression(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='rice_fallback_test_')
        self.addCleanup(self.folder.cleanup)
        self.image_path = Path(self.folder.name) / 'small.jpg'
        Image.new('RGB', (80, 60), (210, 190, 120)).save(self.image_path)

    def infer(self, steps, category='rice', text=None, threshold=.23,
              automatic_selection=None):
        model = ScriptedModel(self, steps)
        runtime = dict(model=model, device='cpu',
                       transform=lambda image, target: (torch.zeros(3, 30, 40), target),
                       nested_tensor=lambda tensor: tensor)
        selected = automatic_selection or dict(accepted=True, category='rice')
        # Guard against accidental future checkpoint loading or CUDA selection.
        with patch.object(worker, 'load_runtime', return_value=(runtime, 0.)) as loader, \
             patch.object(worker, 'probe_category', return_value=selected) as probe, \
             patch.object(torch, 'load', side_effect=AssertionError('Weights must not load')), \
             patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU test')), \
             patch.object(torch.cuda, 'mem_get_info', side_effect=AssertionError('CPU test')), \
             patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CPU test')):
            result = worker.infer(self.image_path, category, device='cpu',
                                  threshold=threshold, text=text)
        loader.assert_called_once()
        self.assertIs(loader.call_args.args[0], torch)
        self.assertEqual(loader.call_args.args[1], 'cpu')
        self.assertEqual(probe.call_count, int(category == 'automatic'))
        self.assertEqual(model.steps, [], 'Expected model invocation was skipped')
        self.assertEqual(result['meta']['device'], 'cpu')
        self.assertEqual(result['meta']['threshold'], threshold)
        self.assertFalse(result['meta']['manual_exemplars'])
        self.assertEqual(result['width'], 80)
        self.assertEqual(result['height'], 60)
        return result, model.calls

    def test_zero_rice_seeds_accept_three_distinct_brown_seeds_and_refine(self):
        output, calls = self.infer([
            ('rice grain .', False, response([.1])),
            ('brown rice grain .', False, response([.9, .8, .7])),
            ('brown rice grain .', True, response([.8, .7, .24, .22])),
        ])
        self.assertEqual([len(call['pseudo']) for call in calls], [0, 0, 3])
        self.assertEqual(output['count'], 3)
        meta = output['meta']
        self.assertTrue(meta['prediction_valid'])
        self.assertEqual(meta['initial_count'], 3)
        self.assertEqual(meta['pseudo_exemplar_count'], 3)
        self.assertEqual(meta['prompt'], 'brown rice grain')
        self.assertEqual(meta['prompt_source'], 'system_preset_fallback')
        self.assertEqual(meta['count_kind'], 'grain_candidates')
        self.assertFalse(meta['count_is_total'])
        self.assertTrue(meta['review_required'])
        self.assertIn('候选', meta['user_message'])
        fallback = meta['prompt_fallback']
        self.assertTrue(fallback['accepted'])
        self.assertTrue(fallback['threshold_unchanged'])
        self.assertEqual(fallback['original_initial_count'], 0)
        self.assertEqual(fallback['distinct_seed_support'], 3)
        self.assertEqual(fallback['minimum_distinct_seeds'], 3)
        # The model receives coordinates at transformed-image size; response
        # exemplars retain original pixels, not the half-sized tensor pixels.
        self.assertTrue(torch.allclose(torch.tensor(output['exemplars']),
                        torch.tensor(calls[-1]['pseudo']) * 2))

    def test_two_distinct_seeds_cannot_enable_fallback(self):
        output, calls = self.infer([
            ('rice grain .', False, response([.1])),
            ('brown rice grain .', False, response([.9, .8])),
        ])
        self.assertEqual(len(calls), 2)
        self.assertEqual(output['count'], 0)
        self.assertEqual(output['exemplars'], [])
        self.assertFalse(output['meta']['prediction_valid'])
        self.assertEqual(output['meta']['prompt'], 'rice grain')
        self.assertEqual(output['meta']['diagnosis'], 'no_initial_exemplars')
        self.assertFalse(output['meta']['prompt_fallback']['accepted'])
        self.assertEqual(output['meta']['prompt_fallback']['distinct_seed_support'], 2)

    def test_three_overlapping_boxes_are_one_seed_not_three(self):
        output, calls = self.infer([
            ('rice grain .', False, response([.1])),
            ('brown rice grain .', False,
             response([.9, .8, .7], [DISTINCT[0]] * 3)),
        ])
        fallback = output['meta']['prompt_fallback']
        self.assertEqual(fallback['alternative_initial_count'], 3)
        self.assertEqual(fallback['distinct_seed_support'], 1)
        self.assertFalse(fallback['accepted'])
        self.assertEqual(output['count'], 0)
        self.assertEqual(len(calls), 2)

    def test_default_positive_seed_refines_without_trying_brown(self):
        output, calls = self.infer([
            ('rice grain .', False, response([.8])),
            ('rice grain .', True, response([.9])),
        ])
        self.assertEqual(output['count'], 1)
        self.assertIsNone(output['meta']['prompt_fallback'])
        self.assertEqual(output['meta']['prompt_source'], 'system_preset')
        self.assertEqual([len(call['pseudo']) for call in calls], [0, 1])

    def test_custom_text_never_enables_system_alias_fallback(self):
        for text in ('rice grain', 'custom grain phrase', ''):
            with self.subTest(text=text):
                caption = (text or 'rice grain') + ' .'
                output, calls = self.infer([(caption, False, response([.1]))], text=text)
                self.assertIsNone(output['meta']['prompt_fallback'])
                self.assertEqual(output['meta']['prompt_source'], 'text_override')
                self.assertEqual(output['count'], 0)
                self.assertEqual(len(calls), 1)

    def test_non_rice_categories_never_try_brown_grain_alias(self):
        for category, prompt in worker.PROMPTS.items():
            if category == 'rice':
                continue
            with self.subTest(category=category):
                output, calls = self.infer([(prompt + ' .', False, response([.1]))],
                                           category=category)
                self.assertIsNone(output['meta']['prompt_fallback'])
                self.assertEqual(output['count'], 0)
                self.assertEqual(len(calls), 1)

    def test_automatic_probe_selecting_rice_still_cannot_try_rice_alias(self):
        output, calls = self.infer([('rice grain .', False, response([.1]))],
                                   category='automatic')
        self.assertTrue(output['meta']['automatic_category'])
        self.assertEqual(output['meta']['detected_category'], 'rice')
        self.assertIsNone(output['meta']['prompt_fallback'])
        self.assertEqual(output['count'], 0)
        self.assertEqual(len(calls), 1)

    def test_custom_threshold_applies_to_both_seed_passes_and_final_boxes(self):
        output, calls = self.infer([
            ('rice grain .', False, response([.35])),
            ('brown rice grain .', False, response([.9, .8, .7, .35])),
            ('brown rice grain .', True, response([.41, .39])),
        ], threshold=.4)
        self.assertEqual(output['meta']['prompt_fallback']['alternative_initial_count'], 3)
        self.assertEqual(output['meta']['prompt_fallback']['distinct_seed_support'], 3)
        self.assertEqual(len(calls[-1]['pseudo']), 3)
        self.assertEqual(output['count'], 1)
        self.assertAlmostEqual(output['points'][0]['score'], .41, places=4)

    def test_successful_seed_fallback_does_not_invent_final_detections(self):
        output, _ = self.infer([
            ('rice grain .', False, response([.1])),
            ('brown rice grain .', False, response([.9, .8, .7])),
            ('brown rice grain .', True, response([.1])),
        ])
        self.assertTrue(output['meta']['prompt_fallback']['accepted'])
        self.assertFalse(output['meta']['prediction_valid'])
        self.assertEqual(output['meta']['diagnosis'], 'no_final_detections')
        self.assertEqual(output['count'], 0)
        self.assertEqual(output['points'], [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
