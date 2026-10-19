"""Restricted end-face review contracts, using synthetic boxes and CPU logits.

No checkpoint, photographs, truth files or GPU are used. These tests establish
dispatch and suppression limits; supplied counts do not measure model accuracy.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

import countgdpp_worker as worker
from end_face_review import mixed_shaft_evidence, suppress_uniform_end_centers


def point(x, y, minor=20., score=.9, aspect=1.):
    width, height = minor*aspect, minor
    return dict(x=x, y=y, score=score,
                box=[x-width/2, y-height/2, x+width/2, y+height/2])


def uniform_grid():
    return [point(25.+30*x, 25.+30*y) for y in range(3) for x in range(4)]


SEEDS = [[10., 10., 30., 30.], [50., 50., 70., 70.]]


class EndFaceShapeEvidence(unittest.TestCase):
    def test_compact_examples_and_much_larger_shaft_trigger(self):
        values = uniform_grid()+[point(120., 120., minor=20, aspect=5)]
        evidence = mixed_shaft_evidence(values, SEEDS)
        self.assertIsNotNone(evidence)
        self.assertEqual(evidence['shaft_box_candidates'], 1)
        self.assertEqual(evidence['original_prediction_count'], 13)
        self.assertEqual(evidence['seed_long_side'], 20.)

    def test_compact_only_predictions_leave_original_path_unchanged(self):
        self.assertIsNone(mixed_shaft_evidence(uniform_grid(), SEEDS))

    def test_missing_single_or_noncompact_examples_cannot_trigger(self):
        values = [point(120., 120., minor=20, aspect=5)]
        for seeds in ([], SEEDS[:1], [SEEDS[0], [50., 50., 120., 60.]],
                      [SEEDS[0], [50., 50., 50., 70.]]):
            with self.subTest(seeds=seeds):
                self.assertIsNone(mixed_shaft_evidence(values, seeds))

    def test_slightly_elongated_or_small_regions_are_not_whole_shafts(self):
        for value in (point(120., 120., minor=30, aspect=2.5),
                      point(120., 120., minor=10, aspect=4)):
            with self.subTest(value=value):
                self.assertIsNone(mixed_shaft_evidence([value], SEEDS))

    def test_separated_uniform_end_faces_are_all_retained(self):
        values = uniform_grid()
        before = copy.deepcopy(values)
        retained, report = suppress_uniform_end_centers(values)
        self.assertEqual(len(retained), len(values))
        self.assertEqual(values, before)
        self.assertTrue(report['applied'])
        self.assertEqual(report['removed'], 0)
        self.assertEqual(report['minimum_center_separation'], 14.)

    def test_near_duplicate_retains_the_higher_score(self):
        values = uniform_grid()
        duplicate = point(27., 26., score=.3)
        values.append(duplicate)
        retained, report = suppress_uniform_end_centers(values)
        self.assertEqual(len(retained), 12)
        self.assertNotIn(duplicate, retained)
        self.assertIn(values[0], retained)
        self.assertEqual(report['removed'], 1)

    def test_small_candidate_sets_never_use_uniform_array_assumption(self):
        values = [point(25., 25.), point(26., 25., score=.3)]
        retained, report = suppress_uniform_end_centers(values)
        self.assertEqual(retained, values)
        self.assertFalse(report['applied'])
        self.assertEqual(report['removed'], 0)

    def test_mixed_size_or_elongated_sets_are_not_merged(self):
        cases = [[point(20+25*i, 20, minor=10 if i < 6 else 30)
                  for i in range(12)],
                 [point(20+25*i, 20, aspect=4) for i in range(12)]]
        for values in cases:
            with self.subTest(values=values):
                retained, report = suppress_uniform_end_centers(values)
                self.assertEqual(retained, values)
                self.assertFalse(report['applied'])

    def test_a_minority_small_face_is_not_absorbed_into_a_larger_neighbor(self):
        # Percentiles alone can conceal a size outlier in a mostly uniform set.
        # Nearby, distinct-size instances must not become a duplicate pair.
        values = uniform_grid()
        smaller = point(35., 25., minor=5, score=.3)
        values.append(smaller)
        retained, report = suppress_uniform_end_centers(values)
        self.assertIn(smaller, retained)
        self.assertEqual(len(retained), 13)
        self.assertEqual(report['removed'], 0)

    def test_invalid_box_dimensions_cannot_enable_suppression(self):
        values = uniform_grid()
        values[-1]['box'][2] = values[-1]['box'][0]
        retained, report = suppress_uniform_end_centers(values)
        self.assertEqual(retained, values)
        self.assertFalse(report['applied'])


class ScriptedModel:
    def __init__(self, test, responses):
        self.test, self.responses = test, list(responses)
        self.calls = []

    def __call__(self, nested, exemplars_image, boxes, targets, labels, captions):
        self.test.assertTrue(self.responses, 'Unexpected extra model forward')
        expected_caption, probabilities, positions = self.responses.pop(0)
        self.test.assertEqual(captions, [expected_caption])
        self.test.assertIs(nested, exemplars_image)
        self.test.assertEqual(boxes[0].device.type, 'cpu')
        self.calls.append(dict(caption=captions[0], pseudo=boxes[0].clone()))

        class Token:
            @staticmethod
            def char_to_token(offset):
                return 0

        return dict(pred_logits=torch.logit(torch.tensor(probabilities)).reshape(1, -1, 1),
                    pred_boxes=torch.tensor(positions).reshape(1, -1, 4), token=[Token()])


class EndFaceWorkerDispatch(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='end_review_test_')
        self.addCleanup(self.folder.cleanup)
        self.image_path = Path(self.folder.name)/'synthetic.png'
        Image.new('RGB', (80, 60), (220, 190, 140)).save(self.image_path)
        self.seed_boxes = [[.2, .25, .12, .16], [.5, .5, .12, .16],
                           [.8, .75, .12, .16]]
        self.mixed_boxes = [self.seed_boxes[0], [.5, .7, .85, .12]]

    def infer(self, responses, category='bamboo', text=None, threshold=.23,
              automatic_selection=None):
        model = ScriptedModel(self, responses)
        runtime = dict(model=model, device='cpu',
                       transform=lambda image, target: (torch.zeros(3, 30, 40), target),
                       nested_tensor=lambda tensor: tensor)
        with patch.object(worker, 'load_runtime', return_value=(runtime, 0.)), \
             patch.object(worker, 'probe_category', return_value=automatic_selection) as probe, \
             patch.object(torch, 'load', side_effect=AssertionError('No weights')), \
             patch.object(torch.cuda, 'is_available', side_effect=AssertionError('CPU only')), \
             patch.object(torch.cuda, 'mem_get_info', side_effect=AssertionError('CPU only')), \
             patch.object(torch.cuda, 'synchronize', side_effect=AssertionError('CPU only')):
            result = worker.infer(self.image_path, category, 'cpu', threshold, text)
        self.assertEqual(model.responses, [])
        self.assertEqual(probe.call_count, int(category == 'automatic'))
        return result, model.calls

    def steps(self, category='bamboo', text=None, refined=None):
        caption = (text if text is not None else worker.PROMPTS[category])+' .'
        steps = [(caption, [.9, .8, .7], self.seed_boxes),
                 (caption, [.9, .8], self.mixed_boxes)]
        if refined is not None:
            probabilities, boxes = refined
            steps.append(('circular wooden end face .', probabilities, boxes))
        return steps

    def test_confirm_only_end_route_and_reuse_exact_same_visual_examples(self):
        result, calls = self.infer(self.steps(refined=([.9, .22], self.seed_boxes[:2])))
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(calls[0]['pseudo']), 0)
        self.assertTrue(torch.equal(calls[1]['pseudo'], calls[2]['pseudo']))
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['meta']['threshold'], .23)
        review = result['meta']['end_face_review']
        self.assertTrue(review['threshold_unchanged'])
        self.assertEqual(review['refinement_prompt'], 'circular wooden end face')
        self.assertEqual(review['original_prediction_count'], 2)
        self.assertEqual(review['raw_refined_count'], 1)
        self.assertEqual(result['meta']['count_kind'], 'flat_end_candidates')
        self.assertFalse(result['meta']['count_is_total'])
        self.assertTrue(result['meta']['review_required'])
        self.assertFalse(result['meta']['manual_exemplars'])
        self.assertFalse(review['center_review']['applied'])
        self.assertFalse(result['meta']['final_nms'])

    def test_nondefault_threshold_is_preserved_for_confirmation(self):
        result, _ = self.infer(self.steps(refined=([.7, .5], self.seed_boxes[:2])),
                               threshold=.6)
        self.assertEqual(result['count'], 1)
        self.assertEqual(result['meta']['threshold'], .6)
        self.assertTrue(result['meta']['end_face_review']['threshold_unchanged'])

    def test_empty_confirmation_never_restores_old_mixed_shaft_predictions(self):
        result, calls = self.infer(self.steps(refined=([.1, .1], self.seed_boxes[:2])))
        self.assertEqual(len(calls), 3)
        self.assertEqual(result['points'], [])
        self.assertFalse(result['meta']['prediction_valid'])
        self.assertEqual(result['meta']['diagnosis'], 'no_final_detections')
        self.assertEqual(result['meta']['end_face_review']['raw_refined_count'], 0)
        self.assertFalse(result['meta']['final_nms'])

    def test_whole_rods_other_categories_and_explicit_text_are_unchanged(self):
        for category in worker.PROMPTS:
            if category == 'bamboo':
                continue
            with self.subTest(category=category):
                result, calls = self.infer(self.steps(category), category=category)
                self.assertEqual(len(calls), 2)
                self.assertEqual(result['count'], 2)
                self.assertNotIn('end_face_review', result['meta'])
                self.assertFalse(result['meta']['final_nms'])
        for text in ('bamboo stick end', 'wooden rod', ''):
            with self.subTest(text=text):
                # Existing empty-text behavior uses the default prompt but must
                # still retain explicit override isolation for review dispatch.
                caption = (text or worker.PROMPTS['bamboo'])+' .'
                steps = [(caption, [.9, .8, .7], self.seed_boxes),
                         (caption, [.9, .8], self.mixed_boxes)]
                result, calls = self.infer(steps, text=text)
                self.assertEqual(len(calls), 2)
                self.assertNotIn('end_face_review', result['meta'])
                self.assertEqual(result['meta']['prompt_source'], 'text_override')

    def test_automatic_probe_selecting_end_faces_can_receive_same_review(self):
        result, calls = self.infer(self.steps(refined=([.9], self.seed_boxes[:1])),
            category='automatic', automatic_selection=dict(accepted=True, category='bamboo'))
        self.assertEqual(len(calls), 3)
        self.assertTrue(result['meta']['automatic_category'])
        self.assertEqual(result['meta']['category'], 'bamboo')
        self.assertIn('end_face_review', result['meta'])

    def test_clean_compact_path_retains_original_two_forwards_and_predictions(self):
        caption = worker.PROMPTS['bamboo']+' .'
        result, calls = self.infer([(caption, [.9, .8, .7], self.seed_boxes),
                                   (caption, [.9, .8, .7], self.seed_boxes)])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['count'], 3)
        self.assertNotIn('end_face_review', result['meta'])
        self.assertFalse(result['meta']['final_nms'])

    def test_single_auto_example_cannot_enable_end_face_confirmation(self):
        caption = worker.PROMPTS['bamboo']+' .'
        result, calls = self.infer([(caption, [.9], self.seed_boxes[:1]),
                                   (caption, [.9, .8], self.mixed_boxes)])
        self.assertEqual(len(calls), 2)
        self.assertEqual(result['count'], 2)
        self.assertNotIn('end_face_review', result['meta'])

    def test_uniform_confirmation_suppresses_only_close_lower_score_candidate(self):
        grid = [[.15+.2*x, .2+.3*y, .1, .12] for y in range(3) for x in range(4)]
        duplicate = [.16, .2, .1, .12]
        result, _ = self.infer(self.steps(refined=([.9]*12+[.3], grid+[duplicate])))
        self.assertEqual(result['count'], 12)
        report = result['meta']['end_face_review']['center_review']
        self.assertTrue(report['applied'])
        self.assertEqual(report['removed'], 1)
        self.assertEqual(result['meta']['end_face_review']['raw_refined_count'], 13)
        self.assertTrue(result['meta']['final_nms'])
        self.assertFalse(any(p['score'] == .3 for p in result['points']))

    def test_user_point_visual_mode_does_not_apply_uniform_end_face_review(self):
        references = [dict(x=20., y=20., diameter=12.)]
        expected = dict(count=1, points=[dict(x=20., y=20.)], meta={})
        with patch.object(worker, 'infer_point_examples', return_value=expected) as visual, \
             patch.object(worker, '_infer_once', side_effect=AssertionError('No auto review')), \
             patch('end_face_review.suppress_uniform_end_centers',
                   side_effect=AssertionError('No uniform assumption for user examples')):
            result = worker.infer(self.image_path, 'bamboo', 'cpu', exemplar_points=references)
        self.assertIs(result, expected)
        visual.assert_called_once_with(self.image_path, references, 'cpu', .23)


if __name__ == '__main__':
    unittest.main(verbosity=2)
