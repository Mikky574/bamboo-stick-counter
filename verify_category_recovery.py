"""Bounded semantic recovery contracts; no photographs, weights or GPU needed.

Scripted outcomes exercise dispatch and failure handling only. Their supplied
counts are not model accuracy measurements or image-specific expected answers.
"""
import copy
from pathlib import Path
import unittest
from unittest.mock import call, patch

import numpy as np

import auto_counter
import countgdpp_worker as worker


def failed(category='rice', diagnosis='no_initial_exemplars', device='cpu'):
    return dict(count=0, points=[], width=80, height=60, exemplars=[],
                meta=dict(category=category, detected_category=category,
                          device=device, diagnosis=diagnosis,
                          prediction_valid=False, elapsed_seconds=2.25,
                          model_load_seconds=1.5, model_reused=False,
                          automatic_category=False))


def detected(category='insects', device='cpu'):
    return dict(count=1, points=[dict(x=20., y=30., score=.8)],
                width=80, height=60, exemplars=[],
                meta=dict(category=category, detected_category=category,
                          device=device, diagnosis='detections_returned',
                          prediction_valid=True, elapsed_seconds=.75,
                          model_load_seconds=0., model_reused=True,
                          automatic_category=True,
                          category_selection=dict(accepted=True, category=category),
                          manual_exemplars=False, count_is_total=False,
                          review_required=True, count_kind='insect_candidates'))


class BoundedCategoryRecovery(unittest.TestCase):
    def setUp(self):
        self.image = Path('not-a-real-image.png')

    def test_failure_recovers_once_on_same_cpu_or_cuda_and_threshold(self):
        for device in ('cpu', 'cuda'):
            for diagnosis in ('no_initial_exemplars', 'no_final_detections'):
                with self.subTest(device=device, diagnosis=diagnosis):
                    initial = failed(diagnosis=diagnosis, device=device)
                    replacement = detected(device=device)
                    with patch.object(worker, '_infer_once',
                                      side_effect=[initial, replacement]) as infer_once:
                        result = worker.infer(self.image, 'rice', device, .4,
                                              allow_category_recovery=True)
                    self.assertEqual(infer_once.call_args_list, [
                        call(self.image, 'rice', device, .4, None),
                        call(self.image, 'automatic', device, .4, None),
                    ])
                    self.assertIs(result, replacement)
                    self.assertEqual(result['count'], len(result['points']))
                    meta = result['meta']
                    self.assertEqual(meta['category'], 'insects')
                    self.assertEqual(meta['detected_category'], 'insects')
                    self.assertTrue(meta['prediction_valid'])
                    self.assertTrue(meta['automatic_category'])
                    self.assertFalse(meta['manual_exemplars'])
                    self.assertFalse(meta['count_is_total'])
                    self.assertTrue(meta['review_required'])
                    self.assertEqual(meta['elapsed_seconds'], 3.)
                    self.assertEqual(meta['model_load_seconds'], 1.5)
                    self.assertFalse(meta['model_reused'])
                    self.assertEqual(meta['category_recovery'], dict(
                        attempted=True, original_category='rice',
                        original_diagnosis=diagnosis, accepted=True,
                        category_selection=dict(accepted=True, category='insects')))

    def test_default_and_false_leave_existing_failure_untouched(self):
        for options in ({}, dict(allow_category_recovery=False)):
            with self.subTest(options=options):
                initial = failed()
                original = copy.deepcopy(initial)
                with patch.object(worker, '_infer_once', return_value=initial) as infer_once:
                    result = worker.infer(self.image, 'rice', 'cpu', **options)
                self.assertIs(result, initial)
                self.assertEqual(result, original)
                infer_once.assert_called_once_with(self.image, 'rice', 'cpu', .23, None)

    def test_custom_text_including_empty_string_is_never_replaced(self):
        for text in ('insects', 'rice grain', ''):
            with self.subTest(text=text):
                initial = failed()
                with patch.object(worker, '_infer_once', return_value=initial) as infer_once:
                    result = worker.infer(self.image, 'rice', 'cpu', text=text,
                                          allow_category_recovery=True)
                infer_once.assert_called_once_with(self.image, 'rice', 'cpu', .23, text)
                self.assertIs(result, initial)
                self.assertNotIn('category_recovery', result['meta'])

    def test_valid_nonzero_result_has_no_second_probe(self):
        initial = detected('rice')
        with patch.object(worker, '_infer_once', return_value=initial) as infer_once:
            result = worker.infer(self.image, 'rice', 'cpu', allow_category_recovery=True)
        self.assertIs(result, initial)
        infer_once.assert_called_once()
        self.assertNotIn('category_recovery', result['meta'])

    def test_automatic_request_does_not_reenter_recovery(self):
        initial = failed('automatic')
        with patch.object(worker, '_infer_once', return_value=initial) as infer_once:
            result = worker.infer(self.image, 'automatic', 'cpu',
                                  allow_category_recovery=True)
        self.assertIs(result, initial)
        infer_once.assert_called_once_with(self.image, 'automatic', 'cpu', .23, None)

    def test_unsupported_diagnoses_do_not_trigger_semantic_retry(self):
        for diagnosis in ('ambiguous_category', 'target_shape_mismatch',
                          'gpu_resources_unavailable', 'unknown'):
            with self.subTest(diagnosis=diagnosis):
                initial = failed(diagnosis=diagnosis)
                with patch.object(worker, '_infer_once', return_value=initial) as infer_once:
                    result = worker.infer(self.image, 'rice', 'cpu',
                                          allow_category_recovery=True)
                self.assertIs(result, initial)
                infer_once.assert_called_once()

    def test_missing_explicit_invalid_flag_cannot_trigger_recovery(self):
        initial = failed()
        del initial['meta']['prediction_valid']
        with patch.object(worker, '_infer_once', return_value=initial) as infer_once:
            result = worker.infer(self.image, 'rice', 'cpu', allow_category_recovery=True)
        self.assertIs(result, initial)
        infer_once.assert_called_once()

    def test_same_or_ambiguous_alternative_retains_original_invalid_result(self):
        same = detected('rice')
        ambiguous = failed('automatic', 'ambiguous_category')
        ambiguous['meta']['category_selection'] = dict(accepted=False, category=None)
        invalid = failed('insects', 'no_initial_exemplars')
        for retry in (same, ambiguous, invalid):
            with self.subTest(retry=retry['meta']['diagnosis'],
                              category=retry['meta']['category']):
                initial = failed()
                with patch.object(worker, '_infer_once',
                                  side_effect=[initial, copy.deepcopy(retry)]) as infer_once:
                    result = worker.infer(self.image, 'rice', 'cpu',
                                          allow_category_recovery=True)
                self.assertIs(result, initial)
                self.assertIsNone(result['count'])
                self.assertEqual(result['points'], [])
                self.assertFalse(result['meta']['prediction_valid'])
                self.assertEqual(result['meta']['category'], 'rice')
                self.assertEqual(result['meta']['diagnosis'], 'no_initial_exemplars')
                self.assertFalse(result['meta']['category_recovery']['accepted'])
                self.assertEqual(infer_once.call_count, 2)

    def test_claimed_valid_but_empty_retry_cannot_be_accepted(self):
        retry = detected()
        retry.update(count=0, points=[])
        with patch.object(worker, '_infer_once', side_effect=[failed(), retry]) as infer_once:
            result = worker.infer(self.image, 'rice', 'cpu', allow_category_recovery=True)
        self.assertIsNone(result['count'])
        self.assertFalse(result['meta']['category_recovery']['accepted'])
        self.assertEqual(infer_once.call_count, 2)

    def test_runtime_errors_from_first_or_second_pass_propagate(self):
        for responses, expected_calls in (([RuntimeError('model import failed')], 1),
                                          ([failed(), RuntimeError('model forward failed')], 2)):
            with self.subTest(expected_calls=expected_calls):
                with patch.object(worker, '_infer_once', side_effect=responses) as infer_once:
                    with self.assertRaisesRegex(RuntimeError, 'model .* failed'):
                        worker.infer(self.image, 'rice', 'cpu', allow_category_recovery=True)
                self.assertEqual(infer_once.call_count, expected_calls)

    def test_recovery_requires_real_boolean_before_model_execution(self):
        for value in ('true', 'false', 0, 1, None):
            with self.subTest(value=value):
                with patch.object(worker, '_infer_once') as infer_once:
                    with self.assertRaises(ValueError):
                        worker.infer(self.image, 'rice', 'cpu',
                                     allow_category_recovery=value)
                infer_once.assert_not_called()


class RecoveryBoundaryContracts(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((60, 80, 3), dtype=np.uint8)
        self.image[15:45, 20:50] = 200  # Pass the existing blank-image gate.

    def test_auto_counter_adds_setting_only_when_true(self):
        for value in (False, True):
            with self.subTest(value=value), \
                    patch.object(auto_counter, 'readiness', return_value=dict(installed=True)), \
                    patch.object(auto_counter, '_execute', return_value=detected()) as execute:
                auto_counter.detect_auto(self.image, 'rice', 'cpu',
                                         allow_category_recovery=value)
            self.assertIs(execute.call_args.args[0], self.image)
            self.assertEqual(execute.call_args.args[1:3], ('countgdpp', 'cpu'))
            self.assertEqual(execute.call_args.args[3],
                             dict(category='rice', allow_category_recovery=True)
                             if value else dict(category='rice'))
            self.assertEqual(execute.call_args.kwargs, dict(keep_warm=False))

    def test_auto_counter_rejects_strings_numbers_and_null_before_launch(self):
        for value in ('true', 'false', 0, 1, None):
            with self.subTest(value=value), \
                    patch.object(auto_counter, 'readiness') as readiness, \
                    patch.object(auto_counter, '_execute') as execute:
                with self.assertRaises(ValueError):
                    auto_counter.detect_auto(self.image, 'rice', 'cpu',
                                             allow_category_recovery=value)
            readiness.assert_not_called()
            execute.assert_not_called()

    def test_insects_is_supported_without_retraining_classifier_contract(self):
        import supervised_scene
        self.assertEqual(worker.PROMPTS['insects'], 'insect')
        self.assertIn('insects', worker.PROBE_CATEGORIES)
        self.assertIn('insects', auto_counter.CATEGORIES)
        self.assertNotIn('insects', supervised_scene.CLASSES)
        self.assertEqual(len(supervised_scene.CLASSES), 8)

    def test_api_explicit_cpu_gateway_payload_forwards_recovery_switch(self):
        import app as application
        for options in ({}, dict(allow_category_recovery=True)):
            with self.subTest(options=options), \
                    patch.object(application, 'decode', return_value=self.image), \
                    patch.object(auto_counter, 'detect_auto', return_value=detected()) as detect:
                response = application.app.test_client().post('/api/count', json=dict(
                    image='scripted', engine='countgdpp', category='rice',
                    device='cpu', keep_warm=False, **options))
            self.assertEqual(response.status_code, 200)
            self.assertIs(detect.call_args.args[0], self.image)
            self.assertEqual(detect.call_args.args[1:], ('rice', 'cpu'))
            self.assertEqual(detect.call_args.kwargs, dict(
                keep_warm=False, allow_category_recovery=bool(options)))

    def test_api_invalid_switch_is_client_error_and_never_launches_model(self):
        import app as application
        with patch.object(application, 'decode', return_value=self.image), \
                patch.object(auto_counter, '_execute') as execute:
            response = application.app.test_client().post('/api/count', json=dict(
                image='scripted', engine='countgdpp', category='rice',
                device='cpu', allow_category_recovery='true'))
        self.assertEqual(response.status_code, 400)
        execute.assert_not_called()

    def test_smart_system_route_enables_recovery_without_overwriting_selected_category(self):
        import smart_counter
        selected = detected()
        selected['meta']['category_recovery'] = dict(accepted=True)
        with patch('particle_gate.detect_separated', return_value=None), \
                patch.object(smart_counter, 'scene_hint',
                             return_value=dict(scene='rice', accepted=True)), \
                patch.object(smart_counter, 'inference_device', return_value='cuda'), \
                patch.object(auto_counter, 'detect_auto', return_value=selected) as detect:
            result = smart_counter.detect_smart(self.image)
        self.assertIs(detect.call_args.args[0], self.image)
        self.assertEqual(detect.call_args.args[1:], ('rice', 'cuda'))
        self.assertEqual(detect.call_args.kwargs,
                         dict(keep_warm=True, allow_category_recovery=True))
        self.assertEqual(result['meta']['category'], 'insects')
        self.assertEqual(result['meta']['scene_classification']['scene'], 'rice')
        self.assertEqual(result['meta']['routing'], 'empty_scene_hint_then_semantic_recovery')
        self.assertFalse(result['meta']['count_is_total'])
        self.assertTrue(result['meta']['review_required'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
