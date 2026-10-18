"""Routing contracts, plus real end-face pixels that must not become whole rods.

Mock geometry verifies dispatch isolation only; its supplied count is not an
accuracy measurement. Actual package-photo counting is tested separately.
"""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from smart_counter import detect_smart


ROOT = Path(__file__).resolve().parent


def conflict_hint(first='flat_end', second='laid_rods'):
    return dict(accepted=False, scene='unknown', suggested_scene=first,
                candidates=[dict(scene=first, score=.53567),
                            dict(scene=second, score=.46148)], margin=.07419)


def sample_geometry(framework='CardedRodTerminalTracks'):
    return ([dict(x=30., y=35., segment=[[30., 15.], [30., 55.]])],
            dict(framework=framework, prediction_valid=True, category='laid_rods',
                 count_kind='rod_candidates', device='cpu',
                 routing_evidence='complete_individual_terminal_tracks',
                 user_message='Only complete geometric evidence is accepted.'))


class AmbiguousRodRouting(unittest.TestCase):
    def setUp(self):
        self.image = np.full((80, 80, 3), 240, np.uint8)
        cv2.rectangle(self.image, (20, 10), (45, 65), (70, 140, 180), -1)
        self.enterContext(patch('particle_gate.detect_separated', return_value=None))
        self.scene = self.enterContext(patch('smart_counter.scene_hint',
                                             return_value=conflict_hint()))
        self.geometry = self.enterContext(patch('smart_counter._carded_rod_detection',
                                                return_value=sample_geometry()))
        self.model = self.enterContext(patch('auto_counter.detect_auto',
            side_effect=AssertionError('Unexpected neural inference.')))
        self.device = self.enterContext(patch('smart_counter.inference_device',
            side_effect=AssertionError('Unexpected GPU probing.')))

    def unavailable_runtime(self):
        self.device.side_effect = None
        self.device.return_value = None

    def test_conflict_uses_complete_tracks_without_accepting_scene_or_probing_gpu(self):
        for first, second in [('flat_end', 'laid_rods'), ('flat_end', 'chopsticks'),
                              ('laid_rods', 'flat_end'), ('chopsticks', 'flat_end'),
                              ('sharp', 'laid_rods'), ('laid_rods', 'sharp'),
                              ('flat_end', 'sharp')]:
            with self.subTest(first=first, second=second):
                self.scene.return_value = conflict_hint(first, second)
                result = detect_smart(self.image)
                self.assertEqual(result['count'], 1)  # Mock dispatch, not accuracy.
                self.assertEqual(result['meta']['routing'],
                                 'scene_then_strict_terminal_tracks')
                self.assertFalse(result['meta']['scene_classification']['accepted'])
                self.assertEqual(result['meta']['scene_classification']['scene'], 'unknown')
                self.assertEqual(result['meta']['user_message'],
                                 sample_geometry()[1]['user_message'])
        self.device.assert_not_called()
        self.model.assert_not_called()

    def test_accepted_rod_like_scenes_use_only_one_strict_track_count(self):
        with patch('laid_rod_counter.detect_laid_rods',
                   side_effect=AssertionError('Successful tracks must not be counted again.')):
            for scene in ['flat_end', 'sharp', 'laid_rods', 'chopsticks']:
                with self.subTest(scene=scene):
                    self.geometry.reset_mock()
                    self.scene.return_value = dict(accepted=True, scene=scene, candidates=[])
                    result = detect_smart(self.image)
                    self.assertEqual(result['count'], 1)  # Supplied route fixture.
                    self.assertEqual(result['meta']['framework'], 'CardedRodTerminalTracks')
                    self.geometry.assert_called_once_with(self.image)
        self.device.assert_not_called()
        self.model.assert_not_called()

    def test_failed_strict_tracks_preserve_accepted_end_on_neural_route(self):
        self.unavailable_runtime()
        self.geometry.return_value = None
        for scene in ['flat_end', 'sharp']:
            with self.subTest(scene=scene):
                self.scene.return_value = dict(accepted=True, scene=scene, candidates=[])
                result = detect_smart(self.image)
                self.assertIsNone(result['count'])
                self.assertEqual(result['meta']['diagnosis'], 'gpu_resources_unavailable')
        self.model.assert_not_called()

    def test_failed_strict_tracks_preserve_accepted_rod_geometry_route(self):
        self.geometry.return_value = None
        for scene in ['laid_rods', 'chopsticks']:
            with self.subTest(scene=scene), patch('laid_rod_counter.detect_laid_rods',
                                                 return_value=None) as legacy:
                self.scene.return_value = dict(accepted=True, scene=scene, candidates=[])
                result = detect_smart(self.image)
                self.assertIsNone(result['count'])
                self.assertEqual(result['meta']['diagnosis'], 'unresolved_individual_rods')
                legacy.assert_called_once_with(self.image)
        self.device.assert_not_called()
        self.model.assert_not_called()

    def test_legacy_geometry_cannot_override_uncertain_end_on_scene(self):
        self.unavailable_runtime()
        for framework in ['RodSilhouetteConsensus', 'LaidRodSections',
                          'BoxedRodTerminalTracks', 'CloseupRodTerminalFaces']:
            with self.subTest(framework=framework):
                self.geometry.return_value = sample_geometry(framework)
                result = detect_smart(self.image)
                self.assertIsNone(result['count'])
                self.assertEqual(result['meta']['diagnosis'], 'gpu_resources_unavailable')
        self.model.assert_not_called()

    def test_incomplete_or_invalid_tracks_do_not_supply_a_count(self):
        self.unavailable_runtime()
        cases = [None, ([], sample_geometry()[1])]
        for key, value in [('routing_evidence', 'partial_terminal_tracks'),
                           ('prediction_valid', False)]:
            points, meta = copy.deepcopy(sample_geometry())
            meta[key] = value
            cases.append((points, meta))
        points, meta = copy.deepcopy(sample_geometry())
        points[0].pop('segment')
        cases.append((points, meta))
        for index, geometry in enumerate(cases):
            with self.subTest(case=index):
                self.geometry.return_value = geometry
                result = detect_smart(self.image)
                self.assertIsNone(result['count'])
                self.assertEqual(result['points'], [])
                self.assertFalse(result['meta']['prediction_valid'])
        self.model.assert_not_called()

    def test_failed_rod_first_conflict_retains_rod_abstention(self):
        self.scene.return_value = conflict_hint('laid_rods', 'flat_end')
        self.geometry.return_value = None
        result = detect_smart(self.image)
        self.assertIsNone(result['count'])
        self.assertEqual(result['meta']['diagnosis'], 'uncertain_whole_rod_view')
        self.device.assert_not_called()
        self.model.assert_not_called()

    def test_nonconflicting_unaccepted_or_other_scene_never_attempts_rod_geometry(self):
        self.unavailable_runtime()
        hints = [None,
                 dict(accepted=True, scene='cotton', candidates=[]),
                 conflict_hint('flat_end', 'cotton'),
                 dict(accepted=False, scene='unknown', suggested_scene='flat_end',
                      candidates=[dict(scene='flat_end', score=.6)]),
                 dict(accepted=False, scene='unknown', suggested_scene='laid_rods')]
        for hint in hints:
            with self.subTest(hint=hint):
                self.scene.return_value = hint
                result = detect_smart(self.image)
                self.assertIsNone(result['count'])
        self.geometry.assert_not_called()
        self.model.assert_not_called()

    def test_real_end_face_photos_do_not_become_whole_rod_counts(self):
        # Stop mocking geometry: real pixel rejection is essential because the
        # conflict includes true bundles. The classifier remains a route fixture.
        from carded_rod_counter import detect_carded_rods
        self.geometry.side_effect = detect_carded_rods
        self.unavailable_runtime()
        for name in ['dense_flat_craft.jpg', 'dense_flat_custom.jpg', 'bamboo_original.png']:
            with self.subTest(photo=name):
                path = ROOT / 'samples' / 'counting_eval' / name
                image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(image)
                for hint in [conflict_hint(), dict(accepted=True, scene='flat_end'),
                             dict(accepted=True, scene='sharp')]:
                    self.scene.return_value = hint
                    result = detect_smart(image)
                    self.assertIsNone(result['count'])
                    self.assertEqual(result['points'], [])
        self.model.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
