"""Pure-score category-selection regressions; no model, image or GPU is loaded."""
import builtins
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

from countgdpp_worker import PROBE_CATEGORIES, choose_category


def candidates(**rows):
    """Handwritten score fixtures test decisions, not prediction accuracy."""
    result = []
    for category in PROBE_CATEGORIES:
        row = dict(category=category, phrase=category, evidence=0., peak=0.,
                   support=0, seed_support=0)
        row.update(rows.get(category, {}))
        result.append(row)
    return result


class ProbeSelectionRegression(unittest.TestCase):
    def test_worker_module_import_does_not_require_model_libraries(self):
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'torchvision', 'transformers', 'numpy', 'PIL'}:
                raise AssertionError('Worker module import must not load model libraries: ' + name)
            return original_import(name, *args, **kwargs)

        spec = importlib.util.spec_from_file_location(
            'worker_import_regression', Path(__file__).with_name('countgdpp_worker.py'))
        module = importlib.util.module_from_spec(spec)
        with patch('builtins.__import__', side_effect=guarded_import):
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.choose_category))

    def test_flat_and_sharp_same_family_do_not_veto_each_other(self):
        for strongest in ('bamboo', 'bamboo_sharp'):
            with self.subTest(strongest=strongest):
                other = 'bamboo_sharp' if strongest == 'bamboo' else 'bamboo'
                rows = candidates(**{
                    strongest: dict(evidence=.48, support=12, seed_support=12),
                    other: dict(evidence=.47, support=10, seed_support=10),
                    'cotton': dict(evidence=.12, support=0, seed_support=0)})
                for independent in (False, True):
                    result = choose_category(rows, independent=independent)
                    self.assertTrue(result['accepted'])
                    self.assertFalse(result['competing_targets'])
                    self.assertEqual(result['category'], strongest)
                    self.assertAlmostEqual(result['margin'], .36)

    def test_cross_family_strong_candidate_is_rejected(self):
        rows = candidates(bamboo=dict(evidence=.55, support=15, seed_support=15),
                          cotton=dict(evidence=.34, support=8, seed_support=8))
        for independent in (False, True):
            with self.subTest(independent=independent):
                result = choose_category(rows, independent=independent)
                self.assertFalse(result['accepted'])
                self.assertTrue(result['competing_targets'])
                self.assertIsNone(result['category'])

    def test_low_combined_evidence_can_pass_independent_confirmation(self):
        rows = candidates(mung=dict(evidence=.212, peak=.23, support=3, seed_support=3))
        combined = choose_category(rows)
        independent = choose_category(rows, independent=True)
        self.assertFalse(combined['accepted'])
        self.assertIsNone(combined['category'])
        self.assertEqual(combined['method'], 'multi_phrase_image_probe')
        self.assertTrue(independent['accepted'])
        self.assertEqual(independent['category'], 'mung')
        self.assertEqual(independent['method'], 'independent_phrase_confirmation')
        self.assertEqual(independent['candidates'], rows)
        self.assertEqual(independent['score_type'], 'uncalibrated_mean_token_evidence')

    def test_insufficient_support_is_rejected(self):
        rows = candidates(mung=dict(evidence=.40, support=2, seed_support=2))
        for independent in (False, True):
            with self.subTest(independent=independent):
                self.assertFalse(choose_category(rows, independent=independent)['accepted'])

    def test_close_cross_family_scores_are_rejected_without_strong_competitor(self):
        rows = candidates(mung=dict(evidence=.36, support=7, seed_support=7),
                          rice=dict(evidence=.32, support=2, seed_support=2))
        for independent in (False, True):
            with self.subTest(independent=independent):
                result = choose_category(rows, independent=independent)
                self.assertFalse(result['competing_targets'])
                self.assertFalse(result['accepted'])
                self.assertIsNone(result['category'])
                self.assertAlmostEqual(result['margin'], .04)

    def test_independent_confirmation_still_requires_absolute_evidence(self):
        rows = candidates(mung=dict(evidence=.17, support=8, seed_support=8))
        self.assertFalse(choose_category(rows, independent=True)['accepted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
