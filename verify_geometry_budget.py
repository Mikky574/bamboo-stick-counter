"""Bounded-work regressions for the conservative packaged-rod geometry path."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import boxed_rod_counter as counter


ROOT = Path(__file__).resolve().parent


class GeometryBudgetChecks(unittest.TestCase):
    def test_dense_frame_edges_are_rejected_before_pair_scans(self):
        gray = np.zeros((1200, 1200), np.uint8)
        frames = [(0., float(x), 0, 1199) for x in range(
            counter.PARAMETERS['maximum_frame_edges'] + 1)]
        # Complexity guards reject the scene; they must not truncate a dense
        # edge list and later accept a plausible subset of instances.
        with patch.object(counter.np, 'mean', side_effect=AssertionError(
                'Dense frames must not enter quadratic row scanning')):
            self.assertEqual(counter._frame_pairs(frames, gray), [])

    def test_expired_budget_skips_even_a_small_frame_pair(self):
        gray = np.zeros((100, 100), np.uint8)
        frames = [(0., 10., 0, 99), (0., 90., 0, 99)]
        with patch.object(counter.time, 'perf_counter', return_value=2.):
            with patch.object(counter.np, 'mean', side_effect=AssertionError(
                    'Expired budget must not scan another pair')):
                self.assertEqual(counter._frame_pairs(frames, gray, 1.), [])

    def test_one_budget_covers_all_rotations_instead_of_restarting(self):
        clock = [100.]
        deadlines = []

        def unsupported_view(image, deadline):
            deadlines.append(deadline)
            clock[0] += counter.PARAMETERS['maximum_processing_seconds'] * .7
            return None

        with patch.object(counter.time, 'perf_counter', side_effect=lambda: clock[0]):
            with patch.object(counter, '_detect_boxed_rods_upright',
                              side_effect=unsupported_view):
                self.assertIsNone(counter.detect_boxed_rods(
                    np.zeros((100, 100, 3), np.uint8)))
        self.assertEqual(len(deadlines), 2)
        self.assertEqual(deadlines[0], deadlines[1])

    def test_real_dense_hough_inputs_finish_in_a_bounded_subprocess(self):
        # Before the guard, just 40 long strokes (80 Hough edges) took 17.57 s.
        # Run in another process so a future regression cannot hang this suite.
        worker = """
import cv2, json, numpy as np, time
from boxed_rod_counter import detect_boxed_rods
results = []
for count in (40, 80):
    image = np.full((1200, 1200, 3), 245, np.uint8)
    for x in np.linspace(30, 1170, count).astype(int):
        cv2.line(image, (int(x), 20), (int(x), 1180), (30, 30, 30), 2)
    started = time.perf_counter()
    result = detect_boxed_rods(image)
    results.append(dict(lines=count, count=None if result is None else len(result[0]),
                        seconds=time.perf_counter()-started))
print(json.dumps(results))
"""
        try:
            process = subprocess.run([sys.executable, '-c', worker], cwd=ROOT,
                                     capture_output=True, text=True, timeout=8)
        except subprocess.TimeoutExpired:
            self.fail('Dense geometry exceeded the outer 8 second safety limit')
        self.assertEqual(process.returncode, 0, process.stderr)
        measurements = json.loads(process.stdout)
        self.assertEqual(len(measurements), 2)
        for measurement in measurements:
            self.assertIsNone(measurement['count'], measurement)
            # The cooperative deadline cannot interrupt one OpenCV kernel;
            # allow scheduling and bounded 1200 px Hough work to complete.
            self.assertLess(measurement['seconds'], 3., measurement)


if __name__ == '__main__':
    unittest.main(verbosity=2)
