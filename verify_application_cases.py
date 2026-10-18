"""Lightweight CPU API regression checks for application counting examples.

Run directly with Python. No web server, deep model, or GPU is started.
"""
from __future__ import annotations

import base64
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import cv2
import numpy as np

from app import app


ROOT = Path(__file__).resolve().parent
ORIGINALS = ROOT / "samples/application_eval_20261018/originals"


def file_data_url(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def forbidden_model_call(*args, **kwargs):
    raise AssertionError("A CPU regression check attempted to invoke a deep model")


class ApplicationCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cv2.setNumThreads(4)
        cls.client = app.test_client()
        cls.mung = file_data_url(ORIGINALS / "mung_dark.jpeg")
        cls.coins = file_data_url(ORIGINALS / "coins_white.jpeg")
        ok, encoded = cv2.imencode(".png", np.full((256, 256, 3), 255, np.uint8))
        if not ok:
            raise AssertionError("Could not encode blank regression input")
        cls.blank = "data:image/png;base64," + base64.b64encode(encoded).decode("ascii")
        # A route regression must fail the check instead of loading model weights.
        guard = types.ModuleType("auto_counter")
        guard.detect_auto = forbidden_model_call
        guard.detect_geco2 = forbidden_model_call
        cls.model_guard = mock.patch.dict(sys.modules, {"auto_counter": guard})
        cls.model_guard.start()
        cls.addClassCleanup(cls.model_guard.stop)

    def count(self, image, **options):
        response = self.client.post("/api/count", json={"image": image, "device": "cpu", **options})
        self.assertTrue(response.is_json)
        return response, response.get_json()

    def assert_points_valid(self, result, expected):
        self.assertEqual(result["count"], expected)
        self.assertEqual(len(result["points"]), expected)
        for point in result["points"]:
            self.assertGreaterEqual(point["x"], 0)
            self.assertLess(point["x"], result["width"])
            self.assertGreaterEqual(point["y"], 0)
            self.assertLess(point["y"], result["height"])

    def test_real_mung_color_count(self):
        response, result = self.count(self.mung, engine="beans", category="mung")
        self.assertEqual(response.status_code, 200, result)
        self.assertEqual(result["meta"]["framework"], "MungColor")
        self.assertEqual(result["meta"]["diagnosis"], "detections_returned")
        self.assert_points_valid(result, 17)

    def test_real_coins_auto_count(self):
        response, result = self.count(self.coins, engine="auto", category="coins")
        self.assertEqual(response.status_code, 200, result)
        self.assertEqual(result["meta"]["framework"], "SeparatedGrains")
        self.assert_points_valid(result, 8)

    def test_hidden_rice_layout_does_not_block_coins(self):
        first, baseline = self.count(self.coins, engine="auto", category="coins", rice_layout="separated")
        second, result = self.count(self.coins, engine="auto", category="coins", rice_layout="touching")
        self.assertEqual(first.status_code, 200, baseline)
        self.assertEqual(second.status_code, 200, result)
        self.assert_points_valid(result, 8)
        self.assertEqual(result["points"], baseline["points"])
        # Direct quick mode follows the same category-scoped rule.
        direct, direct_result = self.count(self.coins, engine="grains", category="coins", rice_layout="touching")
        self.assertEqual(direct.status_code, 200, direct_result)
        self.assertEqual(direct_result["points"], baseline["points"])

    def test_beans_engine_rejects_other_categories(self):
        for category in ("bamboo", "cotton", "rice", "coins", "unknown"):
            with self.subTest(category=category):
                response, result = self.count(self.mung, engine="beans", category=category)
                self.assertEqual(response.status_code, 400, result)
                self.assertIn("绿豆", result["error"])

    def test_blank_inputs_report_no_detections(self):
        for engine, category in (("beans", "mung"), ("grains", "rice")):
            with self.subTest(engine=engine):
                response, result = self.count(self.blank, engine=engine, category=category, rice_layout="separated")
                self.assertEqual(response.status_code, 200, result)
                self.assert_points_valid(result, 0)
                self.assertEqual(result["meta"]["diagnosis"], "no_final_detections")
                self.assertTrue(result["meta"].get("limitation"))

    def test_touching_rice_is_still_rejected(self):
        for engine in ("auto", "grains"):
            with self.subTest(engine=engine):
                response, result = self.count(self.blank, engine=engine, category="rice", rice_layout="touching")
                self.assertEqual(response.status_code, 400, result)
                self.assertIn("米粒", result["error"])

    def test_application_samples_are_listed_and_readable(self):
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        rows = response.get_json()["samples"]
        ids = [row["id"] for row in rows]
        self.assertEqual(len(ids), len(set(ids)), "Sample IDs must be unique")
        application_rows = [row for row in rows if row["id"].startswith("app_")]
        self.assertGreaterEqual(len(application_rows), 12)
        for expected_id, category in (("mung_dark", "mung"), ("coins_white", "coins")):
            matches = [row for row in rows if row["id"] == expected_id or row["id"].endswith("_" + expected_id)]
            self.assertEqual(len(matches), 1, f"Expected one API sample for {expected_id}; got {ids}")
            row = matches[0]
            self.assertEqual(row["category"], category)
        for row in application_rows:
            image_response = self.client.get("/sample", query_string={"name": row["id"]})
            try:
                self.assertEqual(image_response.status_code, 200)
                self.assertTrue(image_response.mimetype.startswith("image/"),
                                f"{row['id']} returned MIME {image_response.mimetype}")
                image = cv2.imdecode(np.frombuffer(image_response.data, np.uint8), cv2.IMREAD_COLOR)
                self.assertIsNotNone(image)
            finally:
                image_response.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
