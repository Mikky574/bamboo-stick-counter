"""Lightweight regression checks; never starts a model worker or a GPU task.

Run ``.venv/Scripts/python.exe verify_runtime_rice.py`` after the runtime/UI
changes land. ``--core-only`` checks only existing rice paths and saved evidence.
Model endpoints are mocked to verify dispatch without allocating model memory;
the release endpoint is also mocked so this script cannot stop another demo.
"""
import argparse
import base64
import inspect
import json
from html.parser import HTMLParser
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app import app
from grain_counter import detect_grains


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples/counting_eval"


def encoded_image(name):
    suffix = "png" if Path(name).suffix == ".png" else "jpeg"
    raw = (SAMPLES / name).read_bytes()
    return "data:image/" + suffix + ";base64," + base64.b64encode(raw).decode("ascii")


def failed_rice_prediction():
    # A diagnosed inference failure must not be presented as a trusted empty scene.
    return {
        "count": 0,
        "points": [],
        "exemplars": [],
        "width": 377,
        "height": 244,
        "meta": {
            "framework": "CountGD++",
            "category": "rice",
            "initial_count": 0,
            "pseudo_exemplar_count": 0,
            "diagnosis": "no_initial_exemplars",
            "device": "cuda",
        },
    }


class RiceCoreChecks(unittest.TestCase):
    def test_separated_grains_actual_api(self):
        with app.test_client() as client:
            response = client.post("/api/count", json={
                "image": encoded_image("rice_gray.png"),
                "engine": "grains", "category": "rice",
            })
        self.assertEqual(response.status_code, 200, response.get_json())
        result = response.get_json()
        self.assertEqual(result["count"], 30)
        self.assertEqual(len(result["points"]), 30)
        self.assertEqual(result["meta"]["framework"], "SeparatedGrains")
        self.assertIn("touching", result["meta"]["limitation"].lower())
        self.assertTrue(all(
            0 <= p["x"] < result["width"] and 0 <= p["y"] < result["height"]
            for p in result["points"]
        ))

    def test_touching_grains_are_only_four_components(self):
        image = cv2.imdecode(np.frombuffer((SAMPLES / "rice_extra.jpg").read_bytes(),
                                         dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(image)
        points, meta = detect_grains(image)
        # This is a regression value for connected regions, NOT the grain truth.
        self.assertEqual(len(points), 4)
        self.assertEqual(meta["mode"], "connected_components")
        self.assertIn("touching grains may merge", meta["limitation"])
        manifest = json.loads((SAMPLES / "manifest.json").read_text(encoding="utf-8"))
        row = next(item for item in manifest if item["id"] == "rice_touching")
        self.assertIsNone(row["ground_truth"])

    def test_saved_zero_is_missing_auto_examples(self):
        result = json.loads((ROOT / "outputs/automatic_validation/rice_gray.json")
                            .read_text(encoding="utf-8"))
        self.assertEqual(result["count"], 0)
        self.assertEqual(result["meta"]["initial_count"], 0)
        self.assertEqual(result["meta"]["pseudo_exemplar_count"], 0)
        self.assertEqual(result["ground_truth"], 30)
        # Old output predates the diagnosis field. Its recorded cause is enough
        # to establish failure without rerunning the large text-conditioned model.
        self.assertEqual(result["meta"].get("diagnosis", "no_initial_exemplars"),
                         "no_initial_exemplars")


class RuntimeInterfaceChecks(unittest.TestCase):
    def test_production_ui_only_upload_and_identify(self):
        with app.test_client() as client:
            response = client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        class Controls(HTMLParser):
            def __init__(self):
                super().__init__()
                self.items = []

            def handle_starttag(self, tag, attrs):
                if tag in {"input", "button", "select", "textarea"}:
                    self.items.append((tag, dict(attrs)))

        controls = Controls()
        controls.feed(html)
        self.assertEqual([(tag, attrs.get("id")) for tag, attrs in controls.items],
                         [("input", "file"), ("button", "run")])
        self.assertEqual(controls.items[0][1]["type"], "file")
        self.assertIn("image/webp", controls.items[0][1]["accept"])
        self.assertRegex(html, r"JSON\.stringify\(\{\s*image\s*:\s*dataURL\s*\}\)")
        self.assertIn("/api/count", html)
        for old_endpoint in ("/api/samples", "/api/release-model", "/api/model-status"):
            self.assertNotIn(old_endpoint, html)
        self.assertIn("prediction_valid", html)
        self.assertIn("result.count!=null", html)
        self.assertIn("total_elapsed_seconds", html)
        self.assertIn("particles:'独立目标'", html)
        self.assertIn("无法据此判断数量为零", html)

    def test_keep_warm_is_opt_in_and_readiness_exposes_residency(self):
        import auto_counter
        parameter = inspect.signature(auto_counter.detect_auto).parameters.get("keep_warm")
        self.assertIsNotNone(parameter, "detect_auto must accept keep_warm")
        self.assertIs(parameter.default, False, "persistent model loading must be opt-in")
        state = auto_counter.readiness()
        self.assertIn("model_resident", state)
        self.assertIsInstance(state["model_resident"], bool)

    def test_count_api_forwards_cache_choice_and_failure_diagnosis(self):
        import auto_counter
        signature = inspect.signature(auto_counter.detect_auto)
        prediction = failed_rice_prediction()
        with patch("auto_counter.detect_auto", return_value=prediction) as detector:
            with app.test_client() as client:
                response = client.post("/api/count", json={
                    "image": encoded_image("rice_gray.png"),
                    "engine": "countgdpp", "category": "rice", "device": "cuda",
                    "keep_warm": True,
                })
        self.assertEqual(response.status_code, 200, response.get_json())
        detector.assert_called_once()
        bound = signature.bind(*detector.call_args.args, **detector.call_args.kwargs)
        bound.apply_defaults()
        self.assertIs(bound.arguments["keep_warm"], True)
        self.assertEqual(response.get_json()["meta"]["diagnosis"], "no_initial_exemplars")
        self.assertEqual(response.get_json()["meta"]["initial_count"], 0)

    def test_release_endpoint_dispatch_without_releasing_real_worker(self):
        # Do not shut down a real resident model belonging to the active demo.
        with patch("auto_counter.release_model", return_value={
            "released": True, "model_resident": False,
        }) as release:
            with app.test_client() as client:
                response = client.post("/api/release-model", json={})
        self.assertEqual(response.status_code, 200, response.get_json())
        release.assert_called_once()

    def test_model_status_reports_resident_state(self):
        with patch("auto_counter.readiness", return_value={
            "installed": True, "model_resident": True, "busy": False,
        }):
            with app.test_client() as client:
                response = client.get("/api/model-status")
        self.assertEqual(response.status_code, 200)
        self.assertIs(response.get_json()["automatic"]["model_resident"], True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true",
                        help="Only the CPU rice checks; runtime/UI changes may still be pending")
    args = parser.parse_args()
    suite = unittest.TestSuite()
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RiceCoreChecks))
    if not args.core_only:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeInterfaceChecks))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
