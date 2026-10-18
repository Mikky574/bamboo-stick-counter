"""CPU-only unchanged separated-grain baseline on the application photo batch.

Predictions are connected foreground regions, not independently verified counts.
The two manual reference counts were supplied before this run by the reviewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from grain_counter import detect_grains


ROOT = Path(__file__).resolve().parent
MANUAL_REFERENCE_COUNTS = {"mung_dark": 17, "coins_white": 8}


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def numbered_image(image, points, count: int):
    result = image.copy()
    marker = max(4, round(max(image.shape[:2]) / 230))
    font_scale = max(0.4, max(image.shape[:2]) / 2800)
    for index, point in enumerate(points, 1):
        center = (round(point["x"]), round(point["y"]))
        cv2.circle(result, center, marker, (0, 230, 90), 2, cv2.LINE_AA)
        label_position = (center[0] + marker, center[1] - marker)
        cv2.putText(result, str(index), label_position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(result, str(index), label_position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    text = f"Connected-region prediction: {count} (not ground truth)"
    cv2.putText(result, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(result, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "samples/application_eval_20261018/manifest.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/application_grains_baseline")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    results = []
    started = time.perf_counter()
    for row in manifest:
        image_path = args.manifest.parent / row["file"]
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {image_path}")
        points, meta = detect_grains(image)
        reference = MANUAL_REFERENCE_COUNTS.get(row["id"])
        result = {
            "id": row["id"], "category": row["category"],
            "image": str(image_path.resolve()), "source_page": row["source_page"],
            "engine": "unchanged_grain_counter.detect_grains", "device": "cpu",
            "prediction_count": len(points), "prediction_is_ground_truth": False,
            "manual_reference_count": reference,
            "reference_provenance": ("Reviewer independently counted before baseline inference"
                                     if reference is not None else None),
            "signed_count_error": len(points) - reference if reference is not None else None,
            "absolute_count_error": abs(len(points) - reference) if reference is not None else None,
            "localization_verified": False,
            "points": points, "meta": meta,
        }
        write_json(args.output / f"{row['id']}.json", result)
        numbered = numbered_image(image, points, len(points))
        ok, encoded = cv2.imencode(".jpg", numbered, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError(f"Cannot encode numbered image {row['id']}")
        encoded.tofile(str(args.output / f"{row['id']}.numbered.jpg"))
        compact = {key: value for key, value in result.items() if key != "points"}
        compact["json"] = f"{row['id']}.json"
        compact["numbered_image"] = f"{row['id']}.numbered.jpg"
        results.append(compact)
        print(json.dumps({"id": row["id"], "prediction": len(points),
                          "manual_reference": reference, "meta": meta}, ensure_ascii=False), flush=True)
    summary = {
        "engine": "unchanged_grain_counter.detect_grains", "device": "cpu",
        "number_of_images": len(results),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "warning": "Predictions count foreground components. Touching objects can merge; background and reflections can be false positives. Predictions are not labels.",
        "results": results,
    }
    write_json(args.output / "summary.json", summary)
    lines = ["# CPU separated-grain baseline", "",
             "The existing algorithm was used unchanged, with no deep model or GPU.", "",
             "Predicted counts are not truth or training labels. Only two reference counts were independently supplied before inference. Count agreement does not prove point-level correctness.", "",
             "| Image | Predicted regions | Manual reference | Signed error | Algorithm seconds | Numbered output |",
             "|---|---:|---:|---:|---:|---|"]
    for row in results:
        reference = row["manual_reference_count"]
        error = row["signed_count_error"]
        lines.append(f"| {row['id']} | {row['prediction_count']} | {reference if reference is not None else 'unknown'} | {error if error is not None else 'n/a'} | {row['meta']['elapsed_seconds']} | [view]({row['numbered_image']}) |")
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(results)} predictions and numbered images to {args.output}")


if __name__ == "__main__":
    main()
