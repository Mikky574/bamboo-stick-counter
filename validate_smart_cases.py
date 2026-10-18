"""Run real image-only HTTP requests; no category/sample IDs sent to inference."""
import base64
import json
from pathlib import Path
import time

import cv2
import numpy as np
import requests

from counter import annotate

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/smart_validation'
CASES = ('rice_gray', 'app_mung_dark', 'app_coins_white', 'dense_cotton_overhead',
         'dense_flat_craft', 'bamboo_original', 'rice_touching', 'app_mung_paper')


def main():
    OUT.mkdir(exist_ok=True)
    catalogue = json.loads((ROOT/'samples/counting_eval/manifest.json').read_text(encoding='utf-8'))
    session = requests.Session()
    session.trust_env = False
    report = []
    for identity in CASES:
        row = next(r for r in catalogue if r['id'] == identity)
        raw = (ROOT/'samples/counting_eval'/row['file']).read_bytes()
        evaluate(session, identity, raw, row.get('ground_truth'), report)
    blank = cv2.imencode('.png', np.full((256, 256, 3), 255, np.uint8))[1].tobytes()
    evaluate(session, 'blank', blank, None, report)


def evaluate(session, identity, raw, reference, report):
    started = time.perf_counter()
    # This is exactly the new production UI contract, including cold requests.
    response = session.post('http://127.0.0.1:5080/api/count', json={
        'image': 'data:image/jpeg;base64,'+base64.b64encode(raw).decode('ascii'),
    }, timeout=600)
    response.raise_for_status()
    result = response.json()
    result['http_elapsed_seconds'] = round(time.perf_counter()-started, 3)
    (OUT/(identity+'_image_only.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    cv2.imencode('.jpg', annotate(image, result['points']))[1].tofile(OUT/(identity+'_image_only.jpg'))
    meta = result['meta']
    summary = dict(id_for_reporting_only=identity, count=result['count'],
                   provisional_visual_reference=reference,
                   framework=meta['framework'], detected_category=meta.get('detected_category'),
                   diagnosis=meta.get('diagnosis'), prediction_valid=meta.get('prediction_valid'),
                   http_seconds=result['http_elapsed_seconds'],
                   request_keys=['image'])
    report.append(summary)
    (OUT/'image_only_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
