"""Evaluate downloaded real photos without using saved annotations as prompts."""
import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
import requests

from counter import annotate
from grain_counter import detect_grains
from mung_counter import detect_mung

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'samples/application_eval_20261018'
OUT = ROOT / 'outputs/application_eval_20261018'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
    session = requests.Session()
    session.trust_env = False
    report = []
    for row in rows:
        raw = (SOURCE / row['file']).read_bytes()
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        record = dict(id=row['id'], source_page=row['source_page'],
                      ground_truth=row['ground_truth'], screening=row['screening'])
        methods = [('grains', detect_grains)]
        if row['category'] == 'mung':
            methods.append(('beans', detect_mung))
        for name, detector in methods:
            points, meta = detector(image)
            result = dict(count=len(points), points=points, meta=meta)
            save(row['id'], name, image, result)
            record[name] = dict(count=len(points), elapsed_seconds=meta['elapsed_seconds'])
        started = time.perf_counter()
        response = session.post('http://127.0.0.1:5080/api/count', json={
            'image': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii'),
            'engine': 'countgdpp', 'category': row['category'],
            'device': 'cuda', 'keep_warm': True,
        }, timeout=600)
        response.raise_for_status()
        result = response.json()
        result['http_elapsed_seconds'] = round(time.perf_counter() - started, 3)
        save(row['id'], 'countgdpp', image, result)
        record['countgdpp'] = dict(count=result['count'],
                                  elapsed_seconds=result['http_elapsed_seconds'],
                                  diagnosis=result['meta'].get('diagnosis'))
        report.append(record)
        (OUT / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(record, ensure_ascii=True), flush=True)


def save(identity, method, image, result):
    (OUT / (identity + '_' + method + '.json')).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    rendered = annotate(image, result['points'])
    cv2.imencode('.jpg', rendered)[1].tofile(OUT / (identity + '_' + method + '.jpg'))


if __name__ == '__main__':
    main()
