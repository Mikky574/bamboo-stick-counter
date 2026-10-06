"""Integration and NMS comparison checks, NOT an accuracy benchmark."""
import base64
import json
from pathlib import Path
import cv2
import numpy as np
from app import app

root = Path(__file__).parent
(root/'outputs').mkdir(exist_ok=True)
client = app.test_client()
raw = 'data:image/png;base64,'+base64.b64encode((root/'samples/user_bamboo.png').read_bytes()).decode()
boxes = [[315,132,360,172],[358,326,405,369],[287,434,332,479]]
payload = dict(image=raw,engine='geco2',exemplars=boxes,model_threshold=.33)
response = client.post('/api/count',json=payload)
assert response.status_code == 200, response.json
points = response.json
assert points['meta']['framework']=='GeCo2' and points['meta']['mode']=='center_peaks'
assert len(points['points'])>0 and points['count']==len(points['points'])
assert all(0<=p['x']<points['width'] and 0<=p['y']<points['height'] for p in points['points'])
assert all(np.isfinite(p['x']) and np.isfinite(p['y']) for p in points['points'])
nms = client.post('/api/count',json={**payload,'nms_iou':.5})
assert nms.status_code == 200
assert nms.json['count']<=points['count']
assert nms.json['meta']['nms_removed']==points['count']-nms.json['count']
assert nms.json['meta']['cached']
for invalid in [[], [[0,0,1,1]], [[0,0,9999,9999]], [[float('nan'),0,20,20]]]:
    assert client.post('/api/count',json={**payload,'exemplars':invalid}).status_code==400
assert client.post('/api/count',json={**payload,'model_threshold':-1}).status_code==400
export = client.post('/api/export',json=dict(image=raw,points=points['points']))
assert export.status_code == 200
decoded = cv2.imdecode(np.frombuffer(export.data,np.uint8),cv2.IMREAD_COLOR)
assert decoded.shape[:2]==(points['height'],points['width'])
report = dict(point_count=points['count'],box_nms_count=nms.json['count'],
              nms_removed=nms.json['meta']['nms_removed'],meta=points['meta'],
              ground_truth=None,accuracy_verified=False,
              note='User-image inference smoke test. Predicted count is not verified ground truth.')
(root/'outputs/model_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
