"""Example: .venv/Scripts/python model_cli.py samples/user_bamboo.png --demo-exemplars"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from counter import annotate
from model_counter import detect_model

p = argparse.ArgumentParser(description='GeCo2 local pretrained center localization')
p.add_argument('image')
group = p.add_mutually_exclusive_group(required=True)
group.add_argument('--exemplars', help='JSON file containing [[x1,y1,x2,y2],...]')
group.add_argument('--demo-exemplars', action='store_true', help='Use the 3 example boxes for samples/user_bamboo.png only')
p.add_argument('--threshold', type=float, default=.33)
p.add_argument('--nms-iou', type=float, default=None, help='Optional box NMS; default disabled')
p.add_argument('--output', default='outputs/model_result')
a = p.parse_args()
image = cv2.imdecode(np.fromfile(a.image, dtype=np.uint8), cv2.IMREAD_COLOR)
if image is None:
    p.error('Cannot read image')
if a.demo_exemplars:
    if Path(a.image).resolve() != (Path(__file__).parent/'samples/user_bamboo.png').resolve():
        p.error('--demo-exemplars is only valid for samples/user_bamboo.png')
    boxes = [[315,132,360,172],[358,326,405,369],[287,434,332,479]]
else:
    boxes = json.loads(Path(a.exemplars).read_text(encoding='utf-8-sig'))
points, meta = detect_model(image, boxes, a.threshold, a.nms_iou)
out = Path(a.output)
out.parent.mkdir(parents=True, exist_ok=True)
cv2.imencode('.png', annotate(image, points))[1].tofile(str(out)+'.png')
Path(str(out)+'.json').write_text(json.dumps(dict(count=len(points),points=points,
    exemplars=boxes,meta=meta,width=image.shape[1],height=image.shape[0]),indent=2),encoding='utf-8')
print(json.dumps(dict(count=len(points),meta=meta),indent=2))
