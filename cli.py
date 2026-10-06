import argparse,json
from pathlib import Path
import cv2,numpy as np
from counter import detect,annotate
p=argparse.ArgumentParser(description='Detect and number visible end faces.')
p.add_argument('image');p.add_argument('--diameter',type=float,default=24);p.add_argument('--threshold',type=int,default=0);p.add_argument('--polarity',choices=['bright','dark'],default='bright');p.add_argument('--sensitivity',type=float,default=.45);p.add_argument('--output',default='outputs/cli')
a=p.parse_args();im=cv2.imdecode(np.fromfile(a.image,dtype=np.uint8),cv2.IMREAD_COLOR)
if im is None:p.error('Cannot read image')
points,_=detect(im,a.diameter,a.polarity,a.threshold,a.sensitivity)
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
cv2.imencode('.png',annotate(im,points))[1].tofile(str(out)+'.png')
Path(str(out)+'.json').write_text(json.dumps(dict(count=len(points),points=points),indent=2),encoding='utf-8')
print(f'Count: {len(points)}; output: {out}.png / .json')
