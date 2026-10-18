"""Small diagnostic experiment, not a held-out accuracy benchmark."""
import json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from model_counter import detect_model
from counter import annotate
ROOT=Path(__file__).resolve().parent
TRUTH=[(251,200),(205,219),(283,228),(237,263),(188,276),(315,276),
       (274,297),(213,316),(174,335),(245,346),(309,345),(352,323),
       (197,393),(270,401),(344,402),(234,426),(307,440),(262,469)]
def matching(points,truth,tolerance=18):
 if not points:return dict(matched=0,missed=len(truth),extra=0)
 p=np.array([[p['x'],p['y']] for p in points]);t=np.array(truth)
 distances=np.linalg.norm(p[:,None,:]-t[None,:,:],axis=-1)
 # Large penalty first maximizes feasible one-to-one matches, then minimizes distance.
 cost=np.where(distances<=tolerance,distances,1e6)
 a,b=linear_sum_assignment(cost);matched=int((distances[a,b]<=tolerance).sum())
 return dict(matched=matched,missed=len(truth)-matched,extra=len(points)-matched,tolerance_px=tolerance)
def main():
 im=cv2.imdecode(np.fromfile(ROOT/'samples/counting_eval/bamboo_hand.png',np.uint8),cv2.IMREAD_COLOR)
 old=[[249,266,303,331],[218,311,273,381],[323,283,376,357]]
 diverse=[[230,171,275,232],[218,311,273,381],[317,361,369,437]]
 out=ROOT/'outputs/localization_experiment';out.mkdir(exist_ok=True)
 rows=[]
 for name,boxes,roi,nms in [('baseline',old,None,None),('diverse_examples',diverse,None,None),('diverse_examples_nms',diverse,None,.5),('head_crop',diverse,[140,140,410,510],None)]:
  x,y=roi[:2] if roi else (0,0)
  image=im[y:roi[3],x:roi[2]] if roi else im
  local=[[a-x,b-y,c-x,d-y] for a,b,c,d in boxes]
  points,meta=detect_model(image,local,.33,nms)
  for p in points:p['x']+=x;p['y']+=y
  report=dict(experiment=name,count=len(points),ground_truth_count=len(TRUTH),metrics=matching(points,TRUTH),meta=meta,roi=roi,exemplars=boxes)
  rows.append(report);print(json.dumps(report),flush=True)
  cv2.imencode('.jpg',annotate(im,points))[1].tofile(str(out/(name+'.jpg')))
 (out/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
 (ROOT/'samples/counting_eval/bamboo_hand_truth.json').write_text(json.dumps(dict(points=[dict(x=x,y=y) for x,y in TRUTH],count=len(TRUTH),method='Manual single-review visible end-face centers; provisional annotation'),indent=2),encoding='utf-8')
if __name__=='__main__':main()
