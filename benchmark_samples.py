"""Reproducible local sample comparison; predictions are not ground truth."""
import json,time
from pathlib import Path
import cv2,numpy as np
from counter import annotate
from model_counter import detect_model

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'samples/counting_eval'
OUT=ROOT/'outputs/sample_benchmark'
BOXES={
 'bamboo_hand':[[249,266,303,331],[218,311,273,381],[323,283,376,357]],
 'bamboo_bundle':[[537,188,570,228],[599,230,632,267],[627,345,664,389]],
 'rice_gray':[[177,62,201,75],[230,61,245,89],[227,181,260,200]],
 'bamboo_original':[[315,132,360,172],[358,326,405,369],[287,434,332,479]],
 'cotton_dense':[],
}
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rows=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
 reports=[]
 for row in rows:
  boxes=row.get('exemplars') or BOXES[row['id']]
  if not boxes:continue
  row['exemplars']=boxes
  im=cv2.imdecode(np.fromfile(DATA/row['file'],np.uint8),cv2.IMREAD_COLOR)
  for threshold in [.20,.33,.50]:
   points,meta=detect_model(im,boxes,threshold)
   report=dict(sample=row['id'],count=len(points),points=points,exemplars=boxes,meta=meta,ground_truth=row.get('ground_truth'))
   path=OUT/f'{row["id"]}_{threshold:.2f}'
   Path(str(path)+'.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
   cv2.imencode('.jpg',annotate(im,points))[1].tofile(str(path)+'.jpg')
   reports.append({k:report[k] for k in ['sample','count','meta','ground_truth']})
   print(row['id'],threshold,len(points),meta['elapsed_seconds'],flush=True)
 (DATA/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
