"""Real-sample API checks and an explicit single-image optimization comparison."""
import base64,json
from pathlib import Path
import cv2,numpy as np
from app import app
from counter import annotate
from experiment_localization import matching,TRUTH

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/sample_checks';OUT.mkdir(parents=True,exist_ok=True)
client=app.test_client();response=client.get('/api/samples');assert response.status_code==200
rows=response.json['samples'];assert len(rows)>=6
assert next(row for row in rows if row['id']=='bamboo_hand').get('comparison')
reports=[]
for row in rows:
 response=client.get('/sample',query_string={'name':row['id']});assert response.status_code==200
 raw=response.data;image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
 payload=dict(image='data:image/png;base64,'+base64.b64encode(raw).decode(),engine=row['engine'],exemplars=row['exemplars'])
 if row['id']=='rice_gray':
  settings=[('geco2',dict(engine='geco2')),('grains',dict(engine='grains'))]
 elif row.get('comparison'):
  setting=row['comparison']
  settings=[('baseline',{}),('comparison',dict(exemplars=setting['exemplars'],nms_iou=setting['nms_iou'],model_threshold=setting['threshold']))]
 else:settings=[('baseline',{})]
 for name,overrides in settings:
  response=client.post('/api/count',json={**payload,**overrides})
  assert response.status_code==200,response.json
  result=response.json;assert result['count']==len(result['points'])
  assert all(0<=p['x']<result['width'] and 0<=p['y']<result['height'] for p in result['points'])
  if row['id']=='rice_gray' and name=='grains':assert result['count']==30
  metrics=matching(result['points'],TRUTH) if row['id']=='bamboo_hand' else None
  if row['id']=='bamboo_hand' and name=='comparison':assert metrics['missed']==metrics['extra']==0
  report=dict(sample=row['id'],setting=name,predicted_count=result['count'],ground_truth=row['ground_truth'],metrics=metrics,meta=result['meta'])
  reports.append(report);print(row['id'],name,result['count'],flush=True)
  stem=OUT/(row['id']+'_'+name)
  Path(str(stem)+'.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
  cv2.imencode('.jpg',annotate(image,result['points']))[1].tofile(str(stem)+'.jpg')
  exported=client.post('/api/export',json=dict(image=payload['image'],points=result['points']))
  assert exported.status_code==200
  assert cv2.imdecode(np.frombuffer(exported.data,np.uint8),cv2.IMREAD_COLOR).shape==image.shape
(OUT/'report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
print('All sample API, coordinate, export, rice-count and bamboo matching checks passed.')
