"""Run actual CPU text-only inference; record quality without asserting accuracy."""
import base64,json
from pathlib import Path
import cv2,numpy as np
from app import app
from counter import annotate
from experiment_localization import matching,TRUTH
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/automatic_validation';OUT.mkdir(parents=True,exist_ok=True)
client=app.test_client();reports=[]
samples=json.loads((ROOT/'samples/counting_eval/manifest.json').read_text(encoding='utf-8'))
assert client.get('/api/model-status').json['automatic']['installed']
for ident in ['bamboo_original','bamboo_hand','cotton_dense','rice_gray']:
 row=next(r for r in samples if r['id']==ident)
 path=ROOT/'samples/counting_eval'/row['file'];raw=path.read_bytes()
 # Deliberately invalid visual prompts: automatic mode must ignore all manual hints.
 payload=dict(image='data:image/png;base64,'+base64.b64encode(raw).decode(),engine='countgdpp',category=row['category'],device='cpu',exemplars=[[0,0,99999,99999]])
 print('Starting',ident,flush=True)
 response=client.post('/api/count',json=payload);assert response.status_code==200,response.json
 result=response.json;meta=result['meta']
 assert meta['manual_exemplars'] is False and meta['worker_exited'] is True and meta['device']=='cpu'
 assert result['count']==len(result['points'])
 assert all(0<=p['x']<result['width'] and 0<=p['y']<result['height'] for p in result['points'])
 result.update(annotation_status='unreviewed_prediction',source_image=str(path),ground_truth=row.get('ground_truth'))
 if ident=='bamboo_hand':result['metrics']=matching(result['points'],TRUTH)
 if row.get('ground_truth') is not None:result['count_absolute_error']=abs(result['count']-row['ground_truth'])
 (OUT/(ident+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
 cv2.imencode('.jpg',annotate(image,result['points']))[1].tofile(str(OUT/(ident+'.jpg')))
 reports.append({k:v for k,v in result.items() if k not in ['points','exemplars']})
 print(ident,result['count'],meta['elapsed_seconds'],'seconds; worker exited',flush=True)
assert client.post('/api/count',json={**payload,'category':'invalid'}).status_code==400
assert client.post('/api/count',json={**payload,'device':'invalid'}).status_code==400
assert not client.get('/api/model-status').json['automatic']['busy']
(OUT/'report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
print('Automatic API integration passed; quality measurements recorded, not assumed correct.',flush=True)
