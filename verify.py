"""Reproducible smoke and localization checks, including a real-photo crop."""
import base64,json
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from counter import detect,annotate
from app import app
ROOT=Path(__file__).parent

def read_image(path):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)

def write_image(path,image):
    cv2.imencode(".png",image)[1].tofile(path)

def run():
    (ROOT/'outputs').mkdir(exist_ok=True)
    rng=np.random.default_rng(42)
    im=np.full((500,700,3),(30,42,37),np.uint8)
    truth=[]
    for row in range(8):
        for col in range(10):
            x=65+col*60+int(rng.integers(-3,4));y=42+row*58+int(rng.integers(-3,4));r=int(rng.integers(18,23))
            cv2.circle(im,(x,y),r,(155,198,218),-1,cv2.LINE_AA)
            cv2.circle(im,(x-4,y-4),r//2,(171,207,230),-1)
            truth.append([x,y])
    write_image(str(ROOT/'samples/synthetic_80.png'),im)
    (ROOT/'samples/synthetic_80_truth.json').write_text(json.dumps(truth),encoding='utf-8')
    cases=[('synthetic_80',im,truth,40,0,.45)]
    if (ROOT/'samples/real_cotton_crop.jpg').is_file():
        cases.append(('real_cotton_crop',read_image(str(ROOT/'samples/real_cotton_crop.jpg')),[[25,91],[59,78],[103,72]],24,145,.5))
    report=[]
    client=app.test_client()
    for name,img,gt,d,t,s in cases:
        pts,_=detect(img,d,'bright',t,s)
        assert len(pts)==len(gt),(name,len(pts),len(gt))
        pred=np.array([[p['x'],p['y']] for p in pts]);dist=np.linalg.norm(pred[:,None]-np.array(gt)[None,:],axis=-1)
        rows,cols=linear_sum_assignment(dist);maxerr=float(dist[rows,cols].max())
        assert maxerr<8,(name,maxerr)
        write_image(str(ROOT/'outputs'/f'{name}_annotated.png'),annotate(img,pts))
        raw='data:image/png;base64,'+base64.b64encode(cv2.imencode('.png',img)[1]).decode()
        r=client.post('/api/count',json=dict(image=raw,diameter=d,threshold=t,sensitivity=s))
        assert r.status_code==200 and r.json['count']==len(gt)
        r=client.post('/api/export',json=dict(image=raw,points=pts));assert r.status_code==200
        assert cv2.imdecode(np.frombuffer(r.data,np.uint8),cv2.IMREAD_COLOR).shape==img.shape
        (ROOT/'outputs'/f'{name}_points.json').write_text(json.dumps({'count':len(pts),'points':pts},indent=2),encoding='utf-8')
        report.append(dict(case=name,expected=len(gt),detected=len(pts),max_center_error_px=maxerr,diameter=d,threshold=t,sensitivity=s))
    assert client.post('/api/count',json={'image':'not an image'}).status_code==400
    assert client.get('/').status_code==200 and client.get('/sample?name=synthetic').status_code==200
    if (ROOT/'samples/catalog_32_0_Im0.jpg').is_file():
        full=read_image(str(ROOT/'samples/catalog_32_0_Im0.jpg'))
        pts,_=detect(full,24,'bright',145,.5)
        write_image(str(ROOT/'outputs/full_photo_annotated.png'),annotate(full,pts))
        report.append(dict(case='full_real_photo_touching_tips',expected=7,detected=len(pts),limitation='Touching tips on the left merge; crop success does not imply full-photo accuracy.'))
    else:
        report.append(dict(case='optional_cotton_photos',skipped=True,reason='Third-party catalog photos are not redistributed in the public source package.'))
    (ROOT/'outputs/validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':run()
