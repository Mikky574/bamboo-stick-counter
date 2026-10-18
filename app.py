import base64
import io
import json
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from uuid import uuid4
import cv2
import numpy as np
from flask import Flask,g,request,jsonify,render_template,send_file
from counter import detect,annotate
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=28*1024*1024  # Base64 JSON for a <=20 MiB source image.
app.config['TEMPLATES_AUTO_RELOAD']=True
ROOT=Path(__file__).parent
UI_VERSION=hashlib.sha256((ROOT/'templates/index.html').read_bytes()).hexdigest()[:12]

# Keep bounded, image-free diagnostics: HTTP status alone did not explain the
# previous 503s. Never persist uploaded pixels or base64 data.
_request_log=logging.getLogger('bamboo_count_requests')
_request_log.setLevel(logging.INFO)
_request_log.propagate=False
if not _request_log.handlers:
    (ROOT/'outputs').mkdir(exist_ok=True)
    _handler=RotatingFileHandler(ROOT/'outputs/count_requests.jsonl',
        maxBytes=1_000_000,backupCount=2,encoding='utf-8')
    _handler.setFormatter(logging.Formatter('%(message)s'))
    _request_log.addHandler(_handler)


@app.before_request
def begin_count_diagnostics():
    if request.path=='/api/count':
        g.count_started=time.perf_counter()
        g.count_request_id=uuid4().hex[:12]
        g.count_transport='http' if 'werkzeug.socket' in request.environ else 'local_test'
        _request_log.info(json.dumps(dict(event='started',
            request_id=g.count_request_id,time_unix=time.time(),
            transport=g.count_transport),ensure_ascii=False))


@app.after_request
def finish_count_diagnostics(response):
    if request.path=='/' or request.path.startswith('/api/'):
        response.headers['Cache-Control']='no-store, max-age=0'
        response.headers['X-Counter-UI-Version']=UI_VERSION
    if request.path=='/api/count' and hasattr(g,'count_started'):
        payload=response.get_json(silent=True)
        payload=payload if isinstance(payload,dict) else {}
        meta=payload.get('meta')
        meta=meta if isinstance(meta,dict) else {}
        record=dict(event='finished',request_id=g.count_request_id,
            time_unix=time.time(),transport=g.count_transport,status=response.status_code,
            elapsed_seconds=round(time.perf_counter()-g.count_started,3),
            count=payload.get('count'),error=str(payload.get('error') or '')[:700])
        record.update({key:meta.get(key) for key in (
            'framework','device','diagnosis','routing','cache_reused','model_load_seconds',
            'category','prompt','initial_count','pseudo_exemplar_count')})
        hint=meta.get('scene_classification') or {}
        scene_meta=hint.get('meta') or {}
        record['scene_model_load_seconds']=scene_meta.get('model_load_seconds')
        _request_log.info(json.dumps(record,ensure_ascii=False))
        response.headers['X-Count-Request-ID']=g.count_request_id
    return response

def evaluation_samples():
    path=ROOT/'samples/counting_eval/manifest.json'
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []

@app.get('/api/samples')
def list_samples():
    return jsonify(samples=[{k:row.get(k) for k in
        ('id','title','category','exemplars','engine','ground_truth','note','comparison')} for row in evaluation_samples()])

def decode(value):
    if not isinstance(value,str):raise ValueError('请选择有效图片。')
    raw=base64.b64decode(value.split(',')[-1],validate=True)
    if len(raw)>20*1024*1024:raise ValueError('图片文件不能超过 20 MB。')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None: raise ValueError('无法读取图片，请上传 PNG 或 JPEG。')
    if image.shape[0]*image.shape[1]>30_000_000: raise ValueError('图片过大，请缩小至 3000 万像素以内。')
    return image
@app.get('/')
def home(): return render_template('index.html')
@app.get('/sample')
def sample():
    files={'real':'real_cotton_crop.jpg','synthetic':'synthetic_80.png','full':'catalog_32_0_Im0.jpg','bamboo':'user_bamboo.png'}
    name=request.args.get('name','real')
    row=next((r for r in evaluation_samples() if r['id']==name),None)
    if row is not None:
        path=(ROOT/'samples/counting_eval'/row['file']).resolve()
        if path.parent != (ROOT/'samples/counting_eval').resolve() or not path.is_file():
            return jsonify(error='此验证图片不在本机。'),404
        return send_file(path,mimetype='image/webp' if path.suffix.lower()=='.webp' else None)
    if name not in files: return jsonify(error='未知示例'),404
    if not (ROOT/'samples'/files[name]).is_file():
        return jsonify(error='此第三方对照图片未随公开源码分发；请使用竹签图、合成图或自行上传。'),404
    return send_file(ROOT/'samples'/files[name])
@app.get('/api/model-status')
def model_status():
    from auto_counter import readiness
    from scene_classifier import readiness as scene_readiness
    from model_startup import snapshot
    return jsonify(framework='GeCo2',weights_ready=(ROOT/'models/CNTQG_multitrain_ca44.pth').is_file(),
                   gpu='启动预加载并常驻',automatic=readiness(),scene_model=scene_readiness(),
                   startup=snapshot(),points_only=True)
@app.get('/api/service-status')
def service_status():
    """Launcher identity used to avoid confusing another listener with us."""
    provider=app.config.get('LOCAL_SERVICE_STATUS')
    if provider is None:
        return jsonify(state='unmanaged',ready=False)
    return jsonify(provider())
@app.post('/api/release-model')
def release_model():
    from auto_counter import release_model as release
    try:
        result=release()
        from scene_classifier import release_scene_model
        result['scene_model']=release_scene_model()
        return jsonify(result)
    except RuntimeError as error:return jsonify(error=str(error)),409
@app.post('/api/count')
def count():
    try:
        data=request.get_json(); im=decode(data['image'])
        engine=data.get('engine','smart')
        if engine=='smart' or (engine=='auto' and not data.get('category')):
            from smart_counter import detect_smart
            return jsonify(detect_smart(im))
        if engine=='auto':
            if data.get('category','bamboo') in {'rice','coins'}:
                if data.get('category')=='rice' and data.get('rice_layout','separated')!='separated':
                    raise ValueError('粘连或成堆米粒暂不支持可靠的一键计数；请铺开拍摄，或改用示例计数对照并逐点复核。')
                engine='grains'
            else:engine='countgdpp'
        if engine=='countgdpp':
            from auto_counter import detect_auto
            return jsonify(detect_auto(im,data.get('category','bamboo'),data.get('device','cpu'),
                                       keep_warm=data.get('keep_warm',False)))
        if engine=='beans':
            if data.get('category')!='mung':raise ValueError('绿豆颜色模式仅用于绿豆；其他目标请选择对应模式。')
            from mung_counter import detect_mung
            points,meta=detect_mung(im)
            return jsonify(points=points,count=len(points),width=im.shape[1],height=im.shape[0],meta=meta)
        if engine=='grains':
            if data.get('category','rice')=='rice' and data.get('rice_layout','separated')!='separated':
                raise ValueError('颗粒快速模式会把粘连米粒合并，不能用于整堆数量；请先将米粒分开。')
            from grain_counter import detect_grains
            points,meta=detect_grains(im)
            return jsonify(points=points,count=len(points),width=im.shape[1],height=im.shape[0],meta=meta)
        if engine=='geco2':
            from auto_counter import detect_geco2
            return jsonify(detect_geco2(im,data.get('exemplars',[]),data.get('model_threshold',.33),
                                       data.get('nms_iou'),data.get('device','cpu')))
        if engine!='opencv': raise ValueError('未知识别引擎。')
        d=float(data.get('diameter',24)); t=int(data.get('threshold',0)); s=float(data.get('sensitivity',.45))
        if not 4<=d<=300 or not 0<=t<=255 or not .1<=s<=1: raise ValueError('参数超出范围。')
        points,mask=detect(im,d,data.get('polarity','bright'),t,s)
        return jsonify(points=points,count=len(points),width=im.shape[1],height=im.shape[0],meta={'framework':'OpenCV'})
    except (ValueError,KeyError,TypeError) as e: return jsonify(error=str(e)),400
    except (RuntimeError,ImportError) as e: return jsonify(error='模型运行失败：'+str(e)),503
@app.post('/api/export')
def export():
    try:
        data=request.get_json(); im=decode(data['image']); points=data['points']
        if len(points)>10000: raise ValueError('标记数量过多。')
        for p in points:
            if not 0<=float(p['x'])<im.shape[1] or not 0<=float(p['y'])<im.shape[0]: raise ValueError('标记位置无效。')
        ok,buf=cv2.imencode('.png',annotate(im,points))
        return send_file(io.BytesIO(buf.tobytes()),mimetype='image/png',as_attachment=True,download_name='counted.png')
    except (ValueError,KeyError,TypeError) as e: return jsonify(error=str(e)),400
@app.errorhandler(413)
def large(e): return jsonify(error='图片文件不能超过 20 MB。'),413
def start_server():
    from model_startup import prepare_models,shutdown_models
    print('正在预加载计数模型到 GPU，并预热分类模型；就绪后启动网页服务。',flush=True)
    try:
        status=prepare_models()
        (ROOT/'outputs/startup_models.json').write_text(
            json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
        print('模型启动准备：'+json.dumps(status,ensure_ascii=False),flush=True)
        app.run(host='127.0.0.1',port=5080,debug=False)
    finally:
        for error in shutdown_models():
            print('模型释放未完成：'+error,flush=True)


if __name__=='__main__': start_server()
