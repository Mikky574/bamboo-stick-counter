import base64
import io
from pathlib import Path
import cv2
import numpy as np
from flask import Flask,request,jsonify,render_template,send_file
from counter import detect,annotate
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=20*1024*1024
app.config['TEMPLATES_AUTO_RELOAD']=True
ROOT=Path(__file__).parent

def decode(value):
    raw=base64.b64decode(value.split(',')[-1],validate=True)
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
    if name not in files: return jsonify(error='未知示例'),404
    if not (ROOT/'samples'/files[name]).is_file():
        return jsonify(error='此第三方对照图片未随公开源码分发；请使用竹签图、合成图或自行上传。'),404
    return send_file(ROOT/'samples'/files[name])
@app.get('/api/model-status')
def model_status():
    import torch
    return jsonify(framework='GeCo2',weights_ready=(ROOT/'models/CNTQG_multitrain_ca44.pth').is_file(),
                   gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                   points_only=True)
@app.post('/api/count')
def count():
    try:
        data=request.get_json(); im=decode(data['image'])
        engine=data.get('engine','opencv')
        if engine=='geco2':
            from model_counter import detect_model
            points,meta=detect_model(im,data.get('exemplars',[]),data.get('model_threshold',.33),
                                     data.get('nms_iou'))
            return jsonify(points=points,count=len(points),width=im.shape[1],height=im.shape[0],meta=meta)
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
if __name__=='__main__': app.run(host='127.0.0.1',port=5080,debug=False)
