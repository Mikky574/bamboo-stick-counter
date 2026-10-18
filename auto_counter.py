"""Process-isolated inference. The web server never owns a CountGD++ model."""
import json,os,subprocess,tempfile,threading,time
from pathlib import Path
import cv2
from inference_process import ResidentWorker
from runtime_capacity import COUNTGDPP_IDLE_TIMEOUT_SECONDS

ROOT=Path(__file__).resolve().parent
_job_lock=threading.Lock()
_warm_worker=ResidentWorker(idle_timeout_seconds=COUNTGDPP_IDLE_TIMEOUT_SECONDS)
CATEGORIES={'bamboo','cotton','rice','mung','coins','bamboo_sharp','chopsticks','rods','automatic'}
WARM_REQUEST_TIMEOUT_SECONDS=60

def prewarm_model():
 """Prepare the same CUDA session that image requests will reuse."""
 started=time.perf_counter()
 if not readiness()['installed']:raise RuntimeError('免框选模型未准备好，请运行 setup_auto.ps1。')
 if not _job_lock.acquire(blocking=False):raise RuntimeError('本项目已有识别任务正在运行，请等待完成。')
 try:
  out=ROOT/'outputs';out.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='countgdpp_warmup_',dir=out) as folder:
   result=Path(folder)/'result.json'
   env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
      TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8')
   command=[str(ROOT/'.venv-countgdpp/Scripts/python.exe'),str(ROOT/'countgdpp_worker.py'),
            '--serve','--device','cuda']
   reused=_warm_worker.request(command,dict(action='warmup',output=str(result.resolve())),
       cwd=ROOT,env=env,log_path=out/'countgdpp_server.log',timeout=WARM_REQUEST_TIMEOUT_SECONDS,
       creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
   try:
    payload=json.loads(result.read_text(encoding='utf-8'))
    if (payload.get('ready') is not True or payload.get('meta',{}).get('device')!='cuda'):
     raise RuntimeError('GPU模型预热返回格式无效。')
   except Exception:
    _warm_worker.release()
    raise
   payload['meta'].update(cache_reused=reused,total_elapsed_seconds=round(time.perf_counter()-started,3))
   return payload
 finally:_job_lock.release()


def readiness():
 python=ROOT/'.venv-countgdpp/Scripts/python.exe'
 weights=ROOT/'models/countgdpp/countgd_plusplus.pth'
 bert=ROOT/'models/countgdpp/bert-base-uncased'
 required=[python,weights,bert/'config.json',bert/'vocab.txt',ROOT/'vendor/countgdpp/cfg_app.py']
 return dict(installed=all(p.is_file() for p in required),missing=[str(p.relative_to(ROOT)) for p in required if not p.is_file()],
             runtime='isolated_subprocess',default_device='cpu',busy=_job_lock.locked(),**_warm_worker.snapshot())

def detect_auto(image,category='bamboo',device='cpu',keep_warm=False):
 if category not in CATEGORIES:raise ValueError('未知计数对象，请选择竹签、棉头、米粒、绿豆或硬币。')
 if device not in {'cpu','auto','cuda'}:raise ValueError('未知运行设备。')
 if not isinstance(keep_warm,bool):raise ValueError('模型缓存开关必须是布尔值。')
 if keep_warm and device!='cuda':raise ValueError('模型常驻仅支持显式选择 GPU。')
 if not readiness()['installed']:raise RuntimeError('免框选模型未准备好，请运行 setup_auto.ps1。')
 return _execute(image,'countgdpp',device,dict(category=category),keep_warm=keep_warm)

def release_model():
 if not _job_lock.acquire(blocking=False):raise RuntimeError('识别任务正在运行，完成后才能释放模型。')
 try:released=_warm_worker.release()
 finally:_job_lock.release()
 return dict(released=released,**readiness())

def detect_geco2(image,exemplars,threshold=.33,nms_iou=None,device='cpu'):
 from model_counter import validate_exemplars
 validate_exemplars(exemplars,image.shape[1],image.shape[0])
 if not .125<=float(threshold)<=.95:raise ValueError('模型筛选阈值必须在 0.125–0.95 之间。')
 if nms_iou is not None and not .1<=float(nms_iou)<=.95:raise ValueError('NMS 参数无效。')
 if device not in {'cpu','cuda','auto'}:raise ValueError('未知运行设备。')
 return _execute(image,'geco2',device,dict(exemplars=exemplars,threshold=float(threshold),nms_iou=nms_iou))

def _execute(image,engine,device,settings,keep_warm=False):
 started=time.perf_counter()
 if not _job_lock.acquire(blocking=False):raise RuntimeError('本项目已有识别任务正在运行，请等待完成。')
 try:
  if not keep_warm:_warm_worker.release()
  out=ROOT/'outputs';out.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(prefix='countgdpp_',dir=out) as folder:
   folder=Path(folder);source=folder/'input.png';result=folder/'result.json'
   ok,encoded=cv2.imencode('.png',image)
   if not ok:raise ValueError('图片编码失败。')
   encoded.tofile(source)
   if engine=='countgdpp':
    command=[str(ROOT/'.venv-countgdpp/Scripts/python.exe'),str(ROOT/'countgdpp_worker.py'),str(source),
             '--category',settings['category'],'--device',device,'--output',str(result)]
   else:
    request=folder/'settings.json';request.write_text(json.dumps(settings),encoding='utf-8')
    command=[str(ROOT/'.venv/Scripts/python.exe'),str(ROOT/'geco2_worker.py'),str(source),str(request),str(result),device]
   env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8')
   if device=='cpu':env.update(CUDA_VISIBLE_DEVICES='-1',PYTORCH_NVML_BASED_CUDA_CHECK='0')
   flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
   reused=False
   if keep_warm:
    command=[str(ROOT/'.venv-countgdpp/Scripts/python.exe'),str(ROOT/'countgdpp_worker.py'),'--serve','--device','cuda']
    reused=_warm_worker.request(command,dict(image=str(source.resolve()),output=str(result.resolve()),category=settings['category']),
       cwd=ROOT,env=env,log_path=out/'countgdpp_server.log',timeout=WARM_REQUEST_TIMEOUT_SECONDS,creationflags=flags)
   else:
    try:
     process=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                            timeout=600,creationflags=flags)
    except subprocess.TimeoutExpired as e:
     # subprocess.run kills and waits for its child on timeout.
     raise RuntimeError('识别超过 10 分钟，子进程已结束，模型资源已释放。') from e
    if process.returncode:
     log=process.stderr.decode('utf-8',errors='replace')
     (out/(engine+'_last_error.log')).write_text(log,encoding='utf-8')
     detail=next((line for line in reversed(log.splitlines()) if line.strip()),'unknown error')
     raise RuntimeError(f'推理失败；详情见 outputs/{engine}_last_error.log。'+detail[:250])
   try:
    payload=json.loads(result.read_text(encoding='utf-8'))
    if not isinstance(payload,dict) or not isinstance(payload.get('meta'),dict):raise RuntimeError('推理结果格式无效，模型资源已释放。')
   except Exception:
    if keep_warm:_warm_worker.release()
    raise
   payload['meta'].update(worker_exited=not keep_warm,cache_reused=reused)
 finally:_job_lock.release()
 payload['meta']['total_elapsed_seconds']=round(time.perf_counter()-started,3)
 return payload
