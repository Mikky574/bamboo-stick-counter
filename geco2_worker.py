"""Legacy exemplar inference, isolated to release resources after every request."""
import json,os,sys
from pathlib import Path
import cv2,numpy as np

def main():
 source,request,out,device=sys.argv[1:]
 import torch
 torch.set_num_threads(4)
 if device!='cpu' and torch.cuda.is_available() and torch.cuda.mem_get_info()[0]<8*1024**3:
  raise RuntimeError('GPU free memory below 8 GiB; use CPU')
 from model_counter import detect_model
 settings=json.loads(Path(request).read_text(encoding='utf-8'))
 image=cv2.imdecode(np.fromfile(source,np.uint8),cv2.IMREAD_COLOR)
 points,meta=detect_model(image,settings['exemplars'],settings['threshold'],settings['nms_iou'])
 meta.update(process_lifetime='one_request',worker_exited=True)
 Path(out).write_text(json.dumps(dict(points=points,count=len(points),meta=meta,width=image.shape[1],height=image.shape[0])),encoding='utf-8')
if __name__=='__main__':main()
