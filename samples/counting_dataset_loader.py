"""Read a portable center-point counting dataset. No GPU/model initialization."""
import json
from pathlib import Path
import numpy as np
from PIL import Image

class PointCountingDataset:
 def __init__(self,root,split='train',image_size=None):
  self.root=Path(root)
  if split not in {'train','val','test'}:raise ValueError('Unknown split')
  self.records=[json.loads(line) for line in (self.root/'splits'/(split+'.jsonl')).read_text(encoding='utf8').splitlines() if line.strip()]
  if image_size is not None and (len(image_size)!=2 or min(image_size)<=0):raise ValueError('image_size must be positive (width,height)')
  self.image_size=image_size
 def __len__(self):return len(self.records)
 def __getitem__(self,index):
  row=self.records[index];im=Image.open(self.root/row['image_file']).convert('RGB')
  points=np.array(row['points'],dtype=np.float32).reshape(-1,2)
  original_size=im.size
  if self.image_size is not None:
   width,height=self.image_size;points=points*np.array([width/im.width,height/im.height],dtype=np.float32)
   im=im.resize((width,height),Image.Resampling.BILINEAR)
  return im,dict(image_id=row['image_id'],points=points,count=row['count'],labels=np.full(len(points),row['category_id'],dtype=np.int64),original_size=original_size,size=im.size)

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root',nargs='?',default=str(Path(__file__).resolve().parent));p.add_argument('--split',choices=['train','val','test'],default='train');a=p.parse_args()
 ds=PointCountingDataset(a.root,a.split)
 for i in range(len(ds)):
  image,target=ds[i];print(target['image_id'],image.size,target['count'])
