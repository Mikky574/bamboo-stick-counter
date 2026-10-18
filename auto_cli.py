"""Run automatic counting / provisional point annotation with no visual prompts."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from auto_counter import detect_auto
from grain_counter import detect_grains
from counter import annotate

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('input',help='Image file or folder')
 p.add_argument('--category',choices=['bamboo','cotton','rice'],default='bamboo')
 p.add_argument('--device',choices=['cpu','auto','cuda'],default='cpu')
 p.add_argument('--separated-grains',action='store_true',help='Use connected components for separated grains only')
 p.add_argument('--output',default='outputs/auto_labels');a=p.parse_args()
 source=Path(a.input);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 files=[source] if source.is_file() else sorted(f for f in source.iterdir() if f.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'})
 if not files:p.error('No images found')
 reports=[]
 for path in files:
  image=cv2.imdecode(np.fromfile(path,np.uint8),cv2.IMREAD_COLOR)
  if image is None:p.error('Cannot read '+str(path))
  if a.separated_grains:
   points,meta=detect_grains(image);result=dict(count=len(points),points=points,meta=meta,width=image.shape[1],height=image.shape[0])
  else:result=detect_auto(image,a.category,a.device)
  result.update(source_image=str(path.resolve()),annotation_status='unreviewed_prediction',ground_truth=False)
  # Keep source extension in the stem to avoid overwriting same-name JPG/PNG files.
  stem=out/path.name
  Path(str(stem)+'.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
  cv2.imencode('.jpg',annotate(image,result['points']))[1].tofile(str(stem)+'.counted.jpg')
  labelme=dict(version='5.5.0',flags=dict(automatically_generated=True,reviewed=False),
    shapes=[dict(label=a.category,points=[[q['x'],q['y']]],group_id=None,shape_type='point',flags={}) for q in result['points']],
    imagePath=path.name,imageData=None,imageHeight=result['height'],imageWidth=result['width'])
  Path(str(stem)+'.labelme.json').write_text(json.dumps(labelme,ensure_ascii=False,indent=2),encoding='utf-8')
  # Image alongside annotation permits review without resolving external source paths.
  import shutil
  if path.resolve()!=stem.resolve():shutil.copy2(path,stem)
  reports.append(dict(image=path.name,count=result['count'],meta=result['meta']))
  print(path.name,result['count'],flush=True)
 (out/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
