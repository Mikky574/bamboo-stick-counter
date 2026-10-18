"""Real bundled-image inference; never equates a product pack size with truth."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from auto_counter import detect_auto
from counter import annotate

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'samples/dense_cases'
OUT=ROOT/'outputs/dense_cases'
SELECTED=['flat_factory','flat_custom','flat_craft','flat_umall','sharp_relaxdays','round_rods','flat_vice']

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--ids',nargs='+');args=parser.parse_args()
 OUT.mkdir(parents=True,exist_ok=True)
 rows=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
 reports=[]
 for ident in (args.ids or SELECTED):
  row=next(r for r in rows if r['id']==ident)
  if 'file' not in row:continue
  image=cv2.imdecode(np.fromfile(DATA/row['file'],np.uint8),cv2.IMREAD_COLOR)
  out=OUT/(ident+'.json')
  if out.exists():
   result=json.loads(out.read_text(encoding='utf-8'))
  else:
   print('Starting',ident,flush=True)
   result=detect_auto(image,'cotton' if row['category']=='cotton' else 'bamboo','cpu')
   result.update(source_page=row['source_page'],source_file=row['file'],origin='real_web_photo',ground_truth=None,accuracy_verified=False)
   out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
  cv2.imencode('.jpg',annotate(image,result['points']))[1].tofile(str(OUT/(ident+'.jpg')))
  reports.append(dict(id=ident,predicted_count=result['count'],ground_truth=None,accuracy_verified=False,meta=result['meta'],source_page=row['source_page']))
  (OUT/'report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
  print(ident,result['count'],result['meta']['elapsed_seconds'],flush=True)
 sheet=Image.new('RGB',(1500,((len(reports)+2)//3)*440),'#f2f4f3');d=ImageDraw.Draw(sheet)
 font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
 for i,row in enumerate(reports):
  im=Image.open(OUT/(row['id']+'.jpg'));im.thumbnail((480,380));x=i%3*500;y=i//3*440
  sheet.paste(im,(x+(500-im.width)//2,y));d.text((x+10,y+385),f'{row["id"]} | 预测 {row["predicted_count"]}',font=font,fill='#19362d')
  d.text((x+10,y+412),'真实照片 · 数量待逐点复核',font=font,fill='#536b60')
 sheet.save(OUT/'results_preview.jpg',quality=94)
if __name__=='__main__':main()
