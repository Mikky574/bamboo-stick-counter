"""Freeze sample hints and provenance after visual inspection. No model training."""
import json,hashlib
from pathlib import Path
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent/'counting_eval'
rows=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
settings={
 'bamboo_hand':('竹签端面 · 手持小捆',[[249,266,303,331],[218,311,273,381],[323,283,376,357]]),
 'bamboo_bundle':('竹签端面 · 密集成捆',[[537,188,570,228],[599,230,632,267],[627,345,664,389]]),
 'bamboo_original':('竹签端面 · 原项目示例',[[315,132,360,172],[358,326,405,369],[287,434,332,479]]),
 'rice_gray':('米粒 · 分散清晰',[[177,62,201,75],[230,61,245,89],[227,181,260,200]]),
 'cotton_dense':('棉头 · 密集粘连（裁剪）',[[949,104,1097,421],[684,507,832,767],[1430,694,1604,998]]),
}
for row in rows:
 if row['id'] not in settings:
  continue
 row['title'],row['exemplars']=settings[row['id']]
 row['engine']='grains' if row['category']=='rice' else 'geco2'
 row['note']='示例框为预设推理提示，不是训练。数量未人工核验。'
 if row['id']=='bamboo_hand':
  row['ground_truth']=18
  row['ground_truth_method']='Manual single-review centers in bamboo_hand_truth.json; provisional, not independent multi-person annotation.'
  row['note']='人工初标 18 个端面；可比较基线与示例/NMS设置，同图调参结果不代表泛化准确率。'
  row['comparison']=dict(exemplars=[[230,171,275,232],[218,311,273,381],[317,361,369,437]],nms_iou=.5,threshold=.33)
 if row['id']=='cotton_dense':
  im=Image.open(ROOT/'cotton_dense.jpg');box=(400,1040,3000,2450)
  im.crop(box).save(ROOT/'cotton_heads_crop.jpg',quality=95)
  row.update(file='cotton_heads_crop.jpg',original_file='cotton_dense.jpg',crop_xyxy=list(box),scene_group='cotton_product_A')
  row['note']='同一照片仅截取上侧棉头；严重粘连难例，数量未核验。'
 if row['id']=='rice_gray':
  row['ground_truth']=30
  row['ground_truth_method']='Single-image visual count and review of one marker per separated grain; provisional, not independent multi-person annotation.'
  row['note']='人工目视复核 30 粒；单图初步检查，非独立测试集。'
 row['sha256']=hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()
extra=ROOT/'rice_extra.jpg'
if extra.exists() and not any(r['id']=='rice_touching' for r in rows):
 rows.append(dict(id='rice_touching',category='rice',file=extra.name,title='米粒 · 接触粘连（难例）',engine='geco2',exemplars=[[324,72,437,114],[272,170,372,208],[201,228,336,275]],source_page='https://github.com/hkedia321/rice-quality-analysis/blob/HEAD/img1.jpg',image_url='https://raw.githubusercontent.com/hkedia321/rice-quality-analysis/HEAD/img1.jpg',ground_truth=None,license='Image reuse rights not verified; local evaluation candidate',note='接触粘连难例；颗粒连通域可能合并，数量未核验。',sha256=hashlib.sha256(extra.read_bytes()).hexdigest()))
(ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
sheet=Image.new('RGB',(1200,600),'#eee');d=ImageDraw.Draw(sheet)
for i,row in enumerate(rows):
 im=Image.open(ROOT/row['file']).convert('RGB');im.thumbnail((380,235));x=i%3*400;y=i//3*300
 sheet.paste(im,(x+(400-im.width)//2,y));d.text((x+8,y+250),row['id'],fill='black')
sheet.save(ROOT/'preview.jpg')
print('Prepared',len(rows),'samples')
