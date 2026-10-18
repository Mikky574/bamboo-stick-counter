"""Package reviewed center-point labels; quarantine incomplete images."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'samples/real_dataset_v1'
DEST=PROJECT/'datasets/end_head_counting_v1'
SPLITS={'flat_sudrc':'train','flat_custom':'train','cotton_overhead':'train','flat_craft':'val'}
REVIEW={'cotton_niva','cotton_family'}
ROIS={'cotton_niva':(120,365,430,545),'cotton_family':(180,420,750,700)}
CLASSES={'bamboo_flat':(0,'stick_end'),'cotton':(1,'cotton_head')}

def write_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')

def main():
 DEST.mkdir(parents=True,exist_ok=True)
 manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf8'))
 byid={r['id']:r for r in manifest}
 queue=json.loads((SOURCE/'annotation_queue.json').read_text(encoding='utf8'))
 records=[];decisions=[];groups={};hashes={}
 for item in queue:
  ident=item['id'];a=json.loads((SOURCE/'annotations_v1'/(ident+'.points.json')).read_text(encoding='utf8'))
  src=SOURCE/item['file'];ext=src.suffix.lower();digest=hashlib.sha256(src.read_bytes()).hexdigest()
  split=SPLITS.get(ident)
  disposition='primary' if split else ('review_only' if ident in REVIEW else 'quarantine')
  if split:
   assert a['status']=='reviewed_single_pass' and a['ground_truth']==len(a['points'])
   assert all(p['review']=='accepted' for p in a['points']) and not a['ignore_regions']
   assert digest not in hashes,'Duplicate source image in primary dataset'
   assert ident not in groups or groups[ident]==split,'Group leakage'
   hashes[digest]=ident;groups[ident]=split
   image_rel=Path('images')/split/(ident+ext)
   label_rel=Path('annotations')/split/(ident+'.json')
  else:
   image_rel=Path(disposition)/'images'/(ident+ext)
   label_rel=Path(disposition)/'annotations'/(ident+'.json')
  image_path=DEST/image_rel;image_path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,image_path)
  category_id,category_name=CLASSES[item['category']]
  width,height=Image.open(src).size
  points=[[p['x'],p['y']] for p in a['points']]
  # All original points remain traceable, but incomplete files have no count target.
  label=dict(schema='end_head_counting.points.v1',image_id=ident,image_file=image_rel.as_posix(),width=width,height=height,category_id=category_id,category=category_name,points=points,point_review=[p['review'] for p in a['points']],ignore_regions=a['ignore_regions'],count=len(points) if split else None,annotation_status=a['status'],disposition=disposition,group_id=ident,split=split,independently_verified=False,source_page=byid[ident]['source_page'],source_image_url=byid[ident].get('image_url'),image_sha256=digest,annotation_method='manual_visual_points' if ident=='flat_sudrc' else 'model_assisted_single_pass_visual_review',review_notes=a['review_notes'])
  write_json(DEST/label_rel,label)
  decisions.append(dict(image_id=ident,disposition=disposition,split=split,points=len(points),count_target=label['count'],status=a['status'],annotation_file=label_rel.as_posix()))
  if not split:continue
  record=dict(image_id=ident,image_file=image_rel.as_posix(),annotation_file=label_rel.as_posix(),width=width,height=height,category_id=category_id,category=category_name,points=points,count=len(points),group_id=ident,split=split,image_sha256=digest)
  records.append(record)
  lm=dict(version='5.5.0',flags={'model_assisted':ident!='flat_sudrc','independent_review':False},shapes=[dict(label=category_name,points=[p],group_id=None,shape_type='point',flags={}) for p in points],imagePath='../../'+image_rel.as_posix(),imageData=None,imageWidth=width,imageHeight=height)
  write_json(DEST/'labelme'/split/(ident+'.json'),lm)
  preview=DEST/'previews'/(ident+'.jpg');preview.parent.mkdir(exist_ok=True)
  shutil.copy2(SOURCE/'annotations_v1'/(ident+'.review.jpg'),preview)
 # Use only separately inspected clear regions of the two partial originals.
 # Point inclusion is by center position; edge fragments whose center lies outside
 # the crop do not create count targets. Full-source labels remain review-only.
 for parent,box in ROIS.items():
  a=json.loads((SOURCE/'annotations_v1'/(parent+'.points.json')).read_text(encoding='utf8'))
  src=SOURCE/a['file'];original=Image.open(src).convert('RGB');im=original.crop(box)
  selected=[p for p in a['points'] if box[0]<=p['x']<box[2] and box[1]<=p['y']<box[3]]
  assert all(p['review']=='accepted' for p in selected)
  assert parent not in groups or groups[parent]=='train'
  groups[parent]='train';ident=parent+'_roi'
  points=[[round(p['x']-box[0],2),round(p['y']-box[1],2)] for p in selected]
  image_rel=Path('images/train')/(ident+'.png');label_rel=Path('annotations/train')/(ident+'.json')
  (DEST/image_rel).parent.mkdir(parents=True,exist_ok=True);im.save(DEST/image_rel)
  digest=hashlib.sha256((DEST/image_rel).read_bytes()).hexdigest()
  assert digest not in hashes;hashes[digest]=ident
  record=dict(image_id=ident,image_file=image_rel.as_posix(),annotation_file=label_rel.as_posix(),width=im.width,height=im.height,category_id=1,category='cotton_head',points=points,count=len(points),group_id=parent,split='train',image_sha256=digest,source_crop_xyxy=list(box),source_original_size=list(original.size),source_original_id=parent,count_scope='head_centers_inside_crop')
  label=dict(record,schema='end_head_counting.points.v1',point_review=['accepted']*len(points),ignore_regions=[],annotation_status='reviewed_clear_roi_single_pass',disposition='primary',independently_verified=False,source_page=byid[parent]['source_page'],source_image_url=byid[parent].get('image_url'),source_image_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),annotation_method='reviewed_model_assisted_points_then_clear_roi_crop',review_notes=['避开原图忽略区域的清晰局部；裁剪后再次目视检查；计数仅指中心点位于裁剪范围内的棉头。原图完整数量仍未知。'])
  write_json(DEST/label_rel,label);records.append(record)
  decisions.append(dict(image_id=ident,disposition='primary_clear_roi',split='train',points=len(points),count_target=len(points),status=label['annotation_status'],annotation_file=label_rel.as_posix()))
  lm=dict(version='5.5.0',flags={'model_assisted':True,'independent_review':False,'cropped_roi':True},shapes=[dict(label='cotton_head',points=[p],group_id=None,shape_type='point',flags={}) for p in points],imagePath='../../'+image_rel.as_posix(),imageData=None,imageWidth=im.width,imageHeight=im.height)
  write_json(DEST/'labelme/train'/(ident+'.json'),lm)
  draw=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',10)
  for i,(x,y) in enumerate(points,1):
   draw.ellipse((x-2,y-2,x+2,y+2),fill='cyan',outline='black')
   draw.text((x+3,y-5),str(i),font=font,fill='cyan',stroke_width=1,stroke_fill='black')
  im.save(DEST/'previews'/(ident+'.jpg'),quality=95)
 for split in ['train','val','test']:
  selected=[r for r in records if r['split']==split]
  (DEST/'splits').mkdir(exist_ok=True)
  (DEST/'splits'/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf8')
  (DEST/'splits'/(split+'.txt')).write_text(''.join(r['image_id']+'\n' for r in selected),encoding='utf8')
 write_json(DEST/'classes.json',[dict(id=0,name='stick_end',description='可见平端面；包含竹签及木圆棒辅助样本'),dict(id=1,name='cotton_head',description='靠近镜头一侧的可见棉头')])
 write_json(DEST/'provenance/source_manifest.json',[byid[r['id']] for r in queue])
 write_json(DEST/'manifest.json',dict(name='end_head_counting_v1',created_on='2026-10-18',annotation_type='point',coordinate_system='image_pixels_xy_after_declared_crop',primary_images=len(records),primary_source_groups=len(groups),primary_points=sum(r['count'] for r in records),split_policy='grouped_by_source_image_roi_crops_stay_with_parent_no_augmentation',validation_scope='pipeline_smoke_check_only_not_accuracy_benchmark',test_available=False,annotation_quality='single_pass_model_assisted_except_flat_sudrc',annotation_independently_verified=False,review_only_images=2,quarantined_images=3,records=records,decisions=decisions))
 with (DEST/'inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(decisions[0]));w.writeheader();w.writerows(decisions)
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
 sheet=Image.new('RGB',(1200,((len(records)+1)//2)*630),'#f1f4f7');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(records):
  im=Image.open(DEST/'previews'/(r['image_id']+'.jpg')).convert('RGB');im.thumbnail((580,565));x=i%2*600;y=i//2*630
  sheet.paste(im,(x+(600-im.width)//2,y));draw.text((x+15,y+580),f"{r['image_id']} | {r['split']} | {r['count']} points",fill='black',font=font)
 sheet.save(DEST/'dataset_preview.jpg',quality=94)
 (DEST/'README.md').write_text('''# 端头中心点计数数据集 v1

2026-10-18。按用户反馈，暂不纳入拼图中上 cotton_muji、中间 cotton_aijie 和左下 flat_umall。

主数据集6张图、709个端头中心点，来自6个原图组。其中4张使用完整图片，两张使用避开遮挡并再次目视复查的清晰局部图（cotton_niva_roi 65点、cotton_family_roi 150点），不是对整盒总数的标注。

训练5张610点（flat_sudrc 50、flat_custom 238、cotton_overhead 107、cotton_niva_roi 65、cotton_family_roi 150）；验证1张99点（flat_craft）。同一原图、裁剪和增强按group_id保持同组；本批没有数据增强，也没有独立测试集。验证仅供检查训练流程；验证集中没有棉头，不能报告棉头泛化准确率。

## 使用

- images/train、images/val：原始图片。
- annotations/train、annotations/val：逐图像素坐标点标注，count仅对主数据集设置。
- splits/train.jsonl、val.jsonl：可直接读取的训练索引；test.jsonl为空。
- labelme/：可编辑的LabelMe中心点标注，图片引用已配套。
- dataset.py：轻量数据读取器，返回PIL图像和点坐标；缩放时同步变换坐标。示例见文件末尾。
- previews/、dataset_preview.jpg：编号标注预览。
- inventory.csv、manifest.json：原图分组、来源和入库状态。

坐标为当前图像像素[x,y]；完整图与原图一致，局部图按记录的source_crop_xyxy平移坐标。局部图只计中心落在裁剪范围内的棉头，只有边缘碎片而中心位于裁剪范围外的不计入。类别0=stick_end（平端面，包含木圆棒辅助样本），1=cotton_head。所有坐标均在图像范围内，每个点只代表一个近侧可见端头，不标杆身和另一侧头。当前没有尖头类别的合格样本。

## 复核区

review_only/保留 cotton_niva 和 cotton_family 原图的局部可见标注，含待确认点及忽略区域。原图不在训练/验证索引中，count为null；只有单独裁出并复查的两张ROI进入训练。不能仅删掉不确定点就把全图当作完整训练图，否则未标目标会变成错误负例。

quarantine/保留用户指出疑似漏标的 cotton_muji、cotton_aijie，以及被剔除的 flat_umall，供修正。原点位保留追踪，但count为null，不进训练。flat_umall的0个已接受点不代表图中有0根签子。

## 标注质量与训练范围

这是初始点标注数据集：flat_sudrc为逐排视觉点标，其余主集由模型候选辅助并单轮目视检查；未经独立盲标复核。来源为网络商品图，来源和授权核实状态保留在原始来源清单中；不能据此宣称独立测试准确率。709个点不等于709张独立图片，当前只有6个主集原图组，适合打通点监督计数训练流程。

没有制作实例边界框，因此此包不是YOLO检测框格式。不要把点随意扩大成固定方框当作人工框真值。图像版权/训练准入仍需按来源核实；本次只整理数据，没有启动训练或GPU模型。
''',encoding='utf8')
 shutil.copy2(PROJECT/'samples/counting_dataset_loader.py',DEST/'dataset.py')
 # Package only the generated delivery files, retaining relative source paths.
 archive=DEST.with_suffix('.zip')
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for path in sorted(DEST.rglob('*')):
   if path.is_file() and '__pycache__' not in path.parts:z.write(path,path.relative_to(DEST.parent).as_posix())
 print(f'Dataset: {DEST}\nPrimary: {len(records)} images / {sum(r["count"] for r in records)} points\nArchive: {archive}')

if __name__=='__main__':main()
