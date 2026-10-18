"""Record visual screening and prepare an annotation queue without fake labels."""
import hashlib,json,shutil
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent/'real_dataset_v1'
MAIN={'flat_sudrc','cotton_muji','cotton_niva','cotton_family','cotton_aijie','flat_custom','flat_craft','flat_umall','cotton_overhead'}
NOTES={
 'flat_sudrc':'近轴向平端面，木圆棒辅助样本；需逐实例复核。',
 'cotton_muji':'正对盒口候选，主体占比小；标注前放大检查边界。',
 'cotton_niva':'近轴向盒口棉头，外圈和包装边缘需复核。',
 'cotton_family':'近俯视实拍候选；外圈棉头需复核。',
 'cotton_aijie':'近轴向整盒棉头；密集边界需复核。',
 'sharp_bakeyy':'商品拼图，多处重复展示同一捆，不能作为整图数量真值。',
 'cotton_daiso':'盒盖与斜拍遮挡，不进入主计数集。',
 'sharp_packstyle':'图片主要展示平头端且分辨率低，不作为尖头样本。',
}

def dhash(path):
 im=Image.open(path).convert('L').resize((9,8));p=list(im.getdata())
 return sum(int(p[y*9+x]>p[y*9+x+1])<<(y*8+x) for y in range(8) for x in range(8))

def main():
 rows=json.loads((ROOT/'manifest.json').read_text(encoding='utf8'))
 oldroot=ROOT.parent/'dense_cases'
 old=json.loads((oldroot/'manifest.json').read_text(encoding='utf8'))
 (ROOT/'existing_originals').mkdir(exist_ok=True)
 ids={r['id'] for r in rows}
 for r in old:
  if 'file' not in r or r['id'] in ids:continue
  dest=ROOT/'existing_originals'/Path(r['file']).name;shutil.copy2(oldroot/r['file'],dest)
  row=dict(r);row.update(file=str(dest.relative_to(ROOT)),annotation_status='unlabeled',split=None)
  rows.append(row)
 hashes={};phashes=[]
 for r in rows:
  if 'file' not in r:continue
  path=ROOT/r['file'];sha=hashlib.sha256(path.read_bytes()).hexdigest();h=dhash(path)
  r['sha256']=sha;r['dhash']=f'{h:016x}'
  r['screening']='annotation_candidate' if r['id'] in MAIN else 'excluded_from_primary'
  r['review_note']=NOTES.get(r['id'],r.get('review_note','侧拍或端头遮挡，不进入主计数集。'))
  r['ground_truth']=None;r['accuracy_verified']=False;r['reviewed_on']='2026-10-17'
  r['ready_for_training']=False;r['split']=None
  if sha in hashes:r['duplicate_of']=hashes[sha];r['screening']='duplicate'
  else:hashes[sha]=r['id']
  similar=[i for i,v in phashes if (h^v).bit_count()<=4]
  if similar:r['possible_visual_duplicates']=similar
  phashes.append((r['id'],h))
 (ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 queue=[dict(id=r['id'],file=r['file'],category=r['category'],source_page=r['source_page'],group_id=r['id'],status='needs_manual_annotation_and_count_review',ground_truth=None) for r in rows if r.get('screening')=='annotation_candidate']
 (ROOT/'annotation_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf8')
 sheet=Image.new('RGB',(1200,((len(queue)+3)//4)*295),'#eeeeee');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(queue):
  im=Image.open(ROOT/r['file']).convert('RGB');im.thumbnail((288,250));x=i%4*300;y=i//4*295
  sheet.paste(im,(x+(300-im.width)//2,y));d.text((x+7,y+258),r['id'],fill='black')
 sheet.save(ROOT/'primary_candidates.jpg',quality=92)
 counts=Counter(r.get('screening','download_failed') for r in rows)
 text=f'''# 真实计数数据第一批（2026-10-17）

本轮新下载18张，另纳入此前13张；合计31张下载图。目视初筛留下9张待标注候选，其余侧拍、遮挡、拼图等保留记录但排除主计数集。9张中平端面4张（含木圆棒辅助样本），棉头5张，合格尖头0张。尚未训练，所有数量真值为空。

- originals/：本轮原始下载，保留原文件字节。
- existing_originals/：此前收集图片的副本，不重复计为新增。
- manifest.json：来源、哈希、视角筛选、失败请求和排除原因。
- annotation_queue.json：仅包含9张初筛候选，仍需逐图确认能否完整标注。
- primary_candidates.jpg：主候选预览。未通过筛选的图片不混入这里。

## 标注规则

每根只标同一侧端头；平头框紧端面，尖头标尖端区域，棉头只标靠近镜头一侧。不得标整根杆身、反射、宣传插图或另一侧头。完全遮挡不能猜测补标；边界无法确认的照片退回，不作为精确整捆计数验收图。包装标称数量不能作真值。

优先人工逐实例标注，自动候选必须人工补漏删错；不将当前模型结果直接转成训练答案。标注文件未生成空标签，避免把未标注图片误当作无目标负样本。

## 数据隔离与下一步

同一商品、同一实物的不同裁剪和增强版本必须同组，划分训练/验证/测试前人工复核近重复与来源；哈希只辅助排重。当前没有划分测试集，也没有独立真值和准确率。商品图片授权状态未确认，记录保留为本地研究候选，训练准入待核实；生成图片不纳入本批。

下一步先完成这9张的端头标注和数量复核，再扩充真实拍摄数据。尖头正视合格样本仍缺，不能用侧拍凑数。网络图尚不足支撑稳定微调。
'''
 (ROOT/'README.md').write_text(text,encoding='utf8')
 print(json.dumps(dict(screening=counts,queue=len(queue)),ensure_ascii=False))
if __name__=='__main__':main()
