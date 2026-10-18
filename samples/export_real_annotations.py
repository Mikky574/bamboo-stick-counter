"""Build a reviewable annotation delivery and validate geometry/file references."""
import csv,html,json,zipfile
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from annotate_real_dataset import ROOT,OUT

def main():
 queue=json.loads((ROOT/'annotation_queue.json').read_text(encoding='utf8'))
 rows=[json.loads((OUT/(r['id']+'.points.json')).read_text(encoding='utf8')) for r in queue]
 summary=[]
 for r in rows:
  for p in r['points']:assert 0<=p['x']<r['width'] and 0<=p['y']<r['height'],r['id']
  assert r['annotation_count']==len(r['points'])
  if r['status']=='reviewed_single_pass':assert r['ground_truth']==len(r['points'])
  label=OUT/(r['id']+'.labelme.json')
  if r['points']:
   lm=json.loads(label.read_text(encoding='utf8'))
   assert (label.parent/lm['imagePath']).is_file()
   assert len([s for s in lm['shapes'] if s['shape_type']=='point'])==len(r['points'])
  else:assert not label.exists(),'Excluded image must not become a negative label'
  summary.append(dict(id=r['id'],category=r['category'],status=r['status'],annotation_count=len(r['points']),accepted_points=sum(p['review']=='accepted' for p in r['points']),uncertain_points=sum(p['review']!='accepted' for p in r['points']),visible_count=r.get('visible_count'),ground_truth=r.get('ground_truth'),independently_verified=False))
 (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
 with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 # Update the source queue without dropping original provenance and source URLs.
 byid={r['id']:r for r in rows}
 for r in queue:
  a=byid[r['id']];r.update(status=a['status'],ground_truth=a['ground_truth'],annotation_file='annotations_v1/'+r['id']+'.points.json',annotation_count=a['annotation_count'])
 (ROOT/'annotation_queue.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding='utf8')
 manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf8'))
 for r in manifest:
  if r['id'] in byid:
   a=byid[r['id']];r.update(annotation_status=a['status'],ground_truth=a['ground_truth'],annotation_file='annotations_v1/'+r['id']+'.points.json',annotation_count=a['annotation_count'],accuracy_verified=False,ready_for_training=False)
 (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
 sheet=Image.new('RGB',(1500,3*555),'#f1f4f7');d=ImageDraw.Draw(sheet)
 cards=[]
 for i,(r,s) in enumerate(zip(rows,summary)):
  im=Image.open(OUT/(r['id']+'.review.jpg')).convert('RGB')
  if r['id']=='cotton_muji':im=im.crop((295,470,915,1020))
  im.thumbnail((485,475));x=i%3*500;y=i//3*555;sheet.paste(im,(x+(500-im.width)//2,y))
  label='visible count: '+str(s['ground_truth']) if s['ground_truth'] is not None else ('PARTIAL: '+str(s['annotation_count'])+' points' if r['points'] else 'EXCLUDED')
  d.text((x+10,y+485),r['id'],font=font,fill='black');d.text((x+10,y+511),label,font=font,fill='#087044' if s['ground_truth'] is not None else '#a14c00')
  notes=html.escape(' '.join(r['review_notes']))
  cards.append(f'''<article><h2>{r['id']}</h2><p class="status">{html.escape(label)}</p><div class="image"><img src="{r['id']}.review.jpg" data-marked="{r['id']}.review.jpg" data-raw="../{r['file'].replace(chr(92),'/')}" alt="{r['id']}"></div><p>{notes}</p><p>已确认点 {s['accepted_points']}；遮挡待确认点 {s['uncertain_points']}</p><a href="{r['id']}.points.json">点位 JSON</a>{(' · <a href="'+r['id']+'.labelme.json">LabelMe</a>') if r['points'] else ''}</article>''')
 sheet.save(OUT/'annotation_overview.jpg',quality=94)
 page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐实例计数标注</title><style>body{font:16px system-ui;margin:24px;background:#f4f6f8;color:#162537}header{max-width:1100px;margin:auto}button{padding:10px 18px;margin:8px 0;cursor:pointer}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:22px}article{background:white;border-radius:12px;padding:18px}h2{font-size:20px}.status{font-weight:bold}.image{height:520px;overflow:auto;background:#eceff3;display:grid;place-items:center}img{max-width:100%;max-height:520px;cursor:zoom-in}img.zoom{max-width:none;max-height:none}p{line-height:1.6}a{color:#1269c5}</style><header><h1>逐实例计数标注 · 第一轮</h1><p>9 张逐图处理：4 张可见端头计数初标；2 张局部可见标注；2 张经用户反馈撤回待修正；1 张剔除。青色为目视接受点，橙色为遮挡待确认点，紫线为忽略区。全部尚未经独立第二轮盲标复核。</p><p>自动候选经过目视检查与修正；部分样本由同一个计数模型辅助，不能用这些标签证明该模型的独立测试准确率。点标注不等于检测框标注。</p><button id="toggle">切换原图 / 标注</button><p>点击图片放大，再次点击恢复；局部可见的点数不能当作整盒或整捆总数。</p></header><main class="grid">'''+''.join(cards)+'''</main><script>let raw=false;document.getElementById('toggle').onclick=()=>{raw=!raw;document.querySelectorAll('img').forEach(im=>im.src=raw?im.dataset.raw:im.dataset.marked)};document.querySelectorAll('img').forEach(im=>im.onclick=()=>im.classList.toggle('zoom'))</script></html>'''
 (OUT/'index.html').write_text(page,encoding='utf8')
 count=sum(s['annotation_count'] for s in summary)
 lines=['# 逐实例标注交付（2026-10-18）','',f'逐张处理9张：4张可见端头计数初标，2张局部可见标注，2张经用户反馈撤回待修正，1张剔除。8张共保存{count}个点，其中含明确标为uncertain_occlusion的待确认点。尚无独立第二轮复核，不宣称整批标注已经达到精确计数金标准。','','| 图片 | 点数 | 可见计数初标 | 状态 |','|---|---:|---:|---|']
 for s in summary:lines.append(f"| {s['id']} | {s['annotation_count']} | {s['ground_truth'] if s['ground_truth'] is not None else '—'} | {s['status']} |")
 lines+=['','flat_sudrc为从原图逐排人工点标；其余使用CountGD++候选辅助，分区放大目视检查，并修正明确重复和非目标点。保留原始候选与删除记录；不能把本轮作为该模型的独立准确率测试。','', 'cotton_niva、cotton_family、cotton_aijie有遮挡、粘连或重复响应歧义，ground_truth保持null，橙色点不算确认标注，紫线区域应忽略。flat_umall候选存在大量木纹误检和漏点，无法可靠完成整图标注，因此退出本批，未生成空标签负样本。','','文件：每图.points.json为主记录，.labelme.json为点与忽略多边形，.review.jpg为编号预览；summary.csv汇总；index.html可离线切换原图/标注和放大。点标注适合计数监督准备；尚未标逐实例边界框，不能直接当作YOLO检测框训练集。训练导入时必须处理忽略区域、排除橙色点及未完成图；目前未启动训练。','','压缩包包含9张源图供逐图复查、8份LabelMe点标注、9份状态记录和预览汇总。剔除图仅供查看，没有LabelMe标签，禁止当作负样本。']
 (OUT/'README.md').write_text('\n'.join(lines),encoding='utf8')
 with zipfile.ZipFile(ROOT/'annotations_v1.zip','w',zipfile.ZIP_DEFLATED) as z:
  for name in ['README.md','index.html','summary.csv','summary.json','annotation_overview.jpg']:
   z.write(OUT/name,'annotations_v1/'+name)
  for r in rows:
   z.write(ROOT/r['file'],r['file'].replace('\\','/'))
   for suffix in ['.points.json','.review.jpg']:
    z.write(OUT/(r['id']+suffix),'annotations_v1/'+r['id']+suffix)
   if r['points']:
    z.write(OUT/(r['id']+'.labelme.json'),'annotations_v1/'+r['id']+'.labelme.json')
 print(json.dumps(dict(images=len(rows),labels=8,points=count,status_counts=dict(Counter(r['status'] for r in rows)),summary=summary),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
