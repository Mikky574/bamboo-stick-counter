"""Collect visual screening and measured predictions without invented ground truth."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'samples/dense_cases';OUT=ROOT/'outputs/dense_cases'
NOTES={
 'flat_factory':('outside_preferred_view','斜拍，杆身可见且边缘端面被截断；预测中有明显杆身误检。','failed_visual_review'),
 'flat_custom':('near_axis_dense_flat','真实成捆平头；密集端面候选，238个预测尚未逐根独立复核。','unverified'),
 'flat_vice':('too_dense_for_this_resolution','端头过小且排列不规则；存在密集重复与不稳定响应，473不能当实际数量。','failed_visual_review'),
 'flat_umall':('near_axis_dense_flat','真实成捆端面；29个明显低于可见端面数量，大量漏检。','failed_visual_review'),
 'flat_craft':('near_axis_dense_flat','真实成捆端面；99个预测目视覆盖较完整，尚未建立独立人工真值。','promising_unverified'),
 'sharp_relaxdays':('oblique_flat_ends','商品为尖头签，但照片展示的是平头端；勿作为尖头识别成绩。','unverified'),
 'round_rods':('outside_preferred_view','明显斜拍、端面分布前后错落，不纳入垂直端拍主验收。','out_of_scope'),
 'sharp_bamboomn':('oblique_dense_sharp','真实密集尖头，斜拍探索样本，不能据包装数量判定68是否正确。','unverified'),
 'sharp_factory':('oblique_dense_sharp','真实密集尖头；171个预测包含杆身误检和重复，未通过。','failed_visual_review'),
 'sharp_product':('oblique_dense_sharp','真实成捆尖头；54个为预测值，部分遮挡，待更接近轴向拍摄的真值样本。','unverified'),
 'cotton_overhead':('vertical_boxed_cotton','真实整盒棉头，垂直俯拍；107个点目视覆盖较完整，独立数量仍待复核。','promising_unverified'),
 'cotton_round':('reflection_distractor','开盖产生棉头反射，模型也数了反射；253个不能作为盒中数量。','failed_visual_review'),
 'cotton_box':('outside_preferred_view','侧拍整盒，杆身/两端可见，不属于只看一侧棉头的目标视角。','out_of_scope'),
}
manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'));reports=[]
for row in manifest:
 if row['id'] not in NOTES:continue
 scene,note,judgement=NOTES[row['id']]
 row.update(scene=scene,screening=judgement,review_note=note,ground_truth=None,accuracy_verified=False)
 if row['id']=='sharp_relaxdays':row['category']='bamboo_flat'
 path=OUT/(row['id']+'.json')
 if path.is_file():
  result=json.loads(path.read_text(encoding='utf-8'))
  reports.append(dict(id=row['id'],scene=scene,predicted_count=result['count'],ground_truth=None,accuracy_verified=False,visual_review=judgement,note=note,source_page=row['source_page'],elapsed_seconds=result['meta']['elapsed_seconds']))
(DATA/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'report.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
selected=['flat_craft','cotton_overhead','sharp_factory','flat_umall']
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
sheet=Image.new('RGB',(1600,1120),'#eff4f1');draw=ImageDraw.Draw(sheet)
titles={'flat_craft':'成捆平头签 · 预测99','cotton_overhead':'整盒棉头俯拍 · 预测107','sharp_factory':'成捆尖头签 · 预测171','flat_umall':'另一捆平头签 · 预测29'}
sub={'flat_craft':'覆盖较完整；实际数量待独立复核','cotton_overhead':'覆盖较完整；实际数量待独立复核','sharp_factory':'明显重复与杆身误检，未通过','flat_umall':'明显大量漏检，未通过'}
for i,ident in enumerate(selected):
 x=i%2*800;y=i//2*560;im=Image.open(OUT/(ident+'.jpg')).convert('RGB')
 if ident=='flat_craft':im=im.crop((210,60,1380,1080))
 im.thumbnail((770,475));sheet.paste(im,(x+(800-im.width)//2,y+(475-im.height)//2))
 draw.text((x+18,y+480),titles[ident],font=font,fill='#19362d');draw.text((x+18,y+520),sub[ident],font=small,fill='#536b60')
sheet.save(OUT/'real_cases_review.jpg',quality=95)
lines=['# 真实成捆/成盒照片：识别效果初查','', '2026-10-17。下载成功13张真实网络商品照片，均保存来源；真实来源不等于已经核实摄影过程或训练集独立性。生成图放在单独目录，不计入本表。','', '本轮没有手工示例框，没有使用包装标称根数作为真值。使用同一 CountGD++ 预设文字和阈值0.23，在CPU运行，逐图进程退出。以下为预测值和目视错误检查，不是准确率报告。','', '| 图片 | 预测 | 初查 |','| --- | ---: | --- |']
for r in reports:lines.append(f'| {r["id"]} | {r["predicted_count"]} | {r["note"]} |')
lines+=['','## 结论','','成捆平头与垂直俯拍棉头有可继续验证的样本，但同类另一捆端面也有严重漏检；尖头样本出现杆身误检。当前不能保证固定简单场景逐张计数正确。','', '优先验收场景收敛为：相机近轴向正对齐平端头、整捆/盒完整在画面内、每个头有可辨边界、无盒盖反射、无包装遮挡。尖头单独建立类别/计数点定义，不直接套圆端面规则。','', '需对入选样本建立经过复核的逐实例点真值，再报告漏检、多检和每图完全正确率；不能把目视“覆盖完整”、两种算法数量相同或包装标称数量当作真实正确率证明。','', '本次生成的密集平头/尖头图仅作补充，路径samples/dense_cases/generated_supplement/，不替代真实测试。']
(ROOT/'doc/真实成捆成盒图片效果初查.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Reviewed',len(reports),'real image outputs.')
