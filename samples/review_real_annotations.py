"""Apply explicit visual corrections and separate full-count from partial labels."""
import json,shutil
from pathlib import Path
import cv2,numpy as np
from annotate_real_dataset import ROOT,OUT,save,render

def main():
 full={'flat_sudrc':50,'flat_craft':99,'flat_custom':238,'cotton_overhead':107,'cotton_muji':200}
 corrections={
  'cotton_niva':([106],'透明盖中央的结构件误检，删除原候选106；边缘盖下实例不用于整盒总数。'),
  'cotton_family':([10,31,59,67,148],'放大近邻候选，删除同一棉头上的重复点10/31/59/67/148；保留21/49/58/138。上缘棉头重叠，整盒真值保持为空。'),
  'cotton_aijie':([166,212,248,269,419,411,497,493,504],'放大检查后删除9个重复候选；透明盖沿和中心结构附近保留忽略区，整盒真值为空。'),
 }
 for name in full.keys()|corrections.keys():
  p=OUT/(name+'.points.json');row=json.loads(p.read_text(encoding='utf8'))
  if row.get('visual_corrections_applied'):continue
  if name in full:
   for point in row['points']:point['review']='accepted'
   row.update(status='reviewed_single_pass',ground_truth=full[name],visible_count=full[name],count_scope='visible_near_side_heads',count_review='single_pass_not_independent')
   if name=='cotton_muji':row['review_notes']=['分四块放大逐点检查200个候选；可见棉头均有对应点，盒盖文字不另计。非独立盲标。']
  else:
   ids,note=corrections[name]
   original=list(row['points']);row['removed_proposals']=[dict(original_index=i,**original[i-1],reason='visual_duplicate_or_non_target') for i in ids]
   row['points']=[p for i,p in enumerate(original,1) if i not in ids]
   for point in row['points']:point['review']='accepted'
   row.update(status='reviewed_partial_visibility',ground_truth=None,visible_count=None,count_scope='annotated_visible_candidates_not_package_total',count_review='not_eligible_for_exact_count_test')
   row['review_notes']=[note,'全图和四分区目视检查后保留可见点位；不能确认完整数量，不进入整图精确计数验收。']
   if name=='cotton_niva':
    polygons=[([[110,150],[175,100],[260,82],[350,110],[420,175],[400,210],[330,145],[240,120],[160,150]],'透明盒盖上缘遮挡和低对比度区域'),([[231,323],[258,323],[263,354],[234,359]],'透明盖中央结构件，不计作棉头')]
   elif name=='cotton_family':
    polygons=[([[230,45],[570,35],[745,143],[810,235],[745,260],[660,180],[510,115],[300,140]],'上缘棉头前后重叠、同一头上下响应易重复，暂不作完整计数真值')]
   else:
    polygons=[([[265,450],[440,390],[640,370],[850,397],[1040,485],[1070,560],[970,540],[820,465],[600,440],[410,475],[295,520]],'透明盖上缘、盖下头边界不清'),([[620,810],[740,800],[775,910],[650,935]],'透明盒盖中心结构与邻近棉头重叠')]
   row['ignore_regions']=[dict(polygon=poly,reason=reason) for poly,reason in polygons]
   for point in row['points']:
    if any(cv2.pointPolygonTest(np.array(g['polygon'],np.float32),(point['x'],point['y']),False)>=0 for g in row['ignore_regions']):point['review']='uncertain_occlusion'
  row.update(visual_corrections_applied=True,reviewer='Codex_visual_review',independently_verified=False,reviewed_on='2026-10-18',training_eligible=False)
  save(row)

 # Circle proposals on this photo split wood grain into duplicates and omit heads.
 # Preserve them as rejected proposals, never export an empty negative annotation.
 name='flat_umall';p=OUT/(name+'.points.json');row=json.loads(p.read_text(encoding='utf8'))
 if row['status']!='excluded_unreliable_annotation':
  dest=ROOT/'annotation_proposals'/'rejected_flat_umall_circles.json'
  dest.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
 row.update(points=[],annotation_count=0,status='excluded_unreliable_annotation',ground_truth=None,independently_verified=False,training_eligible=False,reviewed_on='2026-10-18',review_notes=['分区放大后端面木纹和边界模糊；自动圆候选存在大量重复、漏点，不能当作逐实例标注。此图退出本批标注集，未导出空标签负样本。'])
 p.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
 (OUT/(name+'.labelme.json')).unlink(missing_ok=True)
 render(row)
 print('Reviewed 9 photos: 5 full visible-count labels, 3 partial labels, 1 excluded.')
if __name__=='__main__':main()
