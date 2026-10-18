"""Local point annotations, with explicit proposal/review provenance."""
import json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent/'real_dataset_v1'
OUT=ROOT/'annotations_v1'

MANUAL_SUDRC=[
[(357,215),(420,213),(479,214)],
[(264,267),(325,263),(389,264),(449,263),(507,265),(565,265)],
[(170,332),(226,315),(296,317),(357,315),(417,314),(477,317),(535,317),(596,320)],
[(211,372),(270,368),(329,365),(387,365),(445,372),(506,367),(565,366),(624,368)],
[(175,416),(234,426),(292,421),(352,417),(412,420),(477,418),(537,419),(597,417)],
[(198,472),(255,481),(315,476),(378,473),(442,469),(502,473),(568,471)],
[(223,522),(290,528),(349,521),(409,524),(474,520),(533,522)],
[(258,573),(319,575),(383,575),(442,573)]]

def source(ident):
 rows=json.loads((ROOT/'annotation_queue.json').read_text(encoding='utf8'))
 return next(r for r in rows if r['id']==ident)

def propose(ident,image):
 im=np.array(image);scale=min(1,1000/max(image.size));small=cv2.resize(im,None,fx=scale,fy=scale)
 gray=cv2.cvtColor(small,cv2.COLOR_RGB2GRAY)
 gray=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(gray)
 gray=cv2.GaussianBlur(gray,(5,5),1)
 # Visual estimates on the scaled original; all output remains unverified until review.
 cfg={
 'cotton_muji':(15,6,13,12,(375,460,175,155)),
 'cotton_niva':(19,7,17,12,(265,335,220,247)),
 'cotton_family':(23,9,21,13,(418,373,380,350)),
 'cotton_aijie':(20,7,17,12,(460,570,353,314)),
 'flat_umall':(25,10,23,14,(510,480,420,480)),
 }
 dist,rmin,rmax,p2,ellipse=cfg[ident]
 circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,minDist=dist,param1=80,param2=p2,minRadius=rmin,maxRadius=rmax)
 pts=[];cx,cy,rx,ry=ellipse
 for x,y,r in ([] if circles is None else circles[0]):
  if ((x-cx)/rx)**2+((y-cy)/ry)**2<=1:
   pts.append(dict(x=round(float(x/scale),2),y=round(float(y/scale),2),origin='opencv_circle_proposal',review='pending'))
 return pts

def render(row):
 im=Image.open(ROOT/row['file']).convert('RGB');draw=ImageDraw.Draw(im)
 size=max(10,round(im.width/100));font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
 for i,p in enumerate(row['points'],1):
  x,y=p['x'],p['y'];r=max(2,im.width/250)
  color='#00e5ff' if p.get('review')=='accepted' else '#ff9500'
  draw.ellipse((x-r,y-r,x+r,y+r),fill=color,outline='black',width=1)
  draw.text((x+r+1,y-size/2),str(i),fill=color,font=font,stroke_width=1,stroke_fill='black')
 for region in row.get('ignore_regions',[]):
  draw.polygon([tuple(p) for p in region['polygon']],outline='#ff00ff',width=max(1,round(im.width/500)))
 im.save(OUT/(row['id']+'.review.jpg'),quality=95)

def save(row):
 row['points'].sort(key=lambda p:(round(p['y']/12),p['x']))
 row['annotation_count']=len(row['points'])
 (OUT/(row['id']+'.points.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
 # Pending points explicitly flagged; no fabricated bounding boxes or negative labels.
 shapes=[dict(label=row['category']+'_head',points=[[p['x'],p['y']]],group_id=None,shape_type='point',flags={'pending_review':p.get('review')!='accepted'},description=p['origin']) for p in row['points']]
 shapes.extend(dict(label='__ignore__',points=g['polygon'],group_id=None,shape_type='polygon',flags={'ignore':True},description=g['reason']) for g in row.get('ignore_regions',[]))
 label=dict(version='5.5.0',flags={'draft':row['status']!='reviewed_single_pass'},shapes=shapes,imagePath='../'+row['file'].replace('\\','/'),imageData=None,imageHeight=row['height'],imageWidth=row['width'])
 (OUT/(row['id']+'.labelme.json')).write_text(json.dumps(label,ensure_ascii=False,indent=2),encoding='utf8')
 render(row)

def main():
 OUT.mkdir(exist_ok=True)
 for item in json.loads((ROOT/'annotation_queue.json').read_text(encoding='utf8')):
  ident=item['id'];path=OUT/(ident+'.points.json')
  if path.exists():continue
  im=Image.open(ROOT/item['file']);row=dict(item,width=im.width,height=im.height,status='draft_needs_instance_review',ground_truth=None,independently_verified=False,created_on='2026-10-18',ignore_regions=[],points=[],review_notes=[])
  if ident=='flat_sudrc':
   row['points']=[dict(x=x,y=y,origin='manual_visual_point',review='accepted') for group in MANUAL_SUDRC for x,y in group]
   row.update(status='reviewed_single_pass',visible_count=50,ground_truth=50)
   row['review_notes']=['逐排人工点标：3+6+8+8+8+7+6+4=50；仅计可见平端面。未做独立第二人复核。']
  elif ident in ['flat_craft','cotton_overhead','flat_custom']:
   prior=json.loads((ROOT.parent.parent/'outputs/dense_cases'/(ident+'.json')).read_text(encoding='utf8'))
   row['points']=[dict(x=p['x'],y=p['y'],origin='countgdpp_proposal',review='pending') for p in prior['points']]
  else:row['points']=propose(ident,im.convert('RGB'))
  save(row);print(ident,len(row['points']),row['status'])
if __name__=='__main__':main()
