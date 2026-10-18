"""Download a small, explicitly sourced local counting evaluation collection."""
import concurrent.futures, hashlib, io, json, shutil, argparse
from pathlib import Path
import requests
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'counting_eval'
SOURCES=[
 ('bamboo_hand','bamboo','https://i.etsystatic.com/23207640/r/il/be2e69/6954342291/il_794xN.6954342291_ficj.jpg','https://www.etsy.com/ca/listing/4314005976/personalized-bamboo-sauna-whisk-natural'),
 ('bamboo_bundle','bamboo','https://i.etsystatic.com/23207640/r/il/3cdcaa/6954342297/il_794xN.6954342297_mkar.jpg','https://www.etsy.com/ca/listing/4314005976/personalized-bamboo-sauna-whisk-natural'),
 ('rice_gray','rice','https://raw.githubusercontent.com/hkedia321/rice-quality-analysis/HEAD/rice.png','https://github.com/hkedia321/rice-quality-analysis'),
]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proxy');a=p.parse_args();DEST.mkdir(exist_ok=True)
 def run(item):
  ident,kind,url,source=item;s=requests.Session();s.trust_env=False
  if a.proxy:s.proxies.update(dict(http=a.proxy,https=a.proxy))
  try:
   r=s.get(url,timeout=(15,40));r.raise_for_status();im=Image.open(io.BytesIO(r.content));im.load()
   filename=ident+'.png';im.convert('RGB').save(DEST/filename)
   print(ident,im.size,flush=True)
   return dict(id=ident,category=kind,file=filename,source_page=source,image_url=url,original_size=list(im.size),ground_truth=None,exemplars=[],license='Image reuse rights not verified; local evaluation candidate',sha256=hashlib.sha256((DEST/filename).read_bytes()).hexdigest())
  except Exception as e:print(ident,str(e),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(filter(None,pool.map(run,SOURCES)))
 for ident,kind,src,source in [
  ('bamboo_original','bamboo',ROOT/'user_bamboo.png','Existing repository sample; not a new independent photo'),
  ('cotton_dense','cotton',ROOT/'web_validation/cotton_dense_02.jpg','https://www.smokecartel.com/products/crud-bud-cotton-swabs-tips-110-pack'),
 ]:
  if src.exists():
   filename=ident+src.suffix;shutil.copy2(src,DEST/filename)
   rows.append(dict(id=ident,category=kind,file=filename,source_page=source,ground_truth=None,exemplars=[],license='See source page; local evaluation candidate'))
 (DEST/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 sheet=Image.new('RGB',(1200,((len(rows)+3)//4)*300),'#eee');draw=ImageDraw.Draw(sheet)
 for i,row in enumerate(rows):
  im=Image.open(DEST/row['file']).convert('RGB');im.thumbnail((285,250));x=i%4*300;y=i//4*300
  sheet.paste(im,(x+(300-im.width)//2,y));draw.text((x+8,y+260),row['id'],fill='black')
 if rows:sheet.save(DEST/'preview.jpg')
if __name__=='__main__':main()
