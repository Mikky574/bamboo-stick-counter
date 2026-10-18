"""Collect real application-photo candidates with provenance, without labels."""
import concurrent.futures,hashlib,io,json
from pathlib import Path
import requests
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent/'application_eval_20261018'
SOURCES=[
('mung_paper','mung','https://images.pexels.com/photos/7420814/pexels-photo-7420814.jpeg?auto=compress&cs=tinysrgb&w=1600','https://www.pexels.com/photo/mung-beans-7420814/'),
('mung_yellow','mung','https://images.pexels.com/photos/6805060/pexels-photo-6805060.jpeg?auto=compress&cs=tinysrgb&w=1600','https://www.pexels.com/photo/6805060/'),
('mung_spoon','mung','https://images.pexels.com/photos/7420815/pexels-photo-7420815.jpeg?auto=compress&cs=tinysrgb&w=1600','https://www.pexels.com/photo/7420815/'),
('mung_bowl','mung','https://images.pexels.com/photos/7420818/pexels-photo-7420818.jpeg?auto=compress&cs=tinysrgb&w=1600','https://www.pexels.com/photo/7420818/'),
('mung_blue_ruler','mung','https://rasteniy10.ru/d/vigna_mash_baoshi_355-17285.jpg','https://rasteniy10.ru/internet-magazin/product/vigna-mash-zelenyye-semena-vigna-radiata-l-r-wilczek-bobovyye-fabaceae-lindl'),
('mung_white_ruler','mung','https://gardencells.com/files/original/products/31.jpg','https://gardencells.com/ar/beans/mungbeans'),
('mung_white','mung','https://kontaktonatura.com/cdn/shop/files/Fagiolo-1.jpg?v=1716479281&width=1214','https://kontaktonatura.com/en/products/organic-mung-bean-seed-100gr'),
('mung_ferwer','mung','https://www.ferwer.cz/img/blog/mungo-fazole-zelena-zrnka.webp','https://www.ferwer.cz/blog/mungo'),
('mung_dark_scatter','mung','https://www.biowaynutrition.com/uploads/Some-green-beans-were-scattered-on-the-table.png','https://www.biowaynutrition.com/news/compare-mung-bean-peptide-with-other-plant-peptides-for-immunity/'),
('mung_dark','mung','https://cdn.reishunger.com/mungobohnen2.jpg?quality=85&width=1600','https://www.reishunger.de/wissen/article/374/bohnensorten-im-uberblick'),
('mung_mostaza','mung','https://negromostaza.com/cdn/shop/products/frijol-mungo-frijoles-legumbres-despensa-basica-Negro-Mostaza-2_grande.jpg?v=1756742544','https://negromostaza.com/products/frijol-mungo'),
('coins_white','coins','https://codewords.recurse.com/images/six/image-processing-101/coins/coins-orig.jpg','https://codewords.recurse.com/issues/six/image-processing-101'),
]

def fetch(item):
 ident,category,url,page=item
 row=dict(id=ident,category=category,source_page=page,image_url=url,origin='web_photo_candidate',downloaded_on='2026-10-18',ground_truth=None,accuracy_verified=False,screening='pending_visual_review',license_status='source_recorded_not_verified_for_training')
 try:
  s=requests.Session();s.trust_env=False;s.proxies.update(http='http://127.0.0.1:10090',https='http://127.0.0.1:10090')
  r=s.get(url,timeout=(12,35));r.raise_for_status();im=Image.open(io.BytesIO(r.content));im.load()
  ext=(im.format or 'jpg').lower();file=ROOT/'originals'/(ident+'.'+ext);file.write_bytes(r.content)
  row.update(file=file.relative_to(ROOT).as_posix(),width=im.width,height=im.height,sha256=hashlib.sha256(r.content).hexdigest())
 except Exception as e:row.update(error=str(e),screening='download_failed')
 print(ident,row.get('error','downloaded'),flush=True);return row

def main():
 (ROOT/'originals').mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,SOURCES))
 seen={}
 for r in rows:
  if 'sha256' not in r:continue
  if r['sha256'] in seen:r.update(screening='duplicate',duplicate_of=seen[r['sha256']])
  else:seen[r['sha256']]=r['id']
 (ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 good=[r for r in rows if 'file' in r];sheet=Image.new('RGB',(1600,((len(good)+3)//4)*350),'#eeeeee');d=ImageDraw.Draw(sheet)
 for i,r in enumerate(good):
  im=Image.open(ROOT/r['file']).convert('RGB');im.thumbnail((390,305));x=i%4*400;y=i//4*350;sheet.paste(im,(x+(400-im.width)//2,y));d.text((x+8,y+315),r['id'],fill='black')
 sheet.save(ROOT/'candidate_preview.jpg',quality=92)
 print('Downloaded',len(good),'/',len(rows))
if __name__=='__main__':main()
