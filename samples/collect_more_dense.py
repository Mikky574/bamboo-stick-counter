"""Supplement dense real pointed skewers and boxed cotton heads."""
import concurrent.futures,hashlib,io,json
from pathlib import Path
import requests
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent/'dense_cases'
SOURCES=[
 ('sharp_bamboomn','bamboo_sharp','https://bamboomn.com/cdn/shop/files/4mm-kebab-grilling-sharp-point-round-skewers-sksp40xx-e_1200x.jpg?v=1746029938','https://bamboomn.com/products/sharp-point-round-skewers-4-millimeters'),
 ('sharp_factory','bamboo_sharp','https://image.made-in-china.com/2f0j00EgskHCSJnpcr/Eco-Friendly-Natural-BBQ-Stick-Fruit-Stick-Bamboo-Skewer.jpg','https://liminwood.en.made-in-china.com/product/lmOpuIYxJRkw/China-Eco-Friendly-Natural-BBQ-Stick-Fruit-Stick-Bamboo-Skewer.html'),
 ('sharp_product','bamboo_sharp','https://conwaykitchen.com/cdn/shop/products/s-l1600_7.jpg?v=1747951241','https://conwaykitchen.com/products/harold-bamboo-skewer-6-inch'),
 ('cotton_round','cotton','https://2.wlimg.com/product_images/bc-full/2021/10/8256304/cotton-ear-buds-1634634677-5844499.jpeg','https://storage.googleapis.com/dpphuyounrafme/cotton-ear-buds-in-india.html'),
 ('cotton_box','cotton','https://assets.cdn.jula.com/v2/preset%3Ajpgoptimized/w%3A1920/56992?lastmodified=20250516102445','https://www.jula.no/catalog/hjem-og-husholdning/personlig-pleie-og-helse/hudstell/ansiktspleie/bomullspinner-008736/'),
 ('cotton_overhead','cotton','https://item-shopping.c.yimg.jp/i/n/kawauchi_2jpp52055_4_d_20230120115616','https://store.shopping.yahoo.co.jp/kawauchi/2jpp52055.html'),
]
def run(item):
 ident,kind,url,page=item
 try:
  s=requests.Session();s.trust_env=False;s.proxies.update(dict(http='http://127.0.0.1:10090',https='http://127.0.0.1:10090'))
  r=s.get(url,timeout=(15,40));r.raise_for_status();im=Image.open(io.BytesIO(r.content)).convert('RGB');name=ident+'.jpg';im.save(ROOT/name,quality=96)
  print(ident,im.size,flush=True)
  return dict(id=ident,category=kind,file=name,origin='real_web_photo',source_page=page,image_url=url,width=im.width,height=im.height,sha256=hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),license='Not verified for redistribution/training; local evaluation candidate',screening='pending',ground_truth=None,downloaded_on='2026-10-17')
 except Exception as e:print(ident,str(e),flush=True);return dict(id=ident,error=str(e),source_page=page)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:added=list(pool.map(run,SOURCES))
rows=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
ids={r['id'] for r in added};rows=[r for r in rows if r['id'] not in ids]+added
(ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
good=[r for r in rows if 'file' in r];sheet=Image.new('RGB',(1200,((len(good)+3)//4)*280),'#eee');d=ImageDraw.Draw(sheet)
for i,row in enumerate(good):
 im=Image.open(ROOT/row['file']);im.thumbnail((285,235));x=i%4*300;y=i//4*280;sheet.paste(im,(x+(300-im.width)//2,y));d.text((x+8,y+245),row['id'],fill='black')
sheet.save(ROOT/'candidate_preview.jpg')
