"""Collect real bundled/boxed counting candidates, with provenance and screening."""
import argparse,concurrent.futures,hashlib,io,json
from pathlib import Path
import requests
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parent/'dense_cases'
SOURCES=[
 ('flat_factory','bamboo_flat','https://image.made-in-china.com/2f0j00wJtlCyDrrGzn/Natural-Color-Sharp-End-Round-Disposable-Bamboo-Stick-Bamboo-Skewer.webp','https://m.made-in-china.com/product/Natural-Color-Sharp-End-Round-Disposable-Bamboo-Stick-Bamboo-Skewer-1977437892.html'),
 ('flat_custom','bamboo_flat','https://image.made-in-china.com/2f0j00ZJzhumYyCUrG/20cm-30cm-40cm-Customzied-Sizes-Skewer-Bamboo-Sticks.webp','https://xizhubamboo.en.made-in-china.com/product/rOvfDkKGhBUJ/China-20cm-30cm-40cm-Customzied-Sizes-Skewer-Bamboo-Sticks.html'),
 ('flat_vice','bamboo_flat','https://images.nexusapp.co/assets/fd/eb/ff/11980696.jpg','https://nl.vicedeal.com/products/90-stks-15-cm-bamboe-spiesjes-sticks-houten-barbecue-spiesjes-natuurlijke-wood-sticks-barbecue-accessoires-koken-tool'),
 ('flat_umall','bamboo_flat','https://www.umall.com.au/cdn/shop/files/3_4412b733-19f7-45d5-99ca-0fcc6a89939d.jpg?v=1737684536','https://www.umall.com.au/products/qingchun-bamboo-skewers-for-bbq-30cm-pack-of-65-x3'),
 ('flat_smallbundle','bamboo_flat','https://microless.com/cdn/products/2caab091a8241136906dfdcfb8f2d441-hi.jpg','https://uae.microless.com/product/natural-bamboo-straws-8mm-10-pieces-per-packet-eco-friendly-and-reusable-straws-perfect-for-smoothies-beverages-and-sustainable-living-perfect-for-parties-and-everyday-use/'),
 ('flat_craft','bamboo_flat','https://www.craftworkz.com.au/cdn/shop/products/PhotoRoom_20230131_150358_1024x1024%402x.jpg?v=1675138476','https://www.craftworkz.com.au/products/wooden-dowel'),
 ('flat_long','bamboo_flat','https://ae01.alicdn.com/kf/Sf8a56f84925f4ae1bbcaae26b8ee01d2a/60cm-x-6mm-50-100Pcs-BBQ-Bamboo-Skewers-Super-Long-Tornado-Potato-Wood-Skewer-Wooden-Spiral.jpg','https://arvar.hu/termek/60-cm-x-6-mm-50-100db-bbq-bambusz-nyars-szuper-hosszu-tornado-burgonya-fa-spiral-tu-bar-etteremellatas/'),
 ('sharp_relaxdays','bamboo_sharp','https://cdn.idealo.com/folder/Product/202168/8/202168848/s1_produktbild_max_7/relaxdays-stockbrot-spiesse-im-200er-set-10034324.jpg','https://www.idealo.de/preisvergleich/OffersOfProduct/202168848_-stockbrot-spiesse-im-200er-set-10034324-relaxdays.html'),
 ('round_rods','bamboo_flat','https://image.made-in-china.com/2f0j00QYUcmLRKqipz/Circular-Bamboo-Stick-Round-Rod-Kindergarten-Toys-Circular-Rod.jpg','https://bambooitems.en.made-in-china.com/product/sQuYUzbGmTpr/China-Circular-Bamboo-Stick-Round-Rod-Kindergarten-Toys-Circular-Rod.html'),
 ('cotton_top','cotton','https://roundmuseum.com/sites/dev-round-museum.dd/files/node_gallery/cotton%20swab%20box-IMG_5413.jpg','https://roundmuseum.com/file/3922'),
]
def main():
 p=argparse.ArgumentParser();p.add_argument('--proxy');a=p.parse_args();ROOT.mkdir(exist_ok=True)
 def run(item):
  ident,kind,url,page=item
  try:
   s=requests.Session();s.trust_env=False
   if a.proxy:s.proxies.update(dict(http=a.proxy,https=a.proxy))
   r=s.get(url,timeout=(15,40));r.raise_for_status();im=Image.open(io.BytesIO(r.content)).convert('RGB')
   file=ident+'.jpg';im.save(ROOT/file,quality=96)
   print(ident,im.size,flush=True)
   return dict(id=ident,category=kind,file=file,origin='real_web_photo',source_page=page,image_url=url,width=im.width,height=im.height,sha256=hashlib.sha256((ROOT/file).read_bytes()).hexdigest(),license='Not verified for redistribution/training; local evaluation candidate',screening='pending',ground_truth=None,downloaded_on='2026-10-17')
  except Exception as e:print(ident,str(e),flush=True);return dict(id=ident,error=str(e),source_page=page)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,SOURCES))
 (ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 good=[r for r in rows if 'file' in r];sheet=Image.new('RGB',(1200,((len(good)+3)//4)*280),'#eee');d=ImageDraw.Draw(sheet)
 for i,row in enumerate(good):
  im=Image.open(ROOT/row['file']);im.thumbnail((285,235));x=i%4*300;y=i//4*280
  sheet.paste(im,(x+(300-im.width)//2,y));d.text((x+8,y+245),row['id'],fill='black')
 if good:sheet.save(ROOT/'candidate_preview.jpg')
if __name__=='__main__':main()
