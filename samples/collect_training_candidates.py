"""Download source-traceable candidates; never assign predicted counts as truth."""
import concurrent.futures, hashlib, io, json
from pathlib import Path
import requests
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent/'real_dataset_v1'
SOURCES=[
('flat_matton','bamboo_flat','https://www.mattonbutiken.se/gfx/pages/31/hui-BOK-6054_kti9f185.png','https://www.mattonbutiken.se/produkt/bokpinne-rundstav-1000x3mm'),
('flat_homedepot','bamboo_flat','https://images.thdstatic.com/productImages/65581340-ce07-4e90-ae0f-3e1690c1400a/svn/angel-sar-lattice-accessories-dt11lt75-64_1000.jpg','https://www.homedepot.com/p/332478281'),
('flat_sudrc','bamboo_flat','https://www.sud-rc.fr/5759-large_default/baguette-ronde-10mm-en-ramin-b252510.jpg','https://www.sud-rc.fr/baguette-ronde/2358-526-baguette-ronde-10mm-en-ramin-b252510.html'),
('flat_ebay','bamboo_flat','https://i.ebayimg.com/images/g/JEkAAOSwEMhkJAGc/s-l500.jpg','https://www.ebay.co.uk/itm/134510707421'),
('flat_caldowel','bamboo_flat','https://woodproducts.caldowel.com/images/products/detail/3-4-Dowel-Rods-x-24-Birch.jpg','https://woodproducts.caldowel.com/3-4-Dowel-Rods-x-24-Birch.aspx'),
('sharp_5mm','bamboo_sharp','https://bamboomn.com/cdn/shop/products/bamboo-sharp-point-round-food-skewers-tips-sksp50_1200x.jpg?v=1744215492','https://bamboomn.com/products/sharp-point-round-skewers-5-millimeters'),
('sharp_fucheng','bamboo_sharp','https://image.made-in-china.com/2f0j00kfAcQzpJAFge/10-Inch-Marshmallow-Bamboo-Sticks-Hot-Dog-Skewers-with-Sharp-Point.jpg','https://bambooskewer.en.made-in-china.com/product/NmEpjXSHHYRi/China-10-Inch-Marshmallow-Bamboo-Sticks-Hot-Dog-Skewers-with-Sharp-Point.html'),
('sharp_kite','bamboo_sharp','https://image.made-in-china.com/2f0j00puFcqzNdbtrg/One-Kite-Making-Bamboo-Stick-Name-for-Bamboo-Stick.webp','https://bambooskewer.en.made-in-china.com/product/yJgYnAkOurUT/China-One-Kite-Making-Bamboo-Stick-Name-for-Bamboo-Stick.html'),
('sharp_bakeyy','bamboo_sharp','https://bakeyy.com/cdn/shop/files/Bamboo_Skewers_8.jpg?v=1773075427&width=1214','https://bakeyy.com/products/bamboo-skewers-12-inch-pack-of-45-wood-skewer-wooden-stick-5mm'),
('sharp_baoyizi','bamboo_sharp','https://image.made-in-china.com/365f3j00VtQcFZTEBarE/Material-Strong-BBQ-Sticks-Bamboo-Skewers-Round-Roasting.webp','https://bambooitems.en.made-in-china.com/product/BQurEPRxzUYp/China-Material-Strong-BBQ-Sticks-Bamboo-Skewers-Round-Roasting.html'),
('cotton_jomi','cotton','https://cdn1.concung.com/2024/01/50885-108304-large_mobile/tam-bong-khang-khuan-jomi-tre-em-160-que.webp','https://concung.com/dung-cu-ve-sinh/tam-bong-khang-khuan-jomi-tre-em-160-que-71392.html'),
('cotton_muji','cotton','https://i.ebayimg.com/images/g/95YAAOSwnpZm8dZr/s-l1600.jpg','https://www.ebay.com/itm/116336765306'),
('cotton_daiso','cotton','https://i.ebayimg.com/images/g/26QAAOSw~y1nDMH6/s-l1200.png','https://www.ebay.com/itm/365216669043'),
('cotton_niva','cotton','https://nhathuocdaicatloc.vn/upload/images/Phuong%20dang/IMG_5236_209fe00078.webp','https://nhathuocdaicatloc.vn/tam-bong-niva-ah2-cotton-buds-hu-xoay-1-dau-tron-1-dau-xoan-ve-sinh-tai-mui-ve-sinh-ca-nhan-vet-thuong-200-que'),
('cotton_family','cotton','https://media.karousell.com/media/photos/products/2025/10/11/cotton_buds_500_pieces_per_pac_1760163153_206bbe6b_progressive','https://www.carousell.ph/p/cotton-buds-500-pieces-per-pack-1397640872/'),
('cotton_aijie','cotton','https://image.made-in-china.com/2f0j00YBfkIsKDfncq/Factory-Wholesale-Q-Tips-500-PCS-Double-Head-Paper-Stick-Cotton-Swabs-in-Plastic-Box-for-Daily-Use.jpg','https://aijieswab.en.made-in-china.com/product/pQYRTtclvdkb/China-Factory-Wholesale-Q-Tips-500-PCS-Double-Head-Paper-Stick-Cotton-Swabs-in-Plastic-Box-for-Daily-Use.html'),
('cotton_daily','cotton','https://stat.ameba.jp/user_images/20260214/13/ayapon0007/4c/20/j/o1080144015750927439.jpg','https://ameblo.jp/ayapon0007/entry-12956690339.html'),
('cotton_pink','cotton','https://pcdn.goldapple.ru/p/p/13380300016/web/696d67416464318ddc3e8bb1ec99b.jpg','https://goldapple.kz/hozjajstvennye-tovary/tualetnye-prinadlezhnosti/vatnye-diski-i-palochki'),
]

def download(item):
 ident,category,url,page=item
 row=dict(id=ident,category=category,image_url=url,source_page=page,origin='web_product_or_daily_photo',ground_truth=None,annotation_status='unlabeled',screening='pending_visual_review',license_status='unverified',split=None)
 try:
  s=requests.Session();s.trust_env=False;s.proxies.update(http='http://127.0.0.1:10090',https='http://127.0.0.1:10090')
  r=s.get(url,timeout=(12,35));r.raise_for_status()
  im=Image.open(io.BytesIO(r.content));im.load()
  path=ROOT/'originals'/(ident+'.'+(im.format or 'jpg').lower());path.write_bytes(r.content)
  row.update(file=str(path.relative_to(ROOT)),width=im.width,height=im.height,sha256=hashlib.sha256(r.content).hexdigest())
 except Exception as e:row['error']=str(e)
 print(ident,row.get('error','downloaded'),flush=True)
 return row

def main():
 (ROOT/'originals').mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(download,SOURCES))
 seen={}
 for row in rows:
  if 'sha256' in row:
   if row['sha256'] in seen:row['duplicate_of']=seen[row['sha256']]
   else:seen[row['sha256']]=row['id']
 (ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
 good=[r for r in rows if 'file' in r]
 sheet=Image.new('RGB',(1500,((len(good)+4)//5)*290),'#eeeeee');d=ImageDraw.Draw(sheet)
 for i,row in enumerate(good):
  im=Image.open(ROOT/row['file']).convert('RGB');im.thumbnail((290,250));x=i%5*300;y=i//5*290
  sheet.paste(im,(x+(300-im.width)//2,y));d.text((x+7,y+257),row['id'],fill='black')
 sheet.save(ROOT/'contact_sheet.jpg',quality=90)
 print('Downloaded',len(good),'of',len(rows))
if __name__=='__main__':main()
