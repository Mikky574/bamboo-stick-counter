"""Real laid-flat whole-stick candidates; no model predictions used for labels."""
import concurrent.futures, hashlib, io, json
from pathlib import Path
import requests
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent / 'flat_rods_eval_20261018'
SOURCES = [
 ('chopsticks_daiso_octagonal','chopsticks','https://jp.daisonet.com/cdn/shop/files/4550480418872_05_1024x.jpg?v=1777279262','https://jp.daisonet.com/products/4550480418872'),
 ('chopsticks_daiso_carved','chopsticks','https://jp.daisonet.com/cdn/shop/files/4550480418896_02_1024x.jpg?v=1777263248','https://jp.daisonet.com/products/4550480418896'),
 ('chopsticks_walmart','chopsticks','https://i5.walmartimages.ca/images/Large/021/906/6000199021906.jpg','https://www.walmart.ca/en/ip/mainstays-5-pairs-bamboo-chopsticks-natural/6000198496604'),
 ('chopsticks_striped','chopsticks','https://cb.scene7.com/is/image/Crate/BlackStrpBambooChopstickS5SSF23/%24web_pdp_main_carousel_high%24/240201142823/black-striped-bamboo-chopstick-set-of-5-pairs.jpg','https://www.crateandbarrel.com/black-striped-bamboo-chopsticks-set-of-5-pairs/s529602'),
 ('chopsticks_patterned','chopsticks','https://www.insiderasia.de/images/product_images/original_images/chopsticks-japan.jpg','https://www.insiderasia.de/essstaebchen-5-paar-blau-weiss.html'),
 ('rods_yolli','wooden_rods','https://www.yolli.com/cdn/shop/files/sam_0641_768x700.jpg?v=1751383267','https://www.yolli.com/products/mixed-wooden-dowel-rods'),
 ('rods_yolli_sizes','wooden_rods','https://www.yolli.com/cdn/shop/files/10_12_inch_wooden_dowels_mixed_sizes_768x700.jpg?v=1751383266','https://www.yolli.com/products/mixed-wooden-dowel-rods'),
 ('rods_white','wooden_rods','https://factorydirectcraft.com/pimages/20171109170454-122548/white_wood_dowel_rods.jpg','https://factorydirectcraft.com/catalog/products/1302_2110_1494-58696-white_wood_dowel_rods.html'),
 ('skewers_pexels','skewers','https://images.pexels.com/photos/8774135/pexels-photo-8774135.jpeg?auto=compress&cs=tinysrgb&w=1600','https://www.pexels.com/photo/heap-of-skewers-on-table-8774135/'),
 ('skewers_fan','skewers','https://drryor7280ntb.cloudfront.net/media/catalog/product/b/z/bz35110041_xmeil1tmvgvdydku.jpg','https://www.foodservicedirect.com/poly-king-bamboo-skewer--9-inch----1600-per-case-266164.html'),
 ('rods_creatology','wooden_rods','https://imgs.michaels.com/MAM/assets/1/5E3C12034D34434F8A9BAAFDDF0F8E1B/img/75D3CFFEE21F4279907289EA512397F2/10031021_21.jpg?fit=inside%7C540%3A540','https://www.michaels.com/product/12-wood-dowels-by-creatology-10031021'),
 ('rods_macrame','wooden_rods','https://welcomeyarn.pt/cdn/shop/files/dowel-6mm.jpg?v=1697039587','https://welcomeyarn.pt/products/wooden-dowel-for-macrame-6mm'),
 ('rods_walmart_six','wooden_rods','https://i5.walmartimages.com/asr/7da88a71-f008-4e6c-a08e-674c271bda20_1.5ddd0ef3d766c97a1de29316f6aa7baf.jpeg?odnBg=ffffff&odnHeight=1000&odnWidth=1000','https://www.walmart.com/ip/501205821'),
 ('rods_woodpeckers','wooden_rods','https://woodpeckerscrafts.com/cdn/shop/files/woodpeckers-crafts-316-x-6-wooden-dowel__10549.1677706173.1280.1280.jpg?v=1738085079&width=1280','https://woodpeckerscrafts.com/products/3-16-x-6-wooden-dowel'),
]

def fetch(item):
 ident, category, url, page = item
 row = dict(id=ident,category=category,source_page=page,image_url=url,
            origin='real_web_product_or_photographer_photo_candidate',downloaded_on='2026-10-18',
            count_unit='individual_whole_stick_not_pair',ground_truth=None,accuracy_verified=False,
            screening='pending_visual_review',license_status='not_verified_for_training_or_redistribution')
 try:
  s=requests.Session();s.trust_env=False
  s.proxies.update(http='http://127.0.0.1:10090',https='http://127.0.0.1:10090')
  r=s.get(url,timeout=(15,40));r.raise_for_status();im=Image.open(io.BytesIO(r.content));im.load()
  ext=(im.format or 'jpg').lower();file=ROOT/'originals'/(ident+'.'+ext)
  if file.exists() and file.read_bytes()!=r.content:raise RuntimeError('refusing to overwrite different bytes')
  file.write_bytes(r.content)
  row.update(file=file.relative_to(ROOT).as_posix(),width=im.width,height=im.height,
             sha256=hashlib.sha256(r.content).hexdigest(),size_bytes=len(r.content))
 except Exception as e:row.update(error=str(e),screening='download_failed')
 print(ident,row.get('error','downloaded'),flush=True);return row

def main():
 (ROOT/'originals').mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(fetch,SOURCES))
 (ROOT/'manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
 good=[r for r in rows if 'file' in r];sheet=Image.new('RGB',(1600,((len(good)+3)//4)*350),'#eeeeee');d=ImageDraw.Draw(sheet)
 for i,row in enumerate(good):
  im=Image.open(ROOT/row['file']).convert('RGB');im.thumbnail((390,305));x=i%4*400;y=i//4*350
  sheet.paste(im,(x+(400-im.width)//2,y));d.text((x+8,y+315),row['id'],fill='black')
 sheet.save(ROOT/'candidate_preview.jpg',quality=92)
 print('Downloaded',len(good),'/',len(rows))

if __name__=='__main__':main()
