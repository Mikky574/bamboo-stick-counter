"""Download and verify the public GeCo2 checkpoint; never uploads images."""
import hashlib
from pathlib import Path
import requests
ROOT=Path(__file__).parent
TARGET=ROOT/'models/CNTQG_multitrain_ca44.pth'
EXPECTED='7841dcfd2d0120e65b40a5c6dd1dfcb014df350a34f320d2fd000dc5f4c005e7'
URL_PATH='/datasets/jerpelhan/geco2-assets/resolve/main/weights/CNTQG_multitrain_ca44.pth'
def checksum(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 TARGET.parent.mkdir(exist_ok=True)
 if TARGET.exists() and checksum(TARGET)==EXPECTED:
  print('GeCo2 weights already present; SHA-256 verified.');return
 temp=TARGET.with_suffix('.partial');errors=[]
 urls=[
  'https://github.com/Mikky574/bamboo-stick-counter/releases/download/v0.1.0/CNTQG_multitrain_ca44.pth',
  'https://huggingface.co'+URL_PATH,
  'https://hf-mirror.com'+URL_PATH,
 ]
 for url in urls:
  try:
   session=requests.Session();session.trust_env=False
   with session.get(url,stream=True,timeout=(15,60)) as response:
    response.raise_for_status();downloaded=0
    with temp.open('wb') as output:
     for chunk in response.iter_content(1024*1024):
      output.write(chunk);downloaded+=len(chunk)
      if downloaded%(50*1024*1024)<1024*1024:print(f'{downloaded//(1024*1024)} MB',flush=True)
   if checksum(temp)!=EXPECTED:raise ValueError('Checkpoint checksum mismatch; existing weights preserved.')
   temp.replace(TARGET);print('Download complete; SHA-256 verified.');return
  except (requests.RequestException,ValueError) as error:errors.append(f'{url.split("/")[2]}: {error}')
 raise RuntimeError('Unable to download model. '+'; '.join(errors))
if __name__=='__main__':main()
