"""Download small BERT assets and apply the documented Windows inference patches."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parent
VENDOR=ROOT/'vendor/countgdpp'
ASSETS=ROOT/'models/countgdpp'

def patch_source():
 path=VENDOR/'models/GroundingDINO/ms_deform_attn.py'
 text=path.read_text(encoding='utf-8')
 text=text.replace('except:\n    warnings.warn(', 'except:\n    _C = None\n    warnings.warn(')
 text=text.replace('if torch.cuda.is_available() and value.is_cuda:', 'if _C is not None and torch.cuda.is_available() and value.is_cuda:')
 path.write_text(text,encoding='utf-8')
 path=VENDOR/'util/get_tokenlizer.py';text=path.read_text(encoding='utf-8')
 old='        return BertModel.from_pretrained(text_encoder_type)'
 new="        if os.environ.get('COUNTGDPP_FULL_CHECKPOINT') == '1':\n            from transformers import BertConfig\n            return BertModel(BertConfig.from_pretrained(text_encoder_type, local_files_only=True))\n"+old
 if 'COUNTGDPP_FULL_CHECKPOINT' not in text:
  assert old in text;text=text.replace(old,new)
 path.write_text(text,encoding='utf-8')
 # The application imports groundingdino.util; both upstream copies must agree.
 (VENDOR/'groundingdino/util/get_tokenlizer.py').write_text(text,encoding='utf-8')

def main():
 p=argparse.ArgumentParser();p.add_argument('--proxy');p.add_argument('--patch-only',action='store_true');a=p.parse_args()
 if a.patch_only:
  patch_source()
  print('Pinned vendor Windows inference patches applied; installed models were not modified.')
  return
 dest=ASSETS/'bert-base-uncased';dest.mkdir(parents=True,exist_ok=True)
 session=requests.Session();session.trust_env=False
 if a.proxy:session.proxies.update(dict(http=a.proxy,https=a.proxy))
 for name in ['config.json','tokenizer_config.json','vocab.txt','tokenizer.json']:
  path=dest/name
  if not path.exists():
   response=session.get('https://huggingface.co/google-bert/bert-base-uncased/resolve/main/'+name,timeout=40);response.raise_for_status();path.write_bytes(response.content)
  print(name,path.stat().st_size,flush=True)
 patch_source()
 weight=ASSETS/'countgd_plusplus.pth'
 manifest=dict(source_repo='https://github.com/niki-amini-naieni/CountGDPlusPlus',commit=subprocess.check_output(['git','-C',str(VENDOR),'rev-parse','HEAD'],text=True).strip(),checkpoint_source='https://drive.google.com/file/d/1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs/view',patches=['Use upstream PyTorch deformable-attention fallback when CUDA extension is unavailable','Initialize BERT from config only; all BERT weights must then load from full checkpoint with strict validation'],files={})
 for path in [weight,*dest.iterdir()]:
  if path.is_file():
   with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
   manifest['files'][str(path.relative_to(ASSETS))]=dict(bytes=path.stat().st_size,sha256=digest)
 (ASSETS/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('Assets prepared, Windows patches applied, hashes recorded.')
if __name__=='__main__':main()
