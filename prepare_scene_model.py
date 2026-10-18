"""Download the pinned official TinyCLIP assets; never modifies environments."""
import hashlib
import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
REPO = 'wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M'
REVISION = 'a2a8c6eaa2549ad66eb7c31b85022bf58273a26c'
TARGET = ROOT/'models/scene_classifier/tinyclip_vit_8m16_text3m_yfcc15m'
FILES = {
    'model.safetensors':'9339ee3d736344d0ddcaa6c03edc9f89688f08caaea5401220885233da726fcc',
    'config.json':'c586d856dbd5589b67525db9e791ee0917402405672c7773aead373efb7d7bca',
    'preprocessor_config.json':'910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd',
    'tokenizer.json':'b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842',
}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proxy',default='')
    arguments=parser.parse_args()
    session=requests.Session()
    if arguments.proxy:
        session.trust_env=False
        session.proxies.update(http=arguments.proxy,https=arguments.proxy)
    TARGET.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for name,expected in FILES.items():
        file=TARGET/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            url=f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}'
            partial=file.with_suffix(file.suffix+'.partial')
            with session.get(url,stream=True,timeout=(20,90)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for chunk in response.iter_content(1024*1024):
                        if chunk:stream.write(chunk)
            if hashlib.sha256(partial.read_bytes()).hexdigest()!=expected:
                raise RuntimeError('Downloaded asset checksum mismatch: '+name)
            partial.replace(file)
        manifest.append(dict(file=name,sha256=expected,bytes=file.stat().st_size))
        print(name,'verified',flush=True)
    (TARGET/'verified_assets.json').write_text(json.dumps(dict(repo=REPO,revision=REVISION,files=manifest),indent=2),encoding='utf-8')


if __name__=='__main__':main()
