"""Create deterministic, allowlisted GitHub Release model archives (no training data)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ASSET_BYTES = 2 * 1024 ** 3 - 1
COUNTGD_FILES = [
    'models/countgdpp/countgd_plusplus.pth',
    'models/countgdpp/provenance.json',
    *['models/countgdpp/bert-base-uncased/' + name for name in
      ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt')],
    'docs/licenses/COUNTGDPP-MIT.txt', 'docs/licenses/BERT-APACHE-2.0.txt',
]
SCENE_FILES = [
    'models/scene_classifier/active_profile.json',
    'models/scene_classifier/convnext_small_v2/profile.json',
    'models/scene_classifier/convnext_small_v2/scene_convnext_small.pt',
    'models/scene_classifier/convnext_small_v2/training_report.json',
    'models/scene_classifier/convnext_small_v1/pretrained_provenance.json',
    'docs/licenses/TORCHVISION-v0.21.0-BSD-3-Clause.txt',
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def create_archive(root, destination, files):
    records = {}
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.partial')
    with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name in sorted(files):
            source = root / name
            if not source.is_file() or source.is_symlink():
                raise RuntimeError('Missing or linked release input: ' + name)
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.file_size = source.stat().st_size
            with source.open('rb') as incoming, archive.open(info, 'w') as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            records[name] = {'bytes': source.stat().st_size, 'sha256': digest(source)}
    if partial.stat().st_size > MAX_ASSET_BYTES:
        raise RuntimeError('Asset exceeds the 2 GiB GitHub limit; split allowlist before publishing.')
    partial.replace(destination)
    return {'name': destination.name, 'bytes': destination.stat().st_size,
            'sha256': digest(destination), 'required': True, 'files': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', default='v0.2.0')
    parser.add_argument('--repository', default='Mikky574/bamboo-stick-counter')
    parser.add_argument('--output', type=Path, default=ROOT / 'release_assets/v0.2.0')
    args = parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-' for c in args.tag):
        parser.error('tag must contain only letters, digits, dot, underscore or hyphen')
    profile = json.loads((ROOT / SCENE_FILES[0]).read_text(encoding='utf-8'))
    countgd_provenance = json.loads((ROOT / 'models/countgdpp/provenance.json').read_text(encoding='utf-8'))
    if countgd_provenance.get('commit') != '8489f836d75eb407ae778c56cff1f72cc6f8baae':
        raise RuntimeError('CountGD provenance does not match the pinned vendor revision.')
    for name, record in countgd_provenance['files'].items():
        source = ROOT / 'models/countgdpp' / Path(name.replace('\\', '/'))
        if not source.is_file() or source.stat().st_size != record['bytes'] or digest(source) != record['sha256']:
            raise RuntimeError('CountGD asset does not match recorded upstream hash: ' + name)
    expected_path = 'models/scene_classifier/convnext_small_v2/scene_convnext_small.pt'
    if profile.get('checkpoint') != expected_path or profile.get('sha256') != digest(ROOT / expected_path):
        raise RuntimeError('Active profile does not match the v2 release checkpoint; review the allowlist.')
    assets = []
    for component, files in [('countgdpp-models', COUNTGD_FILES), ('scene-models', SCENE_FILES)]:
        asset = create_archive(ROOT, args.output / f'{component}-{args.tag}.zip', files)
        asset['url'] = f'https://github.com/{args.repository}/releases/download/{args.tag}/{asset["name"]}'
        assets.append(asset)
        print(asset['name'], asset['bytes'], asset['sha256'], flush=True)
    manifest = {'format_version': 1, 'repository': args.repository, 'tag': args.tag,
        'archive_format': 'zip-stored-deterministic-v1', 'assets': assets,
        'excluded': ['Python environments', 'vendor checkouts', 'training images',
                     'old GeCo2 checkpoint', 'old scene v1 checkpoint', 'ImageNet initialization checkpoint',
                     'optional TinyCLIP checkpoint'],
        'sources': {
            'countgdpp': {'repository': 'https://github.com/niki-amini-naieni/CountGDPlusPlus',
                         'commit': '8489f836d75eb407ae778c56cff1f72cc6f8baae',
                         'checkpoint': 'https://drive.google.com/file/d/1j6N22TtKu2NVcKpgfrf-sJHGeLDqs9hs/view',
                         'code_license': 'MIT; pretrained data and upstream notices retain their terms'},
            'bert': {'repository': 'https://huggingface.co/google-bert/bert-base-uncased',
                     'license': 'Apache-2.0', 'scope': 'config and tokenizer only; BERT weights are in CountGD checkpoint'},
            'scene': {'architecture': 'torchvision ConvNeXt-Small', 'project_model': profile['model_id'],
                      'torchvision_code_license': 'BSD-3-Clause',
                      'note': 'Project fine-tuning; no training images or new license grant for upstream weights.'}}}
    encoded = (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    (ROOT / 'models/release-manifest.json').write_bytes(encoded)
    (args.output / 'release-manifest.json').write_bytes(encoded)
    checksums = ''.join(f'{a["sha256"]}  {a["name"]}\n' for a in assets)
    checksums += f'{digest(args.output / "release-manifest.json")}  release-manifest.json\n'
    (args.output / 'SHA256SUMS.txt').write_text(checksums, encoding='utf-8', newline='\n')
    print('Total assets:', sum(a['bytes'] for a in assets), flush=True)


if __name__ == '__main__':
    main()
