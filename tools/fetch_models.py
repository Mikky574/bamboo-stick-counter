"""Restore verified release models, with resumable downloads and safe ZIP extraction.

Uses the standard library only. The checked-in manifest is the trust anchor;
SHA-256 verifies bytes, not the safety of loading arbitrary third-party models.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def matches(path, record):
    return (path.is_file() and not path.is_symlink() and path.stat().st_size == record['bytes']
            and digest(path) == record['sha256'])


def safe_relative(name):
    if (not isinstance(name, str) or not name or '\\' in name or ':' in name
            or '\x00' in name or name.endswith('/') or name.startswith('/')):
        raise ValueError('Unsafe archive path: ' + repr(name))
    parts = name.split('/')
    if any(part in ('', '.', '..') for part in parts):
        raise ValueError('Unsafe archive path: ' + repr(name))
    if any(part.rstrip(' .').upper().split('.')[0] in {'CON', 'PRN', 'AUX', 'NUL',
             *['COM' + str(i) for i in range(1, 10)], *['LPT' + str(i) for i in range(1, 10)]}
           or part != part.rstrip(' .') for part in parts):
        raise ValueError('Unsafe Windows path: ' + repr(name))
    return Path(*PurePosixPath(name).parts)


def contained(root, name):
    relative = safe_relative(name)
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Destination escapes project: ' + name)
    # Refuse parent symlinks even when their target happens to be inside root.
    probe = root
    for part in relative.parts:
        probe /= part
        if probe.is_symlink():
            raise ValueError('Linked destination is not allowed: ' + name)
    return resolved


def validate_record(record):
    if not isinstance(record, dict) or not isinstance(record.get('bytes'), int) or record['bytes'] < 0:
        raise ValueError('Invalid size in release manifest')
    if not re.fullmatch('[a-f0-9]{64}', record.get('sha256', '')):
        raise ValueError('Invalid SHA-256 in release manifest')


def validate_manifest(manifest):
    if manifest.get('format_version') != 1 or not isinstance(manifest.get('assets'), list):
        raise ValueError('Unsupported release manifest')
    seen = set()
    for asset in manifest['assets']:
        validate_record(asset)
        if '/' in asset['name'] or safe_relative(asset['name']).name != asset['name']:
            raise ValueError('Invalid release asset name')
        if not asset.get('url', '').startswith('https://github.com/'):
            raise ValueError('Release download must use HTTPS GitHub URL')
        for name, record in asset['files'].items():
            safe_relative(name)
            if name.casefold() in seen:
                raise ValueError('Duplicate archive destination: ' + name)
            seen.add(name.casefold())
            validate_record(record)


def download(asset, cache, proxy=''):
    target = cache / asset['name']
    if matches(target, asset):
        return target
    cache.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler(
        {'http': proxy, 'https': proxy} if proxy else {}))
    last_error = None
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset >= asset['bytes']:
            if matches(partial, asset):
                partial.replace(target)
                return target
            partial.unlink()
            offset = 0
        headers = {'User-Agent': 'bamboo-stick-counter-model-restorer/0.2', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            request = urllib.request.Request(asset['url'], headers=headers)
            with opener.open(request, timeout=90) as response:
                status = response.status
                if status == 206:
                    expected_range = f'bytes {offset}-{asset["bytes"] - 1}/{asset["bytes"]}'
                    if response.headers.get('Content-Range') != expected_range:
                        raise RuntimeError('Invalid resumed Content-Range')
                elif status == 200:
                    offset = 0
                else:
                    raise RuntimeError('Unexpected download status: ' + str(status))
                remaining = asset['bytes'] - offset
                declared = response.headers.get('Content-Length')
                if declared is not None and int(declared) != remaining:
                    raise RuntimeError('Downloaded Content-Length differs from release manifest')
                received = 0
                with partial.open('ab' if offset else 'wb') as stream:
                    while chunk := response.read(1024 * 1024):
                        received += len(chunk)
                        if received > remaining:
                            raise RuntimeError('Download exceeds expected size')
                        stream.write(chunk)
                if received != remaining:
                    raise RuntimeError('Truncated download; partial retained for resume')
            if not matches(partial, asset):
                partial.unlink(missing_ok=True)
                raise RuntimeError('Model archive SHA-256 mismatch')
            partial.replace(target)
            return target
        except (OSError, ValueError, RuntimeError, urllib.error.URLError) as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f'Cannot download {asset["name"]}: {last_error}')


def custom_active_files(root, manifest):
    """Keep an already valid user-selected classifier profile/checkpoint intact."""
    name = 'models/scene_classifier/active_profile.json'
    profile_path = root / name
    if not profile_path.is_file():
        return set()
    expected = next((a['files'][name] for a in manifest['assets'] if name in a['files']), None)
    if expected and matches(profile_path, expected):
        return set()
    try:
        profile = json.loads(profile_path.read_text(encoding='utf-8'))
        checkpoint = contained(root, profile['checkpoint'])
        if profile.get('architecture') == 'convnext_small' and checkpoint.is_file() and digest(checkpoint) == profile['sha256']:
            return {name, profile['checkpoint']}
    except (KeyError, ValueError, OSError, TypeError):
        pass
    return set()


def restore_archive(path, asset, root, preserve=(), repair=False):
    if not matches(path, asset):
        raise RuntimeError('Archive checksum mismatch: ' + str(path))
    with zipfile.ZipFile(path) as archive:
        names = [info.filename for info in archive.infolist()]
        if len(set(n.casefold() for n in names)) != len(names) or set(names) != set(asset['files']):
            raise ValueError('Archive member list differs from manifest')
        for info in archive.infolist():
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError('Linked or special ZIP members are not allowed')
            safe_relative(info.filename)
            if info.file_size != asset['files'][info.filename]['bytes']:
                raise ValueError('Archive member size differs from manifest')
        # Stage/check every member before touching installed files.
        with tempfile.TemporaryDirectory(prefix='restore_models_', dir=root) as temporary:
            stage = Path(temporary)
            pending = []
            for info in archive.infolist():
                name = info.filename
                destination = contained(root, name)
                if name in preserve or matches(destination, asset['files'][name]):
                    continue
                prepared = stage / safe_relative(name)
                prepared.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as incoming, prepared.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
                if not matches(prepared, asset['files'][name]):
                    raise RuntimeError('Unpacked file checksum mismatch: ' + name)
                if destination.exists() and not repair:
                    # Git's autocrlf can change checked-in JSON/license bytes on
                    # a fresh clone. Permit only the identical small text with
                    # different newlines; actual custom model/config edits stay.
                    newline_only = (destination.is_file() and
                        destination.suffix.lower() in {'.json', '.txt'} and
                        destination.stat().st_size <= 1024 * 1024 and
                        prepared.stat().st_size <= 1024 * 1024 and
                        destination.read_bytes().replace(b'\r\n', b'\n') ==
                        prepared.read_bytes().replace(b'\r\n', b'\n'))
                    if not newline_only:
                        raise RuntimeError('Existing file differs; preserved: ' + name +
                                           '. Back it up and use --repair only if replacement is intended.')
                pending.append((prepared, destination))
            for prepared, destination in pending:
                destination.parent.mkdir(parents=True, exist_ok=True)
                prepared.replace(destination)
    return len(pending)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'models/release-manifest.json')
    parser.add_argument('--tag', default='v0.2.0')
    parser.add_argument('--archive-dir', type=Path, help='Use downloaded ZIPs offline instead of network')
    parser.add_argument('--proxy', default='')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--repair', action='store_true', help='Replace mismatched files after explicit backup; valid custom profile is still preserved')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    validate_manifest(manifest)
    if args.tag != manifest['tag']:
        parser.error('Tag does not match the trusted manifest; check out that release source first.')
    preserve = custom_active_files(ROOT, manifest)
    if preserve:
        print('Preserving valid user-selected scene profile/checkpoint:', ', '.join(sorted(preserve)))
    missing = []
    for asset in manifest['assets']:
        invalid = [name for name, record in asset['files'].items()
                   if name not in preserve and not matches(contained(ROOT, name), record)]
        if invalid and not args.verify_only:
            path = (args.archive_dir / asset['name'] if args.archive_dir
                    else download(asset, ROOT / 'release_assets/downloads', args.proxy))
            restored = restore_archive(path, asset, ROOT, preserve, args.repair)
            print(asset['name'], 'restored files:', restored, flush=True)
            invalid = [name for name, record in asset['files'].items()
                       if name not in preserve and not matches(contained(ROOT, name), record)]
        missing.extend(invalid)
        if not invalid:
            print(asset['name'], 'verified; existing valid files retained', flush=True)
    if missing:
        raise SystemExit('Missing or mismatched files:\n' + '\n'.join(missing))
    print('Required model assets verified. No GPU inference or environment installation was performed.')


if __name__ == '__main__':
    main()
