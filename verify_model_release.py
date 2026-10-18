"""Release download/restore contract tests; use tiny fixtures, never model/GPU loads."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from tools import fetch_models, package_models


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def asset(self, contents=None):
        contents = contents or {'models/a.bin': b'expected-model', 'docs/licenses/test.txt': b'license'}
        for name, content in contents.items():
            path = self.root / 'source' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        archive = self.root / 'tiny.zip'
        asset = package_models.create_archive(self.root / 'source', archive, list(contents))
        asset['url'] = 'https://github.com/owner/project/releases/download/v0/tiny.zip'
        return archive, asset

    def test_reproducible_archive_and_exact_restore(self):
        archive, asset = self.asset()
        repeated = self.root / 'repeated.zip'
        again = package_models.create_archive(self.root / 'source', repeated, list(reversed(asset['files'])))
        self.assertEqual(asset['sha256'], again['sha256'])
        destination = self.root / 'restored'
        destination.mkdir()
        self.assertEqual(fetch_models.restore_archive(archive, asset, destination), 2)
        for name, record in asset['files'].items():
            self.assertTrue(fetch_models.matches(destination / name, record))
        self.assertEqual(fetch_models.restore_archive(archive, asset, destination), 0)

    def test_mismatched_existing_model_is_preserved_until_explicit_repair(self):
        archive, asset = self.asset()
        destination = self.root / 'restored'
        (destination / 'models').mkdir(parents=True)
        (destination / 'models/a.bin').write_bytes(b'user-custom-weight')
        with self.assertRaisesRegex(RuntimeError, 'Existing file differs'):
            fetch_models.restore_archive(archive, asset, destination)
        self.assertEqual((destination / 'models/a.bin').read_bytes(), b'user-custom-weight')
        self.assertFalse((destination / 'docs/licenses/test.txt').exists())
        fetch_models.restore_archive(archive, asset, destination, repair=True)
        self.assertEqual((destination / 'models/a.bin').read_bytes(), b'expected-model')

    def test_archive_and_member_hashes_are_verified_before_install(self):
        archive, asset = self.asset()
        destination = self.root / 'restored'; destination.mkdir()
        wrong = json.loads(json.dumps(asset))
        wrong['files']['models/a.bin']['sha256'] = '0' * 64
        with self.assertRaisesRegex(RuntimeError, 'Unpacked file checksum'):
            fetch_models.restore_archive(archive, wrong, destination)
        self.assertFalse((destination / 'models/a.bin').exists())
        archive.write_bytes(archive.read_bytes() + b'tamper')
        with self.assertRaisesRegex(RuntimeError, 'Archive checksum'):
            fetch_models.restore_archive(archive, asset, destination)

    def test_unsafe_paths_rejected(self):
        for name in ('../escape', '/absolute', 'C:/escape', 'a\\..\\escape',
                     'a//b', 'a/./b', 'a/../b', 'a/NUL.txt', 'a/trailing.', 'file:stream'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fetch_models.safe_relative(name)

    def test_symlink_zip_member_rejected(self):
        archive, asset = self.asset({'models/a.bin': b'expected-model'})
        with zipfile.ZipFile(archive, 'w') as handle:
            info = zipfile.ZipInfo('models/a.bin')
            info.external_attr = 0o120777 << 16
            handle.writestr(info, b'expected-model')
        asset.update(bytes=archive.stat().st_size, sha256=fetch_models.digest(archive))
        destination = self.root / 'restored'; destination.mkdir()
        with self.assertRaisesRegex(ValueError, 'special ZIP'):
            fetch_models.restore_archive(archive, asset, destination)

    def test_custom_valid_active_scene_profile_survives(self):
        name = 'models/scene_classifier/active_profile.json'
        checkpoint = 'models/scene_classifier/convnext_small_v2/scene_convnext_small.pt'
        path = self.root / checkpoint
        path.parent.mkdir(parents=True)
        path.write_bytes(b'user-trained-newer-valid-model')
        profile = {'architecture': 'convnext_small', 'checkpoint': checkpoint,
                   'sha256': fetch_models.digest(path)}
        (self.root / name).write_text(json.dumps(profile), encoding='utf-8')
        manifest = {'assets': [{'files': {name: {'bytes': 0, 'sha256': '0' * 64}}}]}
        self.assertEqual(fetch_models.custom_active_files(self.root, manifest), {name, checkpoint})

    def test_fresh_clone_autocrlf_licenses_are_restored_without_repair(self):
        archive, asset = self.asset({'docs/licenses/test.txt': b'license\nsecond line\n'})
        destination = self.root / 'restored'
        (destination / 'docs/licenses').mkdir(parents=True)
        path = destination / 'docs/licenses/test.txt'
        path.write_bytes(b'license\r\nsecond line\r\n')
        self.assertEqual(fetch_models.restore_archive(archive, asset, destination), 1)
        self.assertTrue(fetch_models.matches(path, asset['files']['docs/licenses/test.txt']))

    def test_download_resumes_exact_range(self):
        payload = b'1234567890abcdef'
        asset = {'name': 'test.zip', 'bytes': len(payload),
                 'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://github.com/test.zip'}
        (self.root / 'test.zip.partial').write_bytes(payload[:6])
        response = io.BytesIO(payload[6:])
        response.status = 206
        response.headers = {'Content-Range': 'bytes 6-15/16', 'Content-Length': '10'}
        class Opener:
            def open(inner, request, timeout):
                self.assertEqual(request.get_header('Range'), 'bytes=6-')
                return response
        with patch.object(fetch_models.urllib.request, 'build_opener', return_value=Opener()):
            result = fetch_models.download(asset, self.root)
        self.assertEqual(result.read_bytes(), payload)

    def test_download_ignoring_range_restarts_cleanly(self):
        payload = b'abcdef'
        asset = {'name': 'test.zip', 'bytes': len(payload),
                 'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://github.com/test.zip'}
        (self.root / 'test.zip.partial').write_bytes(payload[:3])
        response = io.BytesIO(payload)
        response.status = 200; response.headers = {'Content-Length': '6'}
        class Opener:
            def open(inner, request, timeout): return response
        with patch.object(fetch_models.urllib.request, 'build_opener', return_value=Opener()):
            result = fetch_models.download(asset, self.root)
        self.assertEqual(result.read_bytes(), payload)

    def test_invalid_range_never_installs(self):
        payload = b'abcdef'
        asset = {'name': 'test.zip', 'bytes': len(payload),
                 'sha256': hashlib.sha256(payload).hexdigest(), 'url': 'https://github.com/test.zip'}
        (self.root / 'test.zip.partial').write_bytes(payload[:3])
        class Opener:
            def open(inner, request, timeout):
                response = io.BytesIO(payload[3:]); response.status = 206
                response.headers = {'Content-Range': 'bytes 0-2/6', 'Content-Length': '3'}
                return response
        with patch.object(fetch_models.urllib.request, 'build_opener', return_value=Opener()), patch.object(fetch_models.time, 'sleep'):
            with self.assertRaisesRegex(RuntimeError, 'Content-Range'):
                fetch_models.download(asset, self.root)
        self.assertFalse((self.root / 'test.zip').exists())


if __name__ == '__main__':
    unittest.main()
