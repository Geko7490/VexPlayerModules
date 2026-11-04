import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CatalogTest(unittest.TestCase):
    def setUp(self):
        self.catalog = json.loads((ROOT / 'catalog.json').read_text())

    def test_both_real_model_languages_are_present(self):
        self.assertEqual(self.catalog['schema'], 1)
        self.assertEqual({m['language'] for m in self.catalog['models']}, {'en', 'tr'})
        self.assertEqual(len({m['id'] for m in self.catalog['models']}), 2)

    def test_assets_are_versioned_release_urls_not_placeholders(self):
        for model in self.catalog['models']:
            expected = f"https://github.com/Geko7490/VexPlayerModules/releases/download/ctc-v{model['version']}/{model['id']}-{model['version']}.zip"
            self.assertEqual(model['downloadUrl'], expected)
            self.assertTrue(1024 < model['sizeBytes'] <= 167772160)
            self.assertTrue(re.fullmatch('[0-9a-f]{64}', model['sha256']))

    def test_android_contract_and_license_are_explicit(self):
        for model in self.catalog['models']:
            self.assertEqual(model['runtime'], 'onnxruntime-android-1.23.2')
            self.assertEqual(model['architecture'], 'wav2vec2-character-ctc')
            self.assertEqual(model['license'], 'Apache-2.0')
            self.assertGreaterEqual(model['minAppVersionCode'], 4)
            self.assertIn('android-device-and-singing-pending', model['validation'])

    def test_no_model_binaries_are_committed(self):
        self.assertFalse(list(ROOT.rglob('*.onnx')))
        self.assertFalse(list(ROOT.rglob('*.zip')))


if __name__ == '__main__':
    unittest.main()
