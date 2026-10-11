import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
from aim import analyze, pack, unpack, compare, validate

class AIMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        sr = 22050
        t = np.arange(sr * 3) / sr
        y = np.sin(2 * np.pi * 440 * t) * 0.15
        y[int(sr):int(sr*1.2)] = 0
        self.source = self.root / 'song.wav'
        sf.write(self.source, y, sr, subtype='PCM_16')

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyze(self):
        d = analyze(self.source)
        self.assertTrue(validate(d))
        self.assertEqual(d['format'], 'AIFM')
        self.assertAlmostEqual(d['audio']['duration_sec'], 3, places=1)
        self.assertTrue(d['features']['energy_curve'])
        self.assertIsNone(d['features']['harmony'])

    def test_pack_unpack(self):
        archive = pack(self.source, self.root / 'song.aim')
        result = unpack(archive, self.root / 'out')
        self.assertEqual(self.source.read_bytes(), result.read_bytes())
        self.assertTrue((self.root / 'out' / 'aifm.json').exists())

    def test_compare(self):
        d = analyze(self.source)
        result = compare(d, d)
        self.assertEqual(result['descriptors']['mean_normalized_energy']['difference_b_minus_a'], 0)

    def test_bad_fingerprint(self):
        d = analyze(self.source)
        d['audio']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            pack(self.source, self.root / 'bad.aim', data=d)

if __name__ == '__main__':
    unittest.main()
