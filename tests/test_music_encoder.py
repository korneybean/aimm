import importlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf


class FixtureBackend:
    """Small deterministic vectors isolate I/O and pooling from model inference."""
    model_id = 'fixture-model'
    revision = 'fixture-v1'
    sample_rate = 48000
    window_seconds = 10.0

    def __init__(self):
        self.audio_lengths = []

    def audio_embedding(self, samples):
        self.audio_lengths.append(len(samples))
        return np.array([1., 0.]) if len(self.audio_lengths) == 1 else np.array([0., 1.])

    def text_embeddings(self, prompts):
        return np.array([[1., 0.], [0., 1.]])


class MusicEncoderTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('music_encoder'), 'Music encoder must be implemented')
        self.encoder = importlib.import_module('music_encoder')
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write_audio(self, seconds, rate=16000, silent=False):
        t = np.arange(round(seconds * rate)) / rate
        y = np.zeros(len(t)) if silent else .2 * np.sin(2 * np.pi * 440 * t)
        path = self.root / 'audio.wav'
        sf.write(path, np.column_stack([y, y * .5]), rate)
        return path

    def test_full_track_and_short_tail_have_exact_coverage_and_weighted_embedding(self):
        backend = FixtureBackend()
        result = self.encoder.encode_music(self.write_audio(12), backend=backend,
                                          prompts=['First description', 'Second description'])
        self.assertEqual([(c['start_sec'], c['end_sec']) for c in result['chunks']], [(0., 10.), (10., 12.)])
        self.assertEqual(backend.audio_lengths, [480000, 96000])
        expected = np.array([10., 2.]) / np.linalg.norm([10., 2.])
        np.testing.assert_allclose(result['embedding'], expected, atol=1e-6)
        self.assertEqual(result['matches'][0]['description'], 'First description')
        self.assertEqual(result['score_type'], 'cosine_similarity_not_probability')
        self.assertEqual(result['model']['revision'], 'fixture-v1')
        json.dumps(result, allow_nan=False)

    def test_partial_analysis_does_not_encode_unrequested_tail(self):
        backend = FixtureBackend()
        result = self.encoder.encode_music(self.write_audio(12), backend=backend,
            prompts=['First description', 'Second description'], max_duration=3.25)
        self.assertEqual(result['duration_sec'], 3.25)
        self.assertEqual(len(result['chunks']), 1)
        self.assertEqual(backend.audio_lengths, [156000])

    def test_silence_does_not_receive_made_up_description_matches(self):
        backend = FixtureBackend()
        result = self.encoder.encode_music(self.write_audio(1, silent=True), backend=backend,
                                          prompts=['First description', 'Second description'])
        self.assertEqual(result['status'], 'no_audible_audio')
        self.assertIsNone(result['embedding'])
        self.assertEqual(result['matches'], [])
        self.assertEqual(backend.audio_lengths, [])

    def test_invalid_embedding_fails_instead_of_writing_nan(self):
        class BadBackend(FixtureBackend):
            def audio_embedding(self, samples):
                return np.array([np.nan, 1.])
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.encoder.encode_music(self.write_audio(1), backend=BadBackend(),
                                      prompts=['First description', 'Second description'])

    def test_invalid_duration_is_rejected(self):
        for duration in [0, -1, float('nan'), float('inf')]:
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.encoder.encode_music(self.write_audio(1), backend=FixtureBackend(),
                    prompts=['First description', 'Second description'], max_duration=duration)

    def test_analyze_and_package_preserve_encoder_results(self):
        from aim import analyze, pack, unpack
        source = self.write_audio(1)
        data = analyze(source, encoder='clap', encoder_backend=FixtureBackend(),
                       encoder_prompts=['First description', 'Second description'])
        self.assertEqual(data['features']['music_encoder']['model']['id'], 'fixture-model')
        self.assertIsNone(data['features']['lyrics'])
        self.assertEqual(data['features']['instruments'], [])
        archive = pack(source, self.root / 'song.aim', data=data)
        unpack(archive, self.root / 'out')
        restored = json.loads((self.root / 'out' / 'aifm.json').read_text())
        self.assertEqual(restored['features']['music_encoder'], data['features']['music_encoder'])

    def test_encoder_comparison_rejects_different_model_revisions(self):
        from aim import analyze, compare
        source = self.write_audio(1)
        first = analyze(source, encoder='clap', encoder_backend=FixtureBackend(),
                        encoder_prompts=['First description', 'Second description'])
        second = json.loads(json.dumps(first))
        second['features']['music_encoder']['model']['revision'] = 'different'
        result = compare(first, second)
        self.assertIsNone(result['music_encoder']['cosine_similarity'])
        self.assertEqual(result['music_encoder']['status'], 'incompatible_models')
        self.assertAlmostEqual(compare(first, first)['music_encoder']['cosine_similarity'], 1., places=5)

    def test_encoder_comparison_requires_model_and_protocol_metadata(self):
        valid = self.encoder.encode_music(self.write_audio(1), backend=FixtureBackend(),
            prompts=['First description', 'Second description'])
        incomplete = [{'embedding': [1., 0.]}]
        for key in ('model', 'normalization', 'pooling', 'sample_rate', 'window_seconds',
                    'embedding_dimension', 'channels', 'short_chunk_padding'):
            candidate = json.loads(json.dumps(valid))
            candidate.pop(key)
            incomplete.append(candidate)
        for model in ({}, {'id': 'fixture-model'}, {'revision': 'fixture-v1'},
                      {'id': '', 'revision': 'fixture-v1'}):
            candidate = json.loads(json.dumps(valid))
            candidate['model'] = model
            incomplete.append(candidate)
        for candidate in incomplete:
            with self.subTest(metadata=candidate.get('model')):
                result = self.encoder.compare_embeddings(candidate, candidate)
                self.assertEqual(result['status'], 'incompatible_models')
                self.assertIsNone(result['cosine_similarity'])

    def test_encoder_comparison_rejects_incorrect_declared_dimensions(self):
        first = self.encoder.encode_music(self.write_audio(1), backend=FixtureBackend(),
            prompts=['First description', 'Second description'])
        first['embedding_dimension'] = 3
        result = self.encoder.compare_embeddings(first, first)
        self.assertEqual(result['status'], 'incompatible_dimensions')
        self.assertIsNone(result['cosine_similarity'])

    def test_text_encoder_health_rejects_collapsed_representations(self):
        check = getattr(self.encoder, 'check_text_encoder_health', None)
        self.assertTrue(callable(check), 'A collapsed text encoder must be rejected')
        for vectors in (np.ones((4, 4)), np.array([[1., 0., 0., 0.]]) + .001 * np.eye(4)):
            with self.subTest(vectors=vectors), self.assertRaisesRegex(RuntimeError, 'collapsed'):
                check(vectors)

    def test_text_encoder_health_accepts_distinct_representations(self):
        check = getattr(self.encoder, 'check_text_encoder_health', None)
        self.assertTrue(callable(check), 'Text encoder diagnostics must be available')
        report = check(np.eye(4))
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['mean_pairwise_cosine'], 0.)
        self.assertEqual(report['probe_count'], 4)

    def test_text_encoder_health_rejects_invalid_probe_vectors(self):
        check = getattr(self.encoder, 'check_text_encoder_health', None)
        self.assertTrue(callable(check), 'Invalid text encoder probes must be rejected')
        for vectors in (np.ones((1, 4)), np.zeros((4, 4)), np.full((4, 4), np.nan)):
            with self.subTest(vectors=vectors), self.assertRaises(ValueError):
                check(vectors)


if __name__ == '__main__':
    unittest.main()
