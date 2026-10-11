"""Opt-in checks of actual pretrained inference, including known audio examples.

Run with AIMM_RUN_MODEL_TESTS=1 after installing requirements-encoder.txt.
The model and three documented librosa recordings download on first use.
"""
import os
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get('AIMM_RUN_MODEL_TESTS') == '1',
                     'Set AIMM_RUN_MODEL_TESTS=1 to run pretrained audio/text checks')
class ClapIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from music_encoder import ClapBackend
        torch.set_num_threads(4)
        cls.backend = ClapBackend()

    def test_distinct_descriptions_have_distinct_embeddings(self):
        descriptions = ['An electric guitar playing heavy metal.',
                        'Quiet solo piano music.',
                        'A person reading a book aloud.',
                        'A dog barking outdoors.']
        vectors = np.asarray(self.backend.text_embeddings(descriptions))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        similarities = vectors @ vectors.T
        mean = float(similarities[np.triu_indices(len(vectors), 1)].mean())
        self.assertLess(mean, .9, f'Distinct descriptions collapsed: mean cosine {mean}')

    def test_known_recordings_match_their_sound_descriptions(self):
        import librosa
        descriptions = ['A person speaking, with no music.',
                        'A solo piano playing music.',
                        'A solo trumpet playing music.']
        text = np.asarray(self.backend.text_embeddings(descriptions))
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        # Official descriptions and licensing: https://librosa.org/doc/latest/recordings.html
        for key, expected in [('libri1', 0), ('pistachio', 1), ('trumpet', 2)]:
            with self.subTest(recording=key):
                samples, _ = librosa.load(librosa.ex(key), sr=48000, mono=True, duration=10)
                vector = np.asarray(self.backend.audio_embedding(samples))
                vector /= np.linalg.norm(vector)
                scores = text @ vector
                self.assertEqual(int(np.argmax(scores)), expected,
                                 f'{key}: candidate similarities {scores.tolist()}')

    def test_legacy_collapsed_checkpoint_is_rejected(self):
        from music_encoder import ClapBackend
        class LegacyBackend(ClapBackend):
            model_id = 'laion/larger_clap_music'
            revision = 'a0b4534a14f58e20944452dff00a22a06ce629d1'
        backend = LegacyBackend()
        with self.assertRaisesRegex(RuntimeError, 'collapsed'):
            backend.text_embeddings(['Music played on a piano.'])
        self.assertIsNone(backend.model, 'A rejected model must not remain cached as usable')


if __name__ == '__main__':
    unittest.main()
