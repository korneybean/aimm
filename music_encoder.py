"""Pretrained CLAP music embeddings with complete, timestamped audio coverage."""
from __future__ import annotations

import logging
import math

import librosa
import numpy as np
import soundfile as sf

MODEL_ID = 'laion/larger_clap_general'
MODEL_REVISION = 'ada0c23a36c4e8582805bb38fec3905903f18b41'
TEXT_HEALTH_PROBES = (
    'An electric guitar playing heavy metal.',
    'Quiet solo piano music.',
    'A person reading a book aloud.',
    'A dog barking outdoors.',
)
TEXT_COLLAPSE_LIMIT = .98
DEFAULT_PROMPTS = (
    'A recording of hip hop music with rapping over a beat.',
    'A recording of electronic dance music with synthesizers and drums.',
    'A recording of pop music with singing and a steady beat.',
    'A recording of rock music with electric guitars and drums.',
    'A recording of acoustic folk music with guitar and singing.',
    'A recording of jazz music with improvisation.',
    'A recording of soul and R&B music with melodic singing.',
    'A recording of ambient instrumental music with sustained tones.',
    'A recording of orchestral classical music.',
    'A recording of country music with singing and guitar.',
    'A recording of punk music with fast drums and distorted guitars.',
    'A recording of funk music with a rhythmic bass line.',
    'A recording of a person rapping rhythmically over music.',
    'A recording of melodic singing over music.',
    'A recording of spoken words over music.',
    'A recording of instrumental music without vocals.',
    'Music featuring a synthesizer and electronic drums.',
    'Music featuring an acoustic guitar.',
    'Music featuring a piano.',
    'Music featuring distorted electric guitars.',
    'Music featuring an acoustic drum kit and bass guitar.',
    'Music featuring bowed string instruments.',
    'A recording of energetic, driving music.',
    'A recording of gentle, relaxed music.',
)


class ClapBackend:
    """Loads an official pinned checkpoint lazily; never uses random weights."""
    model_id = MODEL_ID
    revision = MODEL_REVISION
    sample_rate = 48000
    window_seconds = 10.0

    def __init__(self):
        self.model = None
        self.diagnostics = None

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import ClapModel, ClapProcessor
        except ImportError as exc:
            raise RuntimeError(
                'CLAP requires the optional encoder dependencies. '
                'Install them with: pip install -r requirements-encoder.txt'
            ) from exc
        self.torch = torch
        processor = ClapProcessor.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False,
        )
        # The official pinned checkpoint supplies PyTorch weights. Modern torch
        # loads these in weights-only mode; no model repository code is executed.
        model = ClapModel.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False,
            use_safetensors=False, weights_only=True,
        ).eval()
        probes = processor.tokenizer(
            list(TEXT_HEALTH_PROBES), padding=True, truncation=True, return_tensors='pt',
        )
        with torch.inference_mode():
            vectors = model.get_text_features(**probes).cpu().float().numpy()
        health = check_text_encoder_health(vectors)
        health['probe_descriptions'] = list(TEXT_HEALTH_PROBES)
        # Publish the loaded model only after its text path passes. A failed
        # load must never leave a cached model that bypasses the next check.
        self.processor, self.model = processor, model
        self.diagnostics = {'text_encoder_health': health}

    def audio_embedding(self, samples):
        self._load()
        inputs = self.processor.feature_extractor(
            samples, sampling_rate=self.sample_rate, return_tensors='pt',
            padding='repeatpad', truncation='rand_trunc',
        )
        with self.torch.inference_mode():
            return self.model.get_audio_features(**inputs)[0].cpu().float().numpy()

    def text_embeddings(self, prompts):
        self._load()
        inputs = self.processor.tokenizer(
            list(prompts), padding=True, truncation=True, return_tensors='pt',
        )
        with self.torch.inference_mode():
            return self.model.get_text_features(**inputs).cpu().float().numpy()


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError('Encoder embeddings must be nonempty, finite vectors')
    norm = np.linalg.norm(vector)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError('Encoder embeddings must have a finite, nonzero norm')
    return vector / norm


def check_text_encoder_health(vectors):
    """Reject catastrophic collapse on fixed, deliberately different probes.

    This is a basic functionality check, not a measure of musical accuracy.
    It must not be run on user candidates, which may be intentional synonyms.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) < 3:
        raise ValueError('Text encoder health requires at least three probe vectors')
    normalized = np.stack([_unit(v) for v in vectors])
    similarities = np.clip(normalized @ normalized.T, -1., 1.)
    pairs = similarities[np.triu_indices(len(normalized), 1)]
    mean = float(pairs.mean())
    if mean >= TEXT_COLLAPSE_LIMIT:
        raise RuntimeError(
            f'Text encoder collapsed: distinct probe descriptions have mean '
            f'cosine similarity {mean:.6f}. Refusing to produce misleading matches.'
        )
    return {'status': 'passed', 'probe_count': len(normalized),
            'mean_pairwise_cosine': round(mean, 6),
            'max_pairwise_cosine': round(float(pairs.max()), 6),
            'collapse_threshold': TEXT_COLLAPSE_LIMIT,
            'interpretation': 'Basic noncollapse check; not an accuracy or confidence score'}


def _matches(vector, text_vectors, prompts):
    if text_vectors.shape[1] != len(vector):
        raise ValueError('Audio and text embedding dimensions do not match')
    scores = np.clip(text_vectors @ vector, -1., 1.)
    order = np.argsort(-scores, kind='stable')[:5]
    return [{'description': prompts[i], 'cosine_similarity': round(float(scores[i]), 6)}
            for i in order]


def encode_music(path, *, backend=None, prompts=DEFAULT_PROMPTS, max_duration=None):
    """Encode every requested sample in consecutive ten-second windows.

    Short final chunks use the model's documented repeat padding. Their real
    duration, not their padded length, determines their weight in the track
    embedding. Prompt rankings are hypotheses, never calibrated probabilities.
    """
    if max_duration is not None and (not math.isfinite(max_duration) or max_duration <= 0):
        raise ValueError('max_duration must be positive and finite')
    if isinstance(prompts, str) or not 1 <= len(prompts) <= 128 or any(
        not isinstance(p, str) or not p.strip() for p in prompts
    ):
        raise ValueError('Provide between 1 and 128 nonempty text descriptions')
    prompts = tuple(p.strip() for p in prompts)
    backend = backend if backend is not None else ClapBackend()
    chunks, vectors, weights = [], [], []
    text_vectors = None
    with sf.SoundFile(str(path)) as audio:
        source_rate = audio.samplerate
        source_channels = audio.channels
        total = len(audio)
        if max_duration is not None:
            total = min(total, int(max_duration * source_rate))
        if total <= 0:
            raise ValueError('No audio samples to encode')
        block_size = round(backend.window_seconds * source_rate)
        target_size = round(backend.window_seconds * backend.sample_rate)
        offset = 0
        while offset < total:
            frames = audio.read(min(block_size, total - offset), dtype='float32', always_2d=True)
            if not len(frames) or not np.isfinite(frames).all():
                raise ValueError('Audio decoder returned empty or non-finite samples')
            samples = frames.mean(axis=1)
            if source_rate != backend.sample_rate:
                samples = librosa.resample(samples, orig_sr=source_rate, target_sr=backend.sample_rate)
            if len(samples) > target_size:
                raise ValueError('Chunk exceeds the model window; refusing to truncate audio')
            start, end = offset / source_rate, (offset + len(frames)) / source_rate
            rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
            chunk = {'start_sec': round(start, 6), 'end_sec': round(end, 6),
                     'rms': round(rms, 7), 'silent': rms < 1e-5,
                     'embedding': None, 'matches': []}
            if not chunk['silent']:
                vector = _unit(backend.audio_embedding(samples))
                if text_vectors is None:
                    raw_text = np.asarray(backend.text_embeddings(prompts))
                    if raw_text.ndim != 2 or len(raw_text) != len(prompts):
                        raise ValueError('Encoder must return one vector per description')
                    text_vectors = np.stack([_unit(v) for v in raw_text])
                chunk['embedding'] = np.round(vector, 7).tolist()
                chunk['matches'] = _matches(vector, text_vectors, prompts)
                vectors.append(vector)
                weights.append(end - start)
            chunks.append(chunk)
            offset += len(frames)
            logging.getLogger(__name__).info('Encoded %.1f / %.1f seconds', end, total / source_rate)
    embedding = _unit(np.average(np.stack(vectors), axis=0, weights=weights)) if vectors else None
    return {
        'status': 'encoded' if vectors else 'no_audible_audio',
        'model': {'id': backend.model_id, 'revision': backend.revision},
        'backend_diagnostics': getattr(backend, 'diagnostics', None),
        'sample_rate': backend.sample_rate, 'channels': 1,
        'source_sample_rate': source_rate, 'source_channels': source_channels,
        'window_seconds': backend.window_seconds,
        'duration_sec': round(total / source_rate, 6),
        'encoded_duration_sec': round(sum(weights), 6),
        'embedding_dimension': len(embedding) if embedding is not None else None,
        'normalization': 'l2',
        'pooling': 'duration_weighted_mean_of_nonsilent_chunks_then_l2',
        'short_chunk_padding': 'repeatpad; weights use original duration',
        'embedding': np.round(embedding, 7).tolist() if embedding is not None else None,
        'chunks': chunks,
        'candidate_descriptions': list(prompts),
        'score_type': 'cosine_similarity_not_probability',
        'matches': _matches(embedding, text_vectors, prompts) if embedding is not None else [],
        'limitations': [
            'Rankings depend on the supplied candidate descriptions; labels are not verified.',
            'Cosine similarity is not a confidence percentage.',
            'This encoder does not transcribe lyrics or establish chords, key, or section labels.',
            'Stereo channels are averaged for model input; original audio remains preserved.',
        ],
    }


def compare_embeddings(first, second):
    """Compare only vectors produced by the same model and pooling convention."""
    if not first or not second or first.get('embedding') is None or second.get('embedding') is None:
        return {'status': 'not_available', 'cosine_similarity': None}
    compatible = ('model', 'normalization', 'pooling', 'sample_rate', 'window_seconds',
                  'embedding_dimension', 'channels', 'short_chunk_padding')
    for record in (first, second):
        model = record.get('model')
        if not isinstance(model, dict) or any(
            not isinstance(model.get(key), str) or not model[key].strip()
            for key in ('id', 'revision')
        ):
            return {'status': 'incompatible_models', 'cosine_similarity': None}
        if any(record.get(key) is None or record.get(key) == '' for key in compatible):
            return {'status': 'incompatible_models', 'cosine_similarity': None}
    if any(first.get(key) != second.get(key) for key in compatible):
        return {'status': 'incompatible_models', 'cosine_similarity': None}
    a, b = _unit(first['embedding']), _unit(second['embedding'])
    if a.shape != b.shape or len(a) != first['embedding_dimension'] or len(b) != second['embedding_dimension']:
        return {'status': 'incompatible_dimensions', 'cosine_similarity': None}
    return {'status': 'compared', 'cosine_similarity': round(float(np.clip(a @ b, -1., 1.)), 6),
            'interpretation': 'Similarity in this model embedding space, not proof of musical identity'}
