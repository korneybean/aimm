"""AIM v1: audio analysis, portable container and interpretable comparisons."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import soundfile as sf
import librosa

VERSION = '1.0.0'


def _round(value, places=4):
    value = float(value)
    return round(value, places) if math.isfinite(value) else None


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def analyze(path, *, hop_length=512, max_duration=None):
    """Analyze an audio file; descriptors are estimates, not semantic ground truth."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    info = sf.info(str(path))
    if info.frames == 0:
        raise ValueError('Empty audio file')
    y, sr = librosa.load(str(path), sr=22050, mono=True, duration=max_duration)
    if y.size == 0:
        raise ValueError('No audio samples decoded')
    duration = len(y) / sr
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    max_rms = float(rms.max())
    energy = [[_round(t, 3), _round(e / max_rms)] for t, e in zip(times, rms) if t <= duration]
    silent = max_rms < 1e-5
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=False)
    onsets = [_round(t, 3) for t in librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length) if t <= duration] if not silent else []
    tempo = None
    beats = []
    if not silent and len(onsets) >= 4:
        estimated_tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length, trim=False)
        candidate = float(np.asarray(estimated_tempo).reshape(-1)[0])
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        if len(beat_times) >= 3 and 35 <= candidate <= 240:
            tempo = _round(candidate, 2)
            beats = [_round(t, 3) for t in beat_times if t <= duration]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    # Chroma is a pitch-class salience profile, not chord recognition.
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    chroma_avg = np.mean(chroma, axis=1) if not silent else np.zeros(12)
    if chroma_avg.sum():
        chroma_avg = chroma_avg / chroma_avg.sum()
    pitch_classes = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    # Coarse novelty-based section boundaries; do not name verse/chorus without a model.
    sections = []
    n = len(rms)
    block_frames = max(1, int(2 * sr / hop_length))
    if n:
        blocks = [(i, float(np.mean(rms[i:i+block_frames]))) for i in range(0, n, block_frames)]
        changes = np.abs(np.diff([v for _, v in blocks]))
        threshold = float(np.median(changes) + 2.0 * np.std(changes)) if len(changes) else float('inf')
        boundaries = [0.0] + [min(duration, blocks[i+1][0] * hop_length / sr) for i, change in enumerate(changes) if change > threshold and change > max_rms * 0.08] + [duration]
        boundaries = sorted(set(boundaries))
        sections = [{'start': _round(a, 3), 'end': _round(b, 3), 'label': 'segment', 'method': 'energy_novelty'} for a, b in zip(boundaries, boundaries[1:]) if b > a]
    features = {
        'energy_curve': energy,
        'onset_candidates_sec': onsets,
        'tempo_bpm': tempo,
        'beats_sec': beats,
        'sections': sections,
        'spectral_centroid_hz_mean': _round(np.mean(centroid)),
        'zero_crossing_rate_mean': _round(np.mean(zcr)),
        'pitch_class_salience': {k: _round(v) for k, v in zip(pitch_classes, chroma_avg)},
        'harmony': None,
        'instruments': [],
        'mood_tags': [],
        'lyrics': None,
    }
    return {
        'format': 'AIFM', 'version': VERSION,
        'audio': {'uri': path.name, 'sha256': _hash(path), 'duration_sec': _round(duration), 'source_duration_sec': _round(info.duration), 'sample_rate': info.samplerate, 'channels': info.channels, 'analyzed_sample_rate': sr, 'partial_analysis': max_duration is not None and info.duration > duration + 0.01},
        'features': features,
        'provenance': {'energy_curve': 'measured normalized RMS', 'onset_candidates_sec': 'librosa spectral flux onset detector', 'tempo_bpm': 'librosa beat tracker, estimated', 'beats_sec': 'librosa beat tracker, estimated', 'sections': 'heuristic RMS novelty, unlabeled', 'spectral_centroid_hz_mean': 'measured spectral descriptor', 'zero_crossing_rate_mean': 'measured waveform descriptor', 'pitch_class_salience': 'mean STFT chroma, not chords or key'},
        'uncertainty': {'tempo_bpm': 'unknown' if tempo is None else 'estimated; half/double-time ambiguity', 'sections': 'heuristic boundaries; no semantic section labels', 'harmony': 'not inferred', 'instruments': 'not inferred', 'mood_tags': 'not inferred', 'lyrics': 'not transcribed'}
    }


def validate(data):
    if not isinstance(data, dict) or data.get('format') != 'AIFM' or not isinstance(data.get('features'), dict):
        raise ValueError('Invalid AIFM document')
    audio = data.get('audio', {})
    if not isinstance(audio, dict) or not isinstance(audio.get('sha256'), str) or len(audio['sha256']) != 64:
        raise ValueError('Missing audio SHA-256')
    if not isinstance(audio.get('duration_sec'), (int, float)) or audio['duration_sec'] <= 0:
        raise ValueError('Invalid audio duration')
    return True


def pack(audio_path, destination, *, data=None):
    """Create a ZIP-based .aim container with embedded original audio and JSON."""
    audio_path = Path(audio_path)
    data = analyze(audio_path) if data is None else data
    validate(data)
    if _hash(audio_path) != data['audio']['sha256']:
        raise ValueError('Audio fingerprint mismatch')
    destination = Path(destination)
    if destination.resolve() == audio_path.resolve():
        raise ValueError('Container cannot overwrite source audio')
    manifest = {'format': 'AIM', 'version': VERSION, 'aifm': 'aifm.json', 'audio': 'audio/' + audio_path.name}
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        z.writestr('manifest.json', json.dumps(manifest, indent=2))
        z.writestr('aifm.json', json.dumps(data, indent=2))
        z.write(audio_path, manifest['audio'])
    return destination


def unpack(container, destination):
    """Verify and extract an AIM container, protecting against path traversal."""
    destination = Path(destination)
    with zipfile.ZipFile(container) as z:
        manifest = json.loads(z.read('manifest.json'))
        if manifest.get('format') != 'AIM' or manifest.get('aifm') != 'aifm.json':
            raise ValueError('Invalid AIM manifest')
        audio_member = manifest.get('audio')
        if not isinstance(audio_member, str) or not audio_member.startswith('audio/') or Path(audio_member).name != audio_member.split('/')[-1] or audio_member.count('/') != 1 or '\\' in audio_member:
            raise ValueError('Unsafe audio path in manifest')
        data = json.loads(z.read('aifm.json'))
        validate(data)
        h = hashlib.sha256()
        with z.open(audio_member) as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                h.update(chunk)
        if h.hexdigest() != data['audio']['sha256']:
            raise ValueError('Container audio fingerprint mismatch')
        destination.mkdir(parents=True, exist_ok=True)
        audio_dest = destination / Path(audio_member).name
        if audio_dest.exists():
            raise FileExistsError(audio_dest)
        with z.open(audio_member) as src, audio_dest.open('wb') as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(chunk)
        (destination / 'aifm.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return audio_dest


def compare(a, b):
    """Grounded comparison of measurable descriptors; no invented emotion."""
    validate(a)
    validate(b)
    af, bf = a['features'], b['features']
    def mean_energy(f):
        return float(np.mean([point[1] for point in f['energy_curve']])) if f['energy_curve'] else 0.0
    descriptors = {
        'mean_normalized_energy': (mean_energy(af), mean_energy(bf)),
        'spectral_centroid_hz_mean': (af.get('spectral_centroid_hz_mean'), bf.get('spectral_centroid_hz_mean')),
        'zero_crossing_rate_mean': (af.get('zero_crossing_rate_mean'), bf.get('zero_crossing_rate_mean')),
        'tempo_bpm': (af.get('tempo_bpm'), bf.get('tempo_bpm')),
    }
    return {'comparison_type': 'measured_and_estimated_descriptors', 'tracks': [a['audio']['uri'], b['audio']['uri']], 'descriptors': {key: {'a': x, 'b': y, 'difference_b_minus_a': _round(y-x) if x is not None and y is not None else None} for key, (x, y) in descriptors.items()}, 'limitations': ['Normalized energy is not absolute loudness', 'Spectral brightness is not emotion', 'Tempo is an estimate', 'No inferred emotional or lyrical meaning']}


def main():
    p = argparse.ArgumentParser(description='AIM: grounded music feature analysis and portable containers')
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('analyze'); a.add_argument('audio'); a.add_argument('-o', '--output'); a.add_argument('--max-duration', type=float)
    pk = sub.add_parser('pack'); pk.add_argument('audio'); pk.add_argument('-o', '--output')
    up = sub.add_parser('unpack'); up.add_argument('container'); up.add_argument('-d', '--destination', required=True)
    co = sub.add_parser('compare'); co.add_argument('first'); co.add_argument('second'); co.add_argument('-o', '--output')
    args = p.parse_args()
    if args.command == 'analyze':
        result = analyze(args.audio, max_duration=args.max_duration)
        output = Path(args.output or str(Path(args.audio).with_suffix('.aifm.json')))
    elif args.command == 'pack':
        output = pack(args.audio, args.output or str(Path(args.audio).with_suffix('.aim')))
        print(f'Wrote {output}'); return
    elif args.command == 'unpack':
        print(f'Extracted {unpack(args.container, args.destination)}'); return
    else:
        result = compare(json.loads(Path(args.first).read_text()), json.loads(Path(args.second).read_text()))
        output = Path(args.output or 'comparison.json')
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
