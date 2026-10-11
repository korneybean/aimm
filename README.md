# AIM / AIFM v1.1.1 — audio analysis and a pretrained music encoder

AIM is a portable `.aim` ZIP container holding original audio, an `aifm.json` structured analysis and `manifest.json`. AIFM contains measured signal features and, when enabled, pretrained music embeddings and ranked audio-description matches. Source audio is preserved byte-for-byte and verified using SHA-256.

## Install and run

Python 3.10+:

```bash
pip install -r requirements.txt
python aim.py analyze song.wav
python aim.py pack song.wav -o song.aim
python aim.py unpack song.aim -d extracted
python aim.py compare first.aifm.json second.aifm.json
python -m unittest discover -s tests -v
```

Supports audio formats decoded by `soundfile` (e.g. WAV/FLAC/OGG; MP3 availability depends on installed libsndfile). Analysis uses a 22.05 kHz mono copy; original bytes remain intact. Analysis may be slow for long recordings.

## Pretrained music encoder

Enable the optional [LAION general CLAP model](https://huggingface.co/laion/larger_clap_general), trained on general audio, music, and speech. CLAP learns a shared representation of audio and text, allowing audio embeddings to be compared with candidate descriptions. This integration uses the official pretrained checkpoint, not randomly initialized weights.

Version 1.1.1 replaces the previous `larger_clap_music` checkpoint. Its text representations collapsed across different descriptions, a [separately reported model-specific failure](https://github.com/embeddings-benchmark/mteb/issues/5069) reproduced in our real-model tests. Earlier description rankings from that checkpoint should not be used. Reanalyze the original recordings to obtain new embeddings; vectors from the two checkpoints are incompatible.

Install the optional dependencies. For CPU-only use, install PyTorch from its CPU wheel index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-encoder.txt
python aim.py analyze song.mp3 --encoder clap -o song.aifm.json
python aim.py pack song.mp3 --encoder clap -o song.aim
python aim.py compare first.aifm.json second.aifm.json
```

The first encoder run downloads and caches the model. Subsequent inference runs locally; audio is not sent to Hugging Face or LAION. Set `HF_HUB_OFFLINE=1` after the checkpoint is cached to require offline operation. Basic analysis still works with only `requirements.txt`; it defaults to `--encoder none`.

- Model: `laion/larger_clap_general`, revision `ada0c23a36c4e8582805bb38fec3905903f18b41` (Apache-2.0).
- Before a loaded model is used, four deliberately different text probes must have mean pairwise cosine similarity below 0.98. This catches near-constant output; it is not an accuracy test or a confidence threshold. Diagnostics and probe descriptions are saved in AIFM. A failed check stops encoding instead of returning misleading rankings.
- Model input: 48 kHz mono audio, streamed in consecutive ten-second chunks. Every requested sample is covered, including the final short chunk.
- Output: 512-dimensional L2-normalized embeddings for audible chunks, exact start/end timestamps, and a duration-weighted normalized track embedding.
- Final short chunks use the model's repeat padding; pooling weights use their real duration. Effectively silent chunks retain timestamps and receive no description matches.
- Description matching: cosine similarity against the explicit candidate descriptions saved in AIFM. The top five are rankings within that candidate set, **not confidence percentages or verified labels**.
- `analyze --max-duration SECONDS --encoder clap` applies the same duration limit to signal analysis and the encoder.
- `compare` adds a music-embedding cosine score only for compatible model revisions, dimensions, and preprocessing/pooling conventions. Similarity is not proof that two recordings are musically identical.

Encoder data appears under `features.music_encoder`:

```text
model: id + pinned revision
backend_diagnostics: text_encoder_health
duration_sec + encoded_duration_sec
sample_rate + source_sample_rate + source_channels
embedding_dimension + normalization + pooling
embedding: normalized track vector
chunks: start_sec, end_sec, rms, silent, embedding, matches
candidate_descriptions + score_type + matches + limitations
```

Custom candidate descriptions are supported through the Python API:

```python
from aim import analyze
data = analyze('song.mp3', encoder='clap', encoder_prompts=[
    'A recording of rhythmic rapping over electronic music.',
    'A recording of melodic singing over acoustic guitar.',
])
```

Candidate choice affects the result. Including only two descriptions forces a comparison between those descriptions; neither is necessarily correct. The encoder does not transcribe lyrics or confirm instruments, key, chords, moods, or verse/chorus structure. Those legacy fields remain unknown rather than being populated from an uncalibrated match.

## Verification

```bash
python -m unittest discover -s tests -v
```

The default suite needs only the base dependencies. It tests analysis, original-byte archive recovery, wrong fingerprints, encoder chunk coverage, short-tail weighting, partial analysis, silence, finite embeddings, AIFM preservation, model compatibility, and text-collapse rejection. Deterministic fixture vectors isolate data handling from neural inference; they are not evidence of model accuracy.

Run the separate pretrained checks with the optional dependencies installed:

```bash
AIMM_RUN_MODEL_TESTS=1 python -m unittest discover -s tests -v
```

These checks exercise distinct descriptions, verify that three documented recordings rank their correct description above the other two, and confirm that the old collapsed checkpoint is rejected. The first run downloads both pinned checkpoints and three [librosa example recordings](https://librosa.org/doc/latest/recordings.html): `libri1` (Garth Comira narration, CC-BY 4.0), `pistachio` (Lena Orsa solo piano, CC-BY-NC 3.0), and `trumpet` (Mihai Sorohan, CC-BY 3.0). Recordings are used locally and are not bundled in this repository. These are small regression checks, not a benchmark of general music understanding.

## Implemented

- Audio metadata and cryptographic fingerprint
- Time-aligned RMS energy and candidate note/transient onsets
- Estimated beat positions and tempo when evidence is sufficient
- Spectral centroid and zero-crossing-rate descriptors
- Pitch-class salience (not a chord transcription)
- Heuristic unlabeled segmentation using energy changes
- ZIP-based AIM packaging, fingerprint-verified extraction, path safety checks
- JSON-based grounded comparison of two tracks
- Provenance and explicit unknowns for unsupported semantic claims
- Optional pretrained CLAP music embeddings and timestamped description rankings

## Boundaries

**This remains a reference implementation with an optional pretrained encoder.** It does not train an encoder or call a language model. CLAP adds learned music representations and text-audio matching; it does not provide a lyric transcript or independently validated musical labels. A tempo estimate can be half- or double-time. Spectral brightness and normalized energy cannot establish emotional meaning. Unlabeled energy-based segments are not song structure. Both analysis paths mix stereo to mono, so their features omit stereo placement. For a production release, add calibrated task-specific models, adversarial archive limits, schema evolution, and evaluation against annotated music.

Origin: March 9, 2026 AIM/AIFM concept; first code prototype October 10, 2026.
