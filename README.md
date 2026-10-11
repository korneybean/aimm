# AIM / AIFM v1.0 — runnable reference implementation

AIM is a portable `.aim` ZIP container holding original audio, an `aifm.json` structured analysis and `manifest.json`. AIFM is the JSON representation of measured and estimated music features. Source audio is preserved byte-for-byte and verified using SHA-256.

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

## Boundaries

**This is an end-to-end reference implementation, not a full music-perception model.** It cannot reliably identify instruments, chords, key, lyrics, moods, or semantic verse/chorus labels. It does not train an encoder or a language model. A tempo estimate can be half- or double-time. Spectral brightness and normalized energy cannot establish emotional meaning. Unlabeled energy-based segments are not song structure. For a production release, add calibrated ML models, adversarial archive limits, licensing/privacy safeguards, schema evolution, and larger-scale evaluation on real music.

Origin: March 9, 2026 AIM/AIFM concept; first code prototype October 10, 2026.
