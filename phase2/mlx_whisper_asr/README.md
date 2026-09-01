# MLX Whisper ASR helper

This Phase 2b helper is separate from the frozen Phase 1 environment and the
Phase 2a OCR runtime. It runs locally on Apple Silicon through MLX, accepts one
bounded PCM WAV input through the `frameledger-asr-helper-v1` JSON protocol, and
returns raw segment and word timestamps. It does not call a transcription API.

## Isolated environment

```bash
uv venv phase2/mlx_whisper_asr/.venv --python 3.12
UV_CACHE_DIR=.cache/uv uv pip install \
  --python phase2/mlx_whisper_asr/.venv/bin/python \
  -r phase2/mlx_whisper_asr/requirements.lock.txt
chmod +x phase2/mlx_whisper_asr/run
```

`ffmpeg` must also be available. It is a Phase 2b extraction dependency only;
the frozen Phase 1 OpenCV baseline remains FFmpeg-optional.

## Model acquisition

Download one explicitly pinned MLX model revision into the ignored local cache:

```bash
phase2/mlx_whisper_asr/.venv/bin/hf download \
  mlx-community/whisper-large-v3-turbo \
  --revision a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb \
  --local-dir .cache/models/whisper-large-v3-turbo
```

The transcription helper accepts only an existing local model directory and
sets Hugging Face/Transformers offline flags before importing the runtime. The
model file list, byte sizes, individual SHA-256 values, and aggregate tree hash
are included in every successful response.

## Protocol

```json
{
  "protocol": "frameledger-asr-helper-v1",
  "audio_path": "/absolute/path/audio.wav",
  "model_path": "/absolute/path/whisper-large-v3-turbo",
  "language": "zh",
  "task": "transcribe",
  "word_timestamps": true,
  "decoding_profile": "standard_fallback_v1",
  "initial_prompt": "optional exact local context"
}
```

`decoding_profile` is either the compatibility profile `fixed_zero_v1` or
`standard_fallback_v1`. The latter uses Whisper's `0.0–1.0` temperature
fallback sequence with compression-ratio, log-probability, and no-speech
thresholds. When `initial_prompt` is present, it is trimmed, limited to 1000
non-control characters, and its SHA-256 and character count are echoed by the
helper. The orchestrator fails closed if that echo or the requested decoding
profile does not match.

The helper receives no command-line arguments. Standard output is one strict
JSON response containing `engine` and `result`. Errors are JSON on standard
error with an empty standard output. `verbose=None` prevents MLX Whisper from
printing progress or transcript text into the protocol channel.

Successful outputs must also pass the orchestrator's transcript-quality gate.
It rejects segment compression ratios above 2.4, three consecutive identical
non-empty segments, repeated 32-character spans inside one segment, and Unicode
replacement characters. Rejected raw transcripts remain in the failure ledger
and do not receive a `review.html` that could be mistaken for accepted evidence.

Chinese word-level timestamps reflect tokenizer/alignment units rather than a
guaranteed linguistic word segmentation. Segment timestamps are the primary
Phase 2b evidence surface.
