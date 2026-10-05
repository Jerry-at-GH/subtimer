# subtimer (WIP)

alignment of an ordered Japanese transcript to audio. the worker returns one start/end timing per transcript line, in milliseconds. Python source, fitted models, the Linux native encoder, its build sources and third-party licenses are in `subtimer/`.

## install

requires Linux x86-64, an NVIDIA GPU with a CUDA 13-compatible driver, Python 3.12 and `uv`. on Ubuntu 24.04, install the system dependencies and locked Python packages from the repository root:

```bash
sudo apt-get install git ffmpeg espeak-ng libespeak-ng1 libc++1 libc++abi1 libgomp1 libstdc++6 build-essential
uv python install 3.12
uv sync --frozen
```

the external CrispASR libraries and Parakeet model are **not bundled or downloaded automatically**. set the model directory and provision this layout before running:

```bash
export SUBTIMER_MODEL_DIR="$PWD/.cache"
```

```text
.cache/crispasr/
  crispasr-python-linux-x86_64/crispasr/
    libcrispasr.so
    libggml*.so*
  native/
    libcrispasr.so.0.8.24
    libggml-base.so.0
    libggml-cpu.so.0
    libggml-cuda.so.0
    libggml.so.0
  parakeet-tdt_ctc-0.6b-ja-f16.gguf
  parakeet-tdt_ctc-0.6b-ja-tokenizer.model
```

the existing ASR library bundle uses CrispASR 0.8.6. preserve each library bundle's dependencies and symlinks. the native alignment library must be CrispASR 0.8.24 built with CUDA support for your GPU; the bundled encoder is pinned to it. other upstream models download on first use and need internet access and disk space. the fitted models are already included.

to rebuild the encoder, install `g++` and `patch`, then run:

```bash
uv run --frozen python subtimer/native/build.py "$SUBTIMER_MODEL_DIR/crispasr/native"
```

## use

run from the repository root. create `job.json`, replacing the audio path and transcript with your own:

```json
{
  "audio": "/absolute/path/to/audio.wav",
  "dialogues": ["こんにちは。", "今日はいい天気ですね。"],
  "guide_segments": []
}
```

audio can be any format FFmpeg decodes. `dialogues` must be in spoken order and each line must contain alignable text. optional guide entries have the shape `{"text": "こんにちは。", "start": 1000, "end": 2500}`, with times in milliseconds; use an empty array when there is no guide.

```bash
uv run --frozen subtimer-worker < job.json > reports.jsonl
```

stage changes and the final result are JSON lines on stdout; library logs go to stderr. on success, the final report has `"status": "done"` and a `timings` array of `{"start": ..., "end": ...}` objects in transcript order. on failure, the exception propagates and the worker exits nonzero with a traceback on stderr. the output is timing data; subtitle-file formatting belongs to the caller.

each job uses a temporary workspace that is deleted on completion or failure. downloaded models remain available between runs.
