# subtimer

align an ordered Japanese transcript to audio. Python library only; no CLI, audio conversion, transcript cleanup, or subtitle formatting.

## install

requires Linux x86-64, Python 3.12, an NVIDIA GPU with BF16 support, a CUDA 13-compatible driver, and uv.

```bash
uv python install 3.12
uv sync --frozen
```

the two fitted checkpoints are bundled. on first use, pinned [CrisperWhisper](https://huggingface.co/nyralabs/CrisperWhisper2.0_large) (CC BY-NC 4.0) weights and the [Parakeet Japanese tokenizer](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) (CC BY 4.0) download automatically. allow several GB of disk space and internet access. downloads persist in `${XDG_CACHE_HOME:-~/.cache}/subtimer/models`; set `SUBTIMER_MODEL_DIR` before importing to override it.

## use

run Python with `uv run python` from the repository root:

```python
from subtimer.timing import align

pairs = align(["こんにちは。", "今日はいい天気ですね。"], "mono_16k.wav", print)
# [[start_ms, end_ms], ...], one integer pair per input line
```

supply nonempty, precleaned Japanese text lines in spoken order and nonempty mono 16 kHz audio readable by SoundFile (for example, WAV). remove annotations and unspoken lines yourself, and preserve any mapping to your original rows. the library does not transcribe or repair incorrect text.

`progress` is a required callable that receives `"match"` and `"refine"`; pass `lambda stage: None` for silence. exceptions propagate to the caller.

each call loads a fresh backbone and computes fresh features. calls change PyTorch's process-wide thread and determinism settings; run jobs serially or in isolated processes. the optional `backbone` argument consumes and mutates a `(processor, model)` from `load_backbone()`; never reuse that instance for another call.
