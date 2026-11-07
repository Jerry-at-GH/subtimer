# subtimer

automatic subtitle timing for Japanese audio.

give subtimer a recording and its transcript, and it estimates when each line starts and ends, so you only need to review and adjust the timings instead of setting each one by hand.

in our tests on typical Japanese speech, more than 95% of line starts and ends fall within 100 ms of reference timings.

questions are welcome at admin@j-23.top.

## try it online

the easiest way to try subtimer is the website, https://subtimer.j-23.top. upload a video or audio file, paste in the transcript, and you'll get an ASS subtitle file back. use this user key:

```
DFE4DVVKQIAXwptYraG799SUPUeWgmfb6sHoBBcf1tV1pt9SuaogwKVCMRITsPLpISYWuN8pgHSF8pFdpMkZat
```

the trial key is shared and has a daily limit on audio minutes. feel free to [email me](mailto:admin@j-23.top) for a separate allocation.

## run it locally

this repository is only the timing step, as a Python library. you give it cleaned-up lines and audio, and it gives back start and end times. turning media into audio, cleaning up the transcript and writing subtitle files are up to you, which makes it easy to drop into a workflow you already have.

a coding agent can handle the setup, audio conversion and transcript cleanup for you; point it at this README.

### install

requires Linux x86-64, Python 3.12, an NVIDIA GPU with BF16 support, a CUDA 13-compatible driver, and uv.

```bash
uv python install 3.12
uv sync --frozen
```

the two fitted checkpoints are bundled. on first use, pinned [CrisperWhisper](https://huggingface.co/nyralabs/CrisperWhisper2.0_large) (CC BY-NC 4.0) weights and the [Parakeet Japanese tokenizer](https://huggingface.co/nvidia/parakeet-tdt_ctc-0.6b-ja) (CC BY 4.0) download automatically, which takes several GB. they are cached in `${XDG_CACHE_HOME:-~/.cache}/subtimer/models`; set `SUBTIMER_MODEL_DIR` before importing to change that.

### use

```python
from subtimer.timing import align

pairs = align(["こんにちは。", "今日はいい天気ですね。"], "mono_16k.wav", print)
# [[start_ms, end_ms], ...], one pair per line
```

run it with `uv run python` from the repository root.

- **lines**: nonempty Japanese text, one entry per spoken line, in order. leave out annotations and lines nobody says.
- **audio**: mono 16 kHz, in any format SoundFile reads, such as WAV.
- **progress**: called with `"match"` and `"refine"`; pass `lambda stage: None` to stay quiet.

each call loads the model afresh and changes PyTorch's process-wide thread and determinism settings, so run one job at a time per process.
