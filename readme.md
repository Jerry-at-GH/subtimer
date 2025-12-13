# subtimer

it times Japanese subtitles automatically with forced alignment and VAD.

## requirements

first get an Nvidia GPU. then install `python` (>=3.10), `ffmpeg`, `libc++1` (if on Linux). now run (preferably in a new `conda` environment; you might need to `conda install -c conda-forge libstdcxx-ng` first in that case)

1. `pip install -U --force-reinstall git+https://github.com/TEN-framework/ten-vad`
2. `pip install -U 'audio-separator[gpu]' javad librosa "nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo" numpy pysubs2 scipy silero-vad torch 'torchaudio<2.9'`

check each library's documentation for details. things may have changed since last update of this document. [this commit](https://github.com/NVIDIA-NeMo/NeMo/commit/a599d89f66abeb5f61c99c73902567fda7fae76a) of `nemo_toolkit` was tested.

## usage

assume you want to work under `/some/path`, and you've put script `1.txt` and video `1.mkv` in that directory. first clone this repository and enter its directory, then run

```bash
python align.py --work_dir /some/path --lines 1.txt --video 1.mkv --output align.ass
```

and you'll find `align.ass` in `/some/path`. (`--video` and `--output` are optional; default values shown above.)

### input script format

although it's easy to customize how the program handle script files (`align.py` > `load_lines()`), by default it should be either TXT or ASS. if it's TXT, it should follow a syntax like

```
JAPANESE	TRANSLATION
(NOTES HERE
```

where in dialogue lines original text and translation are separated by a tab, and note lines start with `(`. if it's ASS, it should only contains Japanese dialogue and have been approximately timed to the video (the program prevents output that deviates too much from the original timing).

### evaluation

the program aims to provide timings that are accurately at the start and end of human speech (i.e., no lead-in or lead-out). if you have accurate manual timings `ground_truth.ass` that meet the same criteria, you can use them to evaluate the program's output by running

```bash
python evaluate.py --work_dir /some/path --lines 1.txt --video 1.mkv --ground ground_truth.ass --fps 23.97602
```

and you should see something like

```
[total 293]
      | med err | mean abs err |    ≤3    ≤6   ≤12 frames
start |    +0ms |        380ms |   216   235   244
end   |    +0ms |        781ms |   182   212   231
both  |                        |   149   182   206
```

## credits

this project contains code from [NeMo Forced Aligner](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tools/nemo_forced_aligner).
