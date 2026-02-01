# subtimer

it times Japanese subtitles automatically with forced alignment and VAD.

## requirements

first get an Nvidia GPU. then install `libc++1` (if on Linux). now run

```
conda create -n speech python=3.12 -y
conda activate speech
conda install -c conda-forge ffmpeg libstdcxx-ng -y
pip install -U --force-reinstall git+https://github.com/TEN-framework/ten-vad 'nemo_toolkit[asr]@git+https://github.com/NVIDIA/NeMo' 'audio-separator[gpu]' javad librosa numpy pysubs2 scipy silero-vad torch torchaudio
```

where `uv` is optional.

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

where in dialogue lines original text and translation are separated by a tab, and note lines start with `(`. if it's ASS, it should only contains Japanese dialogue and have been approximately timed to the video (in which case you can pass in `--do-check-with-raw` to prevent output from deviating too much from the original timing).

### evaluation

the program aims to provide timings that are accurately at the start and end of human speech (i.e., no lead-in or lead-out). if you have accurate manual timings `ground_truth.ass` that meet the same criteria, you can use them to evaluate the program's output by running

```bash
python evaluate.py --work_dir /some/path --lines 1.txt --video 1.mkv --ground ground_truth.ass --fps 23.97602
```

and you should see something like

```
[total: 297 lines]
      |   p10   p25   p50   p75   p90    | mean abs |   ≤100   ≤500 ms
start |    +0    +0    +0    +0   +30 ms |    106ms |  92.3%  95.6%
end   |  -447   -10    +0    +0    +0 ms |    628ms |  75.4%  87.2%
both  |                                  |    367ms |  71.0%  85.2%
```

## credits

this project contains code from [NeMo Forced Aligner](https://github.com/NVIDIA-NeMo/NeMo/tree/main/tools/nemo_forced_aligner).
