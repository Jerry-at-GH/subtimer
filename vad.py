import os
import subprocess

import numpy as np
import pysubs2
from scipy.signal import resample
import torch


FRAME_HOP_SEC = 0.01  # 10ms frames
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _soften(x, k):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return (x**k) / (x**k + (1 - x) ** k)


def javad_vad(audio_16k_path, vad_dir):
    from javad import Processor
    from silero_vad import read_audio

    processor = Processor(model_name="precise", device=DEVICE)
    wav = read_audio(audio_16k_path, sampling_rate=16000)
    javad = processor.logits(wav).cpu().numpy()
    np.save(os.path.join(vad_dir, "javad.npy"), javad)
    return javad


def silero_vad(audio_16k_path, vad_dir):
    from silero_vad import load_silero_vad, read_audio

    model = load_silero_vad()
    wav = read_audio(audio_16k_path, sampling_rate=16000)
    silero = model.audio_forward(wav, sr=16000).numpy().flatten()
    np.save(os.path.join(vad_dir, "silero.npy"), silero)
    return silero


def ten_vad(audio_16k_path, vad_dir):
    from ten_vad import TenVad
    import scipy.io.wavfile as Wavfile

    _, data = Wavfile.read(audio_16k_path)
    hop_size = 160  # 16000Hz / (1s / 10ms)
    ten_vad_instance = TenVad(hop_size, 0.5)
    num_frames = data.shape[0] // hop_size
    ten = np.array([ten_vad_instance.process(data[i * hop_size : (i + 1) * hop_size])[0] for i in range(num_frames)])[1:]
    np.save(os.path.join(vad_dir, "ten.npy"), ten)
    return ten


def heuristic_vad(video_path, vad_dir):
    import shutil
    from audio_separator.separator import Separator
    import librosa

    ten = np.load(os.path.join(vad_dir, "ten.npy"))
    separator = Separator(model_file_dir=os.path.expanduser("~/audio-separator-models"))
    separator.load_model(model_filename="mel_band_roformer_vocals_fv6_gabox.ckpt")

    input_file = video_path
    separator.separate(input_file, {"Vocals": "vocals", "Instrumental": "inst"})
    shutil.move("vocals.wav", os.path.join(vad_dir, "vocals.wav"))
    shutil.move("inst.wav", os.path.join(vad_dir, "inst.wav"))
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            os.path.join(vad_dir, "vocals.wav"),
            "-ac",
            "1",
            "-ar",
            "16000",
            os.path.join(vad_dir, "vocals_mono_16k.wav"),
        ],
        capture_output=True,
    )

    y, sr = librosa.load(os.path.join(vad_dir, "vocals_mono_16k.wav"))
    import librosa.feature

    rms = librosa.feature.rms(y=y)[0]
    rms_downsampled = np.clip(resample(rms, len(ten)), 0, rms.max())

    rolloff = librosa.feature.spectral_rolloff(y=y + 0.1, sr=sr)[0]
    rolloff_downsampled = np.clip(resample(rolloff, len(ten)), 0, rolloff.max())

    heuristic = rms_downsampled * rolloff_downsampled
    heuristic = (heuristic - heuristic.min()) / (heuristic.max() - heuristic.min())
    np.save(os.path.join(vad_dir, "heuristic.npy"), heuristic)
    return heuristic


def ensemble_from_dir(vad_dir):
    """Ensemble VAD arrays from directory (ten, javad, silero, heuristic with softening)."""
    ten = np.load(os.path.join(vad_dir, "ten.npy"))
    stacks = [
        ten,
        _soften(np.clip(resample(np.load(os.path.join(vad_dir, "javad.npy")), len(ten)), 0, 1), 0.2),
        _soften(np.clip(resample(np.load(os.path.join(vad_dir, "silero.npy")), len(ten)), 0, 1), 0.4),
        _soften(
            np.clip(resample(np.clip((np.load(os.path.join(vad_dir, "heuristic.npy")) - 0.005), 0, 0.1) / 0.1, len(ten)), 0, 1),
            3,
        ),
    ]
    return np.clip(np.mean(np.stack(stacks, axis=1), axis=1), 0, 1)


def pred_to_vad(pred, onset=0.43, offset=0.43, min_on_sec=0.08, min_off_sec=0.08):
    min_on_frames = round(min_on_sec / FRAME_HOP_SEC)
    min_off_frames = round(min_off_sec / FRAME_HOP_SEC)

    # hysteresis threshold
    vad_labels = np.zeros_like(pred, dtype=bool)
    in_speech = False
    for i, val in enumerate(pred):
        if not in_speech:
            if val >= onset:
                in_speech = True
                vad_labels[i] = True
        else:
            if val < offset:
                in_speech = False
            else:
                vad_labels[i] = True

    # utility to get contiguous (start, end) from boolean array
    def get_segments(labels):
        segs = []
        start = None
        for i, val in enumerate(labels):
            if val and start is None:
                start = i
            elif not val and start is not None:
                segs.append((start, i - 1))
                start = None
        if start is not None:
            segs.append((start, len(labels) - 1))
        return segs

    # remove short speech segments
    segments = get_segments(vad_labels)
    filtered = []
    for start, end in segments:
        if end - start >= min_on_frames:
            filtered.append((start, end))

    # merge short non-speech gaps
    merged = []
    for seg in filtered:
        if not merged:
            merged.append(seg)
        else:
            prev_start, prev_end = merged[-1]
            if seg[0] - prev_end < min_off_frames:
                merged[-1] = (prev_start, seg[1])  # merge
            else:
                merged.append(seg)

    binary_pred = np.zeros_like(vad_labels, dtype=bool)
    for s, e in merged:
        binary_pred[s : e + 1] = True

    return merged, binary_pred


def to_srt(merged, srt_file):
    subs = pysubs2.SSAFile()
    for start, end in merged:
        subs.append(pysubs2.SSAEvent(start=int(start * 1000 * FRAME_HOP_SEC), end=int(end * 1000 * FRAME_HOP_SEC), text="VAD"))
    subs.save(srt_file)


def run_vad(work_dir, video_path):
    from contextlib import redirect_stdout, redirect_stderr
    import logging
    import warnings

    audio_16k_path = os.path.join(work_dir, "mono_16k.wav")
    assert os.path.exists(video_path)
    assert os.path.exists(audio_16k_path)

    vad_dir = os.path.join(work_dir, "vad")
    os.makedirs(vad_dir, exist_ok=True)

    if (
        not os.path.exists(os.path.join(vad_dir, "ten.npy"))
        or not os.path.exists(os.path.join(vad_dir, "javad.npy"))
        or not os.path.exists(os.path.join(vad_dir, "silero.npy"))
        or not os.path.exists(os.path.join(vad_dir, "heuristic.npy"))
    ):
        print("running VAD")

    with redirect_stdout(open(os.devnull, "w")), redirect_stderr(open(os.devnull, "w")):
        warnings.filterwarnings("ignore")
        logging.getLogger().setLevel(logging.ERROR)
        if not os.path.exists(os.path.join(vad_dir, "ten.npy")):
            ten_vad(audio_16k_path, vad_dir)
        if not os.path.exists(os.path.join(vad_dir, "javad.npy")):
            javad_vad(audio_16k_path, vad_dir)
        if not os.path.exists(os.path.join(vad_dir, "silero.npy")):
            silero_vad(audio_16k_path, vad_dir)
        if not os.path.exists(os.path.join(vad_dir, "heuristic.npy")):
            heuristic_vad(audio_16k_path, vad_dir)

    pred = ensemble_from_dir(vad_dir)
    merged, _ = pred_to_vad(pred)
    merged_ms = [(s * FRAME_HOP_SEC * 1000, e * FRAME_HOP_SEC * 1000) for s, e in merged]
    return merged_ms
