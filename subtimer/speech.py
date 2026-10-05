import ctypes
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import shutil
import statistics
import subprocess
import tempfile
import traceback
import warnings
import wave
from collections import defaultdict
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import cache as cached
from itertools import pairwise
from pathlib import Path

from . import cache, gpu
from .subtitle import MIN_DURATION, Span, Word

SAMPLE_RATE = 16_000
CHUNK_DURATION = 30_000
FRAME_HOP = 10
CTC_FRAME_DURATION = 80
ATT_CONTEXT = (64, 64)

DEVELOPMENT_MODELS = Path(__file__).resolve().parents[1] / ".cache"
MODEL_DIR = (
    Path(os.environ["SUBTIMER_MODEL_DIR"]).expanduser().resolve()
    if os.environ.get("SUBTIMER_MODEL_DIR")
    else DEVELOPMENT_MODELS
    if DEVELOPMENT_MODELS.exists()
    else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    .expanduser()
    .resolve()
    / "subtimer"
    / "models"
)
CRISPASR = MODEL_DIR / "crispasr"
ASR_LIBRARY = CRISPASR / "crispasr-python-linux-x86_64/crispasr/libcrispasr.so"
NFA_LIBRARY = CRISPASR / "native/libcrispasr.so.0.8.24"
NFA_DEPENDENCIES = [
    CRISPASR / "native" / name
    for name in (
        "libggml-base.so.0",
        "libggml-cpu.so.0",
        "libggml-cuda.so.0",
        "libggml.so.0",
    )
]
PARAKEET_MODEL = CRISPASR / "parakeet-tdt_ctc-0.6b-ja-f16.gguf"
PARAKEET_TOKENIZER = CRISPASR / "parakeet-tdt_ctc-0.6b-ja-tokenizer.model"
ENCODER_LIBRARY = Path(__file__).with_name("native") / "encoder.so"
MODELS = Path(__file__).with_name("models")
CONFIG = json.loads((MODELS / "config.json").read_text())
MANIFEST = CONFIG["manifest"]
TREES = json.loads((MODELS / "trees.json").read_text())

VS_REPOSITORY = "GaboxR67/MelBandRoformers"
VS_FILES = {
    "mel_band_roformer_voc_fv7.ckpt": "melbandroformers/vocals/voc_fv7.ckpt",
    "mel_band_roformer_voc_fv7_config.yaml": "melbandroformers/vocals/v7.yaml",
}
VS_MODEL = "mel_band_roformer_voc_fv7.ckpt"

FVAD_REPOSITORY = "FireRedTeam/FireRedVAD"
FVAD_FILES = ("VAD/model.pth.tar", "VAD/cmvn.ark")
FVAD_WEIGHT = 3.0
TRIM_THRESHOLD = 0.1
SPEECH_ONSET = 0.5
SPEECH_OFFSET = 0.5

BFA_CONTEXT = 300
BFA_MAX_SEGMENT = 30_000
BFA_SILENCE_ANCHORS = 3
BFA_REPOSITORY = "Tabahi/CUPE-2i"
BFA_MODEL = "large_multi_mswc38_ua02g_e03_val_GER=0.5133.ckpt"
BFA_REVISION = "772503ad6605916b88b81b1855f81533243069be"
QFA_MODEL = "17slever17/Qwen3-ForcedAligner-Ultra"
QFA_REVISION = "765dc1d714e4c80196782a5df5bfbcb3e1416736"
QFA_EDGE_DISAGREEMENT = 100
WFA_REPOSITORY = "sakasegawa/japanese-wav2vec2-large-hiragana-ctc"
WFA_REVISION = "d30d246cd24a225821d03d183de7bb2e769e18df"
NEGATIVE = -3.4e38


@cached
def model(name):
    import lightgbm

    return lightgbm.Booster(model_str=TREES[name])


@cached
def weights():
    import torch

    return torch.load(
        MODELS / "weights.pt", map_location="cpu", weights_only=True, mmap=True
    )


@cached
def qfa_path():
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(QFA_MODEL, revision=QFA_REVISION))


@cached
def wfa_path():
    from huggingface_hub import hf_hub_download

    path = MODEL_DIR / "wfa"
    path.mkdir(parents=True, exist_ok=True)
    if not (path / "best-medium-ep5-inference.pt").is_file():
        source = hf_hub_download(
            WFA_REPOSITORY, "best-medium-ep5-inference.pt", revision=WFA_REVISION
        )
        shutil.copy2(source, path / "best-medium-ep5-inference.pt")
    for name, config in CONFIG["wfa"].items():
        cache.save(path / (name + ".json"), config)
    return path


@dataclass(frozen=True, slots=True)
class Audio:
    raw: Path
    vocals: Path
    duration: int


def frames(path):
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes()


def convert(source, destination, preserve_format=False):
    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vn",
    ]
    if not preserve_format:
        command.extend(("-ac", "1", "-ar", str(SAMPLE_RATE)))
    command.extend(("-c:a", "pcm_s16le", str(destination)))
    subprocess.run(command, check=True)


def prepare_audio(extracted_audio, cache_dir, progress=lambda *_: None):
    raw = cache_dir / "mono_16k.wav"
    vocals = cache_dir / "vocals_mono_16k.wav"
    if not (raw.is_file() and vocals.is_file()):
        from huggingface_hub import hf_hub_download

        asset_dir = MODEL_DIR / "vs"
        asset_dir.mkdir(parents=True, exist_ok=True)
        for local_name, remote_name in VS_FILES.items():
            if not (asset_dir / local_name).is_file():
                with tempfile.TemporaryDirectory(dir=asset_dir) as downloads:
                    source = hf_hub_download(
                        VS_REPOSITORY, remote_name, cache_dir=downloads
                    )
                    shutil.copyfile(source, Path(downloads) / local_name)
                    os.replace(Path(downloads) / local_name, asset_dir / local_name)
        torch = gpu.runtime()
        from audio_separator.separator import Separator

        if not getattr(Separator, "_subtimer_registered", False):
            original = Separator.list_supported_model_files

            def list_supported_model_files(self):
                available = original(self)
                available.setdefault("MDXC", {})[
                    "Roformer Model: MelBand Vocals local"
                ] = {
                    "filename": VS_MODEL,
                    "scores": {},
                    "stems": ["vocals"],
                    "target_stem": "vocals",
                    "download_files": list(VS_FILES),
                }
                return available

            Separator.list_supported_model_files = list_supported_model_files
            Separator._subtimer_registered = True

        with tempfile.TemporaryDirectory(prefix=".audio.", dir=cache_dir) as temporary:
            workspace = Path(temporary)
            progress("separate")
            separator = Separator(
                log_level=logging.ERROR,
                model_file_dir=str(asset_dir),
                output_dir=str(workspace),
                output_single_stem="Vocals",
                use_autocast=True,
                mdxc_params={
                    "segment_size": 128,
                    "override_model_segment_size": True,
                    "batch_size": 1,
                    "overlap": 1,
                    "pitch_shift": 0,
                },
            )
            try:
                separator.load_model(model_filename=VS_MODEL)
                if gpu.available() >= 3 * gpu.GIB:
                    model = separator.model_instance.model_run
                    pool = torch.cuda.graph_pool_handle()
                    stream = torch.cuda.Stream()

                    for block in [
                        model.band_split,
                        *[block for pair in model.layers for block in pair],
                        *model.mask_estimators,
                    ]:
                        block.forward = gpu.Graph(block.forward, pool, stream)
                    del model, block
                with open(os.devnull, "w") as devnull, redirect_stderr(devnull):
                    separator.separate(str(extracted_audio), {"Vocals": "vocals"})
                convert(extracted_audio, workspace / "raw_16k.wav")
                convert(workspace / "vocals.wav", workspace / "vocals_16k.wav")
            finally:
                del separator
                gc.collect()
                torch.cuda.empty_cache()
            raw_frames = frames(workspace / "raw_16k.wav")
            if frames(workspace / "vocals_16k.wav") != raw_frames:
                with wave.open(str(workspace / "vocals_16k.wav"), "rb") as wav:
                    data = wav.readframes(raw_frames)
                with wave.open(str(workspace / "vocals_16k.wav"), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(SAMPLE_RATE)
                    wav.writeframes(data.ljust(raw_frames * 2, b"\x00"))
            os.replace(workspace / "raw_16k.wav", raw)
            os.replace(workspace / "vocals_16k.wav", vocals)
    return Audio(raw, vocals, (frames(raw) * 1000 + SAMPLE_RATE // 2) // SAMPLE_RATE)


class AsrSession:
    def __init__(self):
        self.session = None

    def get(self):
        if self.session is None:
            gpu.runtime()
            import crispasr

            self.session = crispasr.Session(
                str(PARAKEET_MODEL),
                lib_path=str(ASR_LIBRARY),
                n_threads=8,
                backend="parakeet",
            )
        return self.session

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None
            gpu.runtime().cuda.empty_cache()


def run_asr(audio, cache_dir, progress, session):
    path = cache_dir / "asr.json"
    encoded = cache.load(path)
    if encoded is None:
        import numpy as np
        import soundfile

        count = frames(audio.raw)
        chunk = SAMPLE_RATE * CHUNK_DURATION // 1000
        pcm, _ = soundfile.read(str(audio.raw), dtype="float32", always_2d=True)
        pcm = np.ascontiguousarray(pcm[:, 0], dtype=np.float32)
        progress("transcribe")
        encoded = []
        for offset in range(0, count, chunk):
            window_start = offset * 1000.0 / SAMPLE_RATE
            window_end = min(offset + chunk, count) * 1000.0 / SAMPLE_RATE
            window = pcm[
                round(window_start * SAMPLE_RATE / 1000) : round(
                    window_end * SAMPLE_RATE / 1000
                )
            ]
            for segment in session.get().transcribe(window, language="ja"):
                items = segment.words or [segment]
                for item in items:
                    if item.text.strip():
                        encoded.append(
                            {
                                "text": item.text.strip(),
                                "start": float(window_start + float(item.start) * 1000),
                                "end": float(window_start + float(item.end) * 1000),
                                "confidence": float(item.confidence)
                                if segment.words
                                else 1.0,
                            }
                        )
        cache.save(path, encoded)
    return [
        Word(w["text"], Span(w["start"], w["end"]), w["confidence"]) for w in encoded
    ]


def soften(values, exponent):
    import numpy as np

    values = np.clip(values, 1e-6, 1 - 1e-6)
    return values**exponent / (values**exponent + (1 - values) ** exponent)


def vad_segments(prediction, onset, offset):
    import numpy as np

    labels = np.zeros_like(prediction, dtype=bool)
    in_speech = False
    for index, value in enumerate(prediction):
        if not in_speech:
            if value >= onset:
                in_speech = True
                labels[index] = True
        elif value < offset:
            in_speech = False
        else:
            labels[index] = True

    spans = []
    start = None
    for index, value in enumerate(labels):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(labels) - 1))

    minimum_frames = round(80 / FRAME_HOP)
    spans = [(start, end) for start, end in spans if end - start >= minimum_frames]
    merged = []
    for span in spans:
        if merged and span[0] - merged[-1][1] < minimum_frames:
            merged[-1] = (merged[-1][0], span[1])
        else:
            merged.append(span)
    return tuple((start * FRAME_HOP, end * FRAME_HOP) for start, end in merged)


def prepare_vad(audio):
    import librosa
    import numpy as np
    from scipy.io import wavfile
    from scipy.signal import resample
    from ten_vad import TenVad

    with (
        open(os.devnull, "w") as devnull,
        redirect_stdout(devnull),
        redirect_stderr(devnull),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        _, raw_samples = wavfile.read(audio.raw)
        hop_size = 160
        tvad_model = TenVad(hop_size, 0.5)
        frame_count = raw_samples.shape[0] // hop_size
        tvad = np.array(
            [
                tvad_model.process(
                    raw_samples[index * hop_size : (index + 1) * hop_size]
                )[0]
                for index in range(frame_count)
            ]
        )[1:]
        vocals_samples, sample_rate = librosa.load(audio.vocals)
        rms = librosa.feature.rms(y=vocals_samples)[0]
        rms = np.clip(resample(rms, len(tvad)), 0, rms.max())
        rolloff = librosa.feature.spectral_rolloff(
            y=vocals_samples + 0.1, sr=sample_rate
        )[0]
        rolloff = np.clip(resample(rolloff, len(tvad)), 0, rolloff.max())
        hvad = rms * rolloff
        value_range = hvad.max() - hvad.min()
        hvad = (hvad - hvad.min()) / value_range if value_range else np.zeros_like(hvad)
    return raw_samples, tvad, hvad


def run_vad(audio, cache_dir, progress, prepared):
    torch = gpu.runtime()
    import numpy as np
    import silero_vad as silero
    from fireredvad import FireRedVad, FireRedVadConfig
    from huggingface_hub import hf_hub_download
    from scipy.signal import resample

    progress("detect")
    raw_samples, tvad, hvad = prepared

    try:
        paths = [
            hf_hub_download(FVAD_REPOSITORY, name, local_files_only=True)
            for name in FVAD_FILES
        ]
    except Exception:
        paths = [hf_hub_download(FVAD_REPOSITORY, name) for name in FVAD_FILES]
    logger = logging.getLogger("fireredvad.vad")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        fvad_model = FireRedVad.from_pretrained(
            str(Path(paths[0]).parent), FireRedVadConfig(use_gpu=True)
        )
        fvad = []
        for path in (audio.raw, audio.vocals):
            _, probabilities = fvad_model.detect(str(path), do_postprocess=False)
            fvad.append(
                np.clip(resample(probabilities.cpu().numpy(), len(tvad)), 0, 1).astype(
                    np.float32
                )
            )
    finally:
        logger.setLevel(old_level)

    svad_samples = (
        torch.as_tensor(raw_samples, dtype=torch.float32, device="cuda") / 32768
    )
    svad_model = silero.load_silero_vad(onnx=False).to("cuda").eval()
    samples = torch.nn.functional.pad(
        svad_samples, (0, -len(svad_samples) % 512)
    ).reshape(-1, 512)
    state = torch.zeros((2, 1, 128), device="cuda")
    context = torch.zeros((1, svad_model._model.context_size_samples), device="cuda")
    step = gpu.Graph(
        lambda chunk: svad_model._model(torch.cat((context, chunk[None]), 1), state),
        torch.cuda.graph_pool_handle(),
        torch.cuda.Stream(),
    )
    probabilities = torch.empty(len(samples), device="cuda")
    with torch.inference_mode():
        for index, chunk in enumerate(samples):
            probability, next_state = step(chunk)
            probabilities[index] = probability.reshape(-1)[0]
            state.copy_(next_state)
            context.copy_(chunk[-context.shape[1] :][None])
    svad = np.clip(
        resample(probabilities.float().cpu().numpy(), len(tvad)), 0, 1
    ).astype(np.float32)

    hvad = soften(
        np.clip(resample(np.clip(hvad - 0.005, 0, 0.1) / 0.1, len(tvad)), 0, 1), 3
    )
    fvad_average = np.clip((fvad[0] + fvad[1]) / 2, 0, 1)
    main = (tvad + hvad + FVAD_WEIGHT * soften(fvad_average, 1.0)) / (2 + FVAD_WEIGHT)
    prediction = np.clip(0.85 * main + 0.15 * soften(svad, 0.4), 0, 1)
    track = {
        "trim": vad_segments(prediction, TRIM_THRESHOLD, TRIM_THRESHOLD),
        "speech": vad_segments(prediction, SPEECH_ONSET, SPEECH_OFFSET),
    }
    if not track["speech"]:
        raise RuntimeError("audio contains no speech")
    np.savez_compressed(
        cache_dir / "vad_tracks.npz",
        vad=prediction.astype(np.float32),
        fvad_raw=fvad[0].astype(np.float32),
        fvad_vocals=fvad[1].astype(np.float32),
        tvad=tvad.astype(np.float32),
        svad=svad.astype(np.float32),
        hvad=hvad.astype(np.float32),
    )
    cache.save(cache_dir / "vad.json", track)
    torch.cuda.empty_cache()
    return track


class NFAContext(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int),
        ("use_flash", ctypes.c_bool),
        ("verbosity", ctypes.c_int),
        ("use_gpu", ctypes.c_bool),
    ]

    @classmethod
    def open(cls, threads):
        handles = [
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            for path in [*NFA_DEPENDENCIES, NFA_LIBRARY]
        ]
        library = ctypes.CDLL(str(ENCODER_LIBRARY), mode=ctypes.RTLD_LOCAL)
        library.parakeet_context_default_params.argtypes = []
        library.parakeet_context_default_params.restype = cls
        library.parakeet_init_from_file.argtypes = [ctypes.c_char_p, cls]
        library.parakeet_init_from_file.restype = ctypes.c_void_p
        library.parakeet_set_att_context.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        library.parakeet_set_att_context.restype = None
        library.parakeet_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        library.parakeet_encode.restype = ctypes.POINTER(ctypes.c_float)
        library.subtimer_parakeet_encode.argtypes = [
            *library.parakeet_encode.argtypes[:3],
            ctypes.c_int,
            *library.parakeet_encode.argtypes[3:],
        ]
        library.subtimer_parakeet_encode.restype = ctypes.POINTER(ctypes.c_float)
        library.subtimer_parakeet_release_workspace.argtypes = [ctypes.c_void_p]
        library.subtimer_parakeet_release_workspace.restype = None
        library.parakeet_free.argtypes = [ctypes.c_void_p]
        library.parakeet_free.restype = None

        params = library.parakeet_context_default_params()
        params.n_threads = threads
        params.verbosity = 0
        params.use_gpu = True
        params._handles = handles
        params.library = library
        params._libc = ctypes.CDLL(None)
        params._libc.free.argtypes = [ctypes.c_void_p]
        params._libc.free.restype = None
        params.context = library.parakeet_init_from_file(
            str(PARAKEET_MODEL).encode(), params
        )
        if not params.context:
            raise RuntimeError("NFA encoder could not load")
        library.parakeet_set_att_context(params.context, *ATT_CONTEXT)
        return params

    def encode(self, pcm, np):
        import torch

        mel_frames = (int(pcm.size) + 159) // 160 + 2
        if mel_frames > 4096:
            self.library.subtimer_parakeet_release_workspace(self.context)
            torch.cuda.empty_cache()
        room = gpu.available()
        if ((mel_frames + 7) // 8) * 1024 * 4 * 48 > room:
            raise RuntimeError("insufficient free VRAM for full-context alignment")
        capacity = room // (4 * 128 * 1024)
        if capacity < 8:
            raise RuntimeError("insufficient free VRAM for the encoder front end")
        tile = min(4096, 1 << (int(capacity).bit_length() - 1))
        count = ctypes.c_int()
        width = ctypes.c_int()
        arguments = (
            self.context,
            pcm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(pcm.size),
        )
        encode = self.library.parakeet_encode
        if mel_frames > tile:
            encode = self.library.subtimer_parakeet_encode
            arguments += (tile,)
        pointer = encode(*arguments, ctypes.byref(count), ctypes.byref(width))
        if not pointer:
            raise RuntimeError("NFA encoder failed")
        try:
            return (
                np.ctypeslib.as_array(pointer, shape=(count.value * width.value,))
                .reshape(count.value, width.value)
                .copy()
            )
        finally:
            self._libc.free(ctypes.cast(pointer, ctypes.c_void_p))

    def close(self):
        if self.context is not None:
            self.library.parakeet_free(self.context)
            self.context = None


def ctc_head():
    import numpy as np
    from gguf import GGUFReader

    tensors = {tensor.name: tensor for tensor in GGUFReader(PARAKEET_MODEL).tensors}
    bias = np.asarray(tensors["ctc.bias"].data, dtype=np.float32).reshape(-1).copy()
    weight = np.asarray(tensors["ctc.weight"].data, dtype=np.float32)
    return weight.reshape(int(bias.size), weight.size // int(bias.size)).copy(), bias


def nfa_worker(connection, text, tracks, threads):
    try:
        torch = gpu.runtime()
        import numpy as np
        import sentencepiece
        import soundfile
        from torch.nn import functional

        from . import ctc

        weight, bias = ctc_head()
        tokenizer = sentencepiece.SentencePieceProcessor(
            model_file=str(PARAKEET_TOKENIZER)
        )
        blank_id = int(bias.size) - 1
        target_ids = [blank_id]
        target_text = [""]
        for segment in (
            segment.strip() for segment in text.split("|") if segment.strip()
        ):
            for part in (part for part in segment.split(" ") if part):
                part_ids = [int(value) for value in tokenizer.encode_as_ids(part)]
                pieces = [str(value) for value in tokenizer.encode_as_pieces(part)]
                for token_id, piece in zip(part_ids, pieces):
                    target_ids.extend((token_id, blank_id))
                    target_text.extend((piece, ""))

        device = torch.device("cuda")
        ids = torch.tensor(target_ids, dtype=torch.long, device=device)
        state_count = len(target_ids)
        with torch.no_grad():
            weight_tensor = torch.from_numpy(weight).to(
                device=device, dtype=torch.float32
            )
            bias_tensor = torch.from_numpy(bias).to(device=device, dtype=torch.float32)
        del weight, bias

        nfa_context = NFAContext.open(threads)
        results = {}
        try:
            for name, path in tracks.items():
                connection.send(("track", name))
                pcm, _ = soundfile.read(path, dtype="float32", always_2d=True)
                pcm = np.ascontiguousarray(
                    pcm.mean(axis=1, dtype=np.float32), dtype=np.float32
                )
                encoded = nfa_context.encode(pcm, np)
                del pcm
                with torch.no_grad():
                    encoded_tensor = torch.from_numpy(encoded).to(
                        device=device, dtype=torch.float32
                    )
                    del encoded
                    logits = functional.linear(
                        encoded_tensor, weight_tensor, bias_tensor
                    )
                    emissions = functional.log_softmax(
                        logits, dim=-1, dtype=torch.float32
                    )
                    del encoded_tensor, logits
                    frame_count = int(emissions.shape[0])
                    previous, backpointers = ctc.viterbi(emissions, ids)
                    final_state = (
                        0
                        if state_count == 1
                        else state_count - 2 + int(torch.argmax(previous[-2:]).item())
                    )
                    if float(previous[final_state].item()) <= NEGATIVE / 2:
                        raise RuntimeError("no exact CTC alignment path exists")
                del emissions

                path = [0] * frame_count
                path[-1] = final_state
                for frame in range(frame_count - 1, 0, -1):
                    final_state -= int(backpointers[frame, final_state])
                    path[frame - 1] = final_state
                timings = []
                run_start = 0
                for frame in range(1, len(path) + 1):
                    if frame < len(path) and path[frame] == path[run_start]:
                        continue
                    timings.append(
                        {
                            "text": target_text[path[run_start]],
                            "start": run_start * CTC_FRAME_DURATION,
                            "end": frame * CTC_FRAME_DURATION,
                        }
                    )
                    run_start = frame
                results[name] = timings
        finally:
            nfa_context.close()
        connection.send(("ok", results))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def run_nfa(audio, joined_text, cache_dir, progress):
    text_hash = hashlib.sha256(joined_text.encode("utf-8")).hexdigest()
    paths = {
        name: cache_dir / "nfa" / f"nfa_{name}_{text_hash}.json"
        for name in ("raw", "vocals")
    }
    results = {name: cache.load(path) for name, path in paths.items()}
    missing = {
        name: str(getattr(audio, name))
        for name, value in results.items()
        if value is None
    }
    if missing:
        context = multiprocessing.get_context("spawn")
        receive, send = context.Pipe(duplex=False)
        process = context.Process(
            target=nfa_worker,
            args=(send, joined_text, missing, len(os.sched_getaffinity(0))),
            name="subtimer-nfa",
        )

        environment = {
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
            "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
        }
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
            filter(
                None,
                [
                    str(NFA_LIBRARY.parent),
                    str(gpu.RUNTIME.parent),
                    environment["LD_LIBRARY_PATH"],
                ],
            )
        )
        os.environ["LD_PRELOAD"] = os.pathsep.join(
            filter(None, [str(gpu.RUNTIME), environment["LD_PRELOAD"]])
        )
        try:
            process.start()
        finally:
            for name, value in environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        send.close()
        try:
            while (message := receive.recv())[0] == "track":
                progress("detect")
        finally:
            process.join()
        if message[0] == "error":
            raise RuntimeError(f"NFA failed:\n{message[1]}")
        for name, timings in message[1].items():
            cache.save(paths[name], timings)
            results[name] = timings
    return {f"nfa_{name}": value for name, value in results.items()}


@contextmanager
def silence():
    devnull = os.open(os.devnull, os.O_WRONLY)
    stdout = os.dup(1)
    stderr = os.dup(2)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(stdout, 1)
        os.dup2(stderr, 2)
        os.close(devnull)
        os.close(stdout)
        os.close(stderr)


def segment_key(segment):
    return json.dumps(
        [round(float(segment["start"])), round(float(segment["end"])), segment["text"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def alignment_windows(lines, audio, target):
    segments = []
    segment_lines = []
    for line in lines:
        seed = line.candidates.get("nfa_seed")
        if seed is None:
            continue
        start = max(0.0, seed.start - BFA_CONTEXT)
        end = min(float(audio.duration), seed.end + BFA_CONTEXT)
        text = target(line.spoken)
        if end - start < MIN_DURATION or end - start > BFA_MAX_SEGMENT or not text:
            continue
        segments.append({"start": float(start), "end": float(end), "text": text})
        segment_lines.append(line)
    return segments, segment_lines


def alignment_bank(path, segments):
    bank = cache.load(path) or {}
    output = {}
    missing = []
    for segment in segments:
        key = segment_key(segment)
        if key in bank:
            output[key] = bank[key]
        else:
            missing.append(segment)
    empty = cache.load(empty_path(path)) or {}
    return output, [segment for segment in missing if empty_key(segment) not in empty]


def save_alignment_bank(path, output, empty, reason):
    bank = cache.load(path) or {}
    bank.update(output)
    cache.save(path, bank)
    if empty:
        known = cache.load(empty_path(path)) or {}
        known.update({empty_key(segment): reason for segment in empty})
        cache.save(empty_path(path), known)


def empty_path(path):
    return path.with_name(path.stem + "-empty.json")


def empty_key(segment):
    return json.dumps(
        [int(segment["start"] * 16), int(segment["end"] * 16), segment["text"]],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def bfa_target(text):
    import pyopenjtalk

    return (
        pyopenjtalk.g2p(text, kana=True).replace("、", " ").replace("　", " ").strip()
    )


def run_bfa(lines, audio, cache_dir, progress):
    segments, segment_lines = alignment_windows(lines, audio, bfa_target)
    if not segments:
        return
    torch = gpu.runtime()
    batch_size = gpu.batch_size(
        16, weights=3 * gpu.GIB // 2, workspace=3 * gpu.GIB // 4
    )
    cache_path = cache_dir / "bfa.json"
    output, missing = alignment_bank(cache_path, segments)
    if missing:
        import soundfile
        from bournemouth_aligner import PhonemeTimestampAligner
        from huggingface_hub import hf_hub_download

        from .ctc import bfa_decode_cuda

        model_path = hf_hub_download(
            BFA_REPOSITORY, f"ckpt/{BFA_MODEL}", revision=BFA_REVISION
        )
        aligner = None
        audio_wav = None
        phonemize = None
        old_logging = logging.root.manager.disable
        progress("align")
        logging.disable(logging.WARNING)
        insufficient = set()
        try:
            with silence():
                aligner = PhonemeTimestampAligner(
                    cupe_ckpt_path=model_path,
                    lang="ja",
                    duration_max=max(10, BFA_MAX_SEGMENT // 1000),
                    device="cuda",
                    silence_anchors=BFA_SILENCE_ANCHORS,
                    enforce_all_targets=True,
                )
                from types import MethodType

                for utilities in (aligner.alignment_utils_p, aligner.alignment_utils_g):
                    decoder = utilities.viterbi_decoder
                    decoder._viterbi_decode = MethodType(bfa_decode_cuda, decoder)
                aligner.phonemizer.unmapped_phonemes.setdefault("(ja)", 0)
                phonemize = aligner.phonemize_sentence

                def observed_phonemize(text):
                    result = phonemize(text)
                    phones = result[aligner.phonemes_key]
                    if not phones or len(phones) < aligner.ph_seq_min:
                        insufficient.add(text)
                    return result

                aligner.phonemize_sentence = observed_phonemize

                pcm, _ = soundfile.read(
                    str(audio.vocals), dtype="float32", always_2d=True
                )
                audio_wav = aligner.load_audio(
                    torch.from_numpy(pcm[:, :1].T.copy()), sr=SAMPLE_RATE
                )
                requests = [
                    {
                        "start": segment["start"] / 1000,
                        "end": segment["end"] / 1000,
                        "text": segment["text"],
                    }
                    for segment in missing
                ]
                for offset in range(0, len(requests), 64):
                    returned = aligner.process_segments(
                        {"segments": requests[offset : offset + 64]},
                        audio_wav,
                        batch_size=batch_size,
                        debug=False,
                    )[0]["segments"]
                    expected = {
                        segment_key(segment)
                        for segment in missing[offset : offset + 64]
                    }
                    for item in returned:
                        phones = [
                            {
                                "start": float(phone["start_ms"]),
                                "end": float(phone["end_ms"]),
                                "target_index": int(phone["target_seq_idx"]),
                                "label": str(phone["phoneme_label"]),
                                "confidence": float(phone["confidence"]),
                                "estimated": bool(phone["is_estimated"]),
                            }
                            for phone in item["phoneme_ts"]
                        ]
                        aligned = {
                            "start": round(float(item["start"]) * 1000),
                            "end": round(float(item["end"]) * 1000),
                            "text": item["text"],
                            "phonemes": phones,
                        }
                        key = segment_key(aligned)
                        if key in expected:
                            output[key] = aligned
                    del returned
        finally:
            logging.disable(old_logging)

            phonemize = None
            del aligner
            del audio_wav
            torch.cuda.empty_cache()
        save_alignment_bank(
            cache_path,
            output,
            [s for s in missing if s["text"] in insufficient],
            "phonemizer produced fewer than the required phonemes",
        )

    lines_by_segment = defaultdict(list)
    for line, segment in zip(segment_lines, segments):
        lines_by_segment[segment_key(segment)].append(line)
    for key, aligned in output.items():
        matches = lines_by_segment.get(key)
        phones = aligned.get("phonemes", [])
        if not matches or not phones:
            continue
        start = min(phone["start"] for phone in phones)
        end = max(phone["end"] for phone in phones)
        if end - start < MIN_DURATION:
            continue
        target_indices = [phone["target_index"] for phone in phones]
        durations = [phone["end"] - phone["start"] for phone in phones]
        detail = {
            "phones": len(phones),
            "target_coverage": len(set(target_indices))
            / max(1, max(target_indices) + 1),
            "estimated_fraction": statistics.mean(
                phone["estimated"] for phone in phones
            ),
            "median_confidence": statistics.median(
                phone["confidence"] for phone in phones
            ),
            "min_confidence": min(phone["confidence"] for phone in phones),
            "max_phone_duration": max(durations),
            "span_per_phone": (end - start) / len(phones),
        }
        if detail["span_per_phone"] > 600:
            detail["duration_warning"] = True
        for line in matches:
            line.evidence["bfa_phones"] = phones
            line.evidence["bfa_detail"] = detail
            line.candidates["bfa"] = Span(float(start), float(end))


def needs_qfa(line):
    bfa_span = line.candidates.get("bfa")
    nfa_span = line.candidates.get("nfa_vocals")
    return (
        bfa_span is None
        or nfa_span is None
        or abs(bfa_span.midpoint - nfa_span.midpoint) > 300
        or abs(bfa_span.start - nfa_span.start) > QFA_EDGE_DISAGREEMENT
        or abs(bfa_span.end - nfa_span.end) > QFA_EDGE_DISAGREEMENT
        or bool(line.evidence.get("bfa_detail", {}).get("duration_warning"))
    )


def run_qfa(lines, audio, cache_dir, progress):
    segments, segment_lines = alignment_windows(
        [line for line in lines if needs_qfa(line)], audio, str
    )
    if not segments:
        return
    torch = gpu.runtime()
    batch_size = gpu.batch_size(8, weights=3 * gpu.GIB // 2, workspace=3 * gpu.GIB // 4)
    cache_path = cache_dir / "qfa.json"
    output, missing = alignment_bank(cache_path, segments)
    if missing:
        import numpy as np
        import soundfile
        from qwen_asr import Qwen3ForcedAligner

        model = None
        speech = None
        results = None
        completed_empty = []
        progress("align")
        try:
            with silence():
                model = Qwen3ForcedAligner.from_pretrained(
                    QFA_MODEL,
                    revision=QFA_REVISION,
                    dtype=torch.bfloat16,
                    device_map="cuda",
                )
            speech, rate = soundfile.read(str(audio.vocals))
            speech = np.asarray(speech, dtype=np.float32)
            for offset in range(0, len(missing), batch_size):
                chunk = missing[offset : offset + batch_size]
                windows = [
                    speech[int(segment["start"] * 16) : int(segment["end"] * 16)]
                    for segment in chunk
                ]
                with silence():
                    results = model.align(
                        audio=[(window, rate) for window in windows],
                        text=[segment["text"] for segment in chunk],
                        language=["Japanese"] * len(chunk),
                    )
                for segment, result in zip(chunk, results):
                    items = [
                        {
                            "text": item.text,
                            "start": segment["start"] + float(item.start_time) * 1000,
                            "end": segment["start"] + float(item.end_time) * 1000,
                        }
                        for item in result
                        if item.text.strip()
                    ]
                    if items:
                        output[segment_key(segment)] = {
                            "start": min(item["start"] for item in items),
                            "end": max(item["end"] for item in items),
                            "items": items,
                        }
                    else:
                        completed_empty.append(segment)
                del results
                results = None
        finally:
            del results
            del model
            del speech
            torch.cuda.empty_cache()
        save_alignment_bank(
            cache_path,
            output,
            completed_empty,
            "alignment completed without timestamp words",
        )
    apply_qfa(segments, segment_lines, output)


def apply_qfa(segments, segment_lines, output):
    lines_by_segment = defaultdict(list)
    for line, segment in zip(segment_lines, segments):
        lines_by_segment[segment_key(segment)].append(line)
    for key, aligned in output.items():
        matches = lines_by_segment.get(key)
        if not matches:
            continue
        start = aligned["start"]
        end = aligned["end"]
        if end - start < MIN_DURATION:
            continue
        items = aligned.get("items", [])
        gaps = [
            max(0.0, following["start"] - current["end"])
            for current, following in pairwise(items)
        ]
        for line in matches:
            if items:
                line.evidence["qfa_items"] = items
                detail = {
                    "items": len(items),
                    "max_gap": max(gaps, default=0.0),
                    "max_item_duration": max(
                        item["end"] - item["start"] for item in items
                    ),
                    "span_per_char": (end - start) / max(1, len(line.key)),
                }
                if detail["span_per_char"] > 800:
                    detail["duration_warning"] = True
                line.evidence["qfa_detail"] = detail
            line.candidates["qfa"] = Span(float(start), float(end))
