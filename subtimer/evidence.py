import functools
import itertools
import logging
import operator
import sys
import unicodedata
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn

from . import gpu
from .cache import Evidence, load
from .ctc import ctc_align
from .speech import Audio, wfa_path
from . import numerics as n


class AudioReaders(ExitStack):
    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self.streams = {}

    def read(self, request):
        view = request["view"]
        if view not in self.streams:
            self.streams[view] = self.enter_context(sf.SoundFile(self.paths[view]))
        stream = self.streams[view]
        start = int(request.get("start", 0) * 16)
        end = int(request.get("end", len(stream) / 16) * 16)

        if len(stream) < end <= len(stream) + 8:
            end = len(stream)
        stream.seek(start)
        return stream.read(end - start, dtype="float32")


def map_words(words, texts):
    def normalize(text):
        return "".join(
            c for c in unicodedata.normalize("NFKC", text).lower() if c.isalnum()
        )

    normalized = [normalize(t) for t in texts]
    tokens = [normalize(w["text"]) for w in words]
    if "".join(tokens) != "".join(normalized):
        return None
    starts = np.r_[0, np.cumsum([len(t) for t in tokens])[:-1]]
    ends = np.cumsum([len(t) for t in tokens])
    boundaries = np.r_[0, np.cumsum([len(t) for t in normalized])]
    out = []
    for a, z in itertools.pairwise(boundaries):
        ids = np.flatnonzero((ends > a) & (starts < z) & (ends > starts))

        if not len(ids) or starts[ids[0]] != a or ends[ids[-1]] != z:
            out.append(None)
        else:
            out.append([int(ids[0]), int(ids[-1])])
    return out


class WFARows:
    def __init__(self, store, plans):
        self.store, self.plans = store, plans

    def __getitem__(self, i):
        pair = self.plans[i]
        if pair[0] is None:
            return {"available": False}, {}
        results = [self.store.get(r) for r in pair]
        win = {
            "start": pair[0]["start"],
            "end": pair[0]["end"],
            "cue": pair[0]["cue"],
            "available": all(r[0].get("available") for r in results),
            "tokens": results[0][0].get("tokens", []),
        }
        return win, {v: r[1] for v, r in zip(["raw", "vocals"], results)}


def components(store, d, texts, progress, *, maximum_workers=4):
    plans = []
    for i in range(len(texts)):
        baseline = d["baseline"]
        selected = [i]
        for j in [i - 1, i + 1]:
            if j < 0 or j >= len(baseline):
                continue
            gap = (
                baseline[i, 0] - baseline[j, 1]
                if j < i
                else baseline[j, 0] - baseline[i, 1]
            )
            if 0 <= gap <= 1200:
                trial = sorted(selected + [j])
                if baseline[trial[-1], 1] - baseline[trial[0], 0] <= 12000:
                    selected = trial
        start = max(0, int(baseline[selected[0], 0]) - 1500)
        end = min(int(d["duration"]), int(baseline[selected[-1], 1]) + 1500)
        if not 200 <= end - start <= 16000:
            plans.append([None, None])
            continue
        plans.append(
            [
                {
                    "kind": "wfa",
                    "view": view,
                    "start": start,
                    "end": end,
                    "texts": [texts[j] for j in selected],
                    "cue": selected.index(i),
                }
                for view in ["raw", "vocals"]
            ]
        )
    requests = [r for pair in plans for r in pair if r is not None]
    packplans = []
    for kind in ["bfa", "qfa"]:
        for i, b in enumerate(d["baseline"]):
            start = max(0, int(b[0]) - 1500)
            end = min(int(d["duration"]), int(b[1]) + 1500)
            packplans.append(
                {
                    "kind": kind + "-packed",
                    "view": "vocals",
                    "start": start,
                    "end": end,
                    "text": texts[i],
                }
                if 200 <= end - start <= 16000
                else None
            )
    store.ensure(
        requests + [r for r in packplans if r is not None],
        progress,
        maximum_workers=maximum_workers,
    )
    output = {"wfa": WFARows(store, plans), "packed": {}}
    with AudioReaders(store.paths) as readers:
        for j, kind in enumerate(["bfa", "qfa"]):
            rows = []
            for i, request in enumerate(
                packplans[j * len(texts) : (j + 1) * len(texts)]
            ):
                row = {"outputs": {}}
                if request is not None:
                    result, _ = store.get(request)
                    b = d["baseline"][i]
                    row.update(
                        start=request["start"],
                        end=request["end"],
                        blocks=result["blocks"],
                        near_seam=result["near_seam"],
                    )
                    for key, p in result["outputs"].items():
                        if n.valid(np.asarray(p), b, d["duration"]):
                            row["outputs"][key] = p
                    control = np.array(row["outputs"].get("control", b))
                    packed = np.array(row["outputs"].get("packed", b))
                    guarded = (
                        control
                        if row["near_seam"] and np.abs(control - packed).max() > 50
                        else packed
                    )
                    pcm = readers.read(request)
                    start = request["start"]
                    row["outputs"].update(
                        guarded=guarded.tolist(),
                        **{
                            "control-tail": (
                                activity_edges(pcm, control - start) + start
                            ).tolist(),
                            "packed-tail": (
                                activity_edges(pcm, guarded - start) + start
                            ).tolist(),
                        },
                    )
                rows.append(row)
            output["packed"][kind] = rows
    return output


def qfa_result(store, request):
    if request is None:
        return {"available": False}, None
    row, arrays = store.get(request)
    result = dict(row, start=request["start"], end=request["end"], text=request["text"])
    result["outputs"] = {
        k: (np.array(p) + request["start"]).tolist()
        for k, p in row.get("outputs", {}).items()
    }
    return result, arrays if row.get("available") else None


def qfa_sources(store, d, texts, windows, anchor, progress):
    plans = []
    for i, b in enumerate(d["baseline"]):
        a = max(0, int(b[0]) - 1500)
        z = min(int(d["duration"]), int(b[1]) + 1500)
        pair = [
            {
                "kind": "qfa",
                "view": v,
                "start": a,
                "end": z,
                "text": texts[i],
                "anchor": (anchor[i] - a).tolist(),
            }
            if 200 <= z - a <= 16000
            else None
            for v in ["vocals", "raw"]
        ]
        win = windows[i]
        if win:
            start, end = int(win["start"]), int(win["end"])
            pair.append(
                {
                    "kind": "qfa",
                    "view": "vocals",
                    "start": start,
                    "end": end,
                    "text": win["text"],
                }
                if 200 <= end - start <= 30000
                else None
            )
        else:
            pair.append(None)
        plans.append(pair)
    store.ensure([r for pair in plans for r in pair if r is not None], progress)
    return [[qfa_result(store, r) for r in pair] for pair in plans]


def packets(b, texts, duration):
    out = []
    core = 0
    while core < len(b):
        stop = min(len(b), core + 3)
        while stop > core + 1 and b[core:stop, 1].max() - b[core:stop, 0].min() > 19000:
            stop -= 1
        first, last = core, stop
        if core and b[core, 0] - b[core - 1, 1] < 3000 and b[core, 0] >= b[core - 1, 0]:
            first -= 1
        if (
            stop < len(b)
            and b[stop, 0] - b[stop - 1, 1] < 3000
            and b[stop, 0] >= b[stop - 1, 0]
        ):
            last += 1
        if b[first:last, 1].max() - b[first:last, 0].min() > 21000:
            first, last = core, stop
        a = max(0, int(b[first:last, 0].min()) - 1500)
        z = min(int(duration), int(b[first:last, 1].max()) + 1500)
        out.append(
            {
                "core": list(range(core, stop)),
                "context": list(range(first, last)),
                "start": a,
                "end": z,
                "texts": texts[first:last],
                "text": "\n".join(texts[first:last]),
            }
        )
        core = stop
    return out


def neighbors(store, b, texts, duration, progress):
    plan = packets(b, texts, duration)
    requests = [
        [
            {
                "kind": "qfa-neighbors",
                "view": v,
                "start": p["start"],
                "end": p["end"],
                "text": p["text"],
                "texts": p["texts"],
                "core": [p["context"].index(i) for i in p["core"]],
            }
            if 200 <= p["end"] - p["start"] <= 24000
            else None
            for v in ["raw", "vocals"]
        ]
        for p in plan
    ]
    store.ensure([r for pair in requests for r in pair if r is not None], progress)
    out = {
        f"context/{view}/{kind}": b.copy()
        for view in ["raw", "vocals"]
        for kind in ["native", "head"]
    }
    valid = {k: np.zeros(len(b), bool) for k in out}
    for packet, pair in zip(plan, requests):
        for view, request in zip(["raw", "vocals"], pair):
            if request is None:
                continue
            r, _ = store.get(request)
            if not r.get("available"):
                continue
            for kind in ["native", "head"]:
                for index, point in r["outputs"][kind].items():
                    i = packet["context"][int(index)]
                    p = np.array(point) + packet["start"]
                    if (
                        np.isfinite(p).all()
                        and p[0] >= packet["start"]
                        and p[1] <= packet["end"]
                        and p[1] - p[0] >= 200
                        and np.abs(p - b[i]).max() <= 3000
                    ):
                        key = f"context/{view}/{kind}"
                        out[key][i] = p
                        valid[key][i] = True
    return out, valid


def ranges(pcm):
    frame = 12800
    hop = 6400
    padded = np.pad(pcm, (frame // 2, frame // 2))
    windows = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop].T
    rms = np.sqrt(np.mean(np.square(windows, dtype=np.float32), axis=0))
    active = rms > np.percentile(rms, 90) * 0.1
    blocks = []
    first = None
    for j, on in enumerate(active):
        if on and first is None:
            first = max(0, j * hop - frame // 4)
        if not on and first is not None:
            blocks.append([first, min(len(pcm), j * hop + frame // 4)])
            first = None
    if first is not None:
        blocks.append([first, len(pcm)])
    merged = []
    for a, b in blocks:
        if b <= a:
            continue
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return np.array(merged or [[0, len(pcm)]], dtype=np.int64)


def original(value, blocks, edge):
    lengths = np.diff(blocks, axis=1).ravel()
    ends = np.cumsum(lengths)
    value = int(np.clip(round(value), 0, ends[-1]))
    j = min(
        int(np.searchsorted(ends, value, side="right" if edge == 0 else "left")),
        len(blocks) - 1,
    )
    before = ends[j - 1] if j else 0
    return int(blocks[j, 0] + value - before)


def near_seam(span, blocks):
    seams = np.cumsum(np.diff(blocks, axis=1).ravel())[:-1] / 16
    return bool(len(seams) and np.min(np.abs(np.asarray(span)[:, None] - seams)) <= 20)


def activity_edges(pcm, span):
    n = len(pcm) // 160
    if n < 2:
        return np.array(span)
    rms = np.sqrt(np.mean(pcm[: n * 160].reshape(n, 160) ** 2, axis=1))
    on = rms > max(1e-4, np.percentile(rms, 90) * 0.1)
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]]) * 10
    ends = (np.flatnonzero(on & ~np.r_[on[1:], False]) + 1) * 10
    out = np.array(span, dtype=float)
    for e, events in enumerate([starts, ends]):
        if len(events):
            best = events[np.argmin(np.abs(events - span[e]))]
            if abs(best - span[e]) <= 200:
                out[e] = best
    return out if out[1] - out[0] >= 200 else np.array(span)


def phone_timestamps(
    model,
    wavs,
    wav_lens,
    sequences,
    start_offset_times=0,
    group_sequences=None,
    extract_embeddings=False,
    do_groups=False,
    debug=False,
):
    import torch
    from bournemouth_aligner.utils import _calculate_confidences, convert_to_ms

    lengths = [len(s) for s in sequences]
    maximum = max(lengths)
    ids = torch.tensor(
        [s + [model.blank_class] * (maximum - len(s)) for s in sequences],
        dtype=torch.long,
        device=model.device,
    )
    logits, _, _, spectral = model._cupe_prediction_batch(wavs, wav_lens, False)
    lens = torch.tensor(lengths, dtype=torch.long, device=model.device)
    spectral = torch.tensor(spectral, dtype=torch.long, device=model.device)
    lp = torch.nn.functional.log_softmax(logits, dim=2)
    phones = model.alignment_utils_p.decode_alignments(
        lp,
        true_seqs=ids,
        pred_lens=spectral,
        true_seqs_lens=lens,
        forced_alignment=True,
        boost_targets=model.boost_targets,
        enforce_minimum=model.enforce_minimum,
        debug=debug,
    )
    phones = model.ensure_target_coverage(
        ids,
        aligned_frames=phones,
        seq_lens=lens,
        _silence_class=model.silence_class,
        debug=debug,
    )
    if model.extend_soft_boundaries:
        phones = model.extend_soft_boundaries_func(
            lp, phones, boundary_softness=model.boundary_softness, debug=debug
        )
    for b in range(len(phones)):
        phones[b] = _calculate_confidences(lp[b], phones[b])
        offset = (
            start_offset_times[b]
            if isinstance(start_offset_times, (list, tuple))
            else start_offset_times
        )
        phones[b] = sorted(
            convert_to_ms(
                phones[b], spectral[b], offset, wav_lens[b], model.resampler_sample_rate
            ),
            key=lambda v: v[6],
        )
    return (
        [{"phoneme_timestamps": p, "group_timestamps": []} for p in phones],
        [None] * len(phones),
        [None] * len(phones),
    )


def make_aligner(kind):
    import torch

    from . import gpu, speech
    from .ctc import bfa_decode_cpu

    torch.set_num_threads(4)
    gpu.configure()
    if kind == "bfa":
        from bournemouth_aligner import PhonemeTimestampAligner
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            speech.BFA_REPOSITORY,
            f"ckpt/{speech.BFA_MODEL}",
            revision=speech.BFA_REVISION,
        )
        logging.disable(logging.WARNING)
        model = PhonemeTimestampAligner(
            cupe_ckpt_path=path,
            lang="ja",
            duration_max=16,
            device="cuda",
            silence_anchors=3,
            enforce_all_targets=True,
        )
        model.phonemizer.unmapped_phonemes.setdefault("(ja)", 0)
        import types

        original_prediction = model._cupe_prediction_batch

        def prediction(*args, **kwargs):
            model.device = "cuda"
            try:
                result = original_prediction(*args, **kwargs)
            finally:
                model.device = "cpu"
            return tuple(v.cpu() if isinstance(v, torch.Tensor) else v for v in result)

        model._cupe_prediction_batch = prediction
        model.device = "cpu"
        model.extract_timestamps_from_segment_batch = types.MethodType(
            phone_timestamps, model
        )
        decoder = model.alignment_utils_p.viterbi_decoder
        decoder._viterbi_decode = types.MethodType(bfa_decode_cpu, decoder)

        def align(signals, texts):
            waves, segments = [], []
            for pcm, text in zip(signals, texts):
                spoken = speech.bfa_target(text)
                waves.append(
                    model.load_audio(torch.from_numpy(pcm[None].copy()), sr=16000)
                )
                segments.append(
                    {
                        "segments": [
                            {"start": 0, "end": len(pcm) / 16000, "text": spoken}
                        ]
                    }
                )
            output = model.process_segments(
                segments, waves, batch_size=len(signals), debug=False
            )
            spans = []
            for result in output:
                phones = [p for item in result["segments"] for p in item["phoneme_ts"]]
                spans.append(
                    np.array(
                        [
                            min(p["start_ms"] for p in phones),
                            max(p["end_ms"] for p in phones),
                        ]
                    )
                    if phones
                    else None
                )
            return spans
    else:
        from qwen_asr import Qwen3ForcedAligner

        return Qwen3ForcedAligner.from_pretrained(
            speech.QFA_MODEL,
            revision=speech.QFA_REVISION,
            dtype=torch.bfloat16,
            device_map="cuda",
        )

    return align


class WFA(nn.Module):
    def __init__(self):
        from transformers import Wav2Vec2Config, Wav2Vec2Model
        from .japanese import JapaneseKanaConverter, KanaVocab, PhonemeVocab

        super().__init__()
        checkpoint = torch.load(
            wfa_path() / "best-medium-ep5-inference.pt",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        self.layer = checkpoint["inter_ctc_layer"]
        state = checkpoint["model_state_dict"]
        config = Wav2Vec2Config.from_pretrained(wfa_path())
        config.mask_time_prob = 0.05
        config.apply_spec_augment = False
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            self.encoder = Wav2Vec2Model(config)
            self.kana_head = nn.Linear(
                config.hidden_size, state["kana_head.weight"].shape[0]
            )
            self.phoneme_head = nn.Linear(
                config.hidden_size, state["phoneme_head.weight"].shape[0]
            )
        self.load_state_dict(state, strict=True, assign=True)
        self.kana = KanaVocab()
        self.phone = PhonemeVocab()
        self.converter = JapaneseKanaConverter()

    def tokens(self, texts):
        import pyopenjtalk

        result = []
        for text in texts:
            ph = pyopenjtalk.g2p(text).split()
            result.append(
                {
                    "phone": self.phone.encode(" ".join(ph)),
                    "kana": self.kana.encode(self.converter.text_to_kana(text)),
                }
            )
        return result

    def forward(self, wav):
        wav = wav.float()
        wav = (
            (wav - wav.mean(-1, keepdim=True))
            / torch.sqrt(wav.var(-1, unbiased=False, keepdim=True) + 1e-7)
        ).to(next(self.parameters()).dtype)
        out = self.encoder(
            wav,
            attention_mask=torch.ones(wav.shape, dtype=torch.long, device=wav.device),
            output_hidden_states=True,
        )
        h12 = out.hidden_states[self.layer]
        h24 = out.last_hidden_state
        return (
            h12,
            h24,
            self.phoneme_head(h12).float().log_softmax(-1),
            self.kana_head(h24).float().log_softmax(-1),
        )


def align_record(win, arrays, head):
    groups = [r[head] for r in win["tokens"]]
    ids = np.array(functools.reduce(operator.iadd, groups, []), np.int64)
    first = sum(len(g) for g in groups[: win["cue"]])
    last = first + len(groups[win["cue"]]) - 1
    starts, ends, score = ctc_align(arrays[head].astype(np.float32), ids)
    if not np.isfinite(score) or np.any(starts < 0):
        raise ValueError("infeasible path")
    return np.array([starts[first], ends[last]]) * 20 + win["start"]


def run(payload):
    audio = Audio(
        payload["paths"]["raw"], payload["paths"]["vocals"], payload["duration"]
    )
    cache = Evidence(audio, payload["root"])
    requests = payload["requests"]
    kind = requests[0]["kind"]
    torch.set_num_threads(4)
    gpu.configure()
    device = "cuda"
    if kind == "wfa":
        model = (
            WFA()
            .to(
                device=device,
                dtype=torch.float16 if device == "cuda" else torch.float32,
            )
            .eval()
        )
    elif kind in ["bfa-packed", "qfa-packed"]:
        model = make_aligner(kind.removesuffix("-packed"))
    elif kind in ["qfa", "qfa-neighbors"]:
        from .qfa import QFA

        model = QFA()
    with AudioReaders(payload["paths"]) as readers:
        if kind == "wfa":
            groups = defaultdict(lambda: defaultdict(list))
            for request in requests:
                size = int(request["end"] * 16) - int(request["start"] * 16)
                groups[size][
                    (request["view"], request["start"], request["end"])
                ].append(request)
            batch_size = gpu.batch_size(4, weights=0, workspace=gpu.GIB // 2)
            for group in groups.values():
                crops = list(group.values())
                for left in range(0, len(crops), batch_size):
                    batch = crops[left : left + batch_size]
                    pcm = np.stack([readers.read(rows[0]) for rows in batch])
                    with torch.inference_mode():
                        h12, h24, phone, kana = model(torch.from_numpy(pcm).to(device))
                    arrays = {
                        k: v.cpu().numpy().astype(np.float16)
                        for k, v in [("h12", h12), ("phone", phone), ("kana", kana)]
                    }
                    del h12, h24, phone, kana
                    for i, rows in enumerate(batch):
                        for request in rows:
                            tokens = model.tokens(request["texts"])
                            available = all(r["phone"] and r["kana"] for r in tokens)
                            result = (
                                {"available": True, "tokens": tokens}
                                if available
                                else {
                                    "available": False,
                                    "reason": "empty token sequence",
                                }
                            )
                            cache.put(
                                request,
                                result,
                                {k: v[i] for k, v in arrays.items()}
                                if available
                                else {},
                            )
            return
        if kind in ["qfa", "qfa-neighbors"]:
            batch_size = gpu.batch_size(4, weights=0, workspace=gpu.GIB // 2)
            requests.sort(
                key=lambda r: (round((r["end"] - r["start"]) / 1000), len(r["text"]))
            )
            for left in range(0, len(requests), batch_size):
                rows = requests[left : left + batch_size]
                usable = []
                for request in rows:
                    words, _ = model.qwen.aligner_processor.encode_timestamp(
                        request["text"], "Japanese"
                    )
                    if words:
                        usable.append(request)
                    else:
                        cache.put(
                            request,
                            {"available": False, "reason": "no timestamp words"},
                            {},
                        )
                if not usable:
                    continue
                signals = [readers.read(r) for r in usable]
                texts = [r["text"] for r in usable]
                if kind == "qfa":
                    outputs = model.run_many(
                        signals, texts, [r.get("anchor") for r in usable]
                    )
                    for request, (result, arrays) in zip(usable, outputs):
                        result["available"] = True
                        cache.put(request, result, arrays)
                    continue
                batches = model.local.align(
                    audio=[(pcm, 16000) for pcm in signals],
                    text=texts,
                    language=["Japanese"],
                )
                for request, items in zip(usable, batches):
                    words = [{"text": w.text} for w in items]
                    mapping = map_words(words, request["texts"])
                    if mapping is None:
                        result = {
                            "available": False,
                            "reason": "normalized transcript/token concatenation mismatch",
                        }
                    else:
                        outputs = {}
                        for arm in ["native", "head"]:
                            spans = (
                                np.array(
                                    [
                                        [
                                            w.unconstrained_start_time,
                                            w.unconstrained_end_time,
                                        ]
                                        if arm == "native"
                                        else [w.start_time, w.end_time]
                                        for w in items
                                    ]
                                )
                                * 1000
                            )
                            outputs[arm] = {
                                str(i): spans[mapping[i], [0, 1]].tolist()
                                for i in request["core"]
                                if mapping[i] is not None
                            }
                        result = {
                            "available": True,
                            "words": words,
                            "mapping": mapping,
                            "outputs": outputs,
                            "duration_limited": sum(w.duration_limited for w in items),
                        }
                    cache.put(request, result, {})
            return
        if kind in ["bfa-packed", "qfa-packed"]:
            batch_size = gpu.batch_size(
                16 if kind == "bfa-packed" else 8, weights=0, workspace=gpu.GIB // 4
            )
            for left in range(0, len(requests), batch_size):
                rows = requests[left : left + batch_size]
                signals, texts, plans = [], [], []
                for request in rows:
                    pcm = readers.read(request)
                    blocks = ranges(pcm)
                    control = len(signals)
                    signals.append(pcm)
                    texts.append(request["text"])
                    packed = control
                    if blocks.tolist() != [[0, len(pcm)]]:
                        packed = len(signals)
                        signals.append(np.concatenate([pcm[a:z] for a, z in blocks]))
                        texts.append(request["text"])
                    plans.append((blocks, control, packed))
                items = []
                for start in range(0, len(signals), batch_size):
                    if kind == "bfa-packed":
                        items.extend(
                            model(
                                signals[start : start + batch_size],
                                texts[start : start + batch_size],
                            )
                        )
                        continue
                    items.extend(
                        model.align(
                            audio=[
                                (pcm, 16000)
                                for pcm in signals[start : start + batch_size]
                            ],
                            text=texts[start : start + batch_size],
                            language=["Japanese"],
                        )
                    )
                for request, (blocks, control, packed) in zip(rows, plans):
                    outputs, errors, seam = {}, {}, False
                    for view, index in [("control", control), ("packed", packed)]:
                        if kind == "bfa-packed":
                            point = items[index]
                            if point is None:
                                errors[view] = "no phones"
                                continue
                        else:
                            words = [word for word in items[index] if word.text.strip()]
                            if not words:
                                errors[view] = "no words"
                                continue
                            point = (
                                np.array(
                                    [
                                        min(w.start_time for w in words),
                                        max(w.end_time for w in words),
                                    ]
                                )
                                * 1000
                            )
                        if view == "packed":
                            seam = near_seam(point, blocks)
                            samples = np.rint(np.asarray(point) * 16).astype(int)
                            point = (
                                np.array(
                                    [original(samples[e], blocks, e) for e in range(2)]
                                )
                                / 16
                            )
                        outputs[view] = (point + request["start"]).tolist()
                    cache.put(
                        request,
                        {
                            "available": True,
                            "outputs": outputs,
                            "blocks": blocks.tolist(),
                            "near_seam": seam,
                            "errors": errors,
                        },
                        {},
                    )
            return


if __name__ == "__main__":
    run(load(sys.argv[1]))
