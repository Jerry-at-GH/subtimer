import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from torch import nn

from . import occurrence, subtitle
from .speech import MANIFEST, model

SAMPLES = (-300, -200, -100, -50, -20, 0, 20, 50, 100, 200, 300)

GRID_TRACKS = ("vad", "rms_raw", "rms_vocals", "tvad", "svad", "hvad")
STRUCT_TRACKS = ("vad", "tvad", "svad")
MARGINS = {mode: tuple(values) for mode, values in MANIFEST["margins"].items()}
MIN_SPAN = int(MANIFEST["min_span_ms"])
SMOOTH_SIGMA = int(MANIFEST["smoothing_sigma_frames"])


class CRNN(nn.Module):
    def __init__(self, n_scalars=2):
        super().__init__()
        self.frontend = nn.Sequential(
            nn.Conv2d(2, 32, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(4, 64),
            nn.GELU(),
            nn.Conv2d(64, 96, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(4, 96),
            nn.GELU(),
            nn.Conv2d(96, 128, (5, 3), stride=(2, 1), padding=(2, 1)),
            nn.GroupNorm(4, 128),
            nn.GELU(),
        )
        self.rnn = nn.GRU(
            128 * 4, 96, num_layers=2, batch_first=True, bidirectional=True
        )
        self.head = nn.Sequential(
            nn.Linear(192 * 2 + n_scalars, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, patch, scalars):
        hidden = self.frontend(patch)
        hidden = hidden.flatten(1, 2).transpose(1, 2)
        output, _ = self.rnn(hidden)
        center = output[:, output.shape[1] // 2]
        pooled = output.mean(dim=1)
        return self.head(torch.cat([center, pooled, scalars], dim=1)).squeeze(-1)


def apply(lines, audio, vad, cache_dir, session, progress, *, capture):
    timed = [
        line for line in lines if line.key and line.candidates.get("result") is not None
    ]
    if not timed:
        return
    progress("refine")

    cache_dir = Path(cache_dir)
    mode = "guide" if any(line.hint_calibrated for line in timed) else "text"
    capture.set_refinement_mode(mode)
    baselines = [
        [line.candidates["result"].start, line.candidates["result"].end]
        for line in timed
    ]
    with np.load(cache_dir / "vad_tracks.npz") as data:
        tracks = {
            name: data[name].astype(np.float32)
            for name in ("vad", "fvad_raw", "fvad_vocals", "tvad", "svad", "hvad")
        }
    tracks["rms_raw"] = _rms(audio.raw)
    tracks["rms_vocals"] = _rms(audio.vocals)
    length = min(len(track) for track in tracks.values())
    tracks = {name: track[:length] for name, track in tracks.items()}
    mel = mel_features(audio, cache_dir)
    device = "cuda"
    schema = MANIFEST["schema"]
    standard_edges = []
    models = {}
    for edge_index, edge in enumerate(("start", "end")):
        features, meta = _grid(
            timed, baselines, tracks, edge, 600, schema, is_guide=mode == "guide"
        )
        booster = model(f"edge-scorer-lgbm-{edge}")
        scores = booster.predict(features)
        central = np.asarray([abs(item[2]) <= 300 for item in meta])
        central_features = features[central]
        central_meta = [item for item, keep in zip(meta, central) if keep]
        crnn = CRNN().to(device)
        crnn.load_state_dict(capture.states[edge_index])
        crnn.eval()
        centers = np.asarray(
            [round(item[1] / 10) for item in central_meta], dtype=np.int64
        )
        scalars = central_features[
            :, [schema.index("off_rel"), schema.index("dur_log")]
        ].astype(np.float32)
        scalars[:, 0] *= 2
        crnn_scores = capture.capture_scores(
            edge_index, crnn, mel, centers, scalars, device
        )
        blend = (scores[central] + crnn_scores) / 2
        central_groups = defaultdict(list)
        for (index, candidate, offset), score in zip(central_meta, blend):
            central_groups[index].append((float(score), candidate, offset))
        wide_groups = defaultdict(list)
        for (index, candidate, offset), score in zip(meta, scores):
            wide_groups[index].append((float(score), candidate, offset))
        capture.remember_curves(timed, edge, central_groups)
        margin_a, margin_b = MARGINS[mode]
        selected = {}
        for key in central_groups:
            base = next(
                candidate for _, candidate, offset in central_groups[key] if offset == 0
            )
            best_advantage, best = 0.0, base
            advantage, candidate = _best_move(central_groups, key)
            if advantage > margin_a:
                best_advantage, best = advantage, candidate
            advantage, candidate = _best_move(wide_groups, key)
            if advantage > margin_b and advantage > best_advantage:
                best = candidate
            selected[key] = best
        standard_edges.append(selected)
        if mode == "guide":
            models[edge] = (booster, crnn)
        else:
            del crnn
            torch.cuda.empty_cache()

    previous = -math.inf
    for index, (line, baseline) in enumerate(zip(timed, baselines)):
        start = round(standard_edges[0].get(index, baseline[0]))
        end = round(standard_edges[1].get(index, baseline[1]))
        if end - start < MIN_SPAN:
            start, end = baseline
        span = subtitle.Span(start, end)
        following = (
            sum(baselines[index + 1]) / 2 if index + 1 < len(baselines) else math.inf
        )
        line.candidates["result"] = (
            span if previous <= span.midpoint <= following else subtitle.Span(*baseline)
        )

        previous = max(sum(baseline) / 2, line.candidates["result"].midpoint)

    for index in occurrence.repair_hints(
        timed, audio, vad, cache_dir, session, progress
    ):
        span = [
            timed[index].candidates["result"].start,
            timed[index].candidates["result"].end,
        ]
        selected = {}
        for edge, edge_index in (("start", 0), ("end", 1)):
            booster, crnn = models[edge]
            features, meta = _grid(
                [timed[index]],
                [span],
                tracks,
                edge,
                600,
                schema,
                is_guide=mode == "guide",
            )
            central = np.asarray([abs(item[2]) <= 300 for item in meta])
            features = features[central]
            meta = [item for item, keep in zip(meta, central) if keep]
            scores = booster.predict(features)
            centers = np.asarray([round(item[1] / 10) for item in meta], dtype=np.int64)
            scalars = features[
                :, [schema.index("off_rel"), schema.index("dur_log")]
            ].astype(np.float32)
            scalars[:, 0] *= 2
            crnn_scores = capture.capture_scores(
                edge_index, crnn, mel, centers, scalars, device
            )
            blend = _smooth((scores + crnn_scores) / 2)
            offsets = np.asarray([item[2] for item in meta])
            zero = int(np.flatnonzero(offsets == 0)[0])
            best = int(np.argmax(blend))
            selected[edge] = (
                round(float(meta[best][1]))
                if best != zero and blend[best] > blend[zero]
                else round(float(span[edge_index]))
            )
        start, end = selected["start"], selected["end"]
        if end - start < MIN_SPAN:
            start, end = span
        timed[index].candidates["result"] = subtitle.Span(start, end)
    torch.cuda.empty_cache()


def _rms(path):
    rate, samples = wavfile.read(path)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    hop = max(1, round(rate * 0.01))
    samples = samples[: len(samples) // hop * hop].astype(np.float32, copy=False)
    values = np.log1p(np.sqrt(np.mean(samples.reshape(-1, hop) ** 2, axis=1)))
    low, high = np.percentile(values, (5, 95))
    return np.clip((values - low) / max(high - low, 1e-6), 0, 1).astype(np.float32)


def mel_features(audio, cache_dir):
    path = cache_dir / "mel.npy"
    if path.is_file():
        return np.load(path, mmap_mode="r")
    import torchaudio

    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64,
        power=2.0,
    )
    views = []
    for wav in (audio.raw, audio.vocals):
        _, samples = wavfile.read(wav)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        pcm = torch.from_numpy(samples.astype(np.float32) / 32768.0)
        views.append(torch.log(transform(pcm) + 1e-6))
    frames = min(view.shape[1] for view in views)
    stack = np.stack([view[:, :frames].numpy() for view in views]).astype(np.float32)
    stack = ((stack - stack.mean()) / (stack.std() + 1e-4)).astype(np.float16)
    np.save(path, stack)
    return stack


def _structure(track, times, edge):
    binary = track >= 0.5
    changes = np.flatnonzero(np.diff(binary.astype(np.int8)))
    onsets = (changes[binary[changes + 1]] + 1) * 10
    offsets = (changes[~binary[changes + 1]] + 1) * 10
    frames = np.clip(np.round(times / 10).astype(int), 0, len(track) - 1)
    inside = binary[frames].astype(np.float32)
    relevant = onsets if edge == "start" else offsets
    signed = np.full(len(times), 5.0, dtype=np.float32)
    if len(relevant):
        position = np.searchsorted(relevant, times)
        best = np.full(len(times), np.inf)
        for index in (position - 1, position):
            valid = (index >= 0) & (index < len(relevant))
            candidate = np.where(
                valid, relevant[np.clip(index, 0, len(relevant) - 1)], np.inf
            )
            closer = np.abs(candidate - times) < np.abs(best - times)
            best = np.where(closer, candidate, best)
        signed = (np.clip(best - times, -5000, 5000) / 1000.0).astype(np.float32)
    starts = np.concatenate(([0], onsets))
    ends = np.concatenate((offsets, [len(track) * 10]))
    boundary = starts if edge == "start" else ends
    position = np.clip(
        np.searchsorted(starts, times, side="right") - 1, 0, len(boundary) - 1
    )
    remaining = np.where(
        inside > 0,
        times - boundary[position] if edge == "start" else boundary[position] - times,
        0.0,
    )
    remaining = (np.clip(remaining, 0, 5000) / 1000.0).astype(np.float32)
    return {
        "inside": inside,
        "rel_signed": signed,
        "rel_abs": np.abs(signed),
        "seg_rem": remaining,
    }


def _fa_features(line, baseline, edge):
    edge_index = 0 if edge == "start" else 1
    base_edge = baseline[edge_index]
    phones = [
        phone
        for phone in line.evidence.get("bfa_phones", [])
        if not phone.get("estimated", True)
    ]
    if edge == "start":
        anchor = min(
            phones,
            key=lambda phone: (phone["target_index"], phone["start"]),
            default=None,
        )
        anchor_time = None if anchor is None else anchor["start"]
    else:
        anchor = max(
            phones,
            key=lambda phone: (phone["target_index"], phone["end"]),
            default=None,
        )
        anchor_time = None if anchor is None else anchor["end"]
    values = {
        "fa_bfa_rel": np.clip(((anchor_time or base_edge) - base_edge) / 1000, -5, 5),
        "fa_bfa_conf": float(anchor.get("confidence", 0.0)) if anchor else 0.0,
        "fa_bfa_present": float(anchor is not None),
    }
    anchors = [] if anchor_time is None else [anchor_time]
    items = line.evidence.get("qfa_items", [])
    if items:
        item = items[0] if edge == "start" else items[-1]
        item_time = item["start"] if edge == "start" else item["end"]
        values["fa_qfa_rel"] = np.clip((item_time - base_edge) / 1000, -5, 5)
        anchors.append(item_time)
    else:
        values["fa_qfa_rel"] = 0.0
    for label, key in (("nfar", "nfa_raw_tokens"), ("nfav", "nfa_vocals_tokens")):
        units = line.evidence.get(key, {}).get("units", [])
        if units:
            unit = units[0] if edge == "start" else units[-1]
            unit_time = unit["start"] if edge == "start" else unit["end"]
            values[f"fa_{label}_rel"] = np.clip((unit_time - base_edge) / 1000, -5, 5)
            anchors.append(unit_time)
        else:
            values[f"fa_{label}_rel"] = 0.0
    values["fa_anchor_spread"] = (
        float(max(anchors) - min(anchors)) / 1000 if len(anchors) > 1 else 0.0
    )
    values["fa_anchor_count"] = len(anchors) / 4.0
    return values


def _fa_boundaries(line):
    phones = [
        phone
        for phone in line.evidence.get("bfa_phones", [])
        if not phone.get("estimated", True)
    ]
    items = line.evidence.get("qfa_items", [])
    units = line.evidence.get("nfa_vocals_tokens", {}).get("units", [])
    return {
        "bfa": np.sort(
            [value for phone in phones for value in (phone["start"], phone["end"])]
        )
        if phones
        else None,
        "qfa": np.sort(
            [value for item in items for value in (item["start"], item["end"])]
        )
        if items
        else None,
        "nfa": np.sort(
            [value for unit in units for value in (unit["start"], unit["end"])]
        )
        if units
        else None,
    }


def _nearest(boundaries, times):
    if boundaries is None or not len(boundaries):
        return np.full(len(times), 5.0, dtype=np.float32)
    position = np.searchsorted(boundaries, times)
    best = np.full(len(times), np.inf)
    for index in (position - 1, position):
        valid = (index >= 0) & (index < len(boundaries))
        candidate = np.where(
            valid, boundaries[np.clip(index, 0, len(boundaries) - 1)], np.inf
        )
        closer = np.abs(candidate - times) < np.abs(best - times)
        best = np.where(closer, candidate, best)
    return (np.clip(best - times, -5000, 5000) / 1000.0).astype(np.float32)


def _grid(lines, baselines, tracks, edge, radius, schema, is_guide=False):
    edge_index = 0 if edge == "start" else 1
    direction = 1.0 if edge == "start" else -1.0
    blocks = []
    meta = []
    needed = set(schema)
    for index, (line, baseline) in enumerate(zip(lines, baselines)):
        times = np.asarray(
            [
                baseline[edge_index] + offset
                for offset in range(-radius, radius + 1, 20)
            ],
            dtype=np.float32,
        )
        columns = {}
        for name in GRID_TRACKS:
            if not any(item.startswith(name) for item in needed):
                continue
            frames = np.clip(np.round(times / 10).astype(int), 0, len(tracks[name]) - 1)
            sampled = [
                tracks[name][
                    np.clip(frames + round(offset / 10), 0, len(tracks[name]) - 1)
                ]
                for offset in SAMPLES
            ]
            for offset, values in zip(SAMPLES, sampled):
                columns[f"{name}@{offset}"] = values
            columns[f"{name}_d20s"] = direction * (sampled[5] - sampled[3])
            columns[f"{name}_d20a"] = np.abs(sampled[5] - sampled[3])
            columns[f"{name}_d100s"] = direction * (sampled[8] - sampled[2])
            columns[f"{name}_d100a"] = np.abs(sampled[8] - sampled[2])
        for name in STRUCT_TRACKS:
            if f"{name}_inside" not in needed:
                continue
            structure = _structure(tracks[name], times, edge)
            columns[f"{name}_inside"] = structure["inside"]
            columns[f"{name}_rel_signed"] = structure["rel_signed"]
            columns[f"{name}_rel_abs"] = structure["rel_abs"]
            columns[f"{name}_seg_rem"] = structure["seg_rem"]
        duration = max(200, baseline[1] - baseline[0])
        offsets = times - baseline[edge_index]
        scale = float(radius)
        count = len(times)
        columns["off_rel"] = offsets / scale
        columns["off_abs"] = np.abs(offsets) / scale
        columns["off_zero"] = (offsets == 0).astype(np.float32)
        columns["dur_log"] = np.full(
            count, np.clip(np.log1p(duration) / 10, 0, 1), np.float32
        )
        agreement = line.evidence.get("alignment_agreement", {})
        detail = line.evidence.get("bfa_detail", {})
        for name, value in {
            "ev_agr_cov": float(agreement.get("coverage", 0.0)),
            "ev_agr_long": math.log1p(float(agreement.get("longest_area", 0.0))),
            "ev_agr_start": float(bool(agreement.get("start_agreement"))),
            "ev_agr_end": float(bool(agreement.get("end_agreement"))),
            "ev_bfa_est": float(detail.get("estimated_fraction", 1.0)),
            "ev_bfa_conf": math.log1p(float(detail.get("edge_confidence", 0.0))),
        }.items():
            columns[name] = np.full(count, value, np.float32)
        if "fa_bfa_rel" in needed:
            for name, value in _fa_features(line, baseline, edge).items():
                columns[name] = np.full(count, value, np.float32)
            bounds = _fa_boundaries(line)
            columns["fb_bfa_dist"] = _nearest(bounds["bfa"], times)
            columns["fb_qfa_dist"] = _nearest(bounds["qfa"], times)
            columns["fb_nfa_dist"] = _nearest(bounds["nfa"], times)
        previous_end = baselines[index - 1][1] if index else baseline[0]
        next_start = (
            baselines[index + 1][0] if index + 1 < len(baselines) else baseline[1]
        )
        columns["gap_prev"] = np.full(
            count,
            np.clip(np.log1p(max(0, baseline[0] - previous_end)) / 10, 0, 1),
            np.float32,
        )
        columns["gap_next"] = np.full(
            count,
            np.clip(np.log1p(max(0, next_start - baseline[1])) / 10, 0, 1),
            np.float32,
        )
        columns["edge_is_end"] = np.full(count, float(edge == "end"), np.float32)
        columns["is_guide"] = np.full(count, float(is_guide), np.float32)
        blocks.append(
            np.stack([columns[name] for name in schema], axis=1).astype(np.float32)
        )
        meta.extend(
            (index, float(time), float(offset)) for time, offset in zip(times, offsets)
        )
    return np.concatenate(blocks), meta


def _smooth(values, sigma=SMOOTH_SIGMA):
    xs = np.arange(-2 * sigma, 2 * sigma + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def _best_move(groups, key):
    entries = sorted(groups[key], key=lambda item: item[1])
    raw = _smooth(np.asarray([item[0] for item in entries]))
    offsets = np.asarray([item[2] for item in entries])
    zero = int(np.flatnonzero(offsets == 0)[0])
    best = int(np.argmax(raw))
    if best == zero:
        return 0.0, entries[zero][1]
    return raw[best] - raw[zero], entries[best][1]
