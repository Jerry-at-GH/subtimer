import contextlib
import difflib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import cache, refine, subtitle
from .speech import MANIFEST, bfa_target
from .speech import model as speech_model

SOURCES = (
    "baseline",
    "bfa",
    "qfa",
    "nfa_raw",
    "nfa_vocals",
    "asr",
    "guide",
    "vad",
)
SOURCE_INDEX = {source: index for index, source in enumerate(SOURCES)}
CONFIG = MANIFEST["selection_config"]
MARGIN = float(CONFIG["margin"])
OVERLAP_COST = float(CONFIG["overlap_cost"])
MICRO_SNAP_MS = int(CONFIG["micro_snap_ms"])
EDGE_NMS_MS = int(CONFIG["edge_nms_ms"])
MAX_DISPLACEMENT_MS = 12_000


class AcousticFeatures:
    def __init__(self, mel, nfa_raw, nfa_vocals):
        raw_mel = mel[0].astype(np.float32)
        vocals_mel = mel[1].astype(np.float32)
        self.raw_formants = raw_mel[4:32].mean(axis=0)
        self.vocals_formants = vocals_mel[4:32].mean(axis=0)
        self.frames = len(self.vocals_formants)
        self.vocal_dominance = np.clip(
            self.vocals_formants / (self.raw_formants + 1e-4), -10.0, 10.0
        )
        self.harmonicity = vocals_mel[4:32].std(axis=0)
        padded = np.pad(self.vocals_formants, (5, 5), mode="edge")
        cumulative = np.cumsum(np.pad(padded, (1, 0)))
        moving = (cumulative[5:] - cumulative[:-5]) / 5.0
        self.start_flux = moving[5 : 5 + self.frames] - moving[: self.frames]
        self.end_flux = -self.start_flux
        self.raw_tokens = _nfa_tokens(nfa_raw)
        self.vocals_tokens = _nfa_tokens(nfa_vocals)

    def extract(self, start, end, text):
        start_frame = min(self.frames - 1, max(0, round(start / 10.0)))
        end_frame = min(self.frames - 1, max(0, round(end / 10.0)))
        if end_frame <= start_frame:
            end_frame = min(self.frames - 1, start_frame + 1)
        start_slice = slice(max(0, start_frame - 3), min(self.frames, start_frame + 4))
        end_slice = slice(max(0, end_frame - 3), min(self.frames, end_frame + 4))
        interior = slice(start_frame, end_frame)
        exterior_start = self.vocal_dominance[max(0, start_frame - 30) : start_frame]
        exterior_end = self.vocal_dominance[
            end_frame : min(self.frames, end_frame + 30)
        ]
        exterior = np.concatenate((exterior_start, exterior_end))
        interior_dominance = float(self.vocal_dominance[interior].mean())
        exterior_dominance = float(exterior.mean()) if len(exterior) else 0.0

        tokens = [
            token
            for token in self.vocals_tokens
            if max(start - 200, token[0]) < min(end + 200, token[1])
        ]
        if not tokens:
            tokens = [
                token
                for token in self.raw_tokens
                if max(start - 200, token[0]) < min(end + 200, token[1])
            ]
        if tokens:
            token_chars = sum(len(token[2]) for token in tokens)
            character_coverage = min(2.0, token_chars / max(1.0, len(text)))
            start_token_distance = math.log1p(abs(start - tokens[0][0])) / 10.0
            end_token_distance = math.log1p(abs(end - tokens[-1][1])) / 10.0
            token_duration_ratio = min(
                2.0, (tokens[-1][1] - tokens[0][0]) / max(1.0, end - start)
            )
        else:
            character_coverage = 0.0
            start_token_distance = 1.0
            end_token_distance = 1.0
            token_duration_ratio = 0.0

        return [
            float(self.start_flux[start_slice].max()),
            float(self.end_flux[end_slice].max()),
            interior_dominance,
            exterior_dominance,
            interior_dominance - exterior_dominance,
            float(self.vocals_formants[interior].mean()),
            float(self.harmonicity[interior].mean()),
            float(self.harmonicity[start_slice].max()),
            character_coverage,
            start_token_distance,
            end_token_distance,
            token_duration_ratio,
        ]

    def micro_snap(self, span):
        values = [span.start, span.end]
        for coordinate, flux in enumerate((self.start_flux, self.end_flux)):
            frame = min(self.frames - 1, max(0, round(values[coordinate] / 10.0)))
            lower = max(0, frame - MICRO_SNAP_MS // 10)
            upper = min(self.frames, frame + MICRO_SNAP_MS // 10 + 1)
            best = lower + int(np.argmax(flux[lower:upper]))
            if flux[best] > flux[frame] + 0.1:
                values[coordinate] = float(best * 10)
        if values[1] - values[0] < subtitle.MIN_DURATION:
            values[1] = values[0] + subtitle.MIN_DURATION
        return subtitle.Span(*values)


def apply(lines, audio, vad, cache_dir, nfa, asr_words, progress, *, capture):
    timed = [line for line in lines if line.key and line.candidates.get("result")]
    if not timed:
        return
    progress("refine")
    cache_dir = Path(cache_dir)
    acoustic = capture.shared.get("acoustic")
    if acoustic is None:
        acoustic = AcousticFeatures(
            refine.mel_features(audio, cache_dir), nfa["nfa_raw"], nfa["nfa_vocals"]
        )
    by_bfa_target = defaultdict(list)
    by_qfa_target = defaultdict(list)
    for name, destination in (("bfa", by_bfa_target), ("qfa", by_qfa_target)):
        for key, value in (cache.load(cache_dir / f"{name}.json") or {}).items():
            phones = [
                phone for phone in value.get("phonemes", []) if not phone["estimated"]
            ]
            destination[json.loads(key)[2]].append(
                subtitle.Span(
                    min(phone["start"] for phone in phones),
                    max(phone["end"] for phone in phones),
                )
                if phones
                else subtitle.Span(value["start"], value["end"])
            )
    local_candidates = []
    with open(os.devnull, "w") as sink, contextlib.redirect_stderr(sink):
        for line in timed:
            text = line.text or line.spoken
            local_candidates.append(
                (by_bfa_target[bfa_target(text)], by_qfa_target[text])
            )
    asr_tokens = [(word.span.start, word.span.end, word.text) for word in asr_words]
    model = speech_model("path-selector")
    standard_spans = capture.select_path(
        timed, audio, local_candidates, vad, asr_tokens, acoustic, model
    )
    capture.collect(
        timed, standard_spans, audio, cache_dir, acoustic, local_candidates, vad
    )
    for line, span in zip(timed, standard_spans):
        line.candidates["result"] = span


def _nfa_tokens(values):
    return [
        (float(token["start"]), float(token["end"]), token["text"])
        for token in values
        if token["text"] not in ("", "▁")
    ]


def _candidates(line, baseline, local_candidates, vad):
    output = []
    identities = set()
    speech_intervals = vad["speech"]

    def add(source, span):
        if span is None:
            return
        span = subtitle.Span(float(span.start), float(span.end))
        if (
            span.duration < subtitle.MIN_DURATION
            or span.start < 0
            or abs(span.midpoint - baseline.midpoint) > MAX_DISPLACEMENT_MS
        ):
            return
        if source != "baseline":
            temporary = subtitle.Line("x", "x", candidates={"candidate": span})
            subtitle.snap(
                [temporary],
                "candidate",
                vad,
                start_radius=800,
                end_radius=800,
                ordered=False,
            )
            span = temporary.candidates["candidate"]
        identity = (round(span.start / 20), round(span.end / 20))
        if identity not in identities:
            identities.add(identity)
            output.append((source, span))

    add("baseline", baseline)
    for source in ("nfa_raw", "nfa_vocals", "asr"):
        add(source, line.candidates.get(source))
    if line.hint is not None:
        add("guide", line.hint)
    for span in local_candidates[0]:
        add("bfa", span)
    for span in local_candidates[1]:
        add("qfa", span)

    duration = max(subtitle.MIN_DURATION, baseline.duration)
    starts = [
        start
        for start, _ in speech_intervals
        if abs(start - baseline.midpoint) <= MAX_DISPLACEMENT_MS
    ]
    ends = [
        end
        for _, end in speech_intervals
        if abs(end - baseline.midpoint) <= MAX_DISPLACEMENT_MS
    ]
    proposals = []
    for start in starts:
        eligible = [end for end in ends if end - start >= subtitle.MIN_DURATION]
        if eligible:
            end = min(eligible, key=lambda value: abs((value - start) - duration))
            proposals.append(subtitle.Span(start, end))
    for end in ends:
        eligible = [start for start in starts if end - start >= subtitle.MIN_DURATION]
        if eligible:
            start = min(eligible, key=lambda value: abs((end - value) - duration))
            proposals.append(subtitle.Span(start, end))
    proposals.sort(key=lambda span: abs(span.midpoint - baseline.midpoint))
    for span in proposals[:8]:
        add("vad", span)
    return output


def _features(
    source,
    candidate,
    baseline,
    previous,
    following,
    text,
    references,
    speech_intervals,
    asr_tokens,
    acoustic,
):
    source_values = [0.0] * len(SOURCES)
    source_values[SOURCE_INDEX[source]] = 1.0
    duration = max(1.0, candidate.duration)
    raw = references.get("nfa_raw", baseline)
    overlap = sum(
        max(0.0, min(candidate.end, end) - max(candidate.start, start))
        for start, end in speech_intervals
    )
    matched = "".join(
        token
        for start, end, token in asr_tokens
        if max(candidate.start - 300, start) < min(candidate.end + 300, end)
    )
    alternatives = [span for span in references.values() if span != candidate]
    agreement = min(
        (abs(candidate.midpoint - span.midpoint) for span in alternatives),
        default=0.0,
    )
    values = source_values + [
        math.log1p(abs(candidate.midpoint - baseline.midpoint)) / 10.0,
        math.log1p(duration) / 10.0,
        math.log1p(duration / max(1, len(text))) / 8.0,
        float(np.clip((candidate.start - previous.end) / 12_000, -1, 1)),
        float(np.clip((following.start - candidate.end) / 12_000, -1, 1)),
        math.log1p(abs(candidate.midpoint - raw.midpoint)) / 10.0,
        overlap / duration,
        difflib.SequenceMatcher(None, text, matched).ratio() if matched else 0.0,
        math.log1p(agreement) / 10.0,
    ]
    return values + acoustic.extract(candidate.start, candidate.end, text)


def _transition(previous, current):
    overlap = max(0.0, previous.end - current.start)
    return -OVERLAP_COST * overlap / 1000.0


def _decode(lines, groups, scores, acoustic):
    local_scores = []
    spans = []
    for candidates, indexes in groups:
        local_scores.append(
            np.asarray(
                [
                    float(scores[index]) - (0.0 if position == 0 else MARGIN)
                    for position, index in enumerate(indexes)
                ],
                dtype=np.float64,
            )
        )
        spans.append([span for _, span in candidates])

    forward = []
    for line_index, local in enumerate(local_scores):
        if line_index == 0:
            forward.append(local.copy())
            continue
        current = np.empty(len(local), dtype=np.float64)
        for position, span in enumerate(spans[line_index]):
            current[position] = local[position] + max(
                forward[-1][previous_position] + _transition(previous_span, span)
                for previous_position, previous_span in enumerate(spans[line_index - 1])
            )
        forward.append(current)

    backward = [None] * len(lines)
    backward[-1] = local_scores[-1].copy()
    for line_index in range(len(lines) - 2, -1, -1):
        current = np.empty(len(local_scores[line_index]), dtype=np.float64)
        for position, span in enumerate(spans[line_index]):
            current[position] = local_scores[line_index][position] + max(
                backward[line_index + 1][next_position] + _transition(span, next_span)
                for next_position, next_span in enumerate(spans[line_index + 1])
            )
        backward[line_index] = current

    selected_spans = []
    for line_index, line in enumerate(lines):
        conditioned = (
            forward[line_index] + backward[line_index] - local_scores[line_index]
        )
        order = sorted(
            range(len(conditioned)),
            key=lambda position: (
                -conditioned[position],
                groups[line_index][1][position],
            ),
        )
        ranked_edges = {"start": [], "end": []}
        selected_span = None
        for position in order:
            span = spans[line_index][position]
            if position != 0:
                span = acoustic.micro_snap(span)
            if selected_span is None:
                selected_span = span
            for edge in ("start", "end"):
                value = getattr(span, edge)
                if all(
                    abs(value - prior) >= EDGE_NMS_MS for prior in ranked_edges[edge]
                ):
                    ranked_edges[edge].append(value)
        line.evidence["selection_alternatives"] = {
            edge: values[:2] for edge, values in ranked_edges.items()
        }
        selected_spans.append(selected_span)
    return selected_spans
