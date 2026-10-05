import hashlib
import math
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

from . import cache, speech
from .speech import PARAKEET_TOKENIZER, NFAContext, ctc_head
from .subtitle import MIN_DURATION, Line, Span, Word, match, normalize_key, snap

SUSPECT_SCORE = 0.9026
TRUSTED_SCORE = 0.9790
MIN_RELOCATION = 2_000
MIN_SEED_RELOCATION = 12_000
VIEW_AGREEMENT = 3_000
AGREEMENT_GATE = 200
SCORE_RADII = (1_500, 3_000, 5_000)
SEARCH_RADIUS = 3_000
SEARCH_OFFSETS_SECONDS = range(-15, 16)
CTC_BLOCK_MARGIN = 1.0
CTC_BLOCK_PADDING_MS = 2_000
GUIDE_DISPLACEMENT_RATIO = 3.0
GUIDE_DURATION_MIN_RATIO = 0.3
GUIDE_DURATION_MAX_RATIO = 2.5
GUIDE_CHARACTER_DURATION_MS = 120
GUIDE_SPEECH_COVERAGE_MIN = 0.5
GUIDE_CORROBORATION_PADDING_MS = 400
GUIDE_ORDER_MARGIN_MS = 200

_MORA = re.compile(r"[あ-んア-ヴ][ゃゅょァィゥェォャュョ]?")


def score_seeds(lines):
    selected = []
    features = []
    for line in lines:
        raw = line.candidates.get("nfa_raw")
        vocals = line.candidates.get("nfa_vocals")
        seed = line.candidates.get("nfa_seed")
        if not line.key or raw is None or vocals is None or seed is None:
            continue
        asr = line.candidates.get("asr")
        detail = line.evidence.get("asr", {})
        raw_tokens = line.evidence.get("nfa_raw_tokens", {})
        vocal_tokens = line.evidence.get("nfa_vocals_tokens", {})
        selected.append(line)
        features.append(
            (
                math.log1p(abs(raw.midpoint - vocals.midpoint)),
                math.log1p(abs(raw.start - vocals.start)),
                math.log1p(abs(raw.end - vocals.end)),
                math.log1p(abs(seed.midpoint - asr.midpoint) if asr else 30000),
                float(bool(detail.get("direct"))),
                float(bool(line.evidence.get("asr_interpolated"))),
                detail.get("coverage", 0.0),
                detail.get("longest", 0) / max(1, len(line.key)),
                math.log1p(detail.get("occurrences", 0)),
                math.log1p(seed.duration / max(1, len(line.key))),
                raw_tokens.get("blank_fraction", 1.0),
                vocal_tokens.get("blank_fraction", 1.0),
                math.log1p(raw_tokens.get("max_blank", 30000)),
                math.log1p(vocal_tokens.get("max_blank", 30000)),
                math.log1p(raw_tokens.get("duration_per_char", 1000)),
                math.log1p(vocal_tokens.get("duration_per_char", 1000)),
            )
        )
    if not features:
        return
    for line, score in zip(selected, speech.model("seed-scorer").predict(features)):
        line.evidence["seed_score"] = round(float(score), 4)


def _groups(indices):
    groups = []
    for index in indices:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(index)
    return groups


def _normalize_kana(value):
    value = unicodedata.normalize("NFKC", value)
    value = "".join(
        chr(ord(character) - 0x60) if "ァ" <= character <= "ヶ" else character
        for character in value
    )
    return normalize_key(value)


def _reading(value):
    import pyopenjtalk

    return _normalize_kana(
        pyopenjtalk.g2p(unicodedata.normalize("NFKC", value), kana=True)
    )


def _text_track(items, start_key, end_key):
    items = [
        (_normalize_kana(item["text"]), float(item[start_key]), float(item[end_key]))
        for item in items
        if item.get("text")
    ]
    items = [item for item in items if item[0]]
    total = sum(len(text) for text, _, _ in items)
    if not items or not total:
        return []

    position = 0
    track = [(0.0, items[0][1])]
    for text, start, end in items:
        track.append(((position + len(text) / 2) / total, (start + end) / 2))
        position += len(text)
    track.append((1.0, items[-1][2]))
    return track


def _phone_track(phones):
    usable = [
        phone
        for phone in phones
        if not phone["estimated"] and phone["confidence"] >= 0.01
    ]
    if len(usable) < 2:
        usable = [phone for phone in phones if not phone["estimated"]]
    if not usable:
        return []

    usable.sort(key=lambda phone: (phone["target_index"], phone["start"]))
    first = usable[0]["target_index"]
    denominator = max(1, usable[-1]["target_index"] - first)
    track = [(0.0, float(usable[0]["start"]))]
    track.extend(
        (
            (phone["target_index"] - first) / denominator,
            (phone["start"] + phone["end"]) / 2,
        )
        for phone in usable
    )
    track.append((1.0, float(usable[-1]["end"])))
    return track


def _agreement(nfa_units, bfa_phones, qfa_items, target):
    tracks = {
        "nfa": _text_track(nfa_units, "start", "end"),
        "bfa": _phone_track(bfa_phones),
        "qfa": _text_track(qfa_items, "start", "end"),
    }
    morae = _MORA.findall(_reading(target))
    progress_points = (
        [index / len(morae) for index in range(len(morae) + 1)]
        if morae
        else [index / 10 for index in range(11)]
    )
    checkpoints = []
    for progress in progress_points:
        values = []
        for name, track in tracks.items():
            if not track:
                continue
            value = track[-1][1]
            for left, right in pairwise(track):
                if left[0] <= progress <= right[0]:
                    if left[0] == right[0]:
                        value = statistics.mean((left[1], right[1]))
                    else:
                        fraction = (progress - left[0]) / (right[0] - left[0])
                        value = left[1] + fraction * (right[1] - left[1])
                    break
            values.append((value, name))
        values.sort()
        best = []
        for offset in range(len(values)):
            cluster = [values[offset]]
            for value in values[offset + 1 :]:
                if value[0] - cluster[0][0] > AGREEMENT_GATE:
                    break
                cluster.append(value)
            spread = cluster[-1][0] - cluster[0][0]
            best_spread = best[-1][0] - best[0][0] if best else float("inf")
            if len(cluster) > len(best) or (
                len(cluster) == len(best) and spread < best_spread
            ):
                best = cluster
        checkpoints.append(
            {
                "progress": progress,
                "count": len(best),
                "time": round(statistics.median(value for value, _ in best))
                if best
                else None,
                "spread": round(best[-1][0] - best[0][0]) if best else None,
                "aligners": [name for _, name in best],
            }
        )

    agreeable = [point["count"] >= 2 for point in checkpoints]
    areas = [
        [checkpoints[group[0]]["progress"], checkpoints[group[-1]]["progress"]]
        for group in _groups(index for index, value in enumerate(agreeable) if value)
    ]
    return {
        "coverage": round(sum(agreeable) / len(agreeable), 4),
        "longest_area": round(
            max((end - start for start, end in areas), default=0.0), 4
        ),
        "start_agreement": agreeable[0],
        "end_agreement": agreeable[-1],
        "areas": areas,
        "checkpoints": checkpoints,
    }


def score_agreement(lines: list[Line]) -> None:
    for line in lines:
        seed = line.candidates.get("nfa_seed")
        raw = line.candidates.get("nfa_raw")
        vocals = line.candidates.get("nfa_vocals")
        if seed is None:
            continue
        use_raw = raw is not None and (
            vocals is None
            or abs(raw.midpoint - seed.midpoint) <= abs(vocals.midpoint - seed.midpoint)
        )
        nfa = line.evidence.get(
            "nfa_raw_tokens" if use_raw else "nfa_vocals_tokens", {}
        )
        line.evidence["alignment_agreement"] = _agreement(
            nfa.get("units", []),
            line.evidence.get("bfa_phones", []),
            line.evidence.get("qfa_items", []),
            line.spoken,
        )


def repair_from_guide(lines, guide_words, local_words, vad):
    dialogue = [line for line in lines if line.candidates.get("result") is not None]
    if not dialogue or not guide_words:
        return

    flat = []
    char_to_word = []
    token_starts, token_ends = [], []
    for word in guide_words:
        text = normalize_key(word.text)
        flat.append(text)
        token_starts.append(word.span.start)
        token_ends.append(word.span.end)
        char_to_word.extend([len(token_starts) - 1] * len(text))
    flat = "".join(flat)
    speech = list(vad["speech"])

    for index, line in enumerate(dialogue):
        key = normalize_key(line.text)
        if not key:
            continue
        hits = []
        position = flat.find(key)
        while key and position != -1:
            first, last = char_to_word[position], char_to_word[position + len(key) - 1]
            hits.append((token_starts[first], token_ends[last]))
            position = flat.find(key, position + 1)
        if len(hits) != 1:
            continue
        start, end = hits[0]
        selected = line.candidates["result"]
        duration = max(selected.end - selected.start, 1)
        if abs(start - selected.start) <= max(
            MIN_RELOCATION, GUIDE_DISPLACEMENT_RATIO * duration
        ):
            continue
        if abs((start + end) / 2 - selected.midpoint) <= MIN_RELOCATION:
            continue
        corroboration = "".join(
            normalize_key(word.text)
            for word in local_words
            if word.span.end >= selected.start - GUIDE_CORROBORATION_PADDING_MS
            and word.span.start <= selected.end + GUIDE_CORROBORATION_PADDING_MS
        )
        if key in corroboration or (
            len(key) > 3 and key[: len(key) // 2] in corroboration
        ):
            continue
        expected = max(len(key), 1) * GUIDE_CHARACTER_DURATION_MS
        if end - start < GUIDE_DURATION_MIN_RATIO * expected:
            continue
        if end - start > GUIDE_DURATION_MAX_RATIO * expected:
            end = start + max(expected, MIN_DURATION)
        covered = sum(
            max(0, min(segment_end, end) - max(segment_start, start))
            for segment_start, segment_end in speech
        )
        if covered / (end - start) < GUIDE_SPEECH_COVERAGE_MIN:
            continue
        lower = (
            dialogue[index - 1].candidates["result"].end + GUIDE_ORDER_MARGIN_MS
            if index
            else 0
        )
        upper = (
            dialogue[index + 1].candidates["result"].start - GUIDE_ORDER_MARGIN_MS
            if index + 1 < len(dialogue)
            else float("inf")
        )
        if start < lower or end > upper:
            continue
        line.candidates["result"] = Span(start, max(end, start + MIN_DURATION))


def _trusted(line):
    raw = line.candidates.get("nfa_raw")
    vocals = line.candidates.get("nfa_vocals")
    raw_vocals_agree = (
        raw is not None
        and vocals is not None
        and abs(raw.midpoint - vocals.midpoint) <= 100
    )
    return line.evidence.get("seed_score", 1.0) > TRUSTED_SCORE and (
        bool(line.evidence.get("asr", {}).get("direct")) or raw_vocals_agree
    )


def _bracket(lines, first, last):
    left = next(
        (index for index in range(first - 1, -1, -1) if _trusted(lines[index])),
        None,
    )
    right = next(
        (index for index in range(last + 1, len(lines)) if _trusted(lines[index])),
        None,
    )
    if left is None or right is None:
        return None
    left_span = lines[left].candidates.get("result")
    right_span = lines[right].candidates.get("result")
    if left_span is None or right_span is None:
        return None
    return [round(left_span.end), round(right_span.start)]


def _reading_words(words, max_gap=500):
    groups = []
    for word in sorted(words, key=lambda item: (item.span.start, item.span.end)):
        if groups and word.span.start - groups[-1][-1].span.end <= max_gap:
            groups[-1].append(word)
        else:
            groups.append([word])

    output = []
    for group in groups:
        reading = _reading("".join(word.text for word in group))
        if not reading:
            continue
        weights = [max(1, len(_normalize_kana(word.text))) for word in group]
        confidence = sum(
            word.confidence * weight for word, weight in zip(group, weights)
        ) / sum(weights)
        output.append(
            Word(
                reading,
                Span(group[0].span.start, group[-1].span.end),
                confidence,
            )
        )
    return output


def _reading_lines(texts):
    return [Line(spoken=(reading := _reading(text)), key=reading) for text in texts]


def _direct(line, detail):
    duration = detail.get("end", 0) - detail.get("start", 0)
    plausible = duration <= max(2_000, len(line.key) * 600 + 1_000)
    if len(line.key) <= 2:
        return bool(detail.get("direct")) and plausible
    return bool(
        detail.get("direct")
        and detail.get("coverage", 0) >= 0.8
        and detail.get("longest", 0) >= 3
        and detail.get("max_gap", 0) <= 2_000
        and plausible
    )


def _snap_spans(spans, vad):
    temporary = [
        Line("x", "x", candidates={"result": Span(float(start), float(end))})
        for start, end in spans
    ]
    snap(temporary, "result", vad, start_radius=800, end_radius=800, ordered=False)
    return [
        [round(line.candidates["result"].start), round(line.candidates["result"].end)]
        for line in temporary
    ]


def _block_proposal(lines, block, words, vad):
    bracket = _bracket(lines, block[0], block[-1])
    if bracket is None or bracket[1] - bracket[0] < MIN_DURATION:
        return None
    source = _reading_words(
        word
        for word in words
        if word.span.end >= bracket[0] and word.span.start <= bracket[1]
    )
    source_lines = lines[block[0] : block[-1] + 1]
    matched_lines = _reading_lines([line.spoken for line in source_lines])
    spans, _, info = match(matched_lines, source)
    matched = [_direct(line, detail) for line, detail in zip(matched_lines, info)]
    distinctive = [
        index for index, line in enumerate(matched_lines) if len(line.key) >= 3
    ]
    if not (
        any(matched[index] for index in distinctive) if distinctive else any(matched)
    ):
        return None

    baseline = [lines[index].candidates["result"] for index in block]
    selected = [
        span if span is not None and matched[index] else baseline[index]
        for index, span in enumerate(spans)
    ]
    selected = _snap_spans(([span.start, span.end] for span in selected), vad)
    if (
        max(
            abs(sum(candidate) / 2 - current.midpoint)
            for candidate, current in zip(selected, baseline)
        )
        <= MIN_RELOCATION
    ):
        return None
    if any(
        sum(current) / 2 > sum(following) / 2
        for current, following in pairwise(selected)
    ):
        return None
    if any(
        end - start < MIN_DURATION or start < bracket[0] or end > bracket[1]
        for start, end in selected
    ):
        return None
    return {
        "kind": "asr_block",
        "lines": block,
        "texts": [line.spoken for line in source_lines],
        "bracket": bracket,
        "baseline": [[round(span.start), round(span.end)] for span in baseline],
        "selected": selected,
    }


def _seed_proposals(lines, index, maximum=4):
    line = lines[index]
    baseline = line.candidates["result"]
    bracket = _bracket(lines, index, index)
    if bracket is None:
        return []
    baseline_isolation = min(
        max(0.0, baseline.start - bracket[0]),
        max(0.0, bracket[1] - baseline.end),
    )

    alternatives = []
    labels = ("nfa_raw", "nfa_vocals", "asr", "bfa", "qfa")
    for priority, label in enumerate(labels):
        candidate = line.candidates.get(label)
        if (
            candidate is None
            or candidate.start < bracket[0]
            or candidate.end > bracket[1]
        ):
            continue
        if candidate.duration > max(max(1, len(line.key)) * 400 + 2_000, 15_000):
            continue
        if abs(candidate.midpoint - baseline.midpoint) <= MIN_SEED_RELOCATION:
            continue
        isolation = min(
            max(0.0, candidate.start - bracket[0]),
            max(0.0, bracket[1] - candidate.end),
        )
        if isolation + 1_000 >= baseline_isolation:
            continue
        detail = line.evidence.get(
            "bfa_detail" if label.startswith("bfa") else "qfa_detail", {}
        )
        warning = int(
            label.startswith(("bfa", "qfa")) and bool(detail.get("duration_warning"))
        )
        alternatives.append((isolation, warning, priority, candidate, label))

    clustered = []
    for alternative in sorted(alternatives, key=lambda item: item[:3]):
        if any(
            abs(alternative[3].midpoint - existing[3].midpoint) <= VIEW_AGREEMENT
            for existing in clustered
        ):
            continue
        clustered.append(alternative)
    return [
        {
            "kind": "nfa_seed",
            "lines": [index],
            "texts": [line.spoken],
            "bracket": bracket,
            "baseline": [[round(baseline.start), round(baseline.end)]],
            "selected": [[round(candidate.start), round(candidate.end)]],
            "source": label,
        }
        for _, _, _, candidate, label in clustered[:maximum]
    ]


def _transcript_score(target, words):
    target = _normalize_kana(target)
    transcript = _normalize_kana("".join(word.text for word in words))
    blocks = SequenceMatcher(
        None, target, transcript, autojunk=False
    ).get_matching_blocks()
    matched = sum(block.size for block in blocks)
    longest = max((block.size for block in blocks), default=0)
    coverage = matched / len(target) if target else 0.0
    weights = [max(1, len(_normalize_kana(word.text))) for word in words]
    confidence = (
        sum(word.confidence * weight for word, weight in zip(words, weights))
        / sum(weights)
        if weights
        else 0.0
    )
    return {
        "coverage": coverage,
        "longest": longest,
        "score": coverage * confidence,
        "direct": matched >= max(2, len(target) // 3)
        and longest >= min(2, len(target)),
    }


class CropRecognition:
    def __init__(self, session, cache_dir):
        self.session = session
        self.path = Path(cache_dir) / "occurrence-crops" / "words.json"
        self.rows = cache.load(self.path) or {}
        self.changed = False

    def save(self):
        if self.changed:
            cache.save(self.path, self.rows)
            self.changed = False

    def words(self, pcm, start, end):
        first, last = round(start * 16), round(end * 16)
        if last <= first:
            return []
        crop = pcm[first:last]

        digest = hashlib.sha256(crop).hexdigest()
        rows = self.rows.get(digest)
        if rows is None:
            rows = []
            for segment in self.session.get().transcribe(crop, language="ja"):
                if segment.words:
                    rows.extend(
                        [
                            str(w.text).strip(),
                            float(w.start),
                            float(w.end),
                            float(w.confidence),
                        ]
                        for w in segment.words
                        if str(w.text).strip()
                    )
                elif str(segment.text).strip():
                    rows.append(
                        [
                            str(segment.text).strip(),
                            float(segment.start),
                            float(segment.end),
                            0.0,
                        ]
                    )
            self.rows[digest] = rows
            self.changed = True

        return [
            Word(text, Span(start + a * 1000, start + z * 1000), confidence)
            for text, a, z, confidence in rows
        ]


def _score_location(current, pcm, target, span):
    midpoint = sum(span) / 2
    duration = len(pcm) / 16
    scores = []
    for radius in SCORE_RADII:
        start = max(0.0, midpoint - radius)
        end = min(duration, midpoint + radius)
        scores.append(_transcript_score(target, current.words(pcm, start, end)))
    return max(scores, key=lambda score: (score["score"], score["longest"]))


def _matched_span(target, words):
    target = _normalize_kana(target)
    source = ""
    times = []
    for word in words:
        text = _normalize_kana(word.text)
        source += text
        times.extend([(word.span.start, word.span.end)] * len(text))

    matched = 0
    matched_times = []
    for _, source_start, size in SequenceMatcher(
        None, target, source, autojunk=False
    ).get_matching_blocks():
        if size >= 2:
            matched += size
            matched_times.extend(times[source_start : source_start + size])
    if matched < max(2, len(target) // 3) or not matched_times:
        return None
    return [round(matched_times[0][0]), round(matched_times[-1][1])]


def _search(current, pcm, target, midpoint):
    duration = len(pcm) / 16
    results = []
    for offset in SEARCH_OFFSETS_SECONDS:
        center = midpoint + offset * 1_000
        start = max(0.0, center - SEARCH_RADIUS)
        end = min(duration, center + SEARCH_RADIUS)
        words = current.words(pcm, start, end)
        results.append(
            {
                **_transcript_score(target, words),
                "span": _matched_span(target, words),
                "center": center,
            }
        )
    return max(
        results,
        key=lambda result: (
            result["score"],
            result["longest"],
            -abs(result["center"] - midpoint),
        ),
    )


def _verify_single(current, views, proposal, vad):
    details = {}
    for view, pcm in views.items():
        details[view] = {
            "baseline": _score_location(
                current, pcm, proposal["texts"][0], proposal["baseline"][0]
            ),
            "selected": _score_location(
                current, pcm, proposal["texts"][0], proposal["selected"][0]
            ),
        }
    score = min(
        detail["selected"]["score"] - detail["baseline"]["score"]
        for detail in details.values()
    )
    accepted = all(
        detail["selected"]["direct"]
        and detail["selected"]["score"] > detail["baseline"]["score"]
        for detail in details.values()
    )
    if accepted or proposal["kind"] != "nfa_seed":
        return accepted, proposal["selected"], details, score

    midpoint = sum(proposal["selected"][0]) / 2
    searched = {
        view: _search(current, pcm, proposal["texts"][0], midpoint)
        for view, pcm in views.items()
    }
    details["search"] = searched
    spans = [searched[view]["span"] for view in ("raw", "vocals")]
    if not all(spans) or abs(sum(spans[0]) / 2 - sum(spans[1]) / 2) > VIEW_AGREEMENT:
        return False, proposal["selected"], details, score
    if not all(
        searched[view]["direct"]
        and searched[view]["score"] > details[view]["baseline"]["score"]
        for view in ("raw", "vocals")
    ):
        return False, proposal["selected"], details, score

    recovered = [round(statistics.mean(edges)) for edges in zip(*spans)]
    if (
        abs(sum(recovered) / 2 - sum(proposal["baseline"][0]) / 2)
        <= MIN_SEED_RELOCATION
    ):
        return False, proposal["selected"], details, score
    score = min(
        searched[view]["score"] - details[view]["baseline"]["score"]
        for view in ("raw", "vocals")
    )
    return True, _snap_spans([recovered], vad), details, score


def _ctc_block_llr(current_pcm, proposal):
    import numpy as np
    import sentencepiece
    import torch
    from torch.nn import functional

    weight, bias = ctc_head()
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(PARAKEET_TOKENIZER))
    blank = len(bias) - 1
    device = torch.device("cuda")
    encoder = NFAContext.open(8)
    ids = [blank]
    for text in proposal["texts"]:
        for part in text.split():
            for value in tokenizer.encode_as_ids(part):
                if ids[-1] != value:
                    ids.append(value)
                    ids.append(blank)
    extended = [blank]
    for value in ids[1:]:
        extended.extend((value, blank))
    states = torch.tensor(extended, dtype=torch.long, device=device)
    doubled = torch.zeros(len(extended), dtype=torch.bool, device=device)
    if len(extended) > 2:
        doubled[2:] = states[2:] == states[:-2]
    weight_t = torch.from_numpy(weight).to(device)
    bias_t = torch.from_numpy(bias).to(device)
    scores = {}
    try:
        for name, location in (
            ("baseline", proposal["baseline"]),
            ("proposed", proposal["selected"]),
        ):
            start = max(
                0, min(int(span[0]) for span in location) - CTC_BLOCK_PADDING_MS
            )
            end = min(
                len(current_pcm) // 16,
                max(int(span[1]) for span in location) + CTC_BLOCK_PADDING_MS,
            )
            crop = np.ascontiguousarray(current_pcm[start * 16 : end * 16])
            encoded = encoder.encode(crop, np)
            with torch.no_grad():
                logits = functional.linear(
                    torch.from_numpy(encoded).to(device), weight_t, bias_t
                )
                log_probs = functional.log_softmax(logits, dim=-1, dtype=torch.float32)
                del logits, encoded
                emission = log_probs[:, states]
                previous = torch.full((len(extended),), -1e30, device=device)
                previous[0] = emission[0, 0]
                if len(extended) > 1:
                    previous[1] = emission[0, 1]
                for frame in range(1, emission.shape[0]):
                    advance = torch.full_like(previous, -1e30)
                    advance[1:] = previous[:-1]
                    skip = torch.full_like(previous, -1e30)
                    skip[2:] = previous[:-2]
                    skip.masked_fill_(doubled, -1e30)
                    previous = (
                        torch.logsumexp(torch.stack((previous, advance, skip)), dim=0)
                        + emission[frame]
                    )
                target = (
                    float(torch.logsumexp(previous[-2:], dim=0))
                    if len(extended) > 1
                    else float(previous[0])
                )
                empty = float(log_probs[:, blank].sum())
                del log_probs
            scores[name] = (target - empty) / max(1, len(ids))
    finally:
        encoder.close()
    return scores["proposed"] - scores["baseline"]


def _verify_joint(current, views, proposal, vad):
    evidence = {}
    for view, pcm in views.items():
        words = _reading_words(current.words(pcm, *proposal["bracket"]))
        lines = _reading_lines(proposal["texts"])
        spans, _, info = match(lines, words)
        evidence[view] = [
            {
                "direct": _direct(line, detail),
                "span": None if span is None else [round(span.start), round(span.end)],
            }
            for line, span, detail in zip(lines, spans, info)
        ]
    selected = [list(span) for span in proposal["selected"]]
    original_changes = {
        index
        for index, (baseline, selected_span) in enumerate(
            zip(proposal["baseline"], proposal["selected"])
        )
        if baseline != selected_span
    }
    for index in range(len(selected)):
        raw = evidence["raw"][index]
        vocals = evidence["vocals"][index]
        valid = (
            raw["direct"]
            and vocals["direct"]
            and raw["span"] is not None
            and vocals["span"] is not None
            and abs(sum(raw["span"]) / 2 - sum(vocals["span"]) / 2) <= VIEW_AGREEMENT
        )
        if not valid:
            if index in original_changes:
                llr = _ctc_block_llr(views["raw"], proposal)
                if llr > CTC_BLOCK_MARGIN:
                    return True, _snap_spans(selected, vad), evidence, llr
                return False, proposal["selected"], evidence, -1.0
            continue

        recovered = [
            round(statistics.mean((raw["span"][edge], vocals["span"][edge])))
            for edge in (0, 1)
        ]
        recovered_midpoint = sum(recovered) / 2
        baseline_midpoint = sum(proposal["baseline"][index]) / 2
        if index in original_changes:
            selected_distance = abs(recovered_midpoint - sum(selected[index]) / 2)
            if selected_distance > VIEW_AGREEMENT or selected_distance + 500 >= abs(
                recovered_midpoint - baseline_midpoint
            ):
                return False, proposal["selected"], evidence, -1.0
        elif abs(recovered_midpoint - baseline_midpoint) > MIN_RELOCATION:
            selected[index] = recovered
    return True, _snap_spans(selected, vad), evidence, 1.0


def _contained(lines, proposal, selected):
    bracket = _bracket(lines, proposal["lines"][0], proposal["lines"][-1])
    if (
        bracket is None
        or bracket != proposal["bracket"]
        or len(selected) != len(proposal["lines"])
    ):
        return False
    if any(
        len(span) != 2
        or span[1] - span[0] < MIN_DURATION
        or span[0] < bracket[0]
        or span[1] > bracket[1]
        for span in selected
    ):
        return False
    return not any(
        sum(current) / 2 > sum(following) / 2
        for current, following in pairwise(selected)
    )


def _ordered_seed_support(lines, replacements):
    replacements = dict(replacements)

    def support(index):
        line = lines[index]
        seed = line.candidates.get("nfa_seed")
        raw = line.candidates.get("nfa_raw")
        vocals = line.candidates.get("nfa_vocals")
        baseline = line.candidates.get("result")
        if (
            seed is None
            or raw is None
            or vocals is None
            or baseline is None
            or abs(raw.midpoint - vocals.midpoint) > 500
            or abs(seed.midpoint - baseline.midpoint) <= MIN_SEED_RELOCATION
        ):
            return None
        return seed

    while True:
        ordered = sorted(replacements)
        first = ordered[0]
        if (
            first
            and lines[first - 1].candidates["result"].midpoint
            > replacements[first].midpoint
        ):
            span = support(first - 1)
            if span is None:
                return None
            replacements[first - 1] = span
            continue
        last = ordered[-1]
        if (
            last + 1 < len(lines)
            and replacements[last].midpoint
            > lines[last + 1].candidates["result"].midpoint
        ):
            span = support(last + 1)
            if span is None:
                return None
            replacements[last + 1] = span
            continue
        break
    spans = [
        replacements.get(index, line.candidates["result"])
        for index, line in enumerate(lines)
    ]
    return (
        replacements
        if not any(a.midpoint > b.midpoint for a, b in pairwise(spans))
        else None
    )


def repair(lines, proposal_words, vad, audio, cache_dir, progress, session):
    dialogue = [line for line in lines if line.candidates.get("result") is not None]
    suspect_blocks = _groups(
        index
        for index, line in enumerate(dialogue)
        if line.evidence.get("seed_score", 1.0) <= SUSPECT_SCORE
    )

    proposals = []
    for block in suspect_blocks:
        proposal = _block_proposal(dialogue, block, proposal_words, vad)
        if proposal is not None:
            proposals.append(proposal)
        for label in ("nfa_vocals", "nfa_raw"):
            words = [
                Word(token["text"], Span(token["start"], token["end"]))
                for index in block
                for token in dialogue[index]
                .evidence.get(f"{label}_tokens", {})
                .get("units", [])
            ]
            if not words:
                continue
            proposal = _block_proposal(dialogue, block, words, vad)
            if proposal is not None:
                proposals.append(proposal)
    for block in suspect_blocks:
        for index in block:
            proposals.extend(_seed_proposals(dialogue, index))
    if not proposals:
        return []
    proposals.sort(key=lambda proposal: proposal["lines"])

    cache_path = cache_dir / "occurrence" / f"local_full_{cache.key(proposals)}.json"
    decisions = cache.load(cache_path)
    if decisions is None:
        import numpy as np
        import soundfile

        views = {}
        for name, path in (("raw", audio.raw), ("vocals", audio.vocals)):
            pcm, _ = soundfile.read(str(path), dtype="float32", always_2d=True)
            views[name] = np.ascontiguousarray(pcm[:, 0], dtype=np.float32)
        progress("refine")
        current = CropRecognition(session, cache_dir)
        decisions = []
        for proposal in proposals:
            verify = _verify_single if len(proposal["lines"]) == 1 else _verify_joint
            accepted, selected, details, verification_score = verify(
                current, views, proposal, vad
            )
            decision = {
                **proposal,
                "accepted": bool(accepted),
                "selected": selected,
                "verification_score": round(float(verification_score), 6),
                "details": details,
            }
            if decision["accepted"] and not _contained(dialogue, decision, selected):
                decision["accepted"] = False
            decisions.append(decision)
        current.save()
        cache.save(cache_path, decisions)

    accepted = [decision for decision in decisions if decision["accepted"]]
    blocks = [decision for decision in accepted if len(decision["lines"]) > 1]
    block_lines = {index for decision in blocks for index in decision["lines"]}
    singles = {}
    for decision in accepted:
        if len(decision["lines"]) != 1:
            continue
        index = decision["lines"][0]
        if index in block_lines:
            continue
        previous = singles.get(index)
        if (
            previous is None
            or decision["verification_score"] > previous["verification_score"]
        ):
            singles[index] = decision
    selected_decisions = sorted(
        [*blocks, *singles.values()], key=lambda decision: decision["lines"]
    )

    for decision in selected_decisions:
        replacements = {
            index: Span(float(selected[0]), float(selected[1]))
            for index, selected in zip(decision["lines"], decision["selected"])
        }
        replacements = _ordered_seed_support(dialogue, replacements)
        if replacements is None:
            decision["accepted"] = False
            continue
        originals = {
            index: dialogue[index].candidates["result"] for index in replacements
        }
        for index, selected in replacements.items():
            dialogue[index].candidates["result"] = selected
        results = [line.candidates["result"] for line in dialogue]
        valid = all(
            0 <= span.start < span.end <= audio.duration for span in results
        ) and not any(
            current.midpoint > following.midpoint
            for current, following in pairwise(results)
        )
        if not valid:
            for index, original in originals.items():
                dialogue[index].candidates["result"] = original
            decision["accepted"] = False
            continue

        for index, baseline, selected in zip(
            decision["lines"], decision["baseline"], decision["selected"]
        ):
            if selected != baseline:
                dialogue[index].evidence.pop("fixed_fallback", None)
            dialogue[index].evidence["occurrence_repair"] = {
                "kind": decision["kind"],
                "score": decision["verification_score"],
                "distance": abs(sum(selected) / 2 - sum(baseline) / 2),
            }
        for index in replacements.keys() - set(decision["lines"]):
            baseline = originals[index]
            selected = replacements[index]
            dialogue[index].evidence.pop("fixed_fallback", None)
            dialogue[index].evidence["occurrence_repair"] = {
                "kind": "nfa_support",
                "score": decision["verification_score"],
                "distance": abs(selected.midpoint - baseline.midpoint),
            }

    return decisions


def repair_hints(dialogue, audio, vad, cache_dir, session, progress):
    flagged = [
        (index, line)
        for index, line in enumerate(dialogue)
        if line.hint_calibrated
        and line.hint is not None
        and line.candidates.get("result") is not None
        and abs(line.hint.midpoint - line.candidates["result"].midpoint)
        > MIN_RELOCATION
    ]
    if not flagged:
        return []

    import numpy as np
    import soundfile

    progress("refine")
    views = {}
    for name, path in (("raw", audio.raw), ("vocals", audio.vocals)):
        pcm, _ = soundfile.read(str(path), dtype="float32", always_2d=True)
        views[name] = np.ascontiguousarray(pcm[:, 0], dtype=np.float32)
    current = CropRecognition(session, cache_dir)
    repaired = []
    for index, line in flagged:
        baseline = line.candidates["result"]
        target = line.spoken
        base_views = {
            name: _hint_view(current, pcm, target, baseline.midpoint)
            for name, pcm in views.items()
        }
        hint_views = {
            name: _hint_view(current, pcm, target, line.hint.midpoint)
            for name, pcm in views.items()
        }
        spans = [hint_views[name]["span"] for name in ("raw", "vocals")]
        if not all(spans):
            continue
        new_mid = sum(sum(span) for span in spans) / 4
        previous_end = (
            dialogue[index - 1].candidates["result"].end if index else new_mid
        )
        following_start = (
            dialogue[index + 1].candidates["result"].start
            if index + 1 < len(dialogue)
            else new_mid
        )
        if not (
            all(hint_views[name]["direct"] for name in hint_views)
            and all(hint_views[name]["score"] >= 0.6 for name in hint_views)
            and abs(new_mid - line.hint.midpoint) <= 3_000
            and abs(sum(spans[0]) / 2 - sum(spans[1]) / 2) <= 3_000
            and (
                all(
                    hint_views[name]["score"] > base_views[name]["score"]
                    for name in hint_views
                )
                or sum(base_views[name]["score"] for name in base_views) < 1.2
            )
            and abs(new_mid - baseline.midpoint) > MIN_RELOCATION
            and previous_end - 60_000 <= new_mid <= following_start + 60_000
        ):
            continue
        start = sum(span[0] for span in spans) / len(spans)
        end = sum(span[1] for span in spans) / len(spans)
        probe = Line(line.spoken, line.key)
        probe.candidates["result"] = Span(start, end).at_least()
        snap([probe], "result", vad, start_radius=800, end_radius=800, ordered=False)
        repaired_span = probe.candidates["result"]
        previous_midpoint = (
            dialogue[index - 1].candidates["result"].midpoint
            if index
            else repaired_span.midpoint
        )
        following_midpoint = (
            dialogue[index + 1].candidates["result"].midpoint
            if index + 1 < len(dialogue)
            else repaired_span.midpoint
        )
        if not previous_midpoint <= repaired_span.midpoint <= following_midpoint:
            continue
        line.candidates["result"] = repaired_span
        repaired.append(index)
    current.save()
    return repaired


def _hint_view(current, pcm, target, midpoint):
    duration = len(pcm) / 16
    start = max(0.0, midpoint - SEARCH_RADIUS)
    end = min(duration, midpoint + SEARCH_RADIUS)
    if end - start < 1_000:
        return {"score": 0.0, "direct": False, "span": None}
    words = current.words(pcm, start, end)
    return {**_transcript_score(target, words), "span": _matched_span(target, words)}
