import re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from itertools import pairwise

MIN_DURATION = 200
WORD_CHARS = (
    "A-Za-z0-9\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
)
NON_WORD = re.compile(rf"[^{WORD_CHARS}]+")
ASS_OVERRIDE = re.compile(r"\{\\[^}]*\}")
ASS_ESCAPE = re.compile(r"\\[Nnh]")


@dataclass(frozen=True, slots=True)
class Span:
    start: float
    end: float

    @property
    def duration(self):
        return self.end - self.start

    @property
    def midpoint(self):
        return (self.start + self.end) / 2

    def shift(self, amount):
        return Span(self.start + amount, self.end + amount)

    def at_least(self, duration=MIN_DURATION):
        return (
            self
            if self.duration >= duration
            else Span(self.start, self.start + duration)
        )


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    span: Span
    confidence: float = 1.0


@dataclass(slots=True)
class Line:
    spoken: str
    key: str
    text: str | None = None
    hint: Span | None = None
    hint_calibrated: bool = False
    candidates: dict = field(default_factory=dict)
    evidence: dict = field(default_factory=dict)


def alignment_text(value):
    value = unicodedata.normalize("NFKC", value)
    return ASS_ESCAPE.sub(" ", ASS_OVERRIDE.sub(" ", value))


def normalize_key(value):
    return NON_WORD.sub("", alignment_text(value))


def make_lines(dialogues):
    lines = []
    for dialogue in dialogues:
        spoken = NON_WORD.sub(" ", alignment_text(dialogue)).strip()
        key = NON_WORD.sub("", spoken)
        if not key:
            raise ValueError("transcript line contains no alignable text")
        lines.append(Line(spoken, key, dialogue))
    return lines


def match(lines, words):
    source_text = ""
    source_times = []
    for word in words:
        key = normalize_key(word.text)
        if not key:
            continue
        for index in range(len(key)):
            offset = index / (len(key) - 1) if len(key) > 1 else 0.5
            point = word.span.start + offset * word.span.duration
            source_times.append(
                (point, point) if len(key) > 1 else (word.span.start, word.span.end)
            )
        source_text += key

    target_text = ""
    ranges = []
    for line in lines:
        start = len(target_text)
        target_text += line.key
        ranges.append((start, len(target_text)))
    spans = [None] * len(lines)
    details = [{} for _ in lines]
    if not source_text or not target_text:
        return spans, 0, details

    times = [None] * len(target_text)
    block_ids = [None] * len(target_text)
    for block_id, block in enumerate(
        SequenceMatcher(
            None, target_text, source_text, autojunk=False
        ).get_matching_blocks()
    ):
        target_start, source_start, size = block
        if size < 2:
            continue
        for offset in range(size):
            times[target_start + offset] = source_times[source_start + offset]
            block_ids[target_start + offset] = block_id

    matched = 0
    for index, line in enumerate(lines):
        if not line.key:
            continue
        start, end = ranges[index]
        line_times = [item for item in times[start:end] if item is not None]
        longest = run = 0
        previous_block = None
        for block_id in block_ids[start:end]:
            run = (
                run + 1
                if block_id is not None and block_id == previous_block
                else int(block_id is not None)
            )
            longest = max(longest, run)
            previous_block = block_id
        if len(line_times) < max(2, len(line.key) // 3):
            continue
        gaps = [current[0] - previous[1] for previous, current in pairwise(line_times)]
        maximum_gap = max(gaps, default=0)
        runs = [[line_times[0]]]
        for previous, current in pairwise(line_times):
            if current[0] - previous[1] > 5000:
                runs.append([])
            runs[-1].append(current)
        selected = max(runs, key=len) if maximum_gap > 5000 else line_times
        details[index] = {
            "coverage": len(line_times) / len(line.key),
            "longest": longest,
            "max_gap": maximum_gap,
            "occurrences": sum(
                1 for _ in re.finditer(f"(?={re.escape(line.key)})", source_text)
            ),
            "start": selected[0][0],
            "end": selected[-1][1],
            "direct": True,
        }
        spans[index] = Span(int(selected[0][0]), int(selected[-1][1])).at_least()
        matched += 1
    return spans, matched, details


def apply_guide_segments(lines, guide_segments):
    segments = []
    for segment in guide_segments:
        text = segment["text"]
        if not text.strip():
            continue
        value = alignment_text(re.split(r"\\[Nn]|\n", text, maxsplit=1)[0])
        segments.append((segment, value))
    segments.sort(key=lambda item: (item[0]["start"], item[0]["end"]))
    words = [
        Word(value, Span(segment["start"], segment["end"]))
        for segment, value in segments
    ]
    spans, _, details = match(lines, words)
    for index, detail in enumerate(details):
        if spans[index] is not None and detail.get("max_gap", 0) > 5000:
            spans[index] = None
    used = set()
    for index, line in enumerate(lines):
        if spans[index] is not None or len(line.key) != 1:
            continue
        previous = next((i for i in range(index - 1, -1, -1) if spans[i]), None)
        following = next((i for i in range(index + 1, len(lines)) if spans[i]), None)
        if previous is None or following is None:
            continue
        if (
            sum(
                spans[i] is None and lines[i].key == line.key
                for i in range(previous + 1, following)
            )
            != 1
        ):
            continue
        choices = [
            (word_index, word)
            for word_index, word in enumerate(words)
            if word_index not in used
            and normalize_key(word.text) == line.key
            and word.span.start >= spans[previous].end
            and word.span.end <= spans[following].start
        ]
        if len(choices) == 1:
            word_index, word = choices[0]
            recovered = word.span.at_least()
            if recovered.end <= spans[following].start:
                spans[index] = recovered
                used.add(word_index)
    for line, span in zip(lines, spans):
        if span:
            line.hint = span
    return words


def snap(lines, label, vad, start_radius=400, end_radius=800, ordered=True):
    starts = [start for start, _ in vad["speech"]]
    ends = [end for _, end in vad["speech"]]
    previous_end = None
    for line in lines:
        candidate = line.candidates.get(label)
        if candidate is None:
            continue
        edges = []
        for value, boundaries, radius in (
            (int(candidate.start), starts, start_radius),
            (int(candidate.end), ends, end_radius),
        ):
            position = bisect_left(boundaries, value)
            choices = boundaries[max(0, position - 1) : position + 1]
            choices = [choice for choice in choices if abs(choice - value) <= radius]
            edges.append(
                min(choices, key=lambda choice: abs(choice - value))
                if choices
                else value
            )
        start, end = edges
        if label == "bfa" and candidate.start - start > 200:
            start = int(candidate.start)
        if label == "qfa" and candidate.end - end > 100:
            end = int(candidate.end)
        if end < start:
            start, end = int(candidate.start), int(candidate.end)
        if ordered and previous_end is not None:
            start = max(start, previous_end + 1)
        if end < start:
            end = start + MIN_DURATION
        line.candidates[label] = Span(start, end).at_least()
        previous_end = int(line.candidates[label].end)
    for line in lines:
        candidate = line.candidates.get(label)
        if candidate is None:
            continue
        overlaps = [
            (start, end)
            for start, end in vad["trim"]
            if end >= candidate.start and start <= candidate.end
        ]
        if overlaps:
            start = max(candidate.start, min(start for start, _ in overlaps))
            end = min(candidate.end, max(end for _, end in overlaps))
            if end >= start:
                line.candidates[label] = Span(start, end).at_least()


def apply_words(lines, words, label, vad):
    words = list(words)
    spans, _, details = match(lines, words)
    anchors = {
        index
        for index, detail in enumerate(details)
        if detail.get("direct")
        and detail.get("coverage", 0) >= 0.8
        and detail.get("max_gap", 0) <= 5000
    }
    used = set()
    for index, line in enumerate(lines):
        if spans[index] is not None or len(line.key) != 1:
            continue
        previous = next((i for i in range(index - 1, -1, -1) if i in anchors), None)
        following = next(
            (i for i in range(index + 1, len(lines)) if i in anchors), None
        )
        if previous is None or following is None:
            continue
        if (
            sum(
                spans[i] is None and lines[i].key == line.key
                for i in range(previous + 1, following)
            )
            != 1
        ):
            continue
        choices = [
            (word_index, word)
            for word_index, word in enumerate(words)
            if word_index not in used
            and normalize_key(word.text) == line.key
            and word.span.start >= details[previous]["end"]
            and word.span.end <= details[following]["start"]
        ]
        if len(choices) == 1:
            word_index, word = choices[0]
            recovered = word.span.at_least()
            if recovered.end <= details[following]["start"]:
                spans[index] = recovered
                details[index] = {
                    "coverage": 1.0,
                    "longest": 1,
                    "max_gap": 0,
                    "occurrences": 1,
                    "start": recovered.start,
                    "end": recovered.end,
                    "direct": True,
                    "exact_single": True,
                }
                used.add(word_index)
    for line, span, detail in zip(lines, spans, details):
        line.evidence[label] = detail
        if span:
            line.candidates[label] = span
    snap(lines, label, vad)

    direct = list(spans)
    spans = [
        line.candidates.get(label) if span is not None else None
        for line, span in zip(lines, spans)
    ]
    index = 0
    while index < len(lines):
        if spans[index] is not None:
            index += 1
            continue
        start_index = index
        while index < len(lines) and spans[index] is None:
            index += 1
        previous = next(
            (spans[i] for i in range(start_index - 1, -1, -1) if spans[i]), None
        )
        following = next((spans[i] for i in range(index, len(lines)) if spans[i]), None)
        count = index - start_index
        if previous and following:
            gap_start, gap_end = previous.end, following.start
        elif previous:
            gap_start, gap_end = previous.end, previous.end + count * 1000
        elif following:
            gap_end, gap_start = following.start, max(0, following.start - count * 1000)
        else:
            continue
        step = (gap_end - gap_start) / (count + 1)
        if step < MIN_DURATION:
            continue
        duration = max(MIN_DURATION, min(1000, step * 0.8))
        for offset, line_index in enumerate(range(start_index, index), 1):
            midpoint = gap_start + step * offset
            spans[line_index] = Span(
                int(max(0, midpoint - duration / 2)), int(midpoint + duration / 2)
            )
    for line, span, was_direct in zip(lines, spans, direct):
        if line.candidates.get(label) is None and span:
            line.candidates[label] = span
            if was_direct is None:
                line.evidence[f"{label}_interpolated"] = True


def apply_nfa(lines, tokens, label, compress_long=False):
    tokens = [{**token, "key": normalize_key(token["text"])} for token in tokens]
    dialogue = iter(lines)
    try:
        line = next(dialogue)
    except StopIteration:
        return
    remaining = line.key
    first_search = 0
    for index, token in enumerate(tokens):
        remaining = remaining.removeprefix(token["key"])
        if remaining:
            continue
        first = next(
            (
                position
                for position in range(first_search, len(tokens))
                if tokens[position]["key"]
            ),
            None,
        )
        if first is None:
            return
        selected = tokens[first : index + 1]
        padding = 0
        for following in tokens[index + 1 : index + 3]:
            if following["key"]:
                break
            padding += following["end"] - following["start"]
        start = tokens[first]["start"]
        end = token["end"] + min(padding, 1200) * 0.5
        if compress_long:
            long_tokens = [
                (
                    (offset + 0.5) / max(1, index - first),
                    token["end"] - token["start"] - 1200,
                )
                for offset, token in enumerate(selected)
                if token["end"] - token["start"] > 1200
            ]
            if long_tokens:
                denominator = sum(excess**2 for _, excess in long_tokens)
                position = (
                    sum(point * excess**2 for point, excess in long_tokens)
                    / denominator
                )
                excess = sum(excess for _, excess in long_tokens)
                start += (1 - position) * excess
                end -= position * excess
        line.candidates[label] = Span(start, end)
        blanks = [
            token["end"] - token["start"] for token in selected if not token["key"]
        ]
        token_duration = sum(
            token["end"] - token["start"] for token in selected if token["key"]
        )
        line.evidence[f"{label}_tokens"] = {
            "blank_fraction": sum(blanks)
            / max(1, token["end"] - tokens[first]["start"]),
            "max_blank": max(blanks, default=0),
            "duration_per_char": token_duration / max(1, len(line.key)),
            "units": [
                {"text": token["key"], "start": token["start"], "end": token["end"]}
                for token in selected
                if token["key"]
            ],
        }
        first_search = index + 1
        try:
            line = next(dialogue)
        except StopIteration:
            return
        remaining = line.key


def summarize(lines, duration):
    timings = []
    previous_midpoint = None
    for line in lines:
        span = line.candidates["result"].at_least()
        start = max(0.0, span.start)
        end = min(float(duration), span.end)
        if end - start < MIN_DURATION:
            start = min(start, duration - MIN_DURATION)
            end = start + MIN_DURATION
        midpoint = (start + end) / 2
        if previous_midpoint is not None and midpoint < previous_midpoint:
            raise ValueError("result timings are out of transcript order")
        previous_midpoint = midpoint
        timings.append({"start": float(start), "end": float(end)})
    return timings
