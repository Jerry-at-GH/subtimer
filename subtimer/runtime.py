import copy
import gc
import statistics
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import cache, gpu, occurrence, refine, selection, speech, subtitle
from .speech import MANIFEST, model
from . import evidence, features, transcript
from .capture import Capture

STAGES = (
    "separate",
    "detect",
    "transcribe",
    "match",
    "align",
    "refine",
)


def align_core(
    dialogues,
    prepared,
    cache_dir,
    *,
    capture,
    guide_segments=None,
    progress=lambda *_: None,
    guide_tolerance_ms=12_000,
):
    lines = subtitle.make_lines(dialogues)
    guide_words = (
        subtitle.apply_guide_segments(lines, guide_segments) if guide_segments else []
    )
    session = speech.AsrSession()
    try:
        minimum = subtitle.MIN_DURATION
        joined = "|".join(line.spoken for line in lines)
        track = cache.load(cache_dir / "vad.json")
        vad = track and {
            name: tuple(tuple(span) for span in spans) for name, spans in track.items()
        }
        tracks = None
        if vad is None:
            with ThreadPoolExecutor(max_workers=1) as executor:
                vad_input = executor.submit(speech.prepare_vad, prepared)
                tracks = speech.run_nfa(prepared, joined, cache_dir, progress)
                vad = speech.run_vad(prepared, cache_dir, progress, vad_input.result())

        local_words = speech.run_asr(prepared, cache_dir, progress, session)

        session.close()
        progress("match")
        subtitle.apply_words(lines, local_words, "asr", vad)
        if tracks is None:
            tracks = speech.run_nfa(prepared, joined, cache_dir, progress)
        subtitle.apply_nfa(lines, tracks["nfa_raw"], "nfa_raw", compress_long=True)
        subtitle.apply_nfa(lines, tracks["nfa_vocals"], "nfa_vocals")
        for line in lines:
            capture.unsnapped[id(line)] = {
                label: [span.start, span.end]
                for label in ["nfa_raw", "nfa_vocals"]
                if (span := line.candidates.get(label)) is not None
            }
        subtitle.snap(lines, "nfa_raw", vad)
        subtitle.snap(lines, "nfa_vocals", vad)

        offsets = []
        for line in lines:
            raw = line.candidates.get("nfa_raw")
            vocals = line.candidates.get("nfa_vocals")
            if (
                line.hint
                and raw
                and vocals
                and len(line.key) >= 6
                and abs(raw.midpoint - vocals.midpoint) <= 500
            ):
                offsets.append(
                    (raw.midpoint + vocals.midpoint) / 2 - line.hint.midpoint
                )
        if len(offsets) >= 5:
            offset = statistics.median(offsets)
            asr_offsets = [
                line.candidates["asr"].midpoint - line.hint.midpoint
                for line in lines
                if line.hint is not None and _strong_asr(line)
            ]
            if (
                statistics.median(abs(value - offset) for value in offsets) <= 2000
                and len(asr_offsets) >= 5
                and abs(statistics.median(asr_offsets) - offset) <= 2000
            ):
                for line in lines:
                    if line.hint is not None:
                        line.hint = line.hint.shift(offset)
                        line.hint_calibrated = True
            else:
                for line in lines:
                    line.hint = None

        def usable(candidate):
            if candidate is None:
                return False
            start = max(0, candidate.start - 300)
            end = min(prepared.duration, candidate.end + 300)
            return minimum <= end - start <= 30_000

        for line in lines:
            raw = line.candidates.get("nfa_raw")
            vocals = line.candidates.get("nfa_vocals")
            seed = vocals if usable(vocals) else raw if usable(raw) else None
            if seed is None:
                continue
            if (
                line.hint_calibrated
                and line.hint
                and usable(raw)
                and vocals
                and abs(raw.midpoint - vocals.midpoint) > 100
                and abs(raw.midpoint - line.hint.midpoint)
                < abs(vocals.midpoint - line.hint.midpoint)
            ):
                seed = raw
            line.candidates["nfa_seed"] = seed
        occurrence.score_seeds(lines)

        windows, selected = speech.alignment_windows(lines, prepared, str)
        capture.windows = {id(line): window for line, window in zip(selected, windows)}
        speech.run_bfa(lines, prepared, cache_dir, progress)
        subtitle.snap(lines, "bfa", vad, start_radius=800)
        capture.local_state = {
            "lines": copy.deepcopy(lines),
            "audio": prepared,
            "vad": vad,
            "tracks": tracks,
            "words": local_words,
            "guide_words": guide_words,
            "unsnapped": [capture.unsnapped.get(id(line), {}) for line in lines],
            "windows": [capture.windows.get(id(line)) for line in lines],
        }
        return finish_core(
            lines,
            prepared,
            vad,
            tracks,
            local_words,
            guide_words,
            cache_dir,
            session,
            progress,
            capture,
            guide_tolerance_ms=guide_tolerance_ms,
        )
    finally:
        session.close()
        capture.close()


def _direct_match(info):
    return bool(
        info.get("exact_single")
        or info.get("coverage", 0) >= 0.8
        and info.get("longest", 0) >= 3
        and info.get("max_gap", 0) <= 5000
    )


def _strong_asr(line):
    span = line.candidates.get("asr")
    info = line.evidence.get("asr", {})
    return bool(
        span
        and span.duration <= max(max(1, len(line.key)) * 400 + 2000, 15_000)
        and info.get("direct")
        and _direct_match(info)
        and 0 <= info.get("occurrences", 0) <= 4
    )


def finish_core(
    lines,
    prepared,
    vad,
    tracks,
    local_words,
    guide_words,
    cache_dir,
    session,
    progress,
    capture,
    qfa_runner=None,
    guide_tolerance_ms=12_000,
):
    minimum = subtitle.MIN_DURATION
    hint_distance = guide_tolerance_ms
    (qfa_runner or speech.run_qfa)(lines, prepared, cache_dir, progress)
    subtitle.snap(lines, "qfa", vad, start_radius=800)

    def near_hint(line, label):
        candidate = line.candidates.get(label)
        return bool(
            candidate
            and (
                not line.hint_calibrated
                or line.hint is None
                or abs(candidate.midpoint - line.hint.midpoint) <= hint_distance
            )
        )

    def plausible(line, label):
        candidate = line.candidates.get(label)
        if candidate is None:
            return False
        maximum = (
            max(len(line.key) * 400 + 2000, 5000, line.hint.duration * 1.5 + 2000)
            if line.hint_calibrated and line.hint
            else max(len(line.key) * 400 + 2000, 15_000)
        )
        return candidate.duration <= maximum

    previous_end = None
    for line in lines:
        bfa = line.candidates.get("bfa")
        qfa = line.candidates.get("qfa")
        vocals = line.candidates.get("nfa_vocals")
        raw = line.candidates.get("nfa_raw")
        if bfa and qfa and vocals:
            midpoints = (bfa.midpoint, qfa.midpoint, vocals.midpoint)
            if max(midpoints) - min(midpoints) <= 300:
                candidate = subtitle.Span(
                    statistics.median((bfa.start, qfa.start, vocals.start)),
                    statistics.median((bfa.end, qfa.end, vocals.end)),
                ).at_least()
                if previous_end is None or candidate.start >= previous_end - 200:
                    line.candidates["result"] = candidate
                    previous_end = candidate.end
                    continue

        asr_span = line.candidates.get("asr")
        asr_info = line.evidence.get("asr", {})
        direct_asr = bool(asr_info.get("direct") and _direct_match(asr_info))
        labels = [
            "bfa",
            "qfa",
            "nfa_vocals",
            "nfa_raw",
            "asr",
        ]
        if (
            bfa
            and qfa
            and asr_span
            and direct_asr
            and abs(bfa.midpoint - qfa.midpoint) > 300
            and abs(qfa.midpoint - asr_span.midpoint)
            < abs(bfa.midpoint - asr_span.midpoint)
        ):
            labels[:2] = labels[1], labels[0]
        if (
            line.hint_calibrated
            and line.hint
            and raw
            and vocals
            and abs(raw.midpoint - vocals.midpoint) > 100
        ):
            labels.sort(
                key=lambda label: (
                    abs(line.candidates[label].midpoint - line.hint.midpoint)
                    if line.candidates.get(label) and (label != "asr" or direct_asr)
                    else float("inf")
                )
            )

        require_plausible = any(plausible(line, label) for label in labels)
        for label in labels:
            candidate = line.candidates.get(label)
            if candidate is None or candidate.duration < minimum:
                continue
            if require_plausible and not plausible(line, label):
                continue
            if previous_end is not None and candidate.start < previous_end - 200:
                continue
            if not near_hint(line, label):
                continue
            line.candidates["result"] = candidate
            break

        selected = line.candidates.get("result")
        asr_span = line.candidates.get("asr")
        if (
            selected
            and asr_span
            and selected.duration > 8000
            and minimum <= asr_span.duration <= 8000
            and near_hint(line, "asr")
            and (previous_end is None or asr_span.start >= previous_end - 200)
        ):
            line.candidates["result"] = asr_span

        selected = line.candidates.get("result")
        strong_span = line.candidates.get("asr")
        if (
            _strong_asr(line)
            and selected
            and strong_span
            and raw
            and vocals
            and abs(raw.midpoint - vocals.midpoint) > 100
            and abs(selected.midpoint - strong_span.midpoint) > 300
            and (previous_end is None or strong_span.start >= previous_end - 200)
            and near_hint(line, "asr")
        ):
            line.candidates["result"] = strong_span
        if line.candidates.get("result"):
            previous_end = line.candidates["result"].end

    speech_index = 0
    previous_end = None
    speech_ranges = vad["speech"]
    for index, line in enumerate(lines):
        if line.candidates.get("result"):
            previous_end = line.candidates["result"].end
            continue
        following = next(
            (
                item.candidates["result"]
                for item in lines[index + 1 :]
                if item.candidates.get("result")
            ),
            None,
        )

        def bounded(span, lower=previous_end, upper=following):
            start = max(span.start, lower + 1) if lower is not None else span.start
            end = min(span.end, upper.start - 1) if upper is not None else span.end
            return subtitle.Span(start, end) if end - start >= minimum else None

        while (
            speech_index < len(speech_ranges)
            and previous_end is not None
            and speech_ranges[speech_index][1] <= previous_end
        ):
            speech_index += 1
        if speech_index < len(speech_ranges):
            start, end = speech_ranges[speech_index]
            close = (
                not line.hint_calibrated
                or line.hint is None
                or abs((start + end) / 2 - line.hint.midpoint) <= hint_distance
            )
            fallback = (
                bounded(subtitle.Span(start, max(end, start + minimum)))
                if close
                else None
            )
            if fallback:
                line.candidates["result"] = fallback
                speech_index += 1
        if line.candidates.get("result") is None and near_hint(line, "asr"):
            line.candidates["result"] = bounded(line.candidates["asr"])
        if line.candidates.get("result") is None and line.hint_calibrated and line.hint:
            line.candidates["result"] = bounded(line.hint)
        if line.candidates.get("result") is None:
            start = (
                previous_end + 1
                if previous_end is not None
                else max(0, following.start - 1000)
                if following
                else 0
            )
            end = min(start + 1000, following.start - 1) if following else start + 1000
            if end - start < minimum and following:
                end = max(minimum, following.start - 1)
                start = max(0, end - minimum)
            start = max(0, min(start, prepared.duration - minimum))
            end = min(prepared.duration, max(end, start + minimum))
            line.candidates["result"] = subtitle.Span(start, end).at_least()
        line.evidence["fixed_fallback"] = True
        previous_end = line.candidates["result"].end

    subtitle.snap(lines, "result", vad, start_radius=800, end_radius=800, ordered=False)
    occurrence.repair(lines, local_words, vad, prepared, cache_dir, progress, session)
    occurrence.repair_from_guide(lines, guide_words, local_words, vad)
    occurrence.score_agreement(lines)
    refine.apply(lines, prepared, vad, cache_dir, session, progress, capture=capture)
    session.close()
    selection.apply(
        lines, prepared, vad, cache_dir, tracks, local_words, progress, capture=capture
    )
    return subtitle.summarize(lines, prepared.duration)


class BranchQFA:
    def __init__(self, store, kind):
        self.store = store
        self.kind = kind

    def __call__(self, lines, audio, cache_dir, progress):
        segments, selected = speech.alignment_windows(
            [line for line in lines if speech.needs_qfa(line)], audio, str
        )
        requests = [
            {
                "kind": "qfa",
                "view": "vocals",

                "start": int(s["start"])
                if float(s["start"]).is_integer()
                else s["start"],
                "end": int(s["end"]) if float(s["end"]).is_integer() else s["end"],
                "text": s["text"],
            }
            for s in segments
        ]
        self.store.ensure(requests, progress)
        output = {}
        for segment, request in zip(segments, requests):
            result, _ = self.store.get(request, load_arrays=False)
            if not result.get("available"):
                continue
            items = [
                {
                    "text": w["text"],
                    "start": float(v[0] + segment["start"]),
                    "end": float(v[1] + segment["start"]),
                }
                for w, v in zip(result["words"], result["word_outputs"][self.kind])
                if w["text"].strip()
            ]
            if items:
                output[speech.segment_key(segment)] = {
                    "start": min(w["start"] for w in items),
                    "end": max(w["end"] for w in items),
                    "items": items,
                }
        cache.save(cache_dir / "qfa.json", output)
        speech.apply_qfa(segments, selected, output)


def branch_cache(cache_dir, name, *, native_qfa=False):
    (cache_dir / "occurrence-crops").mkdir(exist_ok=True)
    root = cache_dir / "timing-branches" / name
    root.mkdir(parents=True, exist_ok=True)
    for path in cache_dir.iterdir():
        if path.name == "qfa.json" or path.name.startswith("timing-"):
            continue
        dest = root / path.name
        if not dest.exists() and not dest.is_symlink():
            dest.symlink_to(path.resolve(), target_is_directory=path.is_dir())
    shared = cache_dir / "timing-evidence"
    if shared.is_dir() and not (root / "timing-evidence").exists():
        (root / "timing-evidence").symlink_to(
            shared.resolve(), target_is_directory=True
        )
    if (
        native_qfa
        and (cache_dir / "qfa.json").is_file()
        and not (root / "qfa.json").exists()
    ):
        (root / "qfa.json").symlink_to((cache_dir / "qfa.json").resolve())
    return root


def finish_common(cap, cache_dir, progress):
    d = cap.dense
    snapshot = cap.local_state
    audio = snapshot["audio"]
    texts = [r["text"] for r in cap.rows]
    windows = [r["window"] for r in cap.rows]
    store = cache.Evidence(audio, cache_dir / "timing-evidence")
    p = features.probabilities(d)
    q = features.n.prepare(d, p)
    anchor = q["anchor"]
    projected = features.n.project(d, q)
    wide = [
        np.array([p[0] for p in features.n.options(b, history, row, d["duration"])[0]])
        for b, history, row in zip(d["baseline"], cap.history, cap.entries)
    ]
    pending = cap.shared.get("components")
    values = (
        pending.result()
        if pending is not None
        else evidence.components(store, d, texts, progress)
    )
    component, component_data = features.component(
        d, anchor, projected, wide, values["wfa"], values["packed"]
    )
    del values
    gc.collect()
    sources = evidence.qfa_sources(store, d, texts, windows, component, progress)
    strongest, old = features.qfa_select(d, component, component_data, sources)
    del component_data
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending_neighbors = (
            pool.submit(
                evidence.neighbors, store, strongest, texts, d["duration"], progress
            )
            if gpu.concurrency(2) > 1
            else None
        )
        points = {}
        for kind in ["native", "head"]:
            branch = Capture(cap.mode, shared=cap.shared)
            lines = copy.deepcopy(snapshot["lines"])
            branch.unsnapped = {
                id(line): value for line, value in zip(lines, snapshot["unsnapped"])
            }
            branch.windows = {
                id(line): window
                for line, window in zip(lines, snapshot["windows"])
                if window is not None
            }
            root = branch_cache(cache_dir, kind)
            session = speech.AsrSession()
            try:
                finish_core(
                    lines,
                    audio,
                    snapshot["vad"],
                    snapshot["tracks"],
                    snapshot["words"],
                    snapshot["guide_words"],
                    root,
                    session,
                    progress,
                    branch,
                    qfa_runner=BranchQFA(store, kind),
                )
            finally:
                session.close()
                branch.close()
            bd = branch.dense
            bp = features.probabilities(bd)
            bq = features.n.prepare(bd, bp)

            points[f"qfa_{kind}/final"] = bq["anchor"]
            points[f"qfa_{kind}/projected"] = features.n.project(bd, bq)
            if kind == "head":
                head = branch
        points["enriched"], _ = features.enriched(
            head.dense,
            points["qfa_head/final"],
            points["qfa_head/projected"],
            head.line_states,
            head.entries,
            [s[2] for s in sources],
        )
        points["calibration"] = features.calibrate(d, strongest)
        points["strongest"] = strongest
        extra, valid = (
            pending_neighbors.result()
            if pending_neighbors
            else evidence.neighbors(store, strongest, texts, d["duration"], progress)
        )
    points.update(extra)
    probability = features.probabilities(d, "recovery")
    data = features.final_features(
        d, strongest, points, old, sources, valid, probability
    )
    scores = model("final-selector").predict(
        data["x"][:, MANIFEST["final_columns"]], num_threads=4
    )
    baseline, ids, spans = data["baseline"], data["indices"], data["spans"]
    first = np.r_[True, ids[1:] != ids[:-1]]
    gain = scores - scores[first][ids]
    result = baseline.copy()
    best = np.zeros(len(baseline))
    for j in np.flatnonzero((gain >= 0.05) & np.any(spans != baseline[ids], axis=1)):
        i = ids[j]
        if gain[j] > best[i] + 1e-12:
            result[i] = spans[j]
            best[i] = gain[j]
    result = features.n.ordered(result, baseline)
    return result, {
        "dense": {**d, "probability": p},
        "points": points,
        "transcript": transcript.build(
            data,
            result,
            scores,
            d,
            probability,
            cap.history,
            cap.entries,
            cap.line_states,
            sources,
            component,
        ),
    }


def align(dialogues, extracted_audio, cache_dir, guide_segments, progress):
    from . import cfa, guide, joint

    lines = subtitle.make_lines(dialogues)
    phase = -1

    def intermediate(stage):
        nonlocal phase

        if STAGES.index(stage) > phase:
            phase = STAGES.index(stage)
            progress(stage)

    audio = speech.prepare_audio(extracted_audio, cache_dir, intermediate)
    evidence_store = cache.Evidence(audio, cache_dir / "timing-evidence")
    (cache_dir / "cfa-context").mkdir(exist_ok=True)

    common_cache = branch_cache(cache_dir, "common", native_qfa=True)
    cap = Capture("text")
    try:
        align_core(
            dialogues, audio, common_cache, progress=intermediate, capture=cap
        )
        output, trace = finish_common(cap, common_cache, intermediate)
    finally:
        cap.release()
    output = joint.align(
        audio,
        [line.spoken for line in lines],
        output,
        trace["transcript"],
        evidence_store,
        progress=intermediate,
    )
    joint_output = output
    output = cfa.align(
        audio, dialogues, output, evidence_store, progress=intermediate
    )
    if guide_segments:
        text_output = output
        output = joint_output
        guide_cache = branch_cache(
            common_cache, "guide", native_qfa=True
        )
        guide_cap = Capture("guide")
        try:
            usable = True
            try:
                align_core(
                    dialogues,
                    audio,
                    guide_cache,
                    guide_segments=guide_segments,
                    progress=intermediate,
                    capture=guide_cap,
                    guide_tolerance_ms=MANIFEST["guide_tolerance_ms"],
                )
            except ValueError as exc:
                if str(exc) != "result timings are out of transcript order":
                    raise

                usable = False
            if usable and any(
                line.hint_calibrated for line in guide_cap.local_state["lines"]
            ):
                data = guide.build(
                    output,
                    trace["dense"],
                    trace["points"],
                    guide_cap,
                    trace["dense"]["probability"],
                    MANIFEST["guide_columns"],
                )
                predictions = model("guide-selector").predict(
                    data["x"], raw_score=True, num_threads=2
                )
                output = guide.decode(data, predictions, **MANIFEST["guide_policy"])
                relocation = MANIFEST["guide_relocation"]
                data = guide.relocations(
                    output, guide_cap, trace["dense"], relocation["columns"]
                )
                prediction = model("relocation-selector").predict(
                    data["x"], num_threads=2
                )
                output = guide.relocate(data, prediction, relocation["margin"])
        finally:
            guide_cap.release()

        unchanged = np.all(output == joint_output, axis=1)
        output = joint.legal_ordered(
            np.where(unchanged[:, None], text_output, output), output, audio.duration
        )
    return [{"start": float(s), "end": float(e)} for s, e in output]
