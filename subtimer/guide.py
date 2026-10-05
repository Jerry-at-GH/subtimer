import numpy as np

from . import subtitle
from . import features
from . import numerics as n
from .numerics import GRID, agreement


def build(common, text_data, text_points, cap, tp, columns):
    rd = cap.dense
    td = text_data
    rows = []
    rp = features.probabilities(rd)
    ranchor = n.prepare(rd, rp)["anchor"]
    lines = cap.local_state["lines"]
    for i, b in enumerate(common):
        r = rd["baseline"][i]
        line = lines[i]
        hint = (
            np.array([line.hint.start, line.hint.end]) if line.hint is not None else r
        )
        sources = cap.line_states[i]["sources"]
        proposals = [b, r, ranchor[i]]
        for values in text_points.values():
            proposals.append(values[i])
        for source in [
            "baseline",
            "bfa",
            "qfa",
            "nfa_raw",
            "nfa_vocals",
            "asr",
            "vad",
        ]:
            group = [v for v in cap.history[i] if v["source"] == source]
            if group:
                best = max(group, key=lambda v: v["score"])
                proposals.extend([best["span"], best["micro"]])
        proposals.extend(p for p in cap.entries[i]["spans"].values() if p is not None)
        if line.hint_calibrated and line.hint is not None:
            proposals.append(hint)
        candidates = [b]
        for point in proposals[1:]:
            for fraction in [0.5, 1.0]:
                p = b + fraction * (np.array(point) - b)
                candidates.extend([p, [p[0], b[1]], [b[0], p[1]]])
        for offset in [-40.0, -20.0, 20.0, 40.0]:
            candidates.extend([b + [offset, 0], b + [0, offset], b + [offset, offset]])
        spans = n.unique_spans(
            candidates,
            lambda p: (
                np.isfinite(p).all()
                and p[0] >= 0
                and p[1] <= rd["duration"]
                and p[1] - p[0] >= 200
                and np.abs(p - b).max() <= 12000
            ),
        )
        count = len(spans)
        delta = (spans - b) / 1000
        fields = [
            delta,
            np.abs(delta),
            ((spans[:, 1] - spans[:, 0]) / 1000)[:, None],
            np.broadcast_to(td["cue"][i], (count, 69)),
            np.broadcast_to(rd["cue"][i], (count, 69)),
        ]
        for point in [
            td["baseline"][i],
            r,
            ranchor[i],
            hint,
            *[p[i] for p in text_points.values()],
        ]:
            diff = (point - spans) / 1000
            fields.append(
                np.column_stack(
                    [
                        np.clip(diff, -12, 12),
                        np.abs(diff).max(1) <= 0.05,
                        np.abs(diff).max(1) <= 0.2,
                    ]
                )
            )
        fields.append(np.full((count, 1), line.hint_calibrated))
        fields.append(np.array([agreement(p, sources) for p in spans]))
        for baseline in [common, rd["baseline"]]:
            for j in [i - 1, i + 1]:
                fields.append(
                    np.clip((spans - baseline[j][::-1]) / 1000, -12, 12)
                    if 0 <= j < len(common)
                    else np.full((count, 2), 12.0)
                )
        for d, prob in [(td, tp), (rd, rp)]:
            for edge in range(2):
                grid = d["baseline"][i, edge] + GRID
                fields.append(
                    np.column_stack(
                        [
                            np.interp(spans[:, edge] + s, grid, d["scores"][i, edge])
                            for s in [-50, 0, 50]
                        ]
                    )
                )
                for h in range(2):
                    fields.append(
                        np.column_stack(
                            [
                                np.interp(spans[:, edge] + s, grid, prob[i, edge, h])
                                for s in [-20, 0, 20]
                            ]
                        )
                    )
                fields.append(
                    ((spans[:, edge] < grid[0]) | (spans[:, edge] > grid[-1]))[:, None]
                )
        x = np.column_stack(fields).astype(np.float32)

        columns = np.asarray(columns)
        selected = x[:, columns % x.shape[1]].copy()
        relative = columns >= x.shape[1]
        selected[:, relative] -= x[0, columns[relative] % x.shape[1]]
        rows.append({"spans": spans, "indices": np.full(count, i), "x": selected})
    data = {k: np.concatenate([r[k] for r in rows]) for k in ["spans", "indices", "x"]}
    data["baseline"] = n.rounded(common)
    return data


def decode(data, prediction, margin=0.1, protect=False):
    ids = data["indices"]
    first = np.r_[True, ids[1:] != ids[:-1]]
    if prediction.ndim == 1:
        gain = prediction - prediction[first][ids]
    else:
        delta = prediction - prediction[first][ids]
        gain = delta.sum(1)
        if protect:
            gain = np.where((delta >= 0).all(1), gain, -np.inf)
    return n.choose(data, gain, margin)


RELOCATION_SOURCES = (
    "baseline",
    "nfa_raw",
    "nfa_vocals",
    "asr",
    "bfa",
    "qfa",
    "guide",
    "vad",
    "cfa",
)


def relocations(baseline, cap, text_data, columns):
    baseline = n.rounded(baseline)
    columns = np.asarray(columns)
    lines = cap.local_state["lines"]
    _, _, matches = subtitle.match(lines, cap.local_state["guide_words"])
    all_spans, all_x, all_ids = [], [], []
    for i, (b, line) in enumerate(zip(baseline, lines)):
        calibrated = line.hint_calibrated and line.hint is not None
        hint = np.array([line.hint.start, line.hint.end]) if calibrated else b
        by_source = {name: [] for name in RELOCATION_SOURCES}
        for name, span in cap.line_states[i]["sources"]:
            if name in by_source:
                by_source[name].append(span)
        by_source["baseline"].append(cap.dense["baseline"][i])
        by_source["cfa"].extend(
            p for p in cap.entries[i]["spans"].values() if p is not None
        )
        if calibrated:
            by_source["guide"].append(hint)
        candidates = [b]
        if calibrated:
            for points in by_source.values():
                for point in points:
                    candidates.extend((point, [point[0], b[1]], [b[0], point[1]]))
        spans = n.unique_spans(
            candidates,
            lambda point: (
                not (
                    not np.isfinite(point).all()
                    or point[0] < 0
                    or point[1] > cap.dense["duration"]
                    or point[1] - point[0] < 200
                    or point[1] - point[0] > max(16000, len(line.key) * 800)
                    or np.abs(point - b).max() < 500
                    or abs(point.mean() - hint.mean()) > 5000
                )
            ),
        )
        count = len(spans)
        delta = (spans - b) / 1000
        fields = [delta, np.abs(delta), ((spans[:, 1] - spans[:, 0]) / 1000)[:, None]]
        match = matches[i]
        info = line.evidence
        context = [
            len(line.key),
            float(calibrated),
            match.get("coverage", 0),
            match.get("longest", 0) / max(1, len(line.key)),
            match.get("max_gap", 0) / 1000,
            match.get("occurrences", 0),
            info.get("seed_score", 0),
            float(info.get("asr_interpolated", False)),
            (b[1] - b[0]) / 1000,
            (hint[1] - hint[0]) / 1000,
        ]
        for source in ("nfa_raw", "nfa_vocals"):
            ev = info.get(source + "_tokens", {})
            context.extend(
                [
                    ev.get("blank_fraction", 0),
                    ev.get("max_blank", 0) / 1000,
                    ev.get("duration_per_char", 0) / 1000,
                ]
            )
        fields.append(np.broadcast_to(context, (count, len(context))))
        fields.extend(
            np.broadcast_to(d["cue"][i], (count, len(d["cue"][i])))
            for d in (text_data, cap.dense)
        )
        for point in (hint, cap.dense["baseline"][i], text_data["baseline"][i]):
            diff = (spans - point) / 1000
            fields.extend(
                [np.clip(diff, -60, 60), np.abs(diff), np.abs(diff.mean(1))[:, None]]
            )
        for points in by_source.values():
            if points:
                distance = (
                    np.abs(spans[:, None, :] - np.asarray(points)[None, :, :]) / 1000
                )
                fields.append(
                    np.column_stack(
                        [
                            distance.min(1),
                            distance.max(2).min(1),
                            (distance.max(2) <= 0.2).sum(1),
                            (distance.max(2) <= 0.5).sum(1),
                        ]
                    )
                )
            else:
                fields.append(np.broadcast_to([60, 60, 60, 0, 0], (count, 5)))
        for j in (i - 1, i + 1):
            for point in (
                baseline[j] if 0 <= j < len(lines) else None,
                np.array([lines[j].hint.start, lines[j].hint.end])
                if 0 <= j < len(lines)
                and lines[j].hint_calibrated
                and lines[j].hint is not None
                else None,
            ):
                fields.append(
                    np.clip((spans - point) / 1000, -60, 60)
                    if point is not None
                    else np.full((count, 2), 60)
                )
        for d in (text_data, cap.dense):
            for edge in (0, 1):
                grid = d["baseline"][i, edge] + GRID
                fields.append(
                    np.column_stack(
                        [
                            np.interp(spans[:, edge] + s, grid, d["scores"][i, edge])
                            for s in (-50, 0, 50)
                        ]
                    )
                )
                fields.append(
                    ((spans[:, edge] < grid[0]) | (spans[:, edge] > grid[-1]))[:, None]
                )
        x = np.column_stack(fields).astype(np.float32)
        selected = x[:, columns % x.shape[1]].copy()
        relative = columns >= x.shape[1]
        selected[:, relative] -= x[0, columns[relative] % x.shape[1]]
        all_x.append(selected)
        all_spans.append(spans)
        all_ids.append(np.full(count, i))
    data = {
        "baseline": baseline,
        "spans": np.concatenate(all_spans),
        "x": np.concatenate(all_x),
        "indices": np.concatenate(all_ids),
    }
    reliable = [
        i
        for i, (line, match) in enumerate(zip(lines, matches))
        if line.hint_calibrated
        and line.hint is not None
        and len(line.key) >= 4
        and match.get("coverage", 0) >= 0.8
    ]
    permitted = np.zeros(len(data["indices"]), dtype=bool)
    for i, match in enumerate(matches):
        rows = data["indices"] == i
        if match.get("coverage", 0) >= 0.8 and match.get("occurrences", 0) == 1:
            permitted[rows] = True
            continue
        previous = next((j for j in reversed(reliable) if i - 8 <= j < i), None)
        following = next((j for j in reliable if i < j <= i + 8), None)
        if previous is not None and following is not None:
            midpoint = data["spans"][rows].mean(1)
            permitted[rows] = (midpoint >= lines[previous].hint.midpoint - 1000) & (
                midpoint <= lines[following].hint.midpoint + 1000
            )
    for i, (line, match) in enumerate(zip(cap.local_state["lines"], matches)):
        if not line.hint_calibrated or line.hint is None or not match:
            continue
        original = np.array([match["start"], match["end"]])
        calibrated = np.array([line.hint.start, line.hint.end])
        if (
            np.abs(original - calibrated).max() <= 2000
            or np.abs(original - data["baseline"][i]).max() > 2000
        ):
            continue
        families = {
            "nfa" if name.startswith("nfa_") else name
            for name, span in cap.line_states[i]["sources"]
            if name in ("nfa_raw", "nfa_vocals", "bfa", "qfa", "asr")
            and np.abs(np.asarray(span) - data["baseline"][i]).max() <= 500
        }
        if len(families) >= 2:
            permitted[data["indices"] == i] = False
    data["supported"] = permitted
    return data


def relocate(data, prediction, margin):
    ids = data["indices"]
    first = np.r_[True, ids[1:] != ids[:-1]]
    gain = prediction - prediction[first][ids]
    gain = np.where(first | data["supported"], gain, -np.inf)

    keep = first | (gain > margin)
    spans = data["spans"][keep]
    indices = ids[keep]
    gain = np.where(first[keep], 0, gain[keep] - margin)
    groups = np.split(np.arange(len(indices)), np.flatnonzero(np.diff(indices)) + 1)
    scores, back = [], []
    for i, group in enumerate(groups):
        if i == 0:
            scores.append(gain[group])
            back.append(np.full(len(group), -1))
            continue
        previous = groups[i - 1]
        valid = spans[previous].mean(1)[:, None] <= spans[group].mean(1)[None, :]
        options = np.where(valid, scores[-1][:, None], -np.inf)
        parent = options.argmax(0)
        scores.append(options[parent, np.arange(len(group))] + gain[group])
        back.append(parent)
    selected = []
    index = int(scores[-1].argmax())
    for i in range(len(groups) - 1, -1, -1):
        selected.append(spans[groups[i][index]])
        index = back[i][index]
    return np.asarray(selected[::-1])
