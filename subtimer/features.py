import operator
from functools import lru_cache, reduce

import numpy as np
import torch

from .ctc import ctc_align, ctc_boundaries
from .speech import model, weights
from . import numerics as n
from .evidence import align_record
from .numerics import GRID, feature_matrix

EDGES = ("start", "end")
from .qfa import evidence as qfa_evidence


def probabilities(d, family="dense"):
    out = np.empty((len(d["baseline"]), 2, 2, len(GRID)), np.float32)
    for edge in range(2):
        x = feature_matrix(d["baseline"], d["scores"], d["cue"], d["median"], edge)
        for h, tolerance in enumerate([50, 200]):
            out[:, edge, h] = (
                model(f"{family}-probability-{EDGES[edge]}-{tolerance}")
                .predict(x, num_threads=4)
                .reshape(-1, len(GRID))
            )
    return out


def select(baseline, spans, ids, x, filename, margin):
    score = model(filename).predict(x, num_threads=4)
    first = np.r_[True, ids[1:] != ids[:-1]]
    base = np.zeros(len(baseline))
    base[ids[first]] = score[first]
    return n.choose(
        {"baseline": baseline, "spans": spans, "indices": ids},
        score - base[ids],
        margin,
    )


@lru_cache(maxsize=3)
def phone_head(arm):
    head = torch.nn.Linear(1024, 43)
    head.load_state_dict(weights()[f"wfa-phone-head-{arm}"])
    return head.eval()


def component(d, anchor, projected, wide, wfa, packed):
    rows = []
    for i, b in enumerate(anchor):
        win, arrays = wfa[i]
        if not win.get("available"):
            rows.append(
                {
                    "spans": np.array([b]),
                    "indices": np.array([i]),
                    "x": np.zeros((1, 305), np.float32),
                }
            )
            continue
        groups = [r["phone"] for r in win["tokens"]]
        T = len(arrays["raw"]["phone"])
        new = []
        for view in ["raw", "vocals"]:
            for head in ["phone", "kana"]:
                try:
                    point = align_record(win, arrays[view], head)
                    if (
                        point[1] - point[0] < 200
                        or np.abs(point - d["baseline"][i]).max() > 3000
                    ):
                        point = d["baseline"][i]
                except ValueError:
                    point = d["baseline"][i]
                new.append(point)
        for kind in ["bfa", "qfa"]:
            new.extend(packed[kind][i].get("outputs", {}).values())
        ids = reduce(operator.iadd, groups, [])
        first = sum(len(g) for g in groups[: win["cue"]])
        last = first + len(groups[win["cue"]]) - 1
        for arm in ["ordinary", "prior", "prior-edge"]:
            head = phone_head(arm)
            with torch.inference_mode():
                lp = (
                    head(torch.tensor(arrays["vocals"]["h12"], dtype=torch.float32))
                    .log_softmax(-1)
                    .numpy()
                )
            starts, ends, score = ctc_align(lp, np.array(ids, np.int64))
            point = np.array([starts[first], ends[last]]) * 20 + win["start"]
            baseline = d["baseline"][i]
            new.append(
                point
                if np.isfinite(score) and n.valid(point, baseline, d["duration"])
                else baseline
            )
        proposals = [b, d["baseline"][i], *wide[i]]
        for point in new:
            for fraction in [0.5, 1.0]:
                p = b + fraction * (np.array(point) - b)
                proposals.extend([p, [p[0], b[1]], [b[0], p[1]]])
        spans = n.unique_spans(
            proposals,
            lambda p: (
                not (
                    p[0] < win["start"]
                    or p[1] > win["start"] + T * 20
                    or p[1] - p[0] < 200
                    or np.abs(p - b).max() > 3000
                    or i > 0
                    and b.mean() >= anchor[i - 1].mean()
                    and p.mean() < anchor[i - 1].mean()
                    or i + 1 < len(anchor)
                    and anchor[i + 1].mean() >= b.mean()
                    and p.mean() > anchor[i + 1].mean()
                )
            ),
        )
        delta = (spans - b) / 1000
        geometry = np.column_stack(
            [
                delta,
                np.abs(delta),
                (spans[:, 1] - spans[:, 0]) / 1000,
                np.full(len(spans), (b[1] - b[0]) / 1000),
                np.any(delta != 0, axis=1),
                np.full(len(spans), len(groups[win["cue"]]) / 100),
            ]
        )
        dc = []
        for view in ["raw", "vocals"]:
            for head in ["phone", "kana"]:
                groups = [t[head] for t in win["tokens"]]
                ids = np.array(reduce(operator.iadd, groups, []), np.int32) - 1
                first = sum(len(g) for g in groups[: win["cue"]])
                last = first + len(groups[win["cue"]]) - 1
                lp = arrays[view][head].astype(np.float64)
                lp = np.column_stack([lp[:, 1:], lp[:, 0]])
                posterior, score = ctc_boundaries(lp, ids, first, last)
                times = (
                    win["start"]
                    + (np.arange(len(lp))[None] + np.array([0, 1])[:, None]) * 20
                )
                dc.append(
                    None
                    if not np.isfinite(score) or score < -1e20
                    else (
                        posterior,
                        times,
                        np.clip((score - lp[:, -1].sum()) / len(ids), -100, 100),
                    )
                )
        df = n.features(spans, dc)
        packs = []
        for kind in ["bfa", "qfa"]:
            outputs = packed[kind][i].get("outputs", {})
            for key in ["control", "packed", "guarded", "control-tail", "packed-tail"]:
                p = np.array(outputs.get(key, spans[0]))
                delta = (p - spans) / 1000
                packs.append(
                    np.column_stack(
                        [
                            np.clip(delta, -3, 3),
                            np.abs(delta).max(1) <= 0.05,
                            np.abs(delta).max(1) <= 0.2,
                            np.full(len(spans), key in outputs),
                        ]
                    )
                )
        pf = np.concatenate(packs, -1)
        base = np.column_stack(
            [geometry, np.broadcast_to(d["cue"][i], (len(spans), 69))]
        )
        x = np.column_stack(
            [
                a.astype(np.float32)
                for a in [
                    base,
                    np.column_stack([df, df - df[0]]),
                    np.column_stack([pf, pf - pf[0]]),
                ]
            ]
        )
        rows.append({"spans": spans, "indices": np.full(len(spans), i), "x": x})

    result = {
        k: np.concatenate([r[k] for r in rows]) for k in ["spans", "indices", "x"]
    }
    selected = select(
        anchor,
        result["spans"],
        result["indices"],
        result["x"],
        "component-selector",
        0.2,
    )
    result["baseline"] = anchor
    retained = np.where(
        np.any(selected != anchor, axis=1)[:, None], selected, projected
    )
    return n.ordered(retained, projected), result


def qfa_features(spans, anchor, cue, sources):
    delta = (spans - anchor) / 1000
    geometry = np.column_stack(
        [
            delta,
            np.abs(delta),
            (spans[:, 1] - spans[:, 0]) / 1000,
            np.full(len(spans), (anchor[1] - anchor[0]) / 1000),
            np.any(delta != 0, axis=1),
            np.full(len(spans), len(sources[0][0].get("words", [])) / 50),
        ]
    )
    signal = np.column_stack([qfa_evidence(spans, r, a, anchor) for r, a in sources])
    anchor_signal = np.column_stack(
        [qfa_evidence(n.rounded(anchor)[None], r, a, anchor) for r, a in sources]
    )
    return np.column_stack(
        [
            geometry,
            np.broadcast_to(cue, (len(spans), 69)),
            signal,
            signal - anchor_signal,
        ]
    ).astype(np.float32)


def qfa_select(d, anchor, component_data, sources):
    rows = []
    for i, b in enumerate(anchor):
        initial = [
            b,
            d["baseline"][i],
            *component_data["spans"][component_data["indices"] == i],
        ]
        proposals = list(initial)
        for record, _ in sources[i]:
            for point in record.get("outputs", {}).values():
                for fraction in [0.5, 1.0]:
                    p = b + fraction * (np.array(point) - b)
                    proposals.extend([p, [p[0], b[1]], [b[0], p[1]]])
        spans = n.unique_spans(
            proposals,
            lambda p: (
                n.valid(p, b, d["duration"])
                and not (
                    i
                    and b.mean() >= anchor[i - 1].mean()
                    and p.mean() < anchor[i - 1].mean()
                )
                and not (
                    i + 1 < len(anchor)
                    and b.mean() <= anchor[i + 1].mean()
                    and p.mean() > anchor[i + 1].mean()
                )
            ),
        )
        rows.append(
            {
                "spans": spans,
                "indices": np.full(len(spans), i),
                "x": qfa_features(spans, b, d["cue"][i], sources[i]),
            }
        )
    data = {k: np.concatenate([r[k] for r in rows]) for k in ["spans", "indices", "x"]}
    data["baseline"] = anchor
    return select(
        anchor, data["spans"], data["indices"], data["x"], "qfa-selector", 0.2
    ), data


def enriched(d, base, projected, states, cfa, seed_sources):
    p = probabilities(d, "qfa")
    q = n.prepare(d, p)
    adapted = [q["anchor"], n.project(d, q)]
    rows = []
    for i, b in enumerate(base):
        state = states[i]
        sources = state["sources"]
        extra = []
        candidates = [
            b,
            projected[i],
            d["baseline"][i],
            *[v[i] for v in adapted],
            *[p for _, p in sources],
            *state["candidates"].values(),
        ]
        record, arrays = seed_sources[i]
        if state.get("window") is not None:
            for p in cfa[i]["spans"].values():
                if p is not None:
                    candidates.append(p)
                    extra.append(("cfa", p))
            if record.get("available"):
                for name in ["native", "safe", "head"]:
                    p = record["outputs"][name]
                    candidates.append(p)
                    extra.append(("qfa_" + name, p))
        for p in list(candidates[1:]):
            candidates.extend([[p[0], b[1]], [b[0], p[1]]])
        spans = n.unique_spans(candidates, lambda p: n.valid(p, b, d["duration"]))
        delta = (spans - b) / 1000
        fields = [
            delta,
            np.abs(delta),
            ((spans[:, 1] - spans[:, 0]) / 1000)[:, None],
            np.broadcast_to(d["cue"][i], (len(spans), 69)),
        ]
        for name in [
            "bfa",
            "qfa",
            "nfa_raw",
            "nfa_vocals",
            "asr",
            "baseline",
            "cfa",
            "qfa_native",
            "qfa_safe",
            "qfa_head",
        ]:
            values = np.asarray([p for k, p in sources + extra if k == name])
            if len(values):
                distance = np.abs(spans[:, None, :] - values[None, :, :])
                nearest = distance.max(2).argmin(1)
                diff = (values[nearest] - spans) / 1000
                fields.append(
                    np.column_stack(
                        [
                            np.ones(len(spans)),
                            np.clip(diff, -3, 3),
                            np.min(distance.max(2), axis=1) / 1000,
                        ]
                    )
                )
            else:
                fields.append(np.zeros((len(spans), 4)))
        for edge in range(2):
            fields.append(
                np.column_stack(
                    [
                        np.interp(
                            spans[:, edge] + shift,
                            d["baseline"][i, edge] + GRID,
                            d["scores"][i, edge],
                        )
                        for shift in [-50, 0, 50]
                    ]
                )
            )
        signal = np.column_stack(fields).astype(np.float32)
        signal = np.column_stack(
            [signal, qfa_evidence(spans, record, arrays, b)]
        ).astype(np.float32)
        rows.append(
            {
                "spans": spans,
                "indices": np.full(len(spans), i),
                "x": np.column_stack([signal, signal - signal[0]]),
            }
        )
    data = {k: np.concatenate([r[k] for r in rows]) for k in ["spans", "indices", "x"]}
    return select(
        base,
        data["spans"],
        data["indices"],
        data["x"],
        "qfa-enriched-selector",
        0.2,
    ), data


def calibrate(d, baseline):
    x = np.column_stack([(baseline[:, 1] - baseline[:, 0]) / 1000, d["cue"]]).astype(
        np.float32
    )
    delta = (
        np.column_stack(
            [
                model(f"boundary-calibrator-{EDGES[edge]}").predict(x, num_threads=4)
                for edge in range(2)
            ]
        )
        * 1000
    )
    p = baseline + np.clip(delta, -200, 200)
    ok = (
        np.isfinite(p).all(1)
        & (p[:, 0] >= 0)
        & (p[:, 1] <= d["duration"])
        & (p[:, 1] - p[:, 0] >= 200)
    )
    p[~ok] = baseline[~ok]
    return n.ordered(p, baseline)


def final_features(d, baseline, points, old, sources, valid, pp):
    b = n.rounded(baseline)
    critic = model("qfa-selector")
    old_scores = critic.predict(old["x"], num_threads=4)
    parts = []
    for i, base in enumerate(b):
        ids = np.flatnonzero(old["indices"] == i)
        top = ids[np.argsort(-old_scores[ids], kind="stable")[:4]]
        candidates = [base, *old["spans"][top]]
        for name, prediction in points.items():
            p = prediction[i]
            if name == "strongest":
                if np.abs(p - base).max() > 200:
                    continue
                fractions = [0.25]
            elif name == "calibration":
                fractions = [0.25, 0.5, 1.0]
            else:
                fractions = [0.5, 1.0]
            for fraction in fractions:
                p2 = base + fraction * (p - base)
                candidates.extend([p2, [p2[0], base[1]], [base[0], p2[1]]])
        for a in [-40.0, -20.0, 20.0, 40.0]:
            candidates.extend(
                [base + [a, 0], base + [0, a], base + [a, a], base + [a, -a]]
            )
        spans = n.unique_spans(candidates, lambda p: n.valid(p, base, d["duration"]))
        count = len(spans)
        delta = (spans - base) / 1000
        geometry = np.column_stack(
            [
                delta,
                np.abs(delta),
                (spans[:, 1] - spans[:, 0]) / 1000,
                np.full(count, (base[1] - base[0]) / 1000),
                np.any(delta != 0, axis=1),
            ]
        )
        fields = [geometry, np.broadcast_to(d["cue"][i], (count, 69))]
        for j in [i - 1, i + 1]:
            fields.append(
                np.clip((spans - b[j][::-1]) / 1000, -10, 10)
                if 0 <= j < len(b)
                else np.full((count, 2), 10.0)
            )
        for name, p in points.items():
            diff = (p[i] - spans) / 1000
            fields.append(
                np.column_stack(
                    [
                        np.clip(diff, -3, 3),
                        np.abs(diff).max(1) <= 0.05,
                        np.abs(diff).max(1) <= 0.2,
                    ]
                )
            )
            if name in valid:
                fields.append(np.full((count, 1), valid[name][i]))
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
                            np.interp(spans[:, edge] + s, grid, pp[i, edge, h])
                            for s in [-20, 0, 20]
                        ]
                    )
                )
            fields.append(
                ((spans[:, edge] < grid[0]) | (spans[:, edge] > grid[-1]))[:, None]
            )
        parts.append(
            {
                "spans": spans,
                "indices": np.full(count, i),
                "base": np.column_stack(fields).astype(np.float32),
                "critic": qfa_features(
                    spans, old["baseline"][i], d["cue"][i], sources[i]
                ),
            }
        )
    result = {
        k: np.concatenate([r[k] for r in parts])
        for k in ["spans", "indices", "base", "critic"]
    }
    cp = critic.predict(result.pop("critic"), num_threads=4)
    x = result.pop("base")
    ids = result["indices"]
    first = np.r_[True, ids[1:] != ids[:-1]]
    result["x"] = np.column_stack(
        [x, x - x[first][ids], cp, cp - cp[first][ids]]
    ).astype(np.float32)
    result["baseline"] = b
    return result
