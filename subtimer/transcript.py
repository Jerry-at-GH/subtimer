import numpy as np
from scipy.stats import rankdata

from .speech import MANIFEST, model
from .numerics import GRID, ordered, rounded, unique_spans
from .qfa import evidence

SOURCES = (
    "baseline",
    "nfa_raw",
    "nfa_vocals",
    "bfa",
    "qfa",
    "asr",
    "vad",
    "legacy",
    "cue",
    "lexical",
    "unicode",
)


def choose(data, prediction, margin, *, protect=False, local=False):
    ids = data["indices"]
    gain = prediction - prediction[data["baseline_rows"]][ids]
    score = gain.sum(1)
    if protect:
        score[(gain < 0).any(1)] = -np.inf
    if local:
        score[np.abs(data["spans"] - data["baseline"][ids]).max(1) > 500] = -np.inf
    first = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    maximum = np.maximum.reduceat(score, first)
    best = np.minimum.reduceat(
        np.where(score == maximum[ids], np.arange(len(ids)), len(ids)),
        first,
    )
    out = data["baseline"].copy()
    changed = maximum >= margin
    out[changed] = data["spans"][best[changed]]
    return ordered(out, data["baseline"])


def build(
    data,
    baseline,
    scores,
    dense,
    probability,
    history,
    entries,
    line_states,
    sources,
    anchor,
):
    ids, spans = data["indices"], data["spans"]

    x = data["x"][:, MANIFEST["relation_columns"]]
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    columns = np.r_[0:7, 76:164, 328:330]
    baseline_rows, relations = [], []
    for i, start in enumerate(starts):
        stop = starts[i + 1] if i + 1 < len(starts) else len(ids)
        points, local, score = spans[start:stop], x[start:stop], scores[start:stop]
        row = int(np.flatnonzero(np.all(points == baseline[i], axis=1))[0])
        baseline_rows.append(start + row)
        delta = (points - baseline[i]) / 1000
        distance = np.abs(points[:, None] - points[None]).max(2)
        support = np.column_stack(
            [(distance <= t).mean(1) for t in (20, 50, 100, 200, 500)]
        )
        ranked = rankdata(local[:, columns], method="average", axis=0) / len(points)
        centered = local[:, columns] - local[:, columns].mean(0)
        spread = np.broadcast_to(np.ptp(local[:, columns], axis=0), centered.shape)
        geometry = np.column_stack(
            [
                delta,
                np.abs(delta),
                delta.mean(1),
                delta[:, 1] - delta[:, 0],
                np.abs(delta).max(1) > 0.5,
                delta[:, 0] * delta[:, 1] > 0,
                score,
                score - score[row],
                score - score.max(),
                rankdata(score) / len(score),
                support,
            ]
        )
        for neighbor in (i - 1, i + 1):
            geometry = np.column_stack(
                [
                    geometry,
                    np.clip((points - baseline[neighbor][::-1]) / 1000, -30, 30)
                    if 0 <= neighbor < len(baseline)
                    else np.full((len(points), 2), 30.0),
                ]
            )
        relations.append(
            np.column_stack([local - local[row], geometry, ranked, centered, spread])
        )
    relation_data = {
        "spans": spans,
        "indices": ids,
        "baseline": baseline,
        "baseline_rows": np.asarray(baseline_rows),
        "x": np.column_stack([x, np.concatenate(relations)]).astype(np.float32),
    }
    predictions = model("relation-selector").predict(
        relation_data["x"], raw_score=True, num_threads=2
    )
    related = choose(relation_data, predictions, 0.1, protect=True, local=True)
    chunks = []
    critic = model("qfa-selector")
    for i, base in enumerate(baseline):
        bank = [(r["source"], np.asarray(r["span"]), r["score"]) for r in history[i]]
        bank += [(r["source"], np.asarray(r["micro"]), r["score"]) for r in history[i]]
        bank += [
            (source, np.asarray(point), 0.0)
            for source, point in line_states[i]["sources"]
        ]
        bank += [
            (source, np.asarray(point), 0.0)
            for source, point in entries[i]["spans"].items()
            if source in SOURCES and point is not None
        ]
        original = spans[ids == i]
        candidates = [base, *original]
        for _, point, _ in bank:
            candidates.extend([point, [point[0], base[1]], [base[0], point[1]]])
        for shift in (-40, -20, 20, 40):
            candidates.extend(
                [base + [shift, 0], base + [0, shift], base + [shift, shift]]
            )
        points = unique_spans(
            candidates,
            lambda point: (
                np.isfinite(point).all()
                and point[0] >= 0
                and point[1] <= dense["duration"]
                and point[1] - point[0] >= 200
                and np.abs(point - base).max() <= 30000
            ),
        )
        count = len(points)
        delta = (points - base) / 1000
        fields = [
            np.column_stack(
                [
                    delta,
                    np.abs(delta),
                    delta.mean(1),
                    delta[:, 1] - delta[:, 0],
                    (points[:, 1] - points[:, 0]) / 1000,
                    np.full(count, (base[1] - base[0]) / 1000),
                    np.abs(delta).max(1) > 0.5,
                    delta[:, 0] * delta[:, 1] > 0,
                ]
            ),
            np.broadcast_to(dense["cue"][i], (count, 69)),
        ]
        for neighbor in (i - 1, i + 1):
            fields.append(
                np.clip((points - baseline[neighbor][::-1]) / 1000, -30, 30)
                if 0 <= neighbor < len(baseline)
                else np.full((count, 2), 30.0)
            )
        for source in SOURCES:
            candidates = [
                (point, score) for name, point, score in bank if name == source
            ]
            if not candidates:
                fields.append(np.zeros((count, 12)))
                continue
            values = np.array([point for point, _ in candidates])
            source_scores = np.array([score for _, score in candidates])
            diff = (values[None] - points[:, None]) / 1000
            distance = np.abs(diff).max(2)
            nearest = distance.argmin(1)
            fields.append(
                np.column_stack(
                    [
                        np.ones(count),
                        np.clip(diff[np.arange(count), nearest], -30, 30),
                        np.minimum(np.abs(diff).min(1), 30),
                        (distance <= 0.05).any(1),
                        (distance <= 0.2).any(1),
                        np.clip(source_scores[nearest], -3, 3),
                        np.max(
                            np.where(distance <= 0.05, source_scores[None], -3), axis=1
                        ),
                        np.max(
                            np.where(distance <= 0.2, source_scores[None], -3), axis=1
                        ),
                        np.full(count, len(values) / 20),
                        np.full(count, np.clip(source_scores.max(), -3, 3)),
                    ]
                )
            )
        for edge in range(2):
            grid = dense["baseline"][i, edge] + GRID
            fields.append(
                np.column_stack(
                    [
                        np.interp(
                            points[:, edge] + shift, grid, dense["scores"][i, edge]
                        )
                        for shift in (-50, 0, 50)
                    ]
                )
            )
            for head in range(2):
                fields.append(
                    np.column_stack(
                        [
                            np.interp(
                                points[:, edge] + shift,
                                grid,
                                probability[i, edge, head],
                            )
                            for shift in (-20, 0, 20)
                        ]
                    )
                )
            fields.append(
                ((points[:, edge] < grid[0]) | (points[:, edge] > grid[-1]))[:, None]
            )
        prior = anchor[i]
        signal = np.column_stack(
            [evidence(points, record, arrays, prior) for record, arrays in sources[i]]
        )
        prior_signal = np.column_stack(
            [
                evidence(rounded(prior)[None], record, arrays, prior)
                for record, arrays in sources[i]
            ]
        )
        offset = (points - prior) / 1000
        geometry = np.column_stack(
            [
                offset,
                np.abs(offset),
                (points[:, 1] - points[:, 0]) / 1000,
                np.full(count, (prior[1] - prior[0]) / 1000),
                np.any(offset != 0, axis=1),
                np.full(count, len(sources[i][0][0].get("words", [])) / 50),
            ]
        )
        critic_x = np.column_stack(
            [
                geometry,
                np.broadcast_to(dense["cue"][i], (count, 69)),
                signal,
                signal - prior_signal,
            ]
        ).astype(np.float32)
        score = critic.predict(critic_x, num_threads=2)
        fields.append(
            np.column_stack(
                [score, score - score[0], score - score.max(), rankdata(score) / count]
            )
        )
        local = np.column_stack(fields).astype(np.float32)
        chunks.append(
            {
                "spans": points,
                "indices": np.full(count, i),
                "x": np.column_stack(
                    [local, local - local[0], signal, signal - signal[0]]
                ).astype(np.float32),
            }
        )
    recovered = {
        key: np.concatenate([chunk[key] for chunk in chunks])
        for key in ("spans", "indices", "x")
    }
    recovered["baseline"] = baseline
    recovered["baseline_rows"] = np.flatnonzero(
        np.r_[True, recovered["indices"][1:] != recovered["indices"][:-1]]
    )
    predictions = model("recovery-selector").predict(
        recovered["x"], raw_score=True, num_threads=2
    )
    output = choose(recovered, predictions, 0.2)
    changed = np.any(output != baseline, axis=1)
    combined = np.where(changed[:, None], output, related)
    valid = (combined[:, 0] >= 0) & (combined[:, 1] - combined[:, 0] >= 200)
    combined[~valid] = baseline[~valid]
    combined = ordered(combined, baseline)
    recovered["baseline"] = combined
    recovered["baseline_rows"] = np.array(
        [
            np.flatnonzero(
                (recovered["indices"] == i)
                & np.all(recovered["spans"] == point, axis=1)
            )[0]
            for i, point in enumerate(combined)
        ]
    )
    return recovered


def posterior_features(data, posterior):
    chunks = []
    for i in range(len(data["baseline"])):
        spans = data["spans"][data["indices"] == i]
        p = posterior["logp"][i].astype(np.float32)
        p = np.exp(p - np.logaddexp.reduce(p, axis=-1)[:, None])
        times = np.arange(p.shape[1]) * 80 + posterior["starts"][i]
        cdf = np.column_stack([np.zeros(2), p.cumsum(-1)])
        columns = []
        for edge in range(2):
            targets = spans[:, edge]
            mass = []
            for radius in (40, 80, 120, 200, 400, 800):
                lo = np.searchsorted(times, targets - radius, side="left")
                hi = np.searchsorted(times, targets + radius, side="right")
                mass.append(cdf[edge, hi] - cdf[edge, lo])
            peak = int(np.argmax(p[edge]))
            near = np.floor((targets - posterior["starts"][i]) / 80 + 0.5).astype(int)
            inside = (near >= 0) & (near < p.shape[1])
            near = np.clip(near, 0, p.shape[1] - 1)
            columns.append(
                np.column_stack(
                    [
                        *mass,
                        np.log(np.maximum(p[edge, near], 1e-12)),
                        np.clip((targets - times[peak]) / 1000, -30, 30),
                        np.clip(
                            (targets - posterior["spans"][i, edge]) / 1000, -30, 30
                        ),
                        inside,
                        np.full(len(spans), p[edge, peak]),
                        np.full(
                            len(spans),
                            -(p[edge] * np.log(np.maximum(p[edge], 1e-12))).sum(),
                        ),
                        np.full(len(spans), posterior["available"][i]),
                    ]
                )
            )
        x = np.column_stack(columns)
        x = np.column_stack(
            [x, x[:, :6] * x[:, 13:19], np.minimum(x[:, :6], x[:, 13:19])]
        )
        if not posterior["available"][i]:
            x[:] = 0
        chunks.append(x)
    return np.concatenate(chunks).astype(np.float32)


def select(data, original, adapted):
    original = posterior_features(data, original)
    adapted = posterior_features(data, adapted)
    extra = np.column_stack([original, adapted, adapted - original])
    x = np.column_stack(
        [
            data["x"],
            extra,
            extra - extra[data["baseline_rows"]][data["indices"]],
        ]
    ).astype(np.float32)
    prediction = model("posterior-selector").predict(
        x, raw_score=True, num_threads=2
    )
    return choose(data, prediction, 0.4)
