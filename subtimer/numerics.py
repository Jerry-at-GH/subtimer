from collections import defaultdict

import numpy as np

GRID = np.arange(-240, 241, 20, dtype=np.float32)
ZERO = 12
FEATURES = 81


def agreement(span, sources):
    result = []
    for label in ["bfa", "qfa", "nfa_raw", "nfa_vocals", "asr", "baseline"]:
        spans = [p for source, p in sources if source == label]
        if not spans:
            result.extend([0.0, 10.0, 10.0, 0.0, 0.0])
            continue
        delta = (np.asarray(spans) - span) / 1000
        nearest = delta[np.abs(delta).max(1).argmin()]
        result.extend(
            [
                1.0,
                *np.clip(nearest, -10, 10),
                float(np.abs(nearest).max() <= 0.05),
                float(np.abs(nearest).max() <= 0.2),
            ]
        )
    return np.asarray(result)


def feature_matrix(baseline, scores, cue, median, edge):
    n = len(baseline)
    p = scores[:, edge]
    index = np.clip(
        np.arange(len(GRID))[:, None] + [-5, -3, -1, 0, 1, 3, 5], 0, len(GRID) - 1
    )
    curve = p[:, index]
    offset = np.broadcast_to(GRID, (n, len(GRID)))
    delta = median[:, edge] - baseline[:, edge]
    extra = np.stack(
        [
            offset / 240,
            np.abs(offset) / 240,
            np.broadcast_to(
                (baseline[:, 1] - baseline[:, 0])[:, None] / 1000, offset.shape
            ),
            (delta[:, None] - offset) / 240,
            p - p[:, ZERO, None],
        ],
        -1,
    )
    context = np.broadcast_to(cue[:, None, :], (n, len(GRID), 69))
    result = (
        np.concatenate([curve, extra, context], -1)
        .reshape(-1, FEATURES)
        .astype(np.float32)
    )
    return result


def decode(baseline, probabilities, duration, margin=0.1):
    baseline = np.asarray(baseline, dtype=float)
    output = baseline.copy()
    for i, base in enumerate(baseline):
        times = base[:, None] + GRID
        score = (
            probabilities[i, 0, 0, :, None] * probabilities[i, 1, 0, None, :]
            + 0.25 * probabilities[i, 0, 1, :, None] * probabilities[i, 1, 1, None, :]
        )
        legal = (
            (times[1, None, :] - times[0, :, None] >= 200)
            & (times[0, :, None] >= 0)
            & (times[1, None, :] <= duration)
        )
        if not legal[ZERO, ZERO]:
            continue
        score = np.where(legal, score, -np.inf)
        j, k = np.unravel_index(score.argmax(), score.shape)
        if score[j, k] - score[ZERO, ZERO] > margin:
            output[i] = times[0, j], times[1, k]
    return ordered(output, baseline)


def features(spans, ctx):
    out = np.zeros((len(spans), 16 * len(ctx)))
    for index, item in enumerate(ctx):
        if item is None:
            continue
        p, times, score = item
        if not np.isfinite(p).all() or np.any(p.sum(1) < 0.9):
            continue
        p = p / p.sum(1, keepdims=True)
        offset = index * 16
        out[:, offset : offset + 2] = (1.0, score)
        for edge in range(2):
            q = p[edge]
            peak = q.argmax()
            entropy = -(q * np.log(q + 1e-12)).sum() / max(1, np.log(len(q)))
            maximum = q.max()
            for row, span in enumerate(spans):
                delta = times[edge] - span[edge]
                start = offset + 2 + edge * 7
                out[row, start : start + 7] = (
                    q[np.abs(delta) <= 50].sum(),
                    q[np.abs(delta) <= 200].sum(),
                    q[delta < 0].sum(),
                    np.clip((q * delta).sum() / 1000, -3, 3),
                    np.clip(delta[peak] / 1000, -3, 3),
                    entropy,
                    maximum,
                )
    return out


def ordered(output, baseline):
    output = np.asarray(output).copy()
    while True:
        changed = np.any(output != baseline, axis=1)
        bad = (np.diff(baseline.mean(1)) >= 0) & (np.diff(output.mean(1)) < 0)
        revert = np.zeros(len(output), bool)
        for i in np.flatnonzero(bad):
            revert[i : i + 2] |= changed[i : i + 2]
        if not revert.any():
            return output
        output[revert] = baseline[revert]


def prepare(d, p):
    anchor = decode(d["baseline"], p, float(d["duration"]), 0.1)
    idx = np.rint((anchor - d["baseline"]) / 20).astype(int) + ZERO
    rows = np.arange(len(anchor))
    p50 = p[:, 0, 0, :, None] * p[:, 1, 0, None, :]
    p200 = p[:, 0, 1, :, None] * p[:, 1, 1, None, :]
    t = d["baseline"][:, :, None] + GRID
    legal = (
        (t[:, 1, None, :] - t[:, 0, :, None] >= 200)
        & (t[:, 0, :, None] >= 0)
        & (t[:, 1, None, :] <= float(d["duration"]))
    )
    movement = np.abs(t[:, 0, :, None] - anchor[:, 0, None, None]) + np.abs(
        t[:, 1, None, :] - anchor[:, 1, None, None]
    )
    return {
        "anchor": anchor,
        "p50": p50,
        "p200": p200,
        "a50": p50[rows, idx[:, 0], idx[:, 1]],
        "a200": p200[rows, idx[:, 0], idx[:, 1]],
        "legal": legal,
        "movement": movement,
    }


def project(d, q, gain=0.05, slack50=0.01, slack200=0.01):
    allowed = q["legal"] & (q["p50"] >= q["a50"][:, None, None] - slack50)
    best = np.where(allowed, q["p200"], -np.inf).max((1, 2))
    near = (
        allowed
        & (q["p200"] >= best[:, None, None] - slack200)
        & (q["p200"] >= q["a200"][:, None, None] + gain)
    )

    cost = np.where(near, q["movement"] - q["p200"] * 0.001, np.inf)
    index = cost.reshape(len(cost), -1).argmin(1)
    j, k = np.unravel_index(index, cost.shape[1:])
    proposed = d["baseline"] + np.column_stack([GRID[j], GRID[k]])
    return ordered(
        np.where(near.any((1, 2))[:, None], proposed, q["anchor"]), q["anchor"]
    )


def rounded(p):
    return np.floor(np.asarray(p) / 10 + 0.5) * 10


def unique_spans(candidates, legal):
    spans, seen = [], set()
    for p in candidates:
        p = rounded(p)
        key = tuple(p)
        if key in seen or spans and not legal(p):
            continue
        seen.add(key)
        spans.append(p)
    return np.array(spans)


def valid(p, base, duration):
    return (
        np.isfinite(p).all()
        and p[0] >= 0
        and p[1] <= duration
        and p[1] - p[0] >= 200
        and np.abs(p - base).max() <= 3000
    )


def options(base, candidates, row, duration):
    proposals = [(base, "baseline", 1.0, 0)]
    bysource = defaultdict(list)
    for c in candidates:
        bysource[c["source"]].append(c)
    source_spans = []
    for source, cs in bysource.items():
        c = max(cs, key=lambda c: c["score"])
        p = np.asarray(c["span"])
        if source != "vad" and valid(p, base, duration):
            source_spans.append(p)
        for field in ["span", "micro"]:
            if field in c:
                proposals.append((np.asarray(c[field]), source, float(c["score"]), 1))
    for p in row["spans"].values():
        if p is not None:
            p = np.asarray(p)
            proposals.append((p, "cfa", 0.0, 2))
    cfa = [
        np.asarray(p)
        for p in row["spans"].values()
        if p is not None and valid(np.asarray(p), base, duration)
    ]
    if cfa:
        source_spans.append(np.median(cfa, axis=0))
    if source_spans:
        a = np.asarray(source_spans)
        for q in [0.25, 0.5, 0.75]:
            proposals.append((np.quantile(a, q, axis=0), "consensus", 0.0, 3))
        centers = []
        for edge in [0, 1]:
            values, anchor = a[:, edge], base[edge]

            grid = np.unique(np.r_[values, values - 190, values + 190, anchor])
            count = (np.abs(grid[:, None] - values) < 200).sum(1)
            ids = np.flatnonzero(count == count.max())
            centers.append(grid[ids[np.abs(grid[ids] - anchor).argmin()]])
        proposals.append((np.array(centers), "consensus", 0.0, 4))
    seen = set()
    out = []
    for p, source, score, kind in proposals:
        for fraction in [1.0] if kind == 0 else [1.0, 0.5]:
            v = base + fraction * (p - base)
            for edge, q in enumerate(
                [v, np.array([v[0], base[1]]), np.array([base[0], v[1]])]
            ):
                q = rounded(q)
                key = tuple(q)
                if key in seen or (kind != 0 and not valid(q, base, duration)):
                    continue
                seen.add(key)
                out.append((q, source, score, kind, fraction, edge))
    return out, np.asarray(source_spans).reshape(-1, 2)


def choose(data, pred, margin):
    baseline = data["baseline"]
    output = baseline.copy()
    scores = np.full(len(baseline), margin)
    for i, p, s in zip(data["indices"], data["spans"], pred):
        if s > scores[i]:
            output[i] = p
            scores[i] = s
    return ordered(output, baseline)
