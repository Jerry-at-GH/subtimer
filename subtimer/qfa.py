import sys

import numpy as np
import torch
from scipy.special import softmax

from . import gpu
from .speech import qfa_path


def boundary_context(hidden, positions, words, *, qwen_dim):
    word_hidden = (
        hidden.index_select(0, positions).float().reshape(len(words), 2, qwen_dim)
    )
    current = word_hidden.mean(dim=1)
    padded = torch.nn.functional.pad(current, (0, 0, 1, 1))
    context = torch.stack(
        (
            word_hidden,
            current[:, None].expand(-1, 2, -1),
            padded[:-2, None].expand(-1, 2, -1),
            padded[2:, None].expand(-1, 2, -1),
        ),
        dim=2,
    ).reshape(-1, 4, qwen_dim)
    mask = torch.ones((len(words), 2, 4), dtype=torch.float32, device=hidden.device)
    mask[0, :, 2] = 0
    mask[-1, :, 3] = 0
    endpoints = torch.arange(2, device=hidden.device).repeat(len(words))
    return context, mask.reshape(-1, 4), endpoints


def finish(fixed, corrections, duration):
    import qwen_fa_ultra.runtime as rt

    intervals = []
    for j in range(len(fixed) // 2):
        a, b = np.asarray(fixed[2 * j : 2 * j + 2], float) + np.asarray(
            corrections[2 * j : 2 * j + 2]
        )
        if not np.isfinite([a, b]).all():
            a, b = fixed[2 * j : 2 * j + 2]
        elif b - a < 10:
            m = (a + b) / 2
            a, b = (0.0, 10.0) if m < 5 else (m - 5, m + 5)
        elif a < 0:
            b -= a
            a = 0.0
        if not np.isfinite([a, b]).all() or a < 0 or b < 0 or b - a < 10:
            a, b = 0.0, 10.0
        intervals.append([float(a), float(b)])
    rt._remove_neighbor_overlaps(intervals)
    rt._bound_intervals_to_duration(intervals, duration)
    return np.asarray(intervals)


class QFA:
    def __init__(self):
        torch.set_num_threads(4)
        gpu.configure()
        sys.path.insert(0, str(qfa_path()))
        from qwen_fa_ultra import LocalBoundaryAligner

        self.local = LocalBoundaryAligner.from_pretrained(
            str(qfa_path()),
            device="cuda",
            dtype=torch.bfloat16,
            head_batch_size=32,
        )
        import qwen_fa_ultra.runtime as rt

        rt._context_for_sample = boundary_context
        self.qwen = self.local._qwen

    def run_many(self, signals, texts, anchors):
        import qwen_fa_ultra.runtime as rt
        from qwen_fa_ultra.head import decode_offsets

        capture = {"fixed": [], "positions": [], "context": []}
        old_feature = rt.extract_fixed_center_features
        old_context = rt._context_for_sample

        def features(audio, centers, **kw):
            capture["fixed"].append(np.array(centers))
            return old_feature(audio, centers, **kw)

        def context(*a, **kw):
            capture["positions"].append(a[1])
            r = old_context(*a, **kw)
            capture["context"].append(r)
            return r

        def logits(module, args, result):
            capture["logits"] = result.logits

        def head(module, args, result):
            capture["head_logits"] = result["boundary_logits"]

        rt.extract_fixed_center_features = features
        rt._context_for_sample = context
        h1 = self.qwen.model.thinker.register_forward_hook(logits)
        h2 = self.local._head.register_forward_hook(head)
        try:
            batches = self.local.align(
                audio=[(pcm, 16000) for pcm in signals],
                text=texts,
                language=["Japanese"],
            )
        finally:
            rt.extract_fixed_center_features = old_feature
            rt._context_for_sample = old_context
            h1.remove()
            h2.remove()
        output = []
        cursor = 0
        for sample, (pcm, anchor, items) in enumerate(zip(signals, anchors, batches)):
            fixed = capture["fixed"][sample]
            safe = finish(fixed, np.zeros(len(fixed)), len(pcm) / 16)
            native = (
                np.array(
                    [
                        [r.unconstrained_start_time, r.unconstrained_end_time]
                        for r in items
                    ]
                )
                * 1000
            )
            refined = np.array([[r.start_time, r.end_time] for r in items]) * 1000
            words = [
                {
                    "text": r.text,
                    "start": float(native[j, 0]),
                    "end": float(native[j, 1]),
                }
                for j, r in enumerate(items)
            ]
            out = {
                name: p[[0, -1], [0, 1]].tolist()
                for name, p in [("native", native), ("safe", safe), ("head", refined)]
            }
            slots = capture["positions"][sample][[0, -1]]
            logp = (
                capture["logits"][sample, slots]
                .float()
                .log_softmax(-1)
                .detach()
                .cpu()
                .numpy()
            )
            context, mask, endpoints = capture["context"][sample]
            context = context[[0, -1]].detach()
            mask = mask[[0, -1]].detach()
            endpoints = endpoints[[0, -1]].detach()
            hp = (
                capture["head_logits"][[cursor, cursor + len(fixed) - 1]]
                .float()
                .detach()
                .cpu()
                .numpy()
            )
            arrays = {
                "logp": logp.astype(np.float16),
                "head_logits": hp.astype(np.float16),
                "head_centers": fixed[[0, -1]],
            }
            if anchor is not None:
                local_features = old_feature(
                    pcm,
                    np.asarray(anchor),
                    batch_size=2,
                    normalizer=self.local._normalizer,
                )
                with torch.inference_mode():
                    r = self.local._head(
                        torch.from_numpy(local_features).to(context.device),
                        context,
                        mask,
                        endpoints,
                    )
                    delta = decode_offsets(r["boundary_logits"]).cpu().numpy()
                out["transfer"] = (np.array(anchor) + delta).tolist()
                arrays["transfer_logits"] = (
                    r["boundary_logits"]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            output.append(
                (
                    {
                        "words": words,
                        "outputs": out,
                        "word_outputs": {
                            k: p.tolist()
                            for k, p in [
                                ("native", native),
                                ("safe", safe),
                                ("head", refined),
                            ]
                        },
                        "duration_limited": sum(r.duration_limited for r in items),
                    },
                    arrays,
                )
            )
            cursor += len(fixed)
        return output


def probability(spans, p, times):
    fields = []
    for edge in range(2):
        q = np.asarray(p[edge], float)
        q = q / max(q.sum(), 1e-30)
        t = times[edge]
        s = spans[:, edge]
        cum = np.r_[0.0, np.cumsum(q)]

        def mass(radius, cum=cum, t=t, s=s):
            return (
                cum[np.searchsorted(t, s + radius, side="right")]
                - cum[np.searchsorted(t, s - radius, side="left")]
            )

        fields += [
            mass(50),
            mass(200),
            cum[np.searchsorted(t, s, side="left")],
            np.clip(((q * t).sum() - s) / 1000, -3, 3),
            np.clip((t[q.argmax()] - s) / 1000, -3, 3),
            np.full(len(s), -(q * np.log(q + 1e-30)).sum() / max(1, np.log(len(q)))),
            np.full(len(s), q.max()),
            np.full(len(s), float(p[edge].sum())),
        ]
    return np.column_stack(fields)


def evidence(spans, r, arrays, anchor):
    if not r.get("available") or arrays is None:
        return np.zeros((len(spans), 88))
    n = len(spans)
    fields = [
        np.broadcast_to(
            [
                1.0,
                min(len(r["words"]) / 50, 3.0),
                (r["end"] - r["start"]) / 30000,
                r.get("duration_limited", 0) / max(1, len(r["words"])),
            ],
            (n, 4),
        )
    ]
    for kind in ["native", "safe", "head", "transfer"]:
        point = r["outputs"].get(kind)
        if point is None:
            fields.append(np.zeros((n, 5)))
            continue
        delta = (np.array(point) - spans) / 1000
        fields.append(
            np.column_stack(
                [
                    np.ones(n),
                    np.clip(delta, -3, 3),
                    (np.abs(delta).max(1) <= 0.05),
                    (np.abs(delta).max(1) <= 0.2),
                ]
            )
        )
    lp = arrays["logp"].astype(float)
    q = softmax(lp, axis=1)
    t = np.broadcast_to(r["start"] + np.arange(q.shape[1]) * 80, q.shape)
    fields.append(probability(spans, q, t))
    masked = q * (t <= r["end"])
    fields.append(probability(spans, masked, t))
    h = softmax(arrays["head_logits"].astype(float) / 1.5, axis=1)
    ht = r["start"] + arrays["head_centers"][:, None] + np.arange(-60, 61, 10)[None]
    fields.append(probability(spans, h, ht))
    if "transfer_logits" in arrays:
        q = softmax(arrays["transfer_logits"].astype(float) / 1.5, axis=1)
        fields.append(
            probability(spans, q, anchor[:, None] + np.arange(-60, 61, 10)[None])
        )
    else:
        fields.append(np.zeros((n, 16)))
    return np.column_stack(fields)
