import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import gpu, refine, selection, subtitle
from .speech import weights
from . import cfa
from .numerics import GRID, agreement


class Capture:
    def __init__(self, mode="text", shared=None):
        self.states = {}
        self.vectors = OrderedDict()
        self.unsnapped = {}
        self.windows = {}
        self.curves = {}
        self.mode = mode
        self.dense = None
        self.local_state = None
        self.history = []
        self.entries = []
        self.line_states = []
        self.rows = []
        self.set_refinement_mode(mode, force=True)
        self.shared = {} if shared is None else shared
        self.vectors = self.shared.setdefault("vectors", self.vectors)

    def close(self):
        self.states.clear()

    def release(self):
        if pool := self.shared.get("context_pool"):
            pool.shutdown(wait=True)
        for value in (
            self.vectors,
            self.states,
            self.unsnapped,
            self.windows,
            self.curves,
            self.shared,
        ):
            value.clear()

    def set_refinement_mode(self, mode, force=False):
        if force or self.mode != mode:
            self.mode = mode
            self.vectors.clear()
            for edge, label in enumerate(["start", "end"]):
                self.states[edge] = weights()[f"edge-scorer-crnn-{mode}-{label}"]

    def select_path(self, lines, audio, local, vad, asr_tokens, acoustic, model):
        if "acoustic" not in self.shared:
            acoustic.extract = lru_cache(maxsize=65536)(acoustic.extract)
            self.shared["acoustic"] = acoustic
        acoustic = self.shared["acoustic"]
        baselines = [line.candidates["result"] for line in lines]
        pools = []
        features = []
        groups = []
        for i, (line, baseline) in enumerate(zip(lines, baselines)):
            stock = selection._candidates(line, baseline, local[i], vad)
            groups.append(
                (stock, list(range(len(features), len(features) + len(stock))))
            )
            references = {source: p for source, p in stock}
            previous = baselines[i - 1] if i else subtitle.Span(0, 0)
            following = (
                baselines[i + 1] if i + 1 < len(lines) else subtitle.Span(1e9, 1e9)
            )
            pool = []
            for source, p in self.sources(
                line, baseline, local[i], vad, audio.duration
            ):
                span = subtitle.Span(*p)
                features.append(
                    selection._features(
                        source,
                        span,
                        baseline,
                        previous,
                        following,
                        line.text or line.spoken,
                        references,
                        vad["speech"],
                        asr_tokens,
                        acoustic,
                    )
                )
                micro = acoustic.micro_snap(span)
                pool.append(
                    {
                        "source": source,
                        "span": p,
                        "micro": [float(micro.start), float(micro.end)],
                    }
                )
            pools.append(pool)
        scores = model.predict(np.asarray(features, np.float32))
        for row, score in zip([r for pool in pools for r in pool], scores):
            row["score"] = float(score)
        self.history = pools
        return selection._decode(lines, groups, scores, acoustic)

    def context(self, lines, spans, audio, cache_dir, acoustic, local, vad):
        root = Path(cache_dir) / "cfa-context"
        rows = [
            {"text": line.spoken, "window": self.windows.get(id(line))}
            for line in lines
        ]
        acoustic = self.shared.get("acoustic", acoustic)
        context_key = json.dumps(rows, sort_keys=True, ensure_ascii=False)
        entries = self.shared.setdefault("context_entries", {}).get(context_key)
        if entries is None:
            entries = cfa.ensure_context(rows, audio, root)
            self.shared["context_entries"][context_key] = entries
        self.rows = rows
        self.entries = [row for _, row in entries]
        self.line_states = []
        baseline = np.array([[p.start, p.end] for p in spans])
        cue = np.zeros((len(lines), 69), np.float32)
        median = baseline.copy()
        for i, (line, (key, row)) in enumerate(zip(lines, entries)):
            feature = None
            if row["available"]:
                feature = self.shared.setdefault("context_arrays", {}).get(key)
                if feature is None:
                    with np.load(root / (key + ".npz"), allow_pickle=False) as z:
                        feature = dict(z)
                    size = sum(a.nbytes for a in feature.values())
                    total = self.shared.get("context_bytes", 0) + size
                    if total <= 64 * 1024**2:
                        self.shared["context_arrays"][key] = feature
                        self.shared["context_bytes"] = total
            groups = row.get("groups", {}).get("lexical", [])
            sources = self.sources(
                line, line.candidates["result"], local[i], vad, audio.duration
            )
            self.line_states.append(
                {
                    **rows[i],
                    "candidates": {
                        k: [p.start, p.end] for k, p in line.candidates.items()
                    },
                    "sources": sources,
                }
            )
            cue[i] = np.r_[
                (baseline[i, 1] - baseline[i, 0]) / 1000,
                len(line.spoken),
                len(groups),
                float(row.get("lexical_fallback", False)),
                acoustic.extract(*baseline[i], line.spoken),
                cfa.evidence(baseline[i], feature, groups),
                agreement(baseline[i], sources),
                float(row["available"]),
                0.0,
                0.0,
            ]
            proposals = [p for p in row["spans"].values() if p is not None]
            if proposals:
                value = np.median(proposals, axis=0)
                if np.abs(value - baseline[i]).max() <= 1000:
                    median[i] = value
        return baseline, cue, median

    def collect(self, lines, spans, audio, cache_dir, acoustic, local, vad):
        from .cache import Evidence
        from . import evidence

        workers = gpu.concurrency(3)
        if (
            self.local_state is not None
            and self.mode == "text"
            and "components" not in self.shared
            and workers > 1
        ):
            pool = ThreadPoolExecutor(max_workers=1)
            self.shared["context_pool"] = pool
            self.shared["components"] = pool.submit(
                evidence.components,
                Evidence(audio, Path(cache_dir) / "timing-evidence"),
                {
                    "baseline": np.array([[p.start, p.end] for p in spans]),
                    "duration": audio.duration,
                },
                [line.spoken for line in lines],
                lambda *_: None,
                maximum_workers=workers - 1,
            )
        baseline, cue, median = self.context(
            lines, spans, audio, cache_dir, acoustic, local, vad
        )
        self.dense = {
            "baseline": baseline,
            "cue": cue,
            "median": median,
            "scores": self.score(baseline, refine.mel_features(audio, cache_dir)),
            "duration": audio.duration,
        }

    def remember_curves(self, lines, edge, groups):
        for i, values in groups.items():
            entries = sorted(values, key=lambda row: row[1])
            smooth = refine._smooth(np.array([r[0] for r in entries]))
            peaks = []
            for _, (_, time_ms, _) in sorted(
                zip(smooth, entries), key=lambda row: -float(row[0])
            ):
                if all(abs(time_ms - t) >= 100 for t in peaks):
                    peaks.append(float(time_ms))
                if len(peaks) == 2:
                    break
            self.curves.setdefault(id(lines[i]), {})[edge] = peaks

    def _put(self, key, value):
        self.vectors[key] = value
        self.vectors.move_to_end(key)
        while len(self.vectors) > 65536:
            self.vectors.popitem(last=False)

    def capture_scores(self, edge, model, mel, centers, scalars, device):
        offset = 0
        actual = np.clip(centers, 32, mel.shape[2] - 33)

        def hook(module, args):
            nonlocal offset
            values = args[0][:, :384].detach().cpu().numpy().astype(np.float16)
            for frame, value in zip(actual[offset : offset + len(values)], values):
                self._put((edge, int(frame)), value)
            offset += len(values)

        handle = model.head.register_forward_pre_hook(hook)
        try:
            scores = np.zeros(len(centers), dtype=np.float32)
            mel_gpu = torch.from_numpy(np.asarray(mel)).to(device)
            with torch.no_grad():
                for start in range(0, len(centers), 1024):
                    batch = np.arange(start, min(start + 1024, len(centers)))
                    center = (
                        torch.from_numpy(centers[batch])
                        .to(device)
                        .clamp(32, mel_gpu.shape[2] - 33)
                    )
                    index = (
                        center[:, None] + torch.arange(-32, 33, device=device)[None, :]
                    )
                    index = index.clamp(0, mel_gpu.shape[2] - 1)
                    patch = mel_gpu[:, :, index].permute(2, 0, 1, 3).float()
                    extra = torch.from_numpy(scalars[batch]).to(device)
                    scores[batch] = torch.sigmoid(model(patch, extra)).cpu().numpy()
        finally:
            handle.remove()
        return scores

    def score(self, baseline, mel):
        device = "cuda"
        n = len(baseline)
        result = np.empty((n, 2, len(GRID)), np.float32)

        for edge in [0, 1]:
            model = refine.CRNN().to(device)
            model.load_state_dict(self.states[edge])
            model.eval()
            try:
                with torch.no_grad():
                    for cue_start in range(0, n, 32):
                        base = baseline[cue_start : cue_start + 32]
                        centers = (
                            np.rint((base[:, edge, None] + GRID) / 10)
                            .astype(int)
                            .reshape(-1)
                        )
                        hidden = np.empty((len(centers), 384), np.float16)
                        missing = {}
                        for j, frame in enumerate(centers):
                            key = edge, int(frame)
                            if key in self.vectors and 32 <= frame < mel.shape[-1] - 32:
                                hidden[j] = self.vectors[key]
                            else:
                                missing.setdefault(int(frame), []).append(j)
                        frames = list(missing)
                        for first in range(0, len(frames), 128):
                            batch = frames[first : first + 128]
                            idx = np.clip(
                                np.asarray(batch)[:, None] + np.arange(-32, 33),
                                0,
                                mel.shape[-1] - 1,
                            )
                            patch = torch.from_numpy(
                                np.asarray(mel[:, :, idx]).transpose(2, 0, 1, 3).copy()
                            ).to(device, dtype=torch.float32)
                            h = model.frontend(patch).flatten(1, 2).transpose(1, 2)
                            h, _ = model.rnn(h)
                            values = (
                                torch.cat([h[:, h.shape[1] // 2], h.mean(1)], 1)
                                .cpu()
                                .numpy()
                                .astype(np.float16)
                            )
                            for frame, value in zip(batch, values):
                                self._put((edge, int(frame)), value)
                                for j in missing[frame]:
                                    hidden[j] = value
                        scalars = np.column_stack(
                            [
                                np.zeros(len(base)),
                                np.clip(np.log1p(base[:, 1] - base[:, 0]) / 10, 0, 1),
                            ]
                        ).astype(np.float32)
                        scalars = np.repeat(scalars, len(GRID), axis=0)
                        h = torch.from_numpy(hidden.astype(np.float32)).to(device)
                        s = torch.from_numpy(scalars).to(device)
                        result[cue_start : cue_start + len(base), edge] = (
                            model.head(torch.cat([h, s], 1))
                            .sigmoid()[:, 0]
                            .cpu()
                            .numpy()
                            .reshape(len(base), len(GRID))
                        )
            finally:
                del model
                torch.cuda.empty_cache()
        return result

    def sources(self, line, baseline, local, vad, duration):
        stock = selection._candidates(line, baseline, local, vad)
        output = [(source, [p.start, p.end]) for source, p in stock]
        seen = {tuple(p) for _, p in output}

        def add(source, p):
            if p is None:
                return
            p = np.asarray(p)
            if (
                p[1] - p[0] < 200
                or p[0] < 0
                or p[1] > duration
                or abs(p.mean() - baseline.midpoint) > selection.MAX_DISPLACEMENT_MS
                or tuple(p) in seen
            ):
                return
            seen.add(tuple(p))
            output.append((source, p.tolist()))

        for source in ["nfa_raw", "nfa_vocals", "asr"]:
            span = line.candidates.get(source)
            add(source, [span.start, span.end] if span else None)
        for source, span in self.unsnapped.get(id(line), {}).items():
            add(source, span)
        for source, spans in zip(["bfa", "qfa"], local):
            for span in spans:
                add(source, [span.start, span.end])
        peaks = self.curves.get(id(line), {})
        for start in [baseline.start, *peaks.get("start", [])]:
            for end in [baseline.end, *peaks.get("end", [])]:
                add("baseline", [start, end])
        return output
