import math
import sys

import numpy as np
import soundfile as sf
import torch
from torch import nn

from . import gpu
from .speech import qfa_path, weights
from .subtitle import alignment_text
from .numerics import ordered

SHIFTS = np.arange(-240, 241, 80)
GRID = np.arange(-300, 301, 10)


class Adapter(nn.Module):
    route = None

    def __init__(self, base, endpoint, jitter, key):
        super().__init__()
        self.base = base
        self.scale = 2.0
        for suffix in ("a", "b"):
            self.register_buffer(suffix, endpoint[key + "." + suffix])
            self.register_buffer("jitter_" + suffix, jitter[key + "." + suffix])

    def forward(self, x):
        result = self.base(x)
        if self.route is None:
            return result
        a, b = (
            (self.a, self.b)
            if self.route == "endpoint"
            else (self.jitter_a, self.jitter_b)
        )
        return result + ((x.float() @ a.T) @ b.T * self.scale).to(x.dtype)


def context(selected):
    words = selected.float().reshape(-1, 2, 1024)
    mean = words.mean(1)
    zero = torch.zeros_like(mean[0])
    ctx = torch.stack(
        [
            torch.stack(
                [words[0, 0], mean[0], zero, mean[1] if len(words) > 1 else zero]
            ),
            torch.stack(
                [words[-1, 1], mean[-1], mean[-2] if len(words) > 1 else zero, zero]
            ),
        ]
    )
    mask = torch.tensor(
        [[1, 1, 0, len(words) > 1], [1, 1, len(words) > 1, 0]],
        device=selected.device,
        dtype=torch.bool,
    )
    return ctx.half().float(), mask


def statistics(logp, centers, available, baseline, duration):
    lp = np.asarray(logp, np.float16).astype(np.float64)
    p = np.exp(lp - np.logaddexp.reduce(lp, axis=-1)[..., None])
    return {
        "mean": (p * (centers[..., None] + GRID)).sum(-1),
        "entropy": -(p * np.log(np.maximum(p, 1e-30))).sum(-1) / np.log(61),
        "available": available,
        "baseline": baseline,
        "duration": duration,
    }


def legal_ordered(proposal, fallback, duration):
    legal = (
        np.isfinite(proposal).all(-1)
        & (proposal[:, 0] >= 0)
        & (proposal[:, 1] <= duration)
        & (proposal[:, 1] - proposal[:, 0] >= 200)
    )
    proposal[~legal] = fallback[~legal]
    return ordered(proposal, fallback)


def proposal(distribution, fallback):
    baseline = distribution["baseline"]
    delta = distribution["mean"] - baseline
    valid = (
        distribution["available"]
        & (abs(delta) <= 100)
        & (distribution["entropy"] <= 0.8)
    )
    out = np.floor(np.where(valid, baseline + 0.25 * delta, fallback) / 10 + 0.5) * 10
    return legal_ordered(out, fallback, distribution["duration"])


def decode(distributions):
    canonical, tail, head = (distributions[k] for k in ("canonical", "tail", "head"))
    retained = proposal(canonical, canonical["baseline"])
    a = proposal(tail, retained) - retained
    b = proposal(head, retained) - retained
    available = canonical["available"] & tail["available"] & head["available"]
    delta = np.where(a * b >= 0, np.where(abs(a) >= abs(b), a, b), 0)
    out = retained + np.where(available & (abs(delta) <= 50), delta, 0)
    return legal_ordered(out, retained, canonical["duration"]), retained


def audio_encoding(head, features):
    audio = head.audio(features.transpose(1, 2)).transpose(1, 2)
    normalized = head.audio_norm(audio)
    return (
        audio,
        normalized,
        head.activity_bias(normalized).squeeze(-1),
        head.boundary_bias(normalized[:, 14:27]).squeeze(-1),
    )


def boundary_logits(head, encoded, context, context_mask, endpoint):
    roles = torch.arange(4, device=context.device)
    tokens = head.context_projection(context) + head.role_embedding(roles)[None]
    base_query = tokens[:, :1] + head.endpoint_embedding(endpoint)[:, None]
    query, _ = head.context_attention(
        base_query,
        tokens,
        tokens,
        key_padding_mask=~context_mask.bool(),
        need_weights=False,
    )
    query = head.query_norm(query + base_query)
    audio, normalized, activity_bias, local_bias = encoded
    attended, _ = head.audio_attention(query, audio, audio, need_weights=False)
    query = head.query_norm(query + attended)
    scale = math.sqrt(float(head.config.model_dim))
    activity = (normalized * query).sum(-1) / scale + activity_bias
    local = (normalized[:, 14:27] * query).sum(-1) / scale + local_bias
    left = torch.cat((activity[:, :1], activity[:, :-1]), dim=1)
    derivative = (activity - left)[:, 14:27]
    direction = torch.where(endpoint[:, None] == 0, derivative, -derivative)
    return local + head.derivative_scale * direction + head.qwen_prior[None]


class Joint:
    def __init__(self):
        torch.set_num_threads(2)
        gpu.configure()
        path = qfa_path()
        sys.path.insert(0, str(path))
        from qwen_fa_ultra.features import extract_fixed_center_features
        from qwen_fa_ultra.head import load_local_head

        self.path = path
        self.qwen = None
        endpoint = {
            name: tensor.cuda()
            for name, tensor in weights()["joint-refiner-endpoint-adapter"].items()
        }
        jitter = {
            name: tensor.cuda()
            for name, tensor in weights()["joint-refiner-jitter-adapter"].items()
        }
        self.adapters = []
        self.endpoint_state = endpoint
        self.jitter_state = jitter
        self.endpoint_delta = endpoint["head_delta"]
        self.jitter_delta = jitter["head_delta"]
        self.endpoint_head, self.normalizer, _ = load_local_head(
            path / "local_head", device="cuda"
        )
        self.endpoint_head.load_state_dict(weights()["joint-refiner-endpoint-head"])
        self.endpoint_head.eval().requires_grad_(False)
        self.jitter_head, _, _ = load_local_head(path / "local_head", device="cuda")
        self.jitter_head.load_state_dict(weights()["joint-refiner-jitter-head"])
        self.jitter_head.eval().requires_grad_(False)
        self.features = extract_fixed_center_features
        self.edge = torch.arange(2, device="cuda")[:, None].expand(-1, 7).flatten()
        self.scatter_ids = torch.tensor(
            ((SHIFTS[:, None] + np.arange(-60, 61, 10) + 300) // 10).flatten(),
            device="cuda",
        )[None].expand(2, -1)

    def route(self, name):
        for module in self.adapters:
            module.route = name

    @torch.no_grad()
    def evidence(self, signals, texts):
        if self.qwen is None:
            from qwen_asr import Qwen3ForcedAligner

            self.qwen = Qwen3ForcedAligner.from_pretrained(
                str(self.path),
                dtype=torch.bfloat16,
                device_map="cuda",
                attn_implementation="sdpa",
            )
            self.qwen.model.eval().requires_grad_(False)
            for i, layer in enumerate(self.qwen.model.thinker.model.layers[-4:]):
                for name in ("q_proj", "v_proj"):
                    module = Adapter(
                        getattr(layer.self_attn, name),
                        self.endpoint_state,
                        self.jitter_state,
                        f"layers.{i}.self_attn.{name}",
                    )
                    setattr(layer.self_attn, name, module)
                    self.adapters.append(module)
        encoded = [
            self.qwen.aligner_processor.encode_timestamp(t, "Japanese") for t in texts
        ]
        usable = [i for i, (words, _) in enumerate(encoded) if words]
        if not usable:
            return [None] * len(texts)
        signals = [signals[i] for i in usable]
        encoded = [encoded[i] for i in usable]
        inp = self.qwen.processor(
            text=[p for w, p in encoded],
            audio=signals,
            return_tensors="pt",
            padding=True,
        )
        inp = inp.to(self.qwen.model.device).to(self.qwen.model.dtype)
        positions = [
            torch.where(ids == self.qwen.timestamp_token_id)[0] for ids in inp.input_ids
        ]
        capture = {}

        def prefix(module, args, kwargs):
            capture["hidden"] = args[0].detach()
            capture["kwargs"] = dict(kwargs)

        handle = self.qwen.model.thinker.model.layers[-4].register_forward_pre_hook(
            prefix, with_kwargs=True
        )
        try:
            original = self.qwen.model.thinker(**inp, use_cache=False).logits.float()
        finally:
            handle.remove()

        def posterior(logits, words):
            parsed = self.qwen.aligner_processor.parse_timestamp(
                words, (logits.argmax(-1) * 80).cpu().numpy()
            )
            return {
                "logp": logits[[0, -1]].log_softmax(-1).cpu().numpy(),
                "spans": np.array([parsed[0]["start_time"], parsed[-1]["end_time"]]),
            }

        result = [{} for _ in signals]
        try:
            for name, delta in [
                ("endpoint", self.endpoint_delta),
                ("jitter", self.jitter_delta),
            ]:
                self.route(name)
                hidden = capture["hidden"]
                for layer in self.qwen.model.thinker.model.layers[-4:]:
                    hidden = layer(hidden, **capture["kwargs"])
                hidden = self.qwen.model.thinker.model.norm(hidden)
                for i, p in enumerate(positions):
                    selected = hidden[i, p]
                    logits = (
                        self.qwen.model.thinker.lm_head(selected).float()
                        + selected.float() @ delta.T
                    )
                    ctx, mask = context(selected)
                    result[i][name] = {
                        "context": ctx.cpu().numpy().astype(np.float16),
                        "context_mask": mask.cpu().numpy(),
                    }
                    if name == "endpoint":
                        result[i][name].update(
                            posterior(logits, encoded[i][0]),
                            original=posterior(original[i, p], encoded[i][0]),
                        )
                    else:
                        result[i][name]["logp"] = (
                            logits[[0, -1]].log_softmax(-1).cpu().numpy()
                        )
        finally:
            self.route(None)
        complete = [None] * len(texts)
        for i, value in zip(usable, result):
            complete[i] = value
        return complete

    def mixture(self, head, encoded, context, context_mask, coarse, valid):
        cm = context_mask.clone()
        cm[~cm.any(-1), 0] = True
        logits = boundary_logits(
            head,
            encoded,
            context[:, None].expand(-1, 7, -1, -1).flatten(0, 1),
            cm[:, None].expand(-1, 7, -1).flatten(0, 1),
            self.edge,
        ).reshape(2, 7, 13)
        mask = valid.clone()
        mask[~mask.any(-1), 3] = True
        p = (
            (logits / 1.5 + coarse[..., None] / 2)
            .masked_fill(~mask[..., None], -1e9)
            .flatten(1)
            .softmax(-1)
        )
        return (
            torch.zeros((2, 61), device=p.device)
            .scatter_add(1, self.scatter_ids, p)
            .clamp_min(1e-12)
            .log()
        )

    @torch.no_grad()
    def refine(self, records, evidence, baseline, duration, read_audio):
        n = len(records)
        centers = np.zeros((n, 2))
        available = np.zeros((n, 2), bool)
        post = {
            k: np.zeros((n, 2, 61), np.float16) for k in ("canonical", "tail", "head")
        }
        for i, (record, ev) in enumerate(zip(records, evidence)):
            bins = (
                np.floor((baseline[i] - record["start"]) / 80 + 0.5).astype(int)[
                    :, None
                ]
                + SHIFTS // 80
            )
            absolute = record["start"] + bins * 80
            centers[i] = absolute[:, 3]
            if ev is None:
                continue
            valid = (
                (bins >= 0)
                & (absolute <= record["end"])
                & (absolute >= 0)
                & (absolute <= duration)
            )
            available[i] = valid.any(-1) & ev["endpoint"]["context_mask"].any(-1)
            start = max(0, int(absolute.min()) - 230)
            end = min(int(duration), int(absolute.max()) + 230)
            features = self.features(
                read_audio(start, end),
                absolute.flatten() - start,
                normalizer=self.normalizer,
            )
            acoustic = torch.tensor(
                features.astype(np.float16), device="cuda", dtype=torch.float32
            )
            inputs = {}
            for name, item in ev.items():
                inputs[name] = [
                    torch.tensor(
                        x,
                        device="cuda",
                        dtype=torch.bool
                        if np.asarray(x).dtype == bool
                        else torch.float32,
                    )
                    for x in (
                        item["context"],
                        item["context_mask"],
                        np.take_along_axis(item["logp"], bins.clip(0, 4999), -1),
                        valid,
                    )
                ]
            encoded = audio_encoding(self.endpoint_head, acoustic)
            for name, source in (("canonical", "endpoint"), ("tail", "jitter")):
                post[name][i] = (
                    self.mixture(self.endpoint_head, encoded, *inputs[source])
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            encoded = audio_encoding(self.jitter_head, acoustic)
            post["head"][i] = (
                self.mixture(self.jitter_head, encoded, *inputs["endpoint"])
                .cpu()
                .numpy()
                .astype(np.float16)
            )
        distributions = {
            name: statistics(lp, centers, available, baseline, duration)
            for name, lp in post.items()
        }
        out, retained = decode(distributions)
        return out, distributions, retained


def align(audio, texts, common, candidates, cache, progress=lambda *_: None):
    from . import transcript

    duration = sf.info(audio.raw).frames / 16
    records = []
    for i, text in enumerate(texts):
        start = max(0, int(common[i, 0]) - 1200)
        end = min(audio.duration, int(common[i, 1]) + 1200)
        records.append({"start": start, "end": end, "text": alignment_text(text)})

    def read(start, end):
        return sf.read(
            audio.raw, start=round(start * 16), stop=round(end * 16), dtype="float32"
        )[0]

    progress("refine")
    model = Joint()
    requests = [
        {"kind": "qfa-joint", "view": "raw", **r}
        if r["text"].strip() and 200 <= r["end"] - r["start"] <= 16000
        else None
        for r in records
    ]
    missing = [
        r
        for r in requests
        if r is not None and not cache.path(r).with_suffix(".json").is_file()
    ]
    missing.sort(key=lambda r: (round((r["end"] - r["start"]) / 1000), len(r["text"])))
    size = gpu.batch_size(4, weights=2 * gpu.GIB, workspace=gpu.GIB // 2)
    for left in range(0, len(missing), size):
        batch = missing[left : left + size]
        values = model.evidence(
            [read(r["start"], r["end"]) for r in batch], [r["text"] for r in batch]
        )
        for request, result in zip(batch, values):
            arrays = {}
            if result is None:
                cache.put(request, {"available": False}, arrays)
                continue
            for name, branch in result.items():
                for key, value in branch.items():
                    if key == "original":
                        arrays.update({"original_" + k: v for k, v in value.items()})
                    else:
                        arrays[name + "_" + key] = value
            cache.put(request, {"available": True}, arrays)
    evidence = []
    for request in requests:
        saved = cache.get(request) if request is not None else None
        result = None
        if saved is not None and saved[0]["available"]:
            arrays = saved[1]
            result = {
                name: {
                    key: arrays[name + "_" + key]
                    for key in ("logp", "context", "context_mask")
                }
                for name in ("endpoint", "jitter")
            }
            result["endpoint"]["spans"] = arrays["endpoint_spans"]
            result["endpoint"]["original"] = {
                key: arrays["original_" + key] for key in ("logp", "spans")
            }
        evidence.append(result)
    posterior = {}
    for name in ("original", "endpoint"):
        rows = [
            (item["endpoint"]["original"] if name == "original" else item["endpoint"])
            if item is not None
            else None
            for item in evidence
        ]
        posterior[name] = {
            "logp": np.asarray(
                [
                    row["logp"] if row is not None else np.full((2, 5000), -100)
                    for row in rows
                ],
                np.float16,
            ),
            "spans": np.asarray(
                [
                    row["spans"] + record["start"] if row is not None else [-1, -1]
                    for row, record in zip(rows, records)
                ]
            ),
            "available": np.asarray(
                [row is not None and row["spans"][1] > row["spans"][0] for row in rows]
            ),
            "starts": np.asarray([record["start"] for record in records]),
        }
    baseline = transcript.select(
        candidates, posterior["original"], posterior["endpoint"]
    )
    return model.refine(records, evidence, baseline, duration, read)[0]
