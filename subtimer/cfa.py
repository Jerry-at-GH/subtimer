import gc
import json
import math
import os
import sys
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.special import log_softmax, softmax
from torch import nn

from . import cache, gpu
from .ctc import optional_path
from .speech import MANIFEST, MODEL_DIR, weights
from .subtitle import MIN_DURATION, alignment_text
from .joint import GRID, proposal, statistics

REVISION = "f4334f6e8193f2691212d49b20fa12d370e13896"
REPOSITORY = "nyralabs/CrisperWhisper2.0_large"


def cfa_path():
    path = MODEL_DIR / "cfa"
    if not (path / "model.safetensors").is_file():
        from huggingface_hub import snapshot_download

        snapshot_download(REPOSITORY, revision=REVISION, local_dir=path)
    return path


def normalized(s):
    return "".join(c for c in unicodedata.normalize("NFKC", s) if not c.isspace())


def groups_for(tokenizer, ids):
    def decode(v):
        return tokenizer.decode(v, clean_up_tokenization_spaces=False)

    units = []
    pending = []
    for i in range(len(ids)):
        pending.append(i)
        piece = decode([ids[j] for j in pending])
        if "\ufffd" not in piece:
            units.append((pending, piece))
            pending = []
    if pending:
        raise ValueError("incomplete UTF-8 token group")
    surface = "".join(piece for _, piece in units)
    if surface != decode(ids):
        raise ValueError("inconsistent token decoding")
    unicode_groups = [idx for idx, piece in units if any(c.isalnum() for c in piece)]
    import pyopenjtalk

    lexical = [r["string"] for r in pyopenjtalk.run_frontend(surface)]
    fallback = normalized("".join(lexical)) != normalized(surface)
    if fallback:
        return unicode_groups, unicode_groups, True
    boundaries = set()
    offset = 0
    for word in lexical:
        offset += len(normalized(word))
        boundaries.add(offset)
    lexical_groups = []
    pending = []
    pending_text = ""
    offset = 0
    for idx, piece in units:
        pending += idx
        pending_text += piece
        offset += len(normalized(piece))
        if offset in boundaries:
            if any(c.isalnum() for c in pending_text):
                lexical_groups.append(pending)
            pending = []
            pending_text = ""
    if pending:
        raise ValueError("unconsumed lexical tokens")
    return lexical_groups, unicode_groups, False


def context_span(logp, blank, groups, origin, duration, legacy=False):
    if not groups:
        return None
    if legacy:
        from crisperwhisper.word_timing import viterbi_align_words_with_blanks

        timings = viterbi_align_words_with_blanks(logp, blank, groups)
        if any(a is None or b is None for a, b in timings):
            return None
        start, end = timings[0][0] * 1000, timings[-1][1] * 1000
    else:
        emissions = np.asarray(
            [np.logaddexp.reduce(logp[g], axis=0) for g in groups], np.float32
        )
        path, score = optional_path(emissions, blank)
        first = np.flatnonzero(path == 1)
        last = np.flatnonzero(path == 2 * len(groups) - 1)
        if not np.isfinite(score) or not len(first) or not len(last):
            return None
        start, end = float(first[0] * 20), float(last[-1] * 20)
    if not 0 <= start < end <= duration or end - start < MIN_DURATION:
        return None
    return [origin + start, origin + end]


def evidence(span, feature, groups):
    if feature is None or not groups:
        return np.zeros(20)
    n = min(
        feature["attention"].shape[1], int(np.ceil(float(feature["duration"]) / 20))
    )
    a, b = (np.asarray(span) - float(feature["origin"])) / 20
    att = feature["attention"][:, :n]
    frames = np.arange(n)
    result = [1.0, float(a < 0), float(b > n)]
    for group, edge in [(groups[0], a), (groups[-1], b)]:
        p = np.maximum(att[group].mean(0), 0).astype(float)
        p /= max(p.sum(), 1e-12)
        result.extend(
            [
                p[frames < a].sum(),
                p[frames > b].sum(),
                p[np.abs(frames - edge) <= 5].sum(),
                np.clip((p.argmax() - edge) * 0.02, -5, 5),
                -(p * np.log(p + 1e-12)).sum() / max(1, np.log(n)),
                p.max(),
            ]
        )
    blank = np.exp(feature["blank"][:n])

    def mean(lo, hi):
        values = blank[(frames >= lo) & (frames < hi)]
        return float(values.mean()) if len(values) else 0.0

    result.extend(
        [mean(a, b), mean(a - 5, a), mean(a, a + 5), mean(b - 5, b), mean(b, b + 5)]
    )
    return np.asarray(result)


def ensure_context(rows, audio, root):
    root.mkdir(parents=True, exist_ok=True)
    keys = [cache.key({"text": row["text"], "window": row["window"]}) for row in rows]
    missing = [
        dict(row, key=name)
        for row, name in zip(rows, keys)
        if not (root / (name + ".json")).is_file()
    ]
    if missing:
        payload = {"rows": missing, "audio": str(audio.vocals), "root": str(root)}
        cache.infer("subtimer.cfa", payload, root, "inference.log")
    return [(name, cache.load(root / (name + ".json"))) for name in keys]


def infer(payload):
    import soundfile as sf
    import torch
    from crisperwhisper import CrisperWhisperModel
    from crisperwhisper.prompt import PromptBuilder
    from crisperwhisper.word_timing import (
        blank_logp_from_mel_energy,
        token_logp_from_attention,
    )

    root = Path(payload["root"])
    rows = payload["rows"]
    torch.set_num_threads(4)
    usable = [
        r
        for r in rows
        if r["window"] and r["window"]["end"] - r["window"]["start"] <= 18000
    ]
    model = None
    if usable:
        directory = cfa_path()
        gpu.configure()
        model = CrisperWhisperModel(
            str(directory),
            backend="transformers",
            device="cuda",
            compute_type="float16",
        )
        engine = model._engine
        engine.enable_attention()
        import copy

        encoder = engine.model.model.encoder
        for layer in encoder.layers:
            layer.self_attn.config = copy.deepcopy(encoder.config)
            layer.self_attn.config._attn_implementation = "sdpa"
        prompt = PromptBuilder(engine, language="ja").verbatim()
        decoder = engine.model.model.decoder
        decoder.layers = decoder.layers[
            : max(i for i, _ in engine._alignment_heads) + 1
        ]
        captured = {}
        for layer in {i for i, _ in engine._alignment_heads}:
            selected = [head for i, head in engine._alignment_heads if i == layer]

            def capture(_, args, output, layer=layer, selected=selected):
                weights = output[1][:, selected]
                for index, head in enumerate(selected):
                    captured[layer, head] = weights[:, index]

            decoder.layers[layer].encoder_attn.register_forward_hook(capture)
        rows.sort(key=lambda r: len(engine.encode_text(r["text"])))
    batch_size = (
        gpu.batch_size(2, weights=0, workspace=gpu.GIB) if model is not None else 1
    )
    with sf.SoundFile(payload["audio"]) as audio:
        for left in range(0, len(rows), batch_size):
            chunk = rows[left : left + batch_size]
            prepared = []
            extracted = {}
            for row in chunk:
                window = row["window"]
                if (
                    model is None
                    or not window
                    or not 0 <= window["start"] < window["end"] <= len(audio) / 16
                    or window["end"] - window["start"] > 18000
                ):
                    continue
                ids = engine.encode_text(row["text"])
                if not ids or len(ids) + len(prompt) > 400:
                    continue
                audio.seek(int(window["start"] * 16))
                pcm = audio.read(
                    int(window["end"] * 16) - int(window["start"] * 16), dtype="float32"
                )
                inputs, mel = engine.extract_features_with_mel(pcm)
                prepared.append((row["key"], ids, inputs, mel, len(pcm) / 16))
            if prepared:
                with torch.inference_mode():
                    encoded = encoder(
                        torch.cat([r[2] for r in prepared]),
                        output_attentions=False,
                        return_dict=False,
                    )[0]
                    maximum = max(len(prompt) + len(r[1]) for r in prepared)
                    tokens = torch.full(
                        (len(prepared), maximum),
                        engine.eot_id,
                        device=engine.device,
                        dtype=torch.long,
                    )
                    mask = torch.zeros_like(tokens)
                    for i, (_, ids, _, _, _) in enumerate(prepared):
                        tokens[i, : len(prompt) + len(ids)] = torch.tensor(
                            prompt + ids, device=engine.device
                        )
                        mask[i, : len(prompt) + len(ids)] = 1
                    captured.clear()
                    decoder(
                        input_ids=tokens,
                        attention_mask=mask,
                        encoder_hidden_states=encoded,
                        output_attentions=False,
                        use_cache=False,
                        return_dict=True,
                    )
                    for i, (name, ids, _, mel, duration) in enumerate(prepared):
                        attention = torch.zeros(
                            (len(ids), 1500), device=engine.device, dtype=torch.float32
                        )
                        for layer, head in engine._alignment_heads:
                            attention += captured[layer, head][
                                i, len(prompt) - 1 : len(prompt) - 1 + len(ids)
                            ].float()
                        extracted[name] = (
                            ids,
                            (attention / len(engine._alignment_heads)).cpu().numpy(),
                            mel,
                            duration,
                        )
                    captured.clear()
                    del encoded
            for row in chunk:
                result = {
                    "available": False,
                    "spans": {a: None for a in ["legacy", "cue", "lexical", "unicode"]},
                    "groups": {},
                    "lexical_fallback": False,
                }
                window = row["window"]
                name = row["key"]
                if name in extracted:
                    ids, attention, mel, duration = extracted[name]
                    lexical, unicode_groups, fallback = groups_for(
                        engine.tokenizer, ids
                    )
                    start = window["start"]
                    logp = token_logp_from_attention(attention, sharpen=5.0)
                    blank = blank_logp_from_mel_energy(
                        mel,
                        target_frames=attention.shape[1],
                        gamma=3.0,
                        penalty=3.0,
                    )
                    groups = {
                        "legacy": [list(range(len(ids)))],
                        "cue": [list(range(len(ids)))],
                        "lexical": lexical,
                        "unicode": unicode_groups,
                    }
                    result.update(
                        available=True, groups=groups, lexical_fallback=fallback
                    )
                    for arm, group in groups.items():
                        result["spans"][arm] = context_span(
                            logp,
                            blank,
                            group,
                            start,
                            duration,
                            arm == "legacy",
                        )
                    temporary = root / (name + ".npz.tmp")
                    with temporary.open("wb") as f:
                        np.savez_compressed(
                            f,
                            attention=attention,
                            blank=blank,
                            origin=start,
                            duration=duration,
                        )
                    os.replace(temporary, root / (name + ".npz"))
                cache.save(root / (name + ".json"), result)


OFFSETS = np.arange(-600, 601, 10)
CHANNELS = 240


def interpolate(values, position):
    low = np.floor(position).astype(int)
    fraction = position - low
    return (
        values[np.clip(low, 0, len(values) - 1)] * (1 - fraction[..., None])
        + values[np.clip(low + 1, 0, len(values) - 1)] * fraction[..., None]
    )


class Extractor:
    def __init__(self, path):
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        gpu.configure()
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                path, dtype=torch.float16, attn_implementation="sdpa"
            )
            .cuda()
            .eval()
            .requires_grad_(False)
        )
        self.tokenizer = self.processor.tokenizer
        config = self.model.generation_config
        self.prefix = self.tokenizer.encode(
            "".join(f"[verbatim_{i}]" for i in range(1, 6)), add_special_tokens=False
        )
        self.suffix = [
            self.model.config.decoder_start_token_id,
            config.lang_to_id["<|ja|>"],
            config.task_to_id["transcribe"],
            config.no_timestamps_token_id,
        ]
        self.heads = [tuple(pair) for pair in config.alignment_heads]
        self.projection = weights()["cfa-refiner-projection"].numpy()
        self.matrix = torch.tensor(self.projection, device="cuda")
        self.captured = {}
        self.rows = []
        self.handles = []
        self.projected = {}
        for layer in sorted({layer for layer, _ in self.heads}):
            module = self.model.model.decoder.layers[layer].encoder_attn
            for name in ["q", "k"]:
                def projected(_, args, output, key=(layer, name)):
                    self.projected[key] = output

                self.handles.append(
                    getattr(module, name + "_proj").register_forward_hook(projected)
                )
            self.handles.append(module.register_forward_hook(self.capture(layer)))
        self.batch_size = gpu.batch_size(4, weights=0, workspace=gpu.GIB)
        encoder = self.model.model.encoder
        self.encoder = lambda x: encoder(x, return_dict=False)[0]
        if gpu.available() >= gpu.GIB:
            self.encoder = gpu.Graph(
                self.encoder,
                torch.cuda.graph_pool_handle(),
                torch.cuda.Stream(),
            )

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.captured.clear()
        self.projected.clear()

    def capture(self, layer):
        def hook(module, args, output):
            batch = len(self.rows)
            q = self.projected.pop((layer, "q")) * module.scaling
            q = q.reshape(batch, -1, module.num_heads, module.head_dim).transpose(1, 2)
            k = self.projected.pop((layer, "k"))
            k = k.reshape(batch, -1, module.num_heads, module.head_dim).transpose(1, 2)
            selected = [head for candidate, head in self.heads if candidate == layer]
            for i, (start, count) in enumerate(self.rows):
                attention = (
                    (
                        q[i : i + 1, selected, start : start + count]
                        @ k[i : i + 1, selected].transpose(-1, -2)
                    )
                    .float()
                    .softmax(-1)
                    .to(q.dtype)
                )
                for index, head in enumerate(selected):
                    self.captured[i, layer, head] = attention[0, index]

        return hook

    @torch.inference_mode()
    def extract_many(self, requests):
        prepared = []
        result = [None] * len(requests)
        for i, (pcm, text, anchor, origin) in enumerate(requests):
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            supplied = (
                self.prefix
                + self.tokenizer.encode(
                    " <vtx> " + text.strip() + " <evtx>", add_special_tokens=False
                )
                + self.suffix
            )
            if tokens and len(supplied) + len(tokens) <= 440:
                prepared.append((i, pcm, anchor, origin, supplied, tokens))
        if not prepared:
            return result
        features = self.processor.feature_extractor(
            [r[1] for r in prepared], sampling_rate=16000, return_tensors="pt"
        ).input_features.to("cuda", dtype=torch.float16)

        encoder = (
            self.encoder(features)
            if len(prepared) == self.batch_size
            else self.model.model.encoder(features, return_dict=False)[0]
        )
        maximum = max(len(r[4]) + len(r[5]) for r in prepared)
        ids = torch.full(
            (len(prepared), maximum),
            self.tokenizer.eos_token_id,
            device="cuda",
            dtype=torch.long,
        )
        mask = torch.zeros_like(ids)
        self.rows = []
        for i, (_, _, _, _, supplied, tokens) in enumerate(prepared):
            count = len(supplied) + len(tokens)
            ids[i, :count] = torch.tensor(supplied + tokens, device="cuda")
            mask[i, :count] = 1
            self.rows.append((len(supplied) - 1, len(tokens)))
        self.captured.clear()
        output = self.model.model.decoder(
            input_ids=ids,
            attention_mask=mask if len(prepared) > 1 else None,
            encoder_hidden_states=encoder,
            use_cache=False,
            return_dict=True,
        )
        for i, (index, pcm, anchor, origin, _, _) in enumerate(prepared):
            mel = features[i].float().cpu().numpy().T
            frames = min(1500, math.ceil(len(pcm) / 320))
            hidden = encoder[i, :frames].float()
            hidden = hidden / hidden.square().mean(-1, keepdim=True).add(1e-6).sqrt()
            semantic = (hidden @ self.matrix).cpu().numpy()
            attention = (
                torch.stack(
                    [self.captured[(i, *pair)][:, :frames] for pair in self.heads]
                )
                .cpu()
                .numpy()
            )
            start, count = self.rows[i]
            context = output.last_hidden_state[i, start : start + count].float()
            query = torch.stack(
                [
                    context[: min(3, count)].mean(0),
                    context[-3:].mean(0),
                    context.mean(0),
                ]
            )
            query = query / query.square().mean(-1, keepdim=True).add(1e-6).sqrt()
            query = (query @ self.matrix).cpu().numpy().astype(np.float16)
            summaries = (
                np.stack([attention[:, 0], attention[:, -1], attention.mean(1)], axis=1)
                .reshape(30, frames)
                .T
            )
            position = np.asarray(anchor)[:, None] + OFFSETS - origin
            block = np.zeros((2, CHANNELS, 121), np.float32)
            block[:, :128] = interpolate(semantic, position / 20).transpose(0, 2, 1)
            block[:, 128:208] = interpolate(mel, position / 10).transpose(0, 2, 1)
            block[:, 208:238] = (
                np.log(
                    np.maximum(interpolate(summaries, position / 20), 1e-8)
                ).transpose(0, 2, 1)
                / 8
            )
            block[:, 238] = (position >= 0) & (position <= len(pcm) / 16)
            block[:, 239] = OFFSETS / 600
            result[index] = {
                "features": block.astype(np.float16),
                "query": query,
                "available": True,
            }
        self.captured.clear()
        return result


class Pointer(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = nn.Sequential(
            nn.Conv1d(240, 128, 5, padding=2), nn.GroupNorm(8, 128), nn.GELU()
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(128, 128, 5, padding=2 * d, dilation=d, groups=128),
                    nn.Conv1d(128, 128, 1),
                    nn.GroupNorm(8, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                )
                for d in (1, 2, 4)
            ]
        )
        self.roles = nn.Embedding(2, 128)
        self.query = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU())
        self.key = nn.Conv1d(128, 128, 1)
        self.score = nn.Sequential(
            nn.Conv1d(256, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1)
        )
        self.outside = nn.Linear(256, 1)
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        nn.init.zeros_(self.outside.weight)
        nn.init.zeros_(self.outside.bias)
        grid = torch.tensor(GRID)
        tight = torch.exp(-0.5 * (grid / 20) ** 2)
        tight /= tight.sum()
        broad = torch.exp(-0.5 * (grid / 120) ** 2)
        broad /= broad.sum()
        self.register_buffer(
            "prior", torch.cat((0.8 * tight + 0.1 * broad, torch.tensor([0.1]))).log()
        )

    def forward(self, features, queries):
        n = len(features)
        x = (
            self.audio(features.flatten(0, 1))
            + self.roles(torch.arange(n * 2, device=features.device) % 2)[:, :, None]
        )
        for block in self.blocks:
            x = x + block(x)
        text = self.query(queries[:, None].expand(-1, 2, -1, -1).flatten(0, 1))
        weight = torch.einsum("bct,bkc->btk", self.key(x), text) / math.sqrt(128)
        attended = torch.einsum("btk,bkc->bct", weight.softmax(-1), text)
        joined = torch.cat((x, attended), dim=1)
        density = self.score(joined)[:, 0, 30:91]
        outside = self.outside(joined.mean(-1))
        return (torch.cat((density, outside), dim=-1).float() + self.prior).reshape(
            n, 2, 62
        )


def requests(audio, texts, baseline):
    result = []
    for text, anchor in zip(texts, baseline):
        result.append(
            {
                "kind": "cfa-pointer",
                "view": "raw",
                "text": alignment_text(text),
                "baseline": np.asarray(anchor).tolist(),
                "start": max(0, math.floor(anchor[0] - 1200)),
                "end": min(audio.duration, math.ceil(anchor[1] + 1200)),
            }
        )
    return result


def predict(audio, requests, store, head_names, progress=lambda *_: None):
    unique = {str(store.path(r)): r for r in requests}
    missing = [
        r for r in unique.values() if not store.path(r).with_suffix(".json").is_file()
    ]
    if missing:
        progress("refine")
        extractor = Extractor(cfa_path())
        try:
            missing.sort(key=lambda r: len(extractor.tokenizer.encode(r["text"])))
            with sf.SoundFile(audio.raw) as stream:
                for left in range(0, len(missing), extractor.batch_size):
                    batch = missing[left : left + extractor.batch_size]
                    usable = [r for r in batch if 200 <= r["end"] - r["start"] <= 16000]
                    signals = []
                    for r in usable:
                        stream.seek(round(r["start"] * 16))
                        pcm = stream.read(
                            round(r["end"] * 16) - round(r["start"] * 16),
                            dtype="float32",
                        )
                        signals.append((pcm, r["text"], r["baseline"], r["start"]))
                    values = iter(extractor.extract_many(signals))
                    for r in batch:
                        value = (
                            next(values)
                            if 200 <= r["end"] - r["start"] <= 16000
                            else None
                        )
                        store.put(
                            r,
                            {"available": value is not None},
                            {}
                            if value is None
                            else {k: value[k] for k in ("features", "query")},
                        )
        finally:
            extractor.close()
            del extractor
            gc.collect()
            torch.cuda.empty_cache()
    heads = [Pointer().cuda().eval() for _ in head_names]
    for head, name in zip(heads, head_names):
        head.load_state_dict(weights()[name])
    del head
    logits = np.zeros((len(requests), 2, 62), np.float32)
    available = np.zeros(len(requests), bool)
    with torch.inference_mode():
        for left in range(0, len(requests), 32):
            values = [store.get(r) for r in requests[left : left + 32]]
            x = np.zeros((len(values), 2, CHANNELS, 121), np.float32)
            q = np.zeros((len(values), 3, 128), np.float32)
            for i, (meta, arrays) in enumerate(values):
                if meta["available"]:
                    x[i], q[i] = arrays["features"], arrays["query"]
                    available[left + i] = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                features = torch.tensor(x, device="cuda")
                queries = torch.tensor(q, device="cuda")
                predictions = [head(features, queries).cpu().numpy() for head in heads]
            if len(heads) == 1:
                logits[left : left + len(values)] = predictions[0]
            else:
                densities = [
                    log_softmax(p.astype(np.float64), axis=-1) for p in predictions
                ]
                logits[left : left + len(values)] = np.logaddexp.reduce(
                    densities, axis=0
                ) - math.log(len(heads))
    del heads
    torch.cuda.empty_cache()
    return logits, available


def align(audio, texts, baseline, store, progress=lambda *_: None):
    config = MANIFEST["cfa"]
    logits, available = predict(
        audio, requests(audio, texts, baseline), store, config["heads"], progress
    )
    valid = available[:, None] & (
        softmax(logits, axis=-1)[:, :, -1] <= config["maximum_outside_probability"]
    )
    return proposal(
        statistics(logits[:, :, :61], baseline, valid, baseline, audio.duration),
        baseline,
    )


if __name__ == "__main__":
    infer(json.loads(Path(sys.argv[1]).read_text()))
