# encode audio, route the transcript with CTC and decode the timing of each line.
#
# the encoder leaves the device before the decoder arrives; features live only for the call.

import gc
import hashlib
import itertools
import math
import os
import re
import shutil
import tarfile
import tempfile
import unicodedata
import weakref
from contextlib import contextmanager
from functools import cache
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import sentencepiece
import torch
import triton
import triton.language as tl
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 16_000
MODEL_DIR = Path(os.environ.get(
    "SUBTIMER_MODEL_DIR",
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "subtimer/models",
)).expanduser().resolve()
MODELS = Path(__file__).with_name("models")
CHECKPOINT = MODELS / "timing.pt"
PROJECTION = MODELS / "projection.pt"
TOKENIZER = MODEL_DIR / "tokenizer.model"

BACKBONE_REPO = "nyralabs/CrisperWhisper2.0_large"
BACKBONE_REVISION = "f4334f6e8193f2691212d49b20fa12d370e13896"
BACKBONE_SHA256 = "f097f853b6992e66fcae802a15770a8242670351198d68dad577db3209fe3fcc"
TOKENIZER_REPO = "nvidia/parakeet-tdt_ctc-0.6b-ja"
TOKENIZER_REVISION = "44edb27eea9317daf89333e75eb830db4b1cc298"
TOKENIZER_MEMBER = "./06c3b8cfecdb48c6a1b3595dfe0a321d_tokenizer.model"
TOKENIZER_SHA256 = "09837fc906acd1cebf1f2daf072728b901173c55773fbd33a860faf7a0ca149d"

NEGATIVE = -3.4e38
NON_WORD = re.compile(r"[^A-Za-z0-9\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# download missing pinned assets into the caller's model cache
@cache
def ensure_models():
    from huggingface_hub import hf_hub_download, snapshot_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    backbone = MODEL_DIR / "backbone"
    marker = backbone / "revision"
    if not marker.is_file() or marker.read_text() != BACKBONE_REVISION:
        snapshot_download(
            BACKBONE_REPO,
            revision=BACKBONE_REVISION,
            local_dir=backbone,
            allow_patterns=["model.safetensors", "*.json", "merges.txt", "LICENSE.md"],
        )
        if sha256(backbone / "model.safetensors") != BACKBONE_SHA256:
            raise ValueError("public backbone checksum mismatch")
        marker.write_text(BACKBONE_REVISION)

    if TOKENIZER.is_file():
        return
    archive = hf_hub_download(
        TOKENIZER_REPO,
        filename="parakeet-tdt_ctc-0.6b-ja.nemo",
        revision=TOKENIZER_REVISION,
    )
    with tempfile.TemporaryDirectory(dir=MODEL_DIR) as temporary:
        target = Path(temporary) / "tokenizer.model"
        with tarfile.open(archive) as handle:
            member = handle.extractfile(TOKENIZER_MEMBER)
            with member, target.open("wb") as output:
                shutil.copyfileobj(member, output)
        if sha256(target) != TOKENIZER_SHA256:
            raise ValueError("public tokenizer checksum mismatch")
        target.replace(TOKENIZER)


def deterministic():
    torch.set_num_threads(3)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# collect and return cached device memory when a stage ends, even on error
@contextmanager
def releasing_cuda():
    try:
        yield
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# each run of equal neighbouring values with its [first, end) bounds
def runs(values):
    position = 0
    for value, run in itertools.groupby(values):
        size = sum(1 for _ in run)
        yield value, position, position + size
        position += size


# interleave 20 ms samples and their midpoints on the 10 ms grid
def upsample(x):
    midpoint = (x[..., :-1] + x[..., 1:]) * .5
    odd = torch.cat((midpoint, x[..., -1:]), -1)
    return torch.stack((x, odd), -1).flatten(-2)


# fuse the frozen 2 × 50 ms + 200 ms masses without intermediate grids
@triton.jit
def window_hits(prefix, output, size: tl.constexpr, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    width: tl.constexpr = size + 40
    valid = index < width * width
    start, end = index // width - 20, index % width - 20
    total = tl.full((block,), 0, tl.float64)
    for radius in tl.static_range(2):
        r = 5 if radius == 0 else 20
        a, b = tl.minimum(tl.maximum(start - r, 0), size), tl.minimum(tl.maximum(start + r + 1, 0), size)
        c, d = tl.minimum(tl.maximum(end - r, 0), size), tl.minimum(tl.maximum(end + r + 1, 0), size)
        bd = tl.load(prefix + b * (size + 1) + d, valid, other=0)
        ad = tl.load(prefix + a * (size + 1) + d, valid, other=0)
        bc = tl.load(prefix + b * (size + 1) + c, valid, other=0)
        ac = tl.load(prefix + a * (size + 1) + c, valid, other=0)
        mass = tl.maximum(bd - ad - bc + ac, 0)
        total += mass * (2 if radius == 0 else 1)
    tl.store(output + index, total, valid)


# accumulate component utility in order, reusing the preceding anchor's overlap
@triton.jit(do_not_specialize=["origin", "duration", "overlap"])
def utility_grid(hits, weight, start_error, end_error, previous, output,
                 size: tl.constexpr, origins, origin,
                 duration, overlap, block: tl.constexpr):
    index = tl.program_id(0) * block + tl.arange(0, block)
    row, column = index // size, index % size
    valid = index < size * size
    current = valid & (column >= overlap)
    value = tl.full((block,), 0, tl.float64)
    for component in tl.static_range(len(hits)):
        shift = (origin - tl.load(origins + component)) // 10
        r, c = row + shift + 20, column + shift + 20
        inside = current & (r >= 0) & (r < size + 40) & (c >= 0) & (c < size + 40)
        mass = tl.load(hits[component] + r * (size + 40) + c, inside, other=0)
        value += tl.load(weight + component) * mass
    a = tl.load(start_error + row, current, other=0)
    b = tl.load(end_error + column - overlap, current, other=0)
    value -= (a + b) * 0.0005
    legal = (column - row >= 2) & (row * 10 + origin >= 0) & (column * 10 + origin <= duration)
    value = tl.where(current & legal, value, -float('inf'))
    copied = valid & (row < overlap) & (column < overlap)
    old = tl.load(previous + (row + size - overlap) * size + column + size - overlap, copied, other=0)
    value = tl.where(copied, old, value)
    tl.store(output + index, value, valid)


# bounded Huber penalty: 5% of residuals are outliers
def penalty(residual):
    absolute = np.abs(residual)
    huber = np.where(absolute < 1, .5 * residual**2, absolute - .5)
    return -np.logaddexp(np.log1p(-.05) - huber, np.log(.05))


# advance the exact CTC Viterbi recurrence over one temporal tile
@triton.jit
def advance(
    emissions,
    ids,
    before,
    after,
    forbidden,
    back,
    states,
    vocabulary: tl.constexpr,
    first,
    stop,
    origin,
    steps: tl.constexpr,
    width: tl.constexpr,
    negative: tl.constexpr,
):
    lane = tl.arange(0, width)
    # a state advances at most twice per frame. this halo contains every
    # dependency of the states we write, throughout the whole temporal tile.
    index = tl.program_id(0) * (width - 2 * steps) + lane - 2 * steps
    valid = (index >= 0) & (index < states)
    write = valid & (lane >= 2 * steps)
    previous = tl.load(before + index, valid, other=negative)
    skip = tl.load(forbidden + index, valid, other=1)
    token = tl.load(ids + index, valid, other=0)
    for frame in range(first, stop):
        score = tl.load(emissions + frame.to(tl.int64) * vocabulary + token, valid, other=0)
        one = tl.gather(previous, tl.maximum(lane - 1, 0), 0)
        two = tl.gather(previous, tl.maximum(lane - 2, 0), 0)
        one = tl.where((lane >= 1) & (index >= 1), one, negative)
        two = tl.where((lane >= 2) & (index >= 2) & (~skip), two, negative)
        # strict comparisons preserve torch.max's stay, +1, +2 tie order
        relative = tl.where(one > previous, 1, 0)
        best = tl.maximum(previous, one)
        relative = tl.where(two > best, 2, relative)
        best = tl.maximum(best, two)
        previous = best + score
        # long recordings can exceed 2**31 traceback entries within one block
        tl.store(back + (frame - origin).to(tl.int64) * states + index, relative, write)
    tl.store(after + index, previous, write)


# CTC token emissions from the shared encoder states
class AcousticHead(nn.Module):

    def __init__(self, state):
        super().__init__()
        self.norm = nn.LayerNorm(1280)
        self.projection = nn.Sequential(nn.Linear(1280, 384), nn.GELU())
        self.temporal = nn.ModuleList([nn.Conv1d(384, 384, 3, padding=d, dilation=d, groups=384) for d in (1, 4, 16)])
        self.output = nn.Linear(384, state["ctc_weight"].shape[1])
        self.register_buffer("ctc_weight", state["ctc_weight"].float().clone())
        self.register_buffer("ctc_bias", state["ctc_bias"].float().clone())
        self.load_state_dict(state)

    def forward(self, hidden):
        x = self.projection(self.norm(hidden))
        for layer in self.temporal:
            x = x + F.gelu(layer(x.transpose(1, 2)).transpose(1, 2))
        x = self.output(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return F.linear(x.float(), self.ctc_weight, self.ctc_bias).log_softmax(-1)


# the joint start and end distribution of one line within one audio chunk
class Pointer(nn.Module):

    def __init__(self, state):
        super().__init__()
        self.audio = nn.Sequential(nn.Conv1d(240, 128, 5, dilation=2, padding=4), nn.GroupNorm(8, 128), nn.GELU())
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 128, 5, padding=4 * d, dilation=2 * d, groups=128),
                nn.Conv1d(128, 128, 1), nn.GroupNorm(8, 128), nn.GELU(), nn.Dropout(.1),
            ) for d in (1, 2, 4)
        ])
        self.roles = nn.Embedding(2, 128)
        self.query = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128), nn.GELU())
        self.key = nn.Conv1d(128, 128, 1)
        self.score = nn.Sequential(nn.Conv1d(256, 64, 1), nn.GELU(), nn.Conv1d(64, 1, 1))
        self.span_projection = nn.Conv1d(256, 32, 1)
        self.coupling = nn.Parameter(torch.zeros(32, 32))
        self.null = nn.Linear(512, 1)
        self.duration = nn.Linear(128, 2)
        self.duration_scale = nn.Parameter(torch.tensor(-4.))
        self.projection_delta = nn.Parameter(torch.zeros(1280, 128))
        self.prior_gate = nn.Linear(1, 1)
        self.load_state_dict(state)
        self.geometry = None

    def forward(self, features, query, valid):
        batch, channels, frames = features.shape
        x = self.audio(features[:, None].expand(-1, 2, -1, -1).reshape(batch * 2, channels, frames))
        x = x + self.roles(torch.arange(batch * 2, device=x.device) % 2)[:, :, None]
        for block in self.blocks:
            x = x + block(x)
        text = self.query(query[:, None].expand(-1, 2, -1, -1).reshape(batch * 2, query.shape[1], 128))
        attention = torch.einsum("bct,bkc->btk", self.key(x), text).float() / math.sqrt(128)
        attended = torch.einsum("btk,bkc->bct", attention.softmax(-1).to(text.dtype), text)
        joined = torch.cat((x, attended), 1)
        density = self.score(joined)[:, 0].float().reshape(batch, 2, frames)
        embedding = self.span_projection(joined).float().reshape(batch, 2, 32, frames)
        mean, spread = self.duration(query[:, 2]).float().unbind(-1)
        spread = F.softplus(spread) + .2
        joined = joined.reshape(batch, 2, 256, frames).mean(1)
        pooled = torch.cat((joined.mean(-1), joined.amax(-1)), -1)
        null = self.null(pooled).float()[:, 0] + 2 * math.log(frames)
        if self.geometry is None:
            indices = torch.arange(frames, device=density.device)
            grid = indices * 10
            log_duration = (grid[None] - grid[:, None]).clamp_min(10).float().log()
            self.geometry = indices, log_duration, grid[None] > grid[:, None]
        indices, log_duration, forward = self.geometry
        correlation = torch.einsum("bci,cd,bdj->bij", embedding[:, 0], self.coupling.float(), embedding[:, 1]) / math.sqrt(32)
        score = density[:, 0, :, None] + density[:, 1, None, :] + correlation
        prior = -.5 * ((log_duration[None] - mean[:, None, None]) / spread[:, None, None]).square()
        score = score + F.softplus(self.duration_scale) * prior
        legal = forward[None] & (indices[None, :, None] < valid[:, None, None])
        legal = legal & (indices[None, None, :] < valid[:, None, None])
        return score.masked_fill(~legal, -1e9), null


# cross-attention over the cached keys and values of each row's audio chunk. alignment-head layers also
# keep, for each row's line, its first, last and mean attention, biased toward the CTC route as far as the
# attention itself supports it.
class CrossAttention(nn.Module):

    def __init__(self, base, layer, model):
        super().__init__()
        self.base, self.layer = base, layer
        self.heads = [head for index, head in model.heads if index == layer]
        self.owner = weakref.ref(model)

    def forward(self, hidden_states, **_):
        base, model = self.base, self.owner()
        batch, length = hidden_states.shape[:-1]
        q = (base.q_proj(hidden_states) * base.scaling).view(batch, length, base.num_heads, base.head_dim).transpose(1, 2)
        k = torch.stack([model.cache[chunk][self.layer][0] for chunk in model.chunks])
        v = torch.stack([model.cache[chunk][self.layer][1] for chunk in model.chunks])
        output = F.scaled_dot_product_attention(q, k, v, scale=1.0).transpose(1, 2).reshape(batch, length, -1)
        if self.heads:
            cue_q = q[:, self.heads].gather(2, model.cue_positions[:, None, :, None].expand(-1, len(self.heads), -1, base.head_dim))
            logits = (cue_q @ k[:, self.heads].transpose(-1, -2)).float()
            prior, line = model.active_prior, model.line
            usable = torch.isfinite(prior).all(1)
            location = torch.nan_to_num(prior)
            times = torch.arange(logits.shape[-1], device=prior.device) * 20
            distance = (location[:, 0, None] - 500. - times).clamp_min(0) + (times - location[:, 1, None] - 500.).clamp_min(0)
            bias = -(distance / 500.).clamp_max(4) * usable[:, None]
            with torch.autocast(device_type=logits.device.type, enabled=False):
                nearby = (times[None] >= prior[:, 0, None] - 500) & (times[None] <= prior[:, 1, None] + 500)
                probability = (logits.softmax(-1) * line[:, None, :, None]).sum((1, 2)) / (len(self.heads) * line.sum(1))[:, None]
                mass = (probability * nearby).sum(-1)
                uniform = nearby.float().mean(-1)
                evidence = ((mass + 1e-4) / (uniform + 1e-4)).log().clamp(-8, 8)[:, None]
                strength = (1 + model.pointer.prior_gate(evidence).squeeze(-1)).clamp(0, 1)
            attention = (logits + (bias * strength[:, None])[:, None, None]).softmax(-1)
            rows = torch.arange(batch, device=attention.device)
            model.summaries[self.layer] = torch.stack((attention[:, :, 0], attention[rows, :, model.counts - 1],
                                                       (attention * line[:, None, :, None]).sum(2) / line.sum(1)[:, None, None]), 2)
        return base.out_proj(output), None


# the frozen transcript decoder with its merged adapters, its alignment heads and the pointer
class Model(nn.Module):

    def __init__(self, processor, full, checkpoint):
        super().__init__()
        self.processor = processor
        self.decoder = full.model.decoder.cuda()
        self.heads = [tuple(pair) for pair in full.generation_config.alignment_heads]
        self.ctc_offset = checkpoint["model_config"]["ctc_offset_ms"]
        parameters = iter(checkpoint["lora"])
        for index, layer in enumerate(self.decoder.layers):
            for attn, names in ((layer.self_attn, ("q_proj", "v_proj")), (layer.encoder_attn, ("q_proj", "k_proj", "v_proj", "out_proj"))):
                for name in names:
                    # the low-rank update is folded into the frozen bf16 projection
                    linear, a, b = getattr(attn, name), next(parameters), next(parameters)
                    linear.weight.data = (linear.weight.float() + b.float().cuda() @ a.float().cuda()).to(linear.weight.dtype)
            layer.encoder_attn = CrossAttention(layer.encoder_attn, index, self)
        # where each alignment head's summary sits among its layer's heads
        self.summary_index = [(layer, self.decoder.layers[layer].encoder_attn.heads.index(head))
                              for layer, head in self.heads]
        self.pointer = Pointer(checkpoint["pointer"]).cuda().float()
        projection = torch.load(PROJECTION, map_location="cpu", weights_only=True)["projection"]
        self.register_buffer("matrix", projection.float().cuda())
        # each lru entry owns its projections, never a view of other chunks
        free, _ = torch.cuda.mem_get_info()
        self.batch_size, self.cache_capacity = (16, 8) if free >= 16 * 1024**3 else (8, 4)
        self.cache, self.summaries = {}, {}
        tokenizer = self.processor.tokenizer
        self.prefix = tokenizer.encode("".join(f"[verbatim_{i}]" for i in range(1, 6)), add_special_tokens=False)
        self.suffix = [full.config.decoder_start_token_id, full.generation_config.lang_to_id["<|ja|>"],
                       full.generation_config.task_to_id["transcribe"], full.generation_config.no_timestamps_token_id]

    def prompt(self, text):
        return self.prefix + self.processor.tokenizer.encode(" <vtx> " + text.strip() + " <evtx>", add_special_tokens=False) + self.suffix

    # score bounded cue/chunk batches with an audio K/V LRU
    def forward(self, hidden, mel, chunks, token_rows, valid, prior):
        active = set(chunks)
        for chunk in chunks:
            if chunk in self.cache:
                self.cache[chunk] = self.cache.pop(chunk)
                continue
            if len(self.cache) >= self.cache_capacity:
                victim = next(key for key in self.cache if key not in active)
                del self.cache[victim]
            states = torch.as_tensor(np.array(hidden[chunk], copy=True), device="cuda", dtype=torch.bfloat16)
            self.cache[chunk] = [
                [projection(states).view(-1, layer.encoder_attn.base.num_heads,
                                         layer.encoder_attn.base.head_dim).transpose(0, 1)
                 for projection in (layer.encoder_attn.base.k_proj, layer.encoder_attn.base.v_proj)]
                for layer in self.decoder.layers
            ]
        hidden = torch.as_tensor(hidden[chunks], device="cuda")
        mel = torch.as_tensor(mel[chunks], device="cuda")
        self.chunks, self.active_prior = chunks, prior
        maximum = max(len(row[0]) for row in token_rows)
        packed = np.zeros((2, len(token_rows), maximum), dtype=np.int64)
        packed[0].fill(self.processor.tokenizer.eos_token_id)
        for i, (tokens, _, _) in enumerate(token_rows):
            packed[0, i, :len(tokens)] = tokens
            packed[1, i, :len(tokens)] = 1
        ids, mask = torch.as_tensor(packed, device="cuda").unbind(0)
        firsts, self.counts = torch.tensor([(first, count) for _, first, count in token_rows], device="cuda").unbind(1)
        positions = torch.arange(max(count for _, _, count in token_rows), device="cuda")
        self.line = (positions < self.counts[:, None]).float()
        # padding repeats a legal cue position, then contributes zero to both means
        self.cue_positions = firsts[:, None] + positions[None].minimum(self.counts[:, None] - 1)
        # the decoder's weights are bf16 already; autocast would only add fp32 casts around its norms
        with torch.autocast("cuda", enabled=False):
            out = self.decoder(input_ids=ids, attention_mask=mask, encoder_hidden_states=hidden.bfloat16(),
                               use_cache=False, output_attentions=False, return_dict=True)
        normalized = hidden.float() / (hidden.float().square().mean(-1, keepdim=True) + 1e-6).sqrt()
        matrix = self.matrix + self.pointer.projection_delta
        semantic = normalized @ matrix
        semantic = upsample(semantic.transpose(1, 2))
        valid = valid * 2
        frames = semantic.shape[-1]
        # each alignment head's first, last and mean attention, in the order of the heads
        summaries = torch.stack([self.summaries[layer][:, index] for layer, index in self.summary_index], 1)
        summaries = upsample(summaries.reshape(len(token_rows), 3 * len(self.heads), hidden.shape[1]))
        queries = []
        for i, (_, first, count) in enumerate(token_rows):
            context = out.last_hidden_state[i, first:first + count].float()
            query = torch.stack([context[:min(3, count)].mean(0), context[-3:].mean(0), context.mean(0)])
            query = query / (query.square().mean(-1, keepdim=True) + 1e-6).sqrt()
            queries.append(query @ matrix)
        batch = len(token_rows)
        feature = torch.cat((semantic, mel.transpose(1, 2), summaries.clamp_min(1e-8).log() / 8,
                             (torch.arange(frames, device="cuda") < valid[:, None]).float()[:, None],
                             torch.linspace(-1, 1, frames, device="cuda").expand(batch, 1, frames)), 1)
        return self.pointer(feature, torch.stack(queries), valid)


# return integer millisecond pairs for precleaned nonempty model text.
#
# audio must be mono 16 kHz. the caller owns all transcript cleanup and original-row
# mapping. report the match and refine stages as they begin. nothing outlives the call. a backbone from
# load_backbone() may be passed in; the call consumes it, so a caller that keeps one forks before passing it.
def align(texts, audio_path, progress, backbone=None):
    # matching encodes the audio, then routes every line through it
    progress("match")
    ensure_models()
    processor, full = backbone or load_backbone()
    return align_features(texts, encode(processor, full, audio_path), processor, full, progress)


# the processor and the whole model, read once in bf16 on the CPU; the encoder runs in fp16 and the decoder in
# bf16. it touches neither CUDA nor thread pools, so a process may load it once and fork a job from it.
def load_backbone():
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    path = MODEL_DIR / "backbone"
    full = AutoModelForSpeechSeq2Seq.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").eval()
    full.requires_grad_(False)
    return AutoProcessor.from_pretrained(path), full


# encoder states and mel features of 30 s chunks every 15 s; the encoder leaves the device afterwards
def encode(processor, full, audio_path):
    import soundfile

    deterministic()
    pcm, rate = soundfile.read(audio_path, dtype="float32")
    if rate != SAMPLE_RATE or pcm.ndim != 1:
        raise ValueError("audio must be mono 16 kHz")
    # 16 samples per millisecond
    origins = np.arange(0, max(1, len(pcm) / 16 - 15000), 15000, dtype=np.int64)
    with releasing_cuda(), torch.inference_mode():
        encoder = full.model.encoder.half().cuda()
        # unlinked files back the features, so a long recording's pages can leave memory under pressure
        hidden = np.memmap(tempfile.TemporaryFile(), np.float16, "w+", shape=(len(origins), 1500, full.config.d_model))
        mels = np.memmap(tempfile.TemporaryFile(), np.float16, "w+", shape=(len(origins), 3000, full.config.num_mel_bins))
        for start in range(0, len(origins), 4):
            starts = origins[start:start + 4]
            mel = processor.feature_extractor([pcm[int(a * 16):int((a + 30000) * 16)] for a in starts],
                                              sampling_rate=16000, return_tensors="pt").input_features.cuda().half()
            hidden[start:start + len(starts)] = encoder(mel, return_dict=False)[0].cpu().numpy()
            mels[start:start + len(starts)] = mel.transpose(1, 2).cpu().numpy()
        # dropping the module frees its device weights; nothing reads the encoder after this
        full.model.encoder = None
        del encoder, mel
    return hidden, mels, origins, len(pcm) / 16


# align with encoded features; report the refine stage as it begins
def align_features(texts, features, processor, full, progress):
    deterministic()
    hidden, mel, origins, duration = features
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)

    # route every line globally with exact ctc on emissions blended on the absolute 80 ms clock.
    # the viterbi buffers below stay referenced through refinement, whose batch size depends on the
    # device memory left.
    with releasing_cuda(), torch.inference_mode():
        head = AcousticHead(checkpoint["shared_audio_ctc"]["state"]).cuda().eval()
        count = math.ceil(duration / 80)
        total = torch.zeros((count, len(head.ctc_bias)), device="cuda", dtype=torch.float32)
        weights = torch.zeros(count, device="cuda", dtype=torch.float32)
        for first in range(0, len(origins), 8):
            chunk_ids = list(range(first, min(first + 8, len(origins))))
            selections = []
            for i in chunk_ids:
                origin = int(origins[i])
                start = math.ceil(origin / 80)
                end = math.ceil(min(origin + 1500 * 20, duration) / 80)
                global_ids = np.arange(start, end, dtype=np.int64)
                local_ids = (global_ids * 80 - origin) // 20
                legal = (local_ids >= 0) & (local_ids < 1500)
                selections.append((local_ids[legal], global_ids[legal]))
            length = max(len(local) for local, _ in selections)
            x = np.zeros((len(chunk_ids), length, hidden.shape[-1]), dtype=np.float32)
            for j, (i, (local, _)) in enumerate(zip(chunk_ids, selections)):
                x[j, :len(local)] = hidden[i, local]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                probability = head(torch.as_tensor(x, device="cuda")).exp()
            # one index_add_ per chunk: indices repeat across chunks, so this fixes the summation order
            for j, (local, global_ids) in enumerate(selections):
                g = torch.as_tensor(global_ids, device="cuda")
                w = torch.as_tensor(np.maximum(.05, 1 - np.abs(local * 20 - 15000) / 15000), device="cuda", dtype=torch.float32)
                total.index_add_(0, g, probability[j, :len(g)] * w[:, None])
                weights.index_add_(0, g, w)
        lp = (total / weights[:, None]).clamp_min(1e-30).log()
        lp = lp - lp.logsumexp(-1, keepdim=True)
        del head, total, weights, probability

        blank = lp.shape[1] - 1
        tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(TOKENIZER))
        ids, owners, boundaries = [blank], [-1], {0}
        for cue, text in enumerate(texts):
            boundaries.add(len(ids) - 1)
            for word in NON_WORD.sub(" ", text).strip().split():
                for token, piece in zip(tokenizer.encode_as_ids(word), tokenizer.encode_as_pieces(word)):
                    ids.extend((int(token), blank))
                    owners.extend((cue if NON_WORD.sub("", unicodedata.normalize("NFKC", piece)) else -1, -1))
        boundaries.add(len(ids) - 1)
        ids = np.asarray(ids, np.int64)
        owners = np.asarray(owners, np.int64)
        labels = ids[1::2]
        required = len(labels) + int((labels[1:] == labels[:-1]).sum())
        factor = next((f for f in (1, 2, 4) if len(lp) * f >= required), None)
        if factor is None:
            raise ValueError("transcript is too long to fit even a 20 ms alignment grid")
        if factor > 1:
            # splitting a frame preserves its total evidence and scales the gap cost
            lp = lp.repeat_interleave(factor, dim=0) / factor
        frame_ms = 80 // factor
        nonblank = lp[:, :blank].amax(1)
        wildcard = torch.maximum(lp[:, blank], nonblank - .5 / factor)
        skip = np.ones(len(ids), dtype=bool)
        skip[2:] = ids[2:] == ids[:-2]
        ids[sorted(boundaries)] = lp.shape[1]
        lp = torch.cat((lp, wildcard[:, None]), 1).contiguous()

        # exact viterbi with cpu backpointers, without a frames × states device copy
        frames, vocabulary = lp.shape
        states = len(ids)
        state_ids = torch.tensor(ids, device="cuda")
        forbidden = torch.tensor(skip, device="cuda")
        forbidden[:2] = True
        previous = torch.full((states,), NEGATIVE, device=lp.device)
        previous[: min(2, states)] = lp[0, state_ids[: min(2, states)]]
        following = torch.empty_like(previous)
        back = torch.zeros((frames, states), dtype=torch.int8)
        if frames > 1:
            # a larger device keeps the entire traceback resident. smaller devices
            # transfer completed blocks; the global recurrence never resets.
            free, memory = torch.cuda.mem_get_info()
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved() - allocated
            limit = int(memory * torch.cuda.get_per_process_memory_fraction()) - allocated
            gib = 1024**3
            wsl = "microsoft" in os.uname().release.lower()
            headroom = max(2 * gib if wsl else gib, min(3 * gib, memory // 10))
            rows_per_block = min(frames - 1, max(0, min(free + cached - headroom, limit)) // (2 * states))
            if rows_per_block < 1:
                raise RuntimeError("insufficient free VRAM for one CTC traceback row")
            block = torch.empty((rows_per_block, states), dtype=torch.int8, device=lp.device)
            steps, width = 32, 512
            for origin in range(1, frames, rows_per_block):
                end = min(origin + rows_per_block, frames)
                for first in range(origin, end, steps):
                    advance[(triton.cdiv(states, width - 2 * steps),)](
                        lp, state_ids, previous, following, forbidden, block, states, vocabulary,
                        first, min(first + steps, end), origin, steps, width, NEGATIVE,
                        num_warps=4, enable_fp_fusion=False,
                    )
                    previous, following = following, previous
                back[origin:end].copy_(block[: end - origin])
        state = states - 2 + int(previous[-1] > previous[-2])
        if float(previous[state]) <= NEGATIVE / 2:
            raise RuntimeError("no exact CTC path")
        back = back.numpy()
        path = np.empty(frames, np.int64)
        path[-1] = state
        for frame in range(frames - 1, 0, -1):
            state -= int(back[frame, state])
            path[frame - 1] = state
        assignment = owners[path]
        spans = np.full((len(texts), 2), np.nan)
        for cue in range(len(texts)):
            visited = np.flatnonzero(assignment == cue)
            if len(visited):
                spans[cue] = [visited[0] * frame_ms, (visited[-1] + 1) * frame_ms]
        if not np.isfinite(spans).all():
            raise ValueError("exact CTC path did not visit every transcript line")
        base = np.clip(spans, 0, duration)

    progress("refine")

    # decode each line from its route; equal neighbouring lines are decoded together as one chain
    with releasing_cuda(), torch.no_grad():
        model = Model(processor, full, checkpoint).eval()
        anchor_offset = np.asarray(model.ctc_offset, float)
        tokenizer = model.processor.tokenizer
        keys = [NON_WORD.sub("", text) for text in texts]
        groups = [list(range(first, end)) for key, first, end in runs(keys) if key and end - first > 1]
        repeat_cues = {i for group in groups for i in group}
        # each line's run of identical neighbouring text, which shares its decoder prompt
        equal_run = [(first, end) for _, first, end in runs(texts) for _ in range(first, end)]
        repeat_candidates = {}
        pairs = []
        for cue in range(len(texts)):
            # the chunks covering this line and its neighbours' routes, widened by one chunk
            covered = set()
            for start, end in base[max(0, cue - 1):cue + 2]:
                covering = np.flatnonzero((origins <= start) & (origins + 30000 >= end))
                if len(covering):
                    covered.update(map(int, covering))
                else:
                    covered.add(int(np.abs(origins + 15000 - (start + end) / 2).argmin()))
            selected = sorted({k for index in covered for k in (index - 1, index, index + 1) if 0 <= k < len(origins)})

            # the decoder prompt is this line among its equal neighbours and one line on each side
            first_equal, end_equal = equal_run[cue]
            context = range(max(0, first_equal - 1), min(len(texts), end_equal + 1))
            token_ids, first, count = [], 0, 0
            for index in context:
                if token_ids:
                    token_ids.extend(tokenizer.encode("\n", add_special_tokens=False))
                tokens = tokenizer.encode(texts[index], add_special_tokens=False)
                if index == cue:
                    first, count = len(token_ids), len(tokens)
                token_ids.extend(tokens)
            supplied = model.prompt("\n".join(texts[i] for i in context))
            # the prompt and the scored tokens share the decoder's positions; a line whose neighbours do not fit
            # is decoded alone, and a line that does not fit alone cannot be timed
            positions = model.decoder.config.max_target_positions
            if not count or len(supplied) + len(token_ids) > positions:
                token_ids = tokenizer.encode(texts[cue], add_special_tokens=False)
                supplied = model.prompt(texts[cue])
                first, count = 0, len(token_ids)
            if len(supplied) + len(token_ids) > positions:
                raise ValueError(f"timed line {cue + 1} is too long to time")
            token_row = (supplied + token_ids, len(supplied) - 1 + first, count)

            pairs.extend((cue, chunk, token_row) for chunk in selected)

        def distributions():
            current, components = 0, []
            offset = 0
            while offset < len(pairs):
                end, active = offset, set()
                while end < min(len(pairs), offset + model.batch_size):
                    chunk = pairs[end][1]
                    if chunk not in active and len(active) == model.cache_capacity:
                        break
                    active.add(chunk)
                    end += 1
                batch, offset = pairs[offset:end], end
                chunk_ids = [chunk for _, chunk, _ in batch]
                valid = torch.tensor([min(1500, math.ceil((duration - origins[k]) / 20)) for k in chunk_ids], device="cuda")
                prior = torch.tensor(np.array([base[cue] + anchor_offset - origins[k]
                                               for cue, k, _ in batch]), device="cuda", dtype=torch.float32)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    score, null = model(hidden, mel, chunk_ids, [row for _, _, row in batch], valid, prior)
                for (cue, k, _), logits, absent in zip(batch, score, null):
                    if cue != current:
                        yield current, components
                        current, components = cue, []
                    # the posterior is formed in fp32; its sums and grids accumulate in float64
                    logits = logits.float()
                    logz = torch.logsumexp(logits.flatten(), 0)
                    components.append({"p": (logits - logz).exp().double(), "origin": int(origins[k]), "logodds": float(logz.double() - absent.double())})
            if components:
                yield current, components

        # a function, so each cue's posterior and utility grids are freed before the next cue decodes
        def select(cue, components):
            # the span with the best expected utility: hits within 50 and 200 ms, less a small mean absolute error
            weight = torch.tensor(np.array([c["logodds"] for c in components], np.float64), device="cuda").softmax(0)
            chunk_origins = torch.tensor([c["origin"] for c in components], device="cuda", dtype=torch.int32)
            for c in components:
                c["p"] = p = c["p"] / c["p"].sum()
                c["prefix"] = F.pad(p, (1, 0, 1, 0)).cumsum(0).cumsum(1)
                c["locations"] = c["origin"] + torch.arange(len(p), device="cuda", dtype=torch.float64) * 10
                for name, probability in (("start", p.sum(1)), ("end", p.sum(0))):
                    probability = probability / probability.sum()
                    cumulative = torch.cat((probability.new_zeros(1), probability.cumsum(0)))
                    moment = torch.cat((probability.new_zeros(1), (probability * c["locations"]).cumsum(0)))
                    c[name] = cumulative, moment
            # chunk origins are whole frames apart, so each chunk's window sums, computed once over its
            # frames and a margin, serve every anchor shifted; outside them a window holds no mass.
            # every chunk's grid has the same number of frames.
            size, margin = len(components[0]["p"]), 200 // 10
            for c in components:
                hits = torch.empty((size + 2 * margin, size + 2 * margin), device="cuda", dtype=torch.float64)
                window_hits[(triton.cdiv(hits.numel(), 256),)](c["prefix"], hits, size, 256, enable_fp_fusion=False)
                c["hits"] = hits
            hit_grids = tuple(c["hits"] for c in components)
            best, chosen = -np.inf, np.asarray(base[cue], float)
            # keep every action and its tie order; reuse values where successive grids overlap
            previous, previous_origin = None, 0
            # every anchor's grid lies on one 10 ms axis, so each component's expected error is computed once
            first = min(c["origin"] for c in components)
            axis = first + torch.arange((max(c["origin"] for c in components) - first) // 10 + size, device="cuda", dtype=torch.float64) * 10
            start_axis, end_axis = torch.zeros_like(axis), torch.zeros_like(axis)
            for w, c in zip(weight, components):
                # how many of this component's frames lie at or before each time on the axis
                count = (torch.div(axis - c["origin"], 10, rounding_mode="floor").long() + 1).clamp(0, size)
                for errors, (cumulative, moment) in ((start_axis, c["start"]), (end_axis, c["end"])):
                    # expected absolute error from the marginal's cumulative mass and moment
                    errors += w * (axis * (2 * cumulative[count] - 1) + moment[-1] - 2 * moment[count]).clamp_min(0)
            for anchor in components:
                grid = anchor["locations"]
                overlap = max(0, size - (anchor["origin"] - previous_origin) // 10) if previous is not None else 0
                utility = torch.empty((size, size), device="cuda", dtype=torch.float64)
                offset = (anchor["origin"] - first) // 10
                start_error, end_error = start_axis[offset:offset + size], end_axis[offset + overlap:offset + size]
                utility_grid[(triton.cdiv(utility.numel(), 256),)](
                    hit_grids, weight, start_error, end_error,
                    previous if previous is not None else utility, utility, size,
                    chunk_origins,
                    anchor["origin"], int(duration), overlap, 256, enable_fp_fusion=False,
                )
                previous, previous_origin = utility, anchor["origin"]
                maximum = float(utility.max())
                if not np.isfinite(maximum):
                    continue
                # ties within 1e-10 choose the first cell
                i, j = divmod(int((utility.reshape(-1) >= maximum - 1e-10).to(torch.int8).argmax()), size)
                value = float(utility[i, j])
                if value > best + 1e-10:
                    best, chosen = value, np.array([float(grid[i]), float(grid[j])])

            if cue in repeat_cues:
                # the chosen span and the strongest separate peaks are this line's chain candidates
                anchor = base[cue] + anchor_offset
                locations = {tuple(map(float, chosen))}
                for c in components:
                    mass = F.avg_pool2d(c["p"][None, None], 11, stride=1, padding=5)[0, 0] * 11**2
                    times = torch.arange(size, device="cuda") * 10 + c["origin"]
                    mask = ((times - anchor[0]).abs() <= 1200)[:, None] & ((times - anchor[1]).abs() <= 1200)[None]
                    for work, number in ((mass.clone(), 16), (mass.masked_fill(~mask, 0), 8)):
                        for _ in range(number):
                            maximum, index = work.flatten().max(0)
                            if maximum <= 0:
                                break
                            s, e = divmod(int(index), size)
                            span = (c["origin"] + s * 10, c["origin"] + e * 10)
                            if span[1] - span[0] >= 20 and span[0] >= 0 and span[1] <= duration:
                                locations.add(span)
                            work[max(0, s - 30):min(size, s + 31), max(0, e - 30):min(size, e + 31)] = 0
                candidates = np.asarray(sorted(locations), dtype=np.float64)
                actions = torch.tensor(candidates, device="cuda", dtype=torch.float64)
                score = actions.new_zeros(len(actions))
                for w, c in zip(weight, components):
                    prefix = c["prefix"]
                    coordinate = (actions - c["origin"]) / 10
                    lo = (coordinate - 50 / 10).ceil().long().clamp(0, size)
                    hi = ((coordinate + 50 / 10).floor().long() + 1).clamp(0, size)
                    score += w * (prefix[hi[:, 0], hi[:, 1]] - prefix[lo[:, 0], hi[:, 1]] - prefix[hi[:, 0], lo[:, 1]] + prefix[lo[:, 0], lo[:, 1]]).clamp_min(0)
                log_mass = score.clamp_min(1e-15).log().cpu().numpy()
                # each candidate's own score: its local mass less a bounded penalty for leaving the route
                repeat_candidates[cue] = (candidates, log_mass - 4. * penalty((candidates - anchor) / 1000.).mean(1))
            return chosen.tolist()

        spans = []
        for cue, components in distributions():
            spans.append(select(cue, components))
            # release per-cue posterior grids before constructing the next cue
            components.clear()

        # exact chain map over each group's candidates with bounded relative-gap and anchor penalties
        for group in groups:
            items = [repeat_candidates[i] for i in group]
            scores, backs = items[0][1], []
            centers = [options.mean(1) for options, _ in items]
            coarse = np.asarray(base[group], float).mean(1)
            for i in range(1, len(items)):
                residual = ((centers[i][None] - centers[i - 1][:, None]) - (coarse[i] - coarse[i - 1])) / 1000.
                transitions = scores[:, None] - 1. * penalty(residual)
                pointer = transitions.argmax(0)
                backs.append(pointer)
                scores = transitions[pointer, np.arange(len(pointer))] + items[i][1]
            state = int(scores.argmax())
            states = [state]
            for pointer in reversed(backs):
                state = int(pointer[state])
                states.append(state)
            states.reverse()
            for i, state, (options, _) in zip(group, states, items):
                spans[i] = options[state]
        return [[int(start), int(end)] for start, end in spans]
