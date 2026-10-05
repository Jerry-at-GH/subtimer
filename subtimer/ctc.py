import numpy as np
import torch
import triton
import triton.language as tl
from numba import njit

from . import gpu

NEGATIVE = -3.4e38


@triton.jit
def _advance(
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
    add_before: tl.constexpr,
    band,
    frames,
    use_band: tl.constexpr,
):
    lane = tl.arange(0, width)

    index = tl.program_id(0) * (width - 2 * steps) + lane - 2 * steps
    valid = (index >= 0) & (index < states)
    write = valid & (lane >= 2 * steps)
    previous = tl.load(before + index, valid, other=negative)
    skip = tl.load(forbidden + index, valid, other=1)
    token = tl.load(ids + index, valid, other=0)
    if use_band:
        pace = (states.to(tl.float64) - 1) / (frames.to(tl.float64) - 1)
    for frame in range(first, stop):
        score = tl.load(emissions + frame * vocabulary + token, valid, other=0)
        one = tl.gather(previous, tl.maximum(lane - 1, 0), 0)
        two = tl.gather(previous, tl.maximum(lane - 2, 0), 0)
        if add_before:
            previous += score
            one += score
            two += score
        one = tl.where((lane >= 1) & (index >= 1), one, negative)
        two = tl.where((lane >= 2) & (index >= 2) & (~skip), two, negative)

        relative = tl.where(one > previous, 1, 0)
        best = tl.maximum(previous, one)
        relative = tl.where(two > best, 2, relative)
        best = tl.maximum(best, two)
        previous = best if add_before else best + score
        if use_band:
            center = frame.to(tl.float64) * pace
            low = (center - band).to(tl.float32)
            high = (center + band).to(tl.float32)
            previous = tl.where((index < low) | (index > high), negative, previous)
        tl.store(back + (frame - origin) * states + index, relative, write)
    tl.store(after + index, previous, write)


def viterbi(emissions, ids, *, negative=NEGATIVE, add_before=False, band=0):
    frames, vocabulary = emissions.shape
    states = ids.numel()
    skip = torch.ones(states, dtype=torch.bool, device=emissions.device)
    skip[2:] = ids[2:] == ids[:-2]
    previous = torch.full((states,), negative, device=emissions.device)
    previous[: min(2, states)] = emissions[0, ids[: min(2, states)]]
    following = torch.empty_like(previous)
    back = torch.zeros((frames, states), dtype=torch.int8)
    if frames == 1:
        return previous, back

    count = min(frames - 1, gpu.available() // (2 * states))
    if count < 1:
        raise RuntimeError("insufficient free VRAM for one CTC traceback row")
    block = torch.empty((count, states), dtype=torch.int8, device=emissions.device)
    steps, width = 32, 512
    for origin in range(1, frames, count):
        end = min(origin + count, frames)
        for first in range(origin, end, steps):
            _advance[(triton.cdiv(states, width - 2 * steps),)](
                emissions,
                ids,
                previous,
                following,
                skip,
                block,
                states,
                vocabulary,
                first,
                min(first + steps, end),
                origin,
                steps,
                width,
                negative,
                add_before,
                band,
                frames,
                band > 0 and frames > 1 and states > 1,
                num_warps=4,
                enable_fp_fusion=False,
            )
            previous, following = following, previous
        back[origin:end].copy_(block[: end - origin])
    return previous, back


def bfa_decode_cuda(
    self,
    log_probs,
    ctc_path,
    ctc_len,
    ctc_path_true_idx=None,
    band_width=0,
    debug=False,
):
    scores, back = viterbi(
        log_probs.contiguous(),
        ctc_path,
        negative=self._neg_inf,
        add_before=True,
        band=band_width,
    )
    scores = scores.cpu().numpy()
    if self.truly_forced:
        state = ctc_len - 1
        if scores[state] <= self._neg_inf and ctc_len >= 2:
            state -= 1
        if scores[state] <= self._neg_inf:
            valid = (scores > self._neg_inf).nonzero()[0]
            state = int(valid[-1]) if len(valid) else ctc_len - 1
    else:
        state = int(scores.argmax())
    states = torch.empty(len(log_probs), dtype=torch.long)
    states[-1] = state
    pointers = back.numpy()
    for frame in range(len(log_probs) - 1, 0, -1):
        state %= ctc_len
        state -= int(pointers[frame, state])
        states[frame - 1] = state
    states = states.to(ctc_path.device)
    return ctc_path[states], (
        ctc_path_true_idx[states] if ctc_path_true_idx is not None else None
    )


@njit(cache=True)
def path_states(lp, path, blank, forced, negative, band):
    T = len(lp)
    S = len(path)
    previous = np.full(S, negative, np.float32)
    back = np.zeros((T, S), np.int32)
    previous[0] = lp[0, blank]
    if S > 1:
        previous[1] = lp[0, path[1]]
    for t in range(1, T):
        current = np.empty(S, np.float32)
        for s in range(S):
            value = np.float32(previous[s] + lp[t, path[s]])
            state = s
            advance = np.float32(previous[s - 1] + lp[t, path[s]]) if s else negative
            if advance > value:
                value = advance
                state = s - 1
            skip = (
                np.float32(previous[s - 2] + lp[t, path[s]])
                if s >= 2 and path[s] != path[s - 2]
                else negative
            )
            if skip > value:
                value = skip
                state = s - 2
            if band > 0 and T > 1 and S > 1:
                center = t * (S - 1) / (T - 1)
                if s < np.float32(center - band) or s > np.float32(center + band):
                    value = negative
            current[s] = value
            back[t, s] = state
        previous = current
    if forced:
        state = S - 1
        if previous[state] <= negative and S >= 2:
            state = S - 2
        if previous[state] <= negative:
            found = False
            for s in range(S):
                if previous[s] > negative:
                    state = s
                    found = True
            if not found:
                state = S - 1
    else:
        state = int(np.argmax(previous))
    result = np.empty(T, np.int64)
    result[-1] = state
    for t in range(T - 2, -1, -1):
        state = back[t + 1, state]
        result[t] = state
    return result


def bfa_decode_cpu(
    self,
    log_probs,
    ctc_path,
    ctc_len,
    ctc_path_true_idx=None,
    band_width=0,
    debug=False,
):
    negative = np.float32(-1e30)
    state = torch.from_numpy(
        path_states(
            log_probs.detach().numpy().astype(np.float32),
            ctc_path.numpy().astype(np.int64),
            self.blank_id,
            self.truly_forced,
            negative,
            band_width,
        )
    )
    return ctc_path[state], ctc_path_true_idx[
        state
    ] if ctc_path_true_idx is not None else None


@njit(cache=True)
def ctc_align(lp, ids):
    T = len(lp)
    S = 2 * len(ids) + 1
    states = np.zeros(S, np.int64)
    states[1::2] = ids
    prev = np.full(S, -np.inf)
    prev[0] = 0.0
    trace = np.zeros((T, S), np.int8)
    for t in range(T):
        current = np.full(S, -np.inf)
        for s in range(S):
            best = prev[s]
            move = 0
            if s > 0 and prev[s - 1] > best:
                best = prev[s - 1]
                move = 1
            if (
                s > 1
                and states[s] != 0
                and states[s] != states[s - 2]
                and prev[s - 2] > best
            ):
                best = prev[s - 2]
                move = 2
            current[s] = best + lp[t, states[s]]
            trace[t, s] = move
        prev = current
    state = S - 1 if prev[S - 1] >= prev[S - 2] else S - 2
    score = prev[state]
    starts = np.full(len(ids), -1, np.int64)
    ends = starts.copy()
    for t in range(T - 1, -1, -1):
        if state % 2:
            j = state // 2
            starts[j] = t
            if ends[j] < 0:
                ends[j] = t + 1
        state -= trace[t, state]
    return starts, ends, score


@njit(cache=True)
def ctc_boundaries(lp, ids, first, last):
    t = len(lp)
    s = 2 * len(ids) + 1
    blank = lp.shape[1] - 1
    states = np.full(s, blank, np.int32)
    states[1::2] = ids
    a = np.full((t, s), -1e30)
    b = np.full((t, s), -1e30)
    a[0, 0] = lp[0, blank]
    a[0, 1] = lp[0, ids[0]]
    for f in range(1, t):
        for j in range(s):
            v = a[f - 1, j]
            if j > 0:
                v = np.logaddexp(v, a[f - 1, j - 1])
            if j > 1 and states[j] != blank and states[j] != states[j - 2]:
                v = np.logaddexp(v, a[f - 1, j - 2])
            a[f, j] = v + lp[f, states[j]]
    b[-1, -1] = 0
    b[-1, -2] = 0
    for f in range(t - 2, -1, -1):
        for j in range(s):
            v = b[f + 1, j] + lp[f + 1, states[j]]
            if j + 1 < s:
                v = np.logaddexp(v, b[f + 1, j + 1] + lp[f + 1, states[j + 1]])
            if j + 2 < s and states[j + 2] != blank and states[j + 2] != states[j]:
                v = np.logaddexp(v, b[f + 1, j + 2] + lp[f + 1, states[j + 2]])
            b[f, j] = v
    z = np.logaddexp(a[-1, -1], a[-1, -2])
    out = np.zeros((2, t))
    j = 2 * first + 1
    k = 2 * last + 1
    for f in range(t):
        v = -1e30
        if f == 0:
            if j == 1:
                v = lp[f, states[j]] + b[f, j]
        else:
            v = a[f - 1, j - 1]
            if j > 1 and states[j] != states[j - 2]:
                v = np.logaddexp(v, a[f - 1, j - 2])
            v += lp[f, states[j]] + b[f, j]
        out[0, f] = np.exp(min(0.0, v - z))
        v = -1e30
        if f == t - 1:
            if k == s - 2:
                v = a[f, k]
        else:
            v = lp[f + 1, states[k + 1]] + b[f + 1, k + 1]
            if k + 2 < s and states[k] != states[k + 2]:
                v = np.logaddexp(v, lp[f + 1, states[k + 2]] + b[f + 1, k + 2])
            v += a[f, k]
        out[1, f] = np.exp(min(0.0, v - z))
    return out, z


@njit(cache=True)
def optional_path(emission, blank):
    n, frames = emission.shape
    states = 2 * n + 1
    dp = np.full(states, -np.inf)
    dp[0] = blank[0]
    dp[1] = emission[0, 0]
    back = np.zeros((frames, states), np.int8)
    skip = np.arange(3, states, 2)
    for f in range(1, frames):
        best = dp.copy()
        step = np.zeros(states, np.int8)
        better = dp[:-1] > best[1:]
        best[1:][better] = dp[:-1][better]
        step[1:][better] = 1
        better = dp[skip - 2] > best[skip]
        best[skip[better]] = dp[skip[better] - 2]
        step[skip[better]] = 2
        best[::2] += blank[f]
        best[1::2] += emission[:, f]
        dp = best
        back[f] = step
    state = states - 1 if dp[-1] > dp[-2] else states - 2
    score = dp[state]
    path = np.empty(frames, np.int32)
    for f in range(frames - 1, -1, -1):
        path[f] = state
        if f:
            state -= back[f, state]
    return path, score
