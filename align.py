import argparse
import json
import os
import re
import subprocess
import sys
from bisect import bisect_left

import pysubs2

from vad import run_vad


WORD_CHARS = "A-Za-z0-9\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
non_word = re.compile(rf"[^{WORD_CHARS}]+")

SNAP_START_WINDOW_MS = 400
SNAP_END_WINDOW_MS = 800
GAP_WINDOW_MS = 200
MIN_DURATION_MS = 200


def load_lines(lines_path):
    if os.path.splitext(lines_path)[1].lower() == ".ass":
        subs = pysubs2.load(lines_path)
        lines = [
            {
                "raw": e.text.strip(),
                "clean": re.sub(non_word, " ", e.text.strip()).strip(),
                "name": e.name,
                "raw_start": e.start,
                "raw_end": e.end,
            }
            for e in subs.events
            if len(e.text.strip()) > 0 and not e.is_comment
        ]
        lines = [{**l, "match": re.sub(non_word, "", l["clean"])} for l in lines]
    elif os.path.splitext(lines_path)[1].lower() == ".txt":
        lines = []
        sep = "\t"
        with open(lines_path, "r", encoding="utf-8") as f:
            for raw in f:
                text = raw.strip()
                if not text:
                    continue
                if text[0] in "(（":
                    clean = ""
                else:
                    assert sep in text, f"no split char: {text}"
                    clean = re.sub(non_word, " ", text.split(sep)[0]).strip()
                lines.append(
                    {
                        "raw": text.replace(sep, r"\N"),
                        "clean": clean,
                        "match": re.sub(non_word, "", clean),
                    }
                )
        return lines
    else:
        raise NotImplementedError("unsupported file type for lines")


def load_ctm_tokens(ctm_filepath):
    seg = []
    with open(ctm_filepath, "r", encoding="utf-8") as f:
        for line in f:
            c = line.strip().split()
            if not c:
                continue
            start = float(c[2])
            dur = float(c[3])
            token = c[4].replace("<b>", "")
            seg.append({"start": start, "duration": dur, "end": start + dur, "text": token})
    return seg


def check_with_raw(line, start, end, tolerance=2000):
    if "raw_start" in line and not (float(line["raw_start"]) - tolerance < start < float(line.get("raw_end", start))):
        start = float(line["raw_start"])
    if "raw_end" in line and not (float(line.get("raw_start", end)) < end < float(line["raw_end"]) + tolerance):
        end = float(line["raw_end"])
    return start, end


def process_segments(
    segments,
    lines,
    label,
    start_pad=0,
    end_pad=2,
    pad_time_frac=0.5,
    optimize_long_seg=True,
    do_check_with_raw=True,
):
    """
    Args:
        start_pad: in number of segments. specify wisely according to model behavior
        end_pad: in number of segments. specify wisely according to model behavior
    """
    for s in segments:
        s["match"] = non_word.sub("", s["text"])

    start_i = 0
    line_iter = iter([l for l in lines if l["match"]])
    try:
        l = next(line_iter)
        match = l["match"]
        for i, s in enumerate(segments):
            if match.startswith(s["match"]):
                match = match[len(s["match"]) :]
            if not match:
                start_i = next(i_ for i_, s_ in enumerate(segments) if i_ >= start_i and s_["match"])
                # this will only increase start_i; it skips non-word segments
                end_i = i

                padded_start_time = 0
                for i in range(start_i - start_pad, start_i):
                    if i >= 0 and not "".join(s["match"] for s in segments[i:start_i]):
                        # apply largest start_pad possible with no word. i is the padded_start_index
                        padded_start_time = sum(s["duration"] for s in segments[i:start_i])
                        break
                padded_end_time = 0  # similarly
                for i in range(end_i + 1 + end_pad, end_i + 1, -1):
                    if i < len(segments) and not "".join(s["match"] for s in segments[end_i + 1 : i + 1]):
                        padded_end_time = sum(s["duration"] for s in segments[end_i + 1 : i + 1])
                        break

                start = (segments[start_i]["start"] - padded_start_time * pad_time_frac) * 1000
                end = (segments[end_i]["end"] + padded_end_time * pad_time_frac) * 1000

                if optimize_long_seg:  # only consider segments that's not added by padding
                    long_seg = [
                        ((i + 0.5) / max(1, end_i - start_i), s["duration"] - 0.8)
                        for i, s in enumerate(segments[start_i : end_i + 1])
                        if s["duration"] > 0.8
                    ]
                    if long_seg:
                        denom = sum(max(0.0, x[1]) ** 2 for x in long_seg) or 1.0
                        avg_position = sum(x[0] * (max(0.0, x[1]) ** 2) for x in long_seg) / denom
                        total_delta = sum(max(0.0, x[1]) for x in long_seg) * 1000
                        start += (1 - avg_position) * total_delta
                        end -= avg_position * total_delta

                if do_check_with_raw:
                    start, end = check_with_raw(l, start, end)

                l[f"{label}_start"] = start
                l[f"{label}_end"] = end

                start_i = end_i + 1
                l = next(line_iter)
                match = str(l["match"])
    except StopIteration:
        pass


def snap_lines_with_vad(lines, vad_segs):
    prev_end = None
    for l in lines:
        if not l.get("clean"):
            continue
        s0 = int(l.get("res_start", 0))
        e0 = int(l.get("res_end", 0))
        # inline nearest onset/offset search within snap window
        starts = [s for s, _ in vad_segs]
        i = bisect_left(starts, s0)
        onset_choices = []
        if i < len(vad_segs):
            onset_choices.append((i, vad_segs[i][0]))
        if i - 1 >= 0:
            onset_choices.append(((i - 1), vad_segs[i - 1][0]))
        onset_choices = [(idx, c) for idx, c in onset_choices if abs(c - s0) <= SNAP_START_WINDOW_MS]
        if onset_choices:
            _, s = min(onset_choices, key=lambda t: abs(t[1] - s0))
        else:
            _, s = None, s0

        ends = [e for _, e in vad_segs]
        j = bisect_left(ends, e0)
        offset_cand = []
        if j < len(ends):
            offset_cand.append(ends[j])
        if j - 1 >= 0:
            offset_cand.append(ends[j - 1])
        offset_cand = [c for c in offset_cand if abs(c - e0) <= SNAP_END_WINDOW_MS]
        e = min(offset_cand, key=lambda c: abs(c - e0)) if offset_cand else e0
        if prev_end is not None:
            s = max(s, prev_end + 1)
        if e - s < MIN_DURATION_MS:
            e = s + MIN_DURATION_MS
        l["res_start"], l["res_end"] = int(s), int(e)
        prev_end = l["res_end"]


def adjust_boundaries_with_vad_gaps(lines, vad_segs):
    if len(vad_segs) < 2:
        return
    vad_gaps = [(vad_segs[i][1], vad_segs[i + 1][0]) for i in range(len(vad_segs) - 1)]
    clean_idx = [i for i, l in enumerate(lines) if l.get("clean")]
    for k in range(len(clean_idx) - 1):
        i = clean_idx[k]
        j = clean_idx[k + 1]
        e_i = int(lines[i]["res_end"]) if lines[i].get("clean") else None
        s_j = int(lines[j]["res_start"]) if lines[j].get("clean") else None
        if e_i is None or s_j is None:
            continue
        boundary = (e_i + s_j) / 2
        nearest_gap = None
        min_distance = 10**10
        for g_start, g_end in vad_gaps:
            if g_end <= g_start:
                continue
            if (g_start <= boundary + GAP_WINDOW_MS / 2) and (g_end >= boundary - GAP_WINDOW_MS / 2):
                distance = abs(g_start - boundary)
                if distance < min_distance:
                    min_distance = distance
                    nearest_gap = (g_start, g_end)
        if nearest_gap:
            lines[i]["res_end"] = int(nearest_gap[0])
            lines[j]["res_start"] = int(max(nearest_gap[1], lines[i]["res_end"] + 1))


def build_ass(lines):
    subs = pysubs2.SSAFile()
    subs.styles["Default"] = pysubs2.SSAStyle(
        fontname="IPAexGothic",
        fontsize=40,
        primarycolor=pysubs2.Color(r=255, g=255, b=255, a=0),
        secondarycolor=pysubs2.Color(r=0, g=213, b=255, a=0),
        outlinecolor=pysubs2.Color(r=0, g=0, b=0, a=0),
        backcolor=pysubs2.Color(r=74, g=74, b=74, a=0),
        bold=True,
        alignment=pysubs2.Alignment.BOTTOM_CENTER,
        shadow=0,
    )
    subs.styles["Top"] = subs.styles["Default"].copy()
    subs.styles["Top"].alignment = pysubs2.Alignment.TOP_CENTER
    subs.info["PlayResX"] = ""
    subs.info["PlayResY"] = ""

    dialogue_events = [
        pysubs2.SSAEvent(start=l["res_start"], end=l["res_end"], text=l["raw"], style="Default", name=l.get("name", ""))
        for l in lines
        if l.get("clean")
    ]
    other_events = []
    for idx, l in enumerate(lines):
        if not l["clean"]:
            prev_event = next((lines[i] for i in range(idx - 1, -1, -1) if lines[i]["clean"]), None)
            next_event = next((lines[i] for i in range(idx + 1, len(lines)) if lines[i]["clean"]), None)
            start = l.get("raw_start", 0)
            end = l.get("raw_end", 1000)
            if prev_event and next_event:
                x = (prev_event["res_end"] + next_event["res_start"]) / 2.0
                start = x - 500
                end = x + 500
            elif next_event:
                start = next_event["res_start"] - 1000
                end = next_event["res_start"]
            elif prev_event:
                start = prev_event["res_end"]
                end = prev_event["res_end"] + 1000
            other_events.append(pysubs2.SSAEvent(start=start, end=end, text=l["raw"], style="Top", name=l.get("name", "")))
    subs.events = dialogue_events + other_events
    return subs, dialogue_events, other_events


def evaluate_alignment(dialogue_events, ground_path, fps):
    import statistics

    ground = pysubs2.load(ground_path)
    ground.events = [e for e in ground.events if not e.is_comment and e.style == "Default"]
    assert len(dialogue_events) == len(ground.events)

    start_mederr = statistics.median((g.start - p.start) for g, p in zip(ground.events, dialogue_events))
    end_mederr = statistics.median((g.end - p.end) for g, p in zip(ground.events, dialogue_events))

    start_mae = statistics.mean(abs(g.start - p.start) for g, p in zip(ground.events, dialogue_events))
    end_mae = statistics.mean(abs(g.end - p.end) for g, p in zip(ground.events, dialogue_events))

    start_counts = []
    end_counts = []
    both_counts = []
    for frame_count in [3, 6, 12]:
        threshold = 1000 / fps * frame_count  # ms
        start_counts.append(sum(1 for g, p in zip(ground.events, dialogue_events) if abs(g.start - p.start) <= threshold))
        end_counts.append(sum(1 for g, p in zip(ground.events, dialogue_events) if abs(g.end - p.end) <= threshold))
        both_counts.append(
            sum(
                1
                for g, p in zip(ground.events, dialogue_events)
                if abs(g.start - p.start) <= threshold and abs(g.end - p.end) <= threshold
            )
        )
    print(f"[total {len(dialogue_events)}]")
    print("      | med err | mean abs err |    ≤3    ≤6   ≤12 frames")
    print(
        f"{'start':5} | {round(start_mederr):+5d}ms |      {round(start_mae):5d}ms | {start_counts[0]:5d} {start_counts[1]:5d} {start_counts[2]:5d}"
    )
    print(
        f"{'end':5} | {round(end_mederr):+5d}ms |      {round(end_mae):5d}ms | {end_counts[0]:5d} {end_counts[1]:5d} {end_counts[2]:5d}"
    )
    print(f"{'both':5} |                        | {both_counts[0]:5d} {both_counts[1]:5d} {both_counts[2]:5d}")


def main():
    parser = argparse.ArgumentParser(description="generate ASS via Forced Alignment + VAD refinement.")
    parser.add_argument("--work_dir", required=True, help="all files below relative to this, unless absolute")
    parser.add_argument("--lines", required=True, help="path to lines TXT/ASS file")
    parser.add_argument("--video", default="1.mkv", help="path to video file for alignment")
    parser.add_argument("--output", default="align.ass", help="output ASS filename")
    parser.add_argument("--ground", default=None, help="ground truth ASS filename for evaluation")
    parser.add_argument("--fps", default=24 / 1.001, help="frames per second for video")

    args = parser.parse_args()

    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    assert os.path.isdir(work_dir)

    lines_path = os.path.join(work_dir, args.lines)
    assert os.path.exists(lines_path)
    lines = load_lines(lines_path)

    # extract mono 16kHz audio if not exists
    video_path = os.path.join(work_dir, args.video)
    audio_16k_path = os.path.join(work_dir, "mono_16k.wav")
    if not os.path.exists(audio_16k_path):
        print("extracting mono 16kHz audio")
        subprocess.run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", audio_16k_path], capture_output=True)

    # VAD
    vad_segs = run_vad(work_dir, video_path=video_path)
    if not vad_segs:
        raise SystemExit("VAD failed")

    # NFA (with optional VAD prior)
    nfa_output_dir = os.path.join(work_dir, "nfa", "nfa_output")
    ctm_path = os.path.join(nfa_output_dir, "ctm", "tokens", "mono_16k.ctm")
    if not os.path.exists(ctm_path):
        print("running NFA")
        os.makedirs(os.path.dirname(ctm_path), exist_ok=True)
        nfa_dir = os.path.join(work_dir, "nfa")
        os.makedirs(nfa_dir, exist_ok=True)
        manifest_path = os.path.join(nfa_dir, "manifest.json")
        text_joined = "|".join([l["clean"] for l in lines if l.get("clean")])
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"audio_filepath": audio_16k_path, "text": text_joined}) + "\n")
        subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "nfa.py"),
                "pretrained_name=nvidia/parakeet-tdt_ctc-0.6b-ja",
                f"manifest_filepath={manifest_path}",
                f"output_dir={nfa_output_dir}",
                "additional_segment_grouping_separator=|",
                "use_vad_prior=True",
                "vad_prior_k=2.0",
            ],
            capture_output=True,
        )
        if not os.path.exists(ctm_path):
            raise SystemExit("NFA failed")

    # refine with VAD
    seg = load_ctm_tokens(ctm_path)
    process_segments(seg, lines, label="nfa")
    for l in lines:
        if l.get("clean"):
            l["res_start"] = l.get("nfa_start", 0)
            l["res_end"] = l.get("nfa_end", 0)

    snap_lines_with_vad(lines, vad_segs)
    adjust_boundaries_with_vad_gaps(lines, vad_segs)

    subs, dialogue_events, _ = build_ass(lines)
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(work_dir, output_path)
    subs.save(output_path)

    # evaluate
    if args.ground:
        ground_path = os.path.join(work_dir, args.ground)
        evaluate_alignment(dialogue_events, ground_path, fps=args.fps)


if __name__ == "__main__":
    main()
