import argparse
import json
import os
import re
import subprocess
import sys
from bisect import bisect_left

import numpy as np
import pysubs2
from vad import run_vad

WORD_CHARS = "A-Za-z0-9\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
non_word = re.compile(rf"[^{WORD_CHARS}]+")

SNAP_START_WINDOW_MS = 400
SNAP_END_WINDOW_MS = 800
GAP_WINDOW_MS = 200
MIN_DURATION_MS = 200
CHECK_WITH_RAW_TOLERANCE_MS = 2000


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
        lines = [{**line, "match": re.sub(non_word, "", line["clean"])} for line in lines]
        return lines
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


def nearest_boundary_value(x, boundaries, cap):
    """
    finds the nearest boundary value within cap range of x.
    """
    if not boundaries:
        return None
    i = bisect_left(boundaries, x)
    candidates = []
    if i < len(boundaries):
        candidates.append(boundaries[i])
    if i > 0:
        candidates.append(boundaries[i - 1])
    candidates = [c for c in candidates if abs(c - x) <= cap]
    return min(candidates, key=lambda c: abs(c - x)) if candidates else None


def apply_to_lines(segments, lines, label, start_pad, end_pad, pad_time_frac, optimize_long_seg):
    """
    for an alignment run that generates `segments`, aligns them to `lines`, and write timings to fields starting with `label` in each line
    `start_pad` and `end_pad` are in segment count, so they depend on how the aligner chunks tokens.
    """
    for segment in segments:
        segment["match"] = non_word.sub("", segment["text"])

    start_i = 0
    line_iter = iter([l for l in lines if l["match"]])
    try:
        line = next(line_iter)
        match = line["match"]
        for i, segment in enumerate(segments):
            if match.startswith(segment["match"]):
                match = match[len(segment["match"]) :]
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

                if optimize_long_seg:
                    long_seg = [
                        ((i + 0.5) / max(1, end_i - start_i), s["duration"] - 0.8)
                        for i, s in enumerate(
                            segments[start_i : end_i + 1]  # only consider segments that's not added by padding
                        )
                        if s["duration"] > 0.8
                    ]
                    if long_seg:
                        denom = sum(max(0.0, x[1]) ** 2 for x in long_seg) or 1.0
                        avg_position = sum(x[0] * (max(0.0, x[1]) ** 2) for x in long_seg) / denom
                        total_delta = sum(max(0.0, x[1]) for x in long_seg) * 1000
                        start += (1 - avg_position) * total_delta
                        end -= avg_position * total_delta

                line[f"{label}_start"] = start
                line[f"{label}_end"] = end

                start_i = end_i + 1
                line = next(line_iter)
                match = str(line["match"])
    except StopIteration:
        pass


def run_nfa(work_dir, lines, audio_path):
    nfa_audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    nfa_dir = os.path.join(work_dir, "nfa_" + nfa_audio_name)
    os.makedirs(nfa_dir, exist_ok=True)

    ctm_path = os.path.join(nfa_dir, "ctm", "tokens", f"{nfa_audio_name}.ctm")
    if not os.path.exists(ctm_path):
        print("running NFA")
        os.makedirs(os.path.dirname(ctm_path), exist_ok=True)

        manifest_path = os.path.join(nfa_dir, "manifest.json")
        text_joined = "|".join([line["clean"] for line in lines if line.get("clean")])
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"audio_filepath": audio_path, "text": text_joined}) + "\n")

        subprocess.run(
            [
                sys.executable,
                os.path.join(os.path.dirname(__file__), "nfa.py"),
                f"manifest_filepath={manifest_path}",
                f"output_dir={nfa_dir}",
                "additional_segment_grouping_separator=|",
                "use_vad_prior=True",
                "vad_prior_k=2.0",
                f"vad_dir_name={os.path.relpath(os.path.join(work_dir, 'vad'), os.path.dirname(audio_path))}",
            ],
            capture_output=True,
            text=True,
        )

        if not os.path.exists(ctm_path):
            raise SystemExit("NFA failed")

    segs = []
    with open(ctm_path, "r", encoding="utf-8") as f:
        for l in f:
            c = l.strip().split()
            if not c:
                continue
            start = float(c[2])
            dur = float(c[3])
            token = c[4].replace("<b>", "")
            segs.append({"start": start, "duration": dur, "end": start + dur, "text": token})
    return segs


def refine_lines(lines, label, heuristic_segs, vad_segs):
    """
    reads VAD results (`heuristic_segs`, `vad_segs`) and refines the timings labeled `label` in `lines` in place.
    """
    starts = [s for s, _ in vad_segs]
    ends = [e for _, e in vad_segs]

    # snap line starts and ends to nearby VAD boundaries
    prev_end = None
    for line in lines:
        if not line.get("clean"):
            continue
        if f"{label}_start" not in line or f"{label}_end" not in line:
            continue

        start = int(line[f"{label}_start"])
        end = int(line[f"{label}_end"])

        snapped_start = nearest_boundary_value(start, starts, SNAP_START_WINDOW_MS)
        snapped_end = nearest_boundary_value(end, ends, SNAP_END_WINDOW_MS)

        start = snapped_start if snapped_start is not None else start
        end = snapped_end if snapped_end is not None else end

        if prev_end is not None:
            start = max(start, prev_end + 1)
        if end - start < MIN_DURATION_MS:
            end = start + MIN_DURATION_MS

        line[f"{label}_start"] = int(start)
        line[f"{label}_end"] = int(end)
        prev_end = int(end)

    # move adjacent line boundaries into a nearby VAD gap
    if len(vad_segs) >= 2:
        vad_gaps = [(vad_segs[i][1], vad_segs[i + 1][0]) for i in range(len(vad_segs) - 1)]
        clean_idx = [
            i for i, line in enumerate(lines) if line.get("clean") and f"{label}_start" in line and f"{label}_end" in line
        ]

        for k in range(len(clean_idx) - 1):
            i = clean_idx[k]
            j = clean_idx[k + 1]
            end_i = int(lines[i][f"{label}_end"])
            start_j = int(lines[j][f"{label}_start"])
            boundary = (end_i + start_j) / 2

            nearest_gap = None
            min_distance = 10**10
            for gap_start, gap_end in vad_gaps:
                if gap_end <= gap_start:
                    continue
                if gap_start <= boundary + GAP_WINDOW_MS / 2 and gap_end >= boundary - GAP_WINDOW_MS / 2:
                    distance = abs(gap_start - boundary)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_gap = (gap_start, gap_end)

            if nearest_gap:
                lines[i][f"{label}_end"] = int(nearest_gap[0])
                lines[j][f"{label}_start"] = int(max(nearest_gap[1], lines[i][f"{label}_end"] + 1))

    # trim line edges to overlapping heuristic speech spans
    for line in lines:
        if not line.get("clean"):
            continue
        if f"{label}_start" not in line or f"{label}_end" not in line:
            continue

        start = line[f"{label}_start"]
        end = line[f"{label}_end"]
        overlaps = [(hs, he) for hs, he in heuristic_segs if he >= start and hs <= end]
        if overlaps:
            start = max(start, min(hs for hs, _ in overlaps))
            end = min(end, max(he for _, he in overlaps))

        line[f"{label}_start"] = start
        line[f"{label}_end"] = end


def total_vad_pred(lines, frame_hop, vad_pred, label):
    total = 0
    vad_len = len(vad_pred)

    for line in lines:
        if line.get("clean"):
            start = int(line.get(f"{label}_start", 0))
            end = int(line.get(f"{label}_end", 0))
            frame_start = max(0, min(int(start / frame_hop), vad_len - 1))
            frame_end = max(frame_start + 1, min(int(np.ceil(end / frame_hop)), vad_len))
            total += np.mean(vad_pred[frame_start:frame_end]).item()
    return total


def build_ass(lines):
    subs = pysubs2.SSAFile()
    subs.styles["Default"] = pysubs2.SSAStyle(
        fontname="GenSenRounded2 JP M",
        fontsize=50,
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
        pysubs2.SSAEvent(
            start=line["res_start"], end=line["res_end"], text=line["raw"], style="Default", name=line.get("name", "")
        )
        for line in lines
        if line.get("clean")
    ]
    other_events = []
    for idx, line in enumerate(lines):
        if not line["clean"]:
            prev_event = next((lines[i] for i in range(idx - 1, -1, -1) if lines[i]["clean"]), None)
            next_event = next((lines[i] for i in range(idx + 1, len(lines)) if lines[i]["clean"]), None)
            start = line.get("raw_start", 0)
            end = line.get("raw_end", 1000)
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
            other_events.append(
                pysubs2.SSAEvent(start=start, end=end, text=line["raw"], style="Top", name=line.get("name", ""))
            )
    subs.events = dialogue_events + other_events
    return subs, dialogue_events, other_events


def evaluate_alignment(dialogue_events, ground_path, fps):
    import statistics

    ground = pysubs2.load(ground_path)
    ground.events = [e for e in ground.events if len(e.text.strip()) > 0 and not e.is_comment]
    assert len(dialogue_events) == len(ground.events)

    start_quantiles = statistics.quantiles(((g.start - p.start) for g, p in zip(ground.events, dialogue_events)), n=100)
    end_quantiles = statistics.quantiles(((g.end - p.end) for g, p in zip(ground.events, dialogue_events)), n=100)

    start_mae = statistics.mean(abs(g.start - p.start) for g, p in zip(ground.events, dialogue_events))
    end_mae = statistics.mean(abs(g.end - p.end) for g, p in zip(ground.events, dialogue_events))
    both_mae = (start_mae + end_mae) / 2

    total = len(dialogue_events)
    start_frac = []
    end_frac = []
    both_frac = []
    for threshold in [100, 500]:
        start_frac.append(sum(1 for g, p in zip(ground.events, dialogue_events) if abs(g.start - p.start) <= threshold) / total)
        end_frac.append(sum(1 for g, p in zip(ground.events, dialogue_events) if abs(g.end - p.end) <= threshold) / total)
        both_frac.append(
            sum(
                1
                for g, p in zip(ground.events, dialogue_events)
                if abs(g.start - p.start) <= threshold and abs(g.end - p.end) <= threshold
            )
            / total
        )
    print(f"[total: {total} lines]")
    print("      |   p10   p25   p50   p75   p90    | mean abs |   ≤100   ≤500 ms")
    print(
        f"{'start':5} | {round(start_quantiles[10]):+5d} {round(start_quantiles[25]):+5d} {round(start_quantiles[50]):+5d} {round(start_quantiles[75]):+5d} {round(start_quantiles[90]):+5d} ms |  {round(start_mae):5d}ms | {start_frac[0] * 100:5.1f}% {start_frac[1] * 100:5.1f}%"
    )
    print(
        f"{'end':5} | {round(end_quantiles[10]):+5d} {round(end_quantiles[25]):+5d} {round(end_quantiles[50]):+5d} {round(end_quantiles[75]):+5d} {round(end_quantiles[90]):+5d} ms |  {round(end_mae):5d}ms | {end_frac[0] * 100:5.1f}% {end_frac[1] * 100:5.1f}%"
    )
    print(
        f"{'both':5} |                                  |  {round(both_mae):5d}ms | {both_frac[0] * 100:5.1f}% {both_frac[1] * 100:5.1f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="generate ASS via Forced Alignment + VAD refinement.")
    parser.add_argument("--work_dir", required=True, help="all files below relative to this, unless absolute")
    parser.add_argument("--lines", required=True, help="path to lines TXT/ASS file")
    parser.add_argument("--do_check_with_raw", action="store_true", help="whether to check alignment against raw timestamps")
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
    mix_path = os.path.join(work_dir, "mono_16k.wav")
    if not os.path.exists(mix_path):
        print("extracting mono 16kHz audio")
        subprocess.run(["ffmpeg", "-i", video_path, "-ac", "1", "-ar", "16000", mix_path], capture_output=True)

    # VAD
    frame_hop, vad_pred, heuristic_segs, vad_segs = run_vad(work_dir, video_path=video_path)
    vocals_path = os.path.join(work_dir, "vad", "vocals_mono_16k.wav")

    # NFA, then apply to `lines`
    apply_to_lines(run_nfa(work_dir, lines, mix_path), lines, label="nfa_mix", start_pad=0, end_pad=2, pad_time_frac=0.5, optimize_long_seg=True)
    apply_to_lines(run_nfa(work_dir, lines, vocals_path), lines, label="nfa_vocals", start_pad=0, end_pad=2, pad_time_frac=0.5, optimize_long_seg=True)

    # refine candidate alignment based on VAD segments
    refine_lines(lines, "nfa_mix", heuristic_segs, vad_segs)
    refine_lines(lines, "nfa_vocals", heuristic_segs, vad_segs)

    # score with VAD predictions and choose the better alignment
    mix_score = total_vad_pred(lines, frame_hop, vad_pred, label="nfa_mix")
    vocals_score = total_vad_pred(lines, frame_hop, vad_pred, label="nfa_vocals")
    chosen_label = "nfa_vocals" if vocals_score > mix_score else "nfa_mix"
    for line in lines:
        if line.get("clean"):
            line["res_start"] = line[f"{chosen_label}_start"]
            line["res_end"] = line[f"{chosen_label}_end"]

    if args.do_check_with_raw:
        # clamp timings back to raw timestamps when they drift too far
        for line in lines:
            if line.get("clean"):
                start = line["res_start"]
                end = line["res_end"]

                if "raw_start" in line and not (
                    float(line["raw_start"]) - CHECK_WITH_RAW_TOLERANCE_MS < start < float(line.get("raw_end", start))
                ):
                    start = float(line["raw_start"])
                if "raw_end" in line and not (
                    float(line.get("raw_start", end)) < end < float(line["raw_end"]) + CHECK_WITH_RAW_TOLERANCE_MS
                ):
                    end = float(line["raw_end"])

                line["res_start"] = start
                line["res_end"] = end

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
