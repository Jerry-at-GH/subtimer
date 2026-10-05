import json
import os
import sys
import tempfile
import threading
from pathlib import Path


def run(payload, report):
    from . import speech
    from .runtime import align

    state = {"stage": "decode"}
    lock = threading.Lock()

    def send(**extra):
        with lock:
            report({**state, **extra})

    def progress(stage):
        with lock:
            if stage != state["stage"]:
                state.update(stage=stage)
                report(dict(state))

    send()
    with tempfile.TemporaryDirectory(prefix="subtimer_") as temporary:
        workspace = Path(temporary)
        extracted = workspace / "source.wav"
        speech.convert(Path(payload["audio"]), extracted, preserve_format=True)
        timings = align(
            payload["dialogues"],
            extracted,
            workspace,
            payload.get("guide_segments", []),
            progress,
        )
    send(status="done", timings=timings)


def main():
    sys.stdout.flush()
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as reports:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        try:
            payload = json.load(sys.stdin)

            def report(value):
                reports.write(json.dumps(value, ensure_ascii=False) + "\n")
                reports.flush()

            run(payload, report)
        finally:
            sys.stdout.flush()
            os.dup2(reports.fileno(), sys.stdout.fileno())
