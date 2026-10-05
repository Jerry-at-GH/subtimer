import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

def load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, encoding="utf-8"
    ) as stream:
        json.dump(data, stream, ensure_ascii=False)
    os.replace(stream.name, path)


def key(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def infer(module, payload, root, log_name, *, gpu_budget=None):
    environment = dict(
        os.environ,
        OMP_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    if gpu_budget is not None:
        environment["SUBTIMER_GPU_BUDGET_BYTES"] = str(gpu_budget)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=root) as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.flush()
        with (root / log_name).open("a") as log:
            subprocess.run(
                [sys.executable, "-m", module, stream.name],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                check=True,
            )


class Evidence:
    def __init__(self, audio, root):
        self.audio = audio
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.paths = {"raw": str(audio.raw), "vocals": str(audio.vocals)}

    def path(self, request):
        return self.root / key(request)

    def get(self, request, *, load_arrays=True):
        path = self.path(request)
        result = load(path.with_suffix(".json"))
        if result is None:
            return None
        arrays = {}
        if load_arrays and path.with_suffix(".npz").is_file():
            with np.load(path.with_suffix(".npz"), allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in stored.files}
        return result, arrays

    def put(self, request, result, arrays):
        path = self.path(request)

        if arrays:
            with tempfile.NamedTemporaryFile(
                "wb", dir=self.root, delete=False
            ) as stream:
                (np.savez if request["kind"] == "wfa" else np.savez_compressed)(
                    stream, **arrays
                )
            os.replace(stream.name, path.with_suffix(".npz"))
        save(path.with_suffix(".json"), result)

    def ensure(
        self,
        requests,
        progress=lambda *_: None,
        *,
        maximum_workers=4,
    ):
        from . import gpu

        missing = [
            r
            for r in {key(r): r for r in requests}.values()
            if not self.path(r).with_suffix(".json").is_file()
        ]
        kinds = list(dict.fromkeys(r["kind"] for r in missing))
        workers = min(maximum_workers, gpu.concurrency(len(kinds))) if kinds else 1

        def run(kind):
            progress("refine")
            payload = {
                "paths": self.paths,
                "root": str(self.root),
                "requests": [r for r in missing if r["kind"] == kind],
                "duration": self.audio.duration,
            }
            infer(
                "subtimer.evidence",
                payload,
                self.root,
                kind + ".log",
                gpu_budget=6 * gpu.GIB if workers > 1 else None,
            )

        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(run, kinds))
