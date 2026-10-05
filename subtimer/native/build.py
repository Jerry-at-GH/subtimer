import argparse
import hashlib
import subprocess
import tarfile
import tempfile
import urllib.request
from pathlib import Path

SOURCES = (
    (
        "CrispASR",
        "754b67289cf1137e3ed722885705f94132fc614f",
        "d0e50eb0ef81c1a309f369f2e64c493df36ad05aa83dafc5a1155a5ea31b6b9e",
    ),
    (
        "ggml",
        "bfe8ea228d8134d03641c9fcf233a9931f3730de",
        "fa60e66aa0196f5da88cf133c4cd4275f28204424e70a8ca667e6f135d9a93cc",
    ),
)


def main():
    parser = argparse.ArgumentParser(description="rebuild the pinned Linux encoder against the installed CrispASR libraries.")
    parser.add_argument("libraries", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    libraries = args.libraries.resolve()
    with tempfile.TemporaryDirectory(prefix="subtimer-native-") as directory:
        work = Path(directory)
        for name, revision, digest in SOURCES:
            archive = work / (name + ".tar.gz")
            with urllib.request.urlopen(
                f"https://codeload.github.com/CrispStrobe/{name}/tar.gz/{revision}",
                timeout=60,
            ) as source:
                archive.write_bytes(source.read())
            if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                raise ValueError("native source checksum mismatch: " + name)
            with tarfile.open(archive) as source:
                source.extractall(work, filter="data")
        source = work / ("CrispASR-" + SOURCES[0][1])
        ggml = work / ("ggml-" + SOURCES[1][1])
        subprocess.run(
            ["patch", "-p1", "-i", str(root / "fastconformer.patch")],
            cwd=source,
            check=True,
        )
        subprocess.run(
            [
                "g++",
                "-O2",
                "-std=c++17",
                "-fPIC",
                "-shared",
                "-fopenmp",
                "-Wl,-Bsymbolic",
                "-Wl,-rpath,$ORIGIN",
                "-I" + str(source / "src"),
                "-I" + str(source / "include"),
                "-I" + str(ggml / "include"),
                "-I" + str(ggml / "src"),
                str(root / "encoder.cpp"),
                str(libraries / "libcrispasr.so.0.8.24"),
                str(libraries / "libggml.so.0"),
                str(libraries / "libggml-base.so.0"),
                str(libraries / "libggml-cpu.so.0"),
                "-o",
                str(root / "encoder.so"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
