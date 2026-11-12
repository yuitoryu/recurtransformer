"""Shared project paths, runtime environment and JSON output."""

import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_environment():
    """Keep library caches, compiler products and temporary files in the project."""
    cache = PROJECT_ROOT / ".cache"
    locations = {
        "TMPDIR": cache / "tmp", "TMP": cache / "tmp", "TEMP": cache / "tmp",
        "XDG_CACHE_HOME": cache,
        "MPLCONFIGDIR": cache / "matplotlib",
        "TORCH_HOME": cache / "torch",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TORCH_EXTENSIONS_DIR": cache / "torch_extensions",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
    }
    for key, path in locations.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tempfile.tempdir = str(locations["TMPDIR"])


def project_path(value):
    return (PROJECT_ROOT / Path(value)).resolve()


def write_json(path, value):
    with path.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


