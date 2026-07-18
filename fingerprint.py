"""Architecture fingerprinting for BO run partitioning."""

import hashlib
import importlib.util
import json
from pathlib import Path

FINGERPRINT_FIELDS = ("DEPTH", "ASPECT_RATIO", "HEAD_DIM", "WINDOW_PATTERN")


def get_fingerprint() -> str:
    """Return a stable short hash of train.py's declared structural constants."""
    train_path = Path(__file__).with_name("train.py")
    spec = importlib.util.spec_from_file_location("autoresearch_train_fingerprint", train_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import {train_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    values = tuple(getattr(module, field) for field in FINGERPRINT_FIELDS)
    canonical = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    print(get_fingerprint())
