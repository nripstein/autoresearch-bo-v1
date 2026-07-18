"""Shared subprocess execution and append-only run-log helpers."""

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RESULT_PREFIX = "AUTORESEARCH_RESULT="
RUNS_PATH = Path(__file__).with_name("runs.csv")
RUN_FIELDS = [
    "timestamp", "iteration", "fingerprint", "lr", "weight_decay", "val_bpb",
    "wall_clock_seconds", "source", "requested_lr", "requested_weight_decay",
    "n_prior_obs", "cold_start",
]


def parse_result_line(line: str) -> dict | None:
    """Parse a train.py structured result line, returning None for other output."""
    if not line.startswith(RESULT_PREFIX):
        return None
    payload = json.loads(line[len(RESULT_PREFIX):])
    required = {"lr", "weight_decay", "val_bpb"}
    if not required <= payload.keys():
        raise ValueError(f"Training result is missing keys: {required - payload.keys()}")
    return {key: float(payload[key]) for key in required}


def run_training(train_args: list[str]) -> tuple[int, dict | None, float]:
    """Run train.py, mirror its output, and return exit code/result/wall time."""
    command = [sys.executable, str(Path(__file__).with_name("train.py")), *train_args]
    started = time.monotonic()
    result = None
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            parsed = parse_result_line(line.rstrip("\n"))
            if parsed is not None:
                result = parsed
        returncode = process.wait()
    return returncode, result, time.monotonic() - started


def append_run(*, fingerprint: str, source: str, wall_clock_seconds: float,
               result: dict | None, requested_lr: float | None = None,
               requested_weight_decay: float | None = None,
               n_prior_obs: int | None = None, cold_start: bool | None = None,
               path: Path = RUNS_PATH) -> dict:
    """Append one row while holding an advisory lock and allocating its iteration."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a+", newline="") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        rows = list(csv.DictReader(handle))
        iteration = max((int(row["iteration"]) for row in rows if row["iteration"]), default=0) + 1
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "iteration": iteration,
            "fingerprint": fingerprint,
            "lr": "" if result is None else result["lr"],
            "weight_decay": "" if result is None else result["weight_decay"],
            "val_bpb": "" if result is None else result["val_bpb"],
            "wall_clock_seconds": f"{wall_clock_seconds:.3f}",
            "source": source,
            "requested_lr": "" if requested_lr is None else requested_lr,
            "requested_weight_decay": "" if requested_weight_decay is None else requested_weight_decay,
            "n_prior_obs": "" if n_prior_obs is None else n_prior_obs,
            "cold_start": "" if cold_start is None else str(cold_start).lower(),
        }
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return row


def load_successful_runs(path: Path = RUNS_PATH) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    successful = []
    for row in rows:
        try:
            row["lr"] = float(row["lr"])
            row["weight_decay"] = float(row["weight_decay"])
            row["val_bpb"] = float(row["val_bpb"])
        except (KeyError, TypeError, ValueError):
            continue
        successful.append(row)
    return successful
