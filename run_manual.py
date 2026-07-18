"""Run a hand-edited training attempt and append it to runs.csv."""

import sys

from fingerprint import get_fingerprint
from run_logging import append_run, run_training


def main(argv=None) -> int:
    train_args = sys.argv[1:] if argv is None else argv
    fingerprint = get_fingerprint()
    returncode, result, elapsed = run_training(train_args)
    row = append_run(fingerprint=fingerprint, source="agent", wall_clock_seconds=elapsed, result=result)
    print(f"Logged agent run {row['iteration']} (fingerprint {fingerprint}).")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
