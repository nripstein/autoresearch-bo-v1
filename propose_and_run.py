"""Propose and execute one same-architecture Bayesian optimization run."""

import math

import torch

from fingerprint import get_fingerprint
from run_logging import append_run, load_successful_runs, run_training

# The current defaults (0.04, 0.2) lie well inside these nanochat-scale ranges.
LR_BOUNDS = (0.005, 0.10)
WEIGHT_DECAY_BOUNDS = (0.0, 0.5)
MIN_PRIOR_OBS = 4
UCB_BETA = 0.2


def _bounds() -> torch.Tensor:
    return torch.tensor([[LR_BOUNDS[0], WEIGHT_DECAY_BOUNDS[0]],
                         [LR_BOUNDS[1], WEIGHT_DECAY_BOUNDS[1]]], dtype=torch.double)


def sobol_candidate() -> tuple[float, float]:
    unit = torch.quasirandom.SobolEngine(dimension=2, scramble=True).draw(1).double()
    point = _bounds()[0] + unit[0] * (_bounds()[1] - _bounds()[0])
    return float(point[0]), float(point[1])


def bo_candidate(rows: list[dict]) -> tuple[float, float]:
    """Fit a GP for negative BPB and maximize standard UCB over normalized inputs."""
    from botorch.acquisition import UpperConfidenceBound
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Normalize, Standardize
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    bounds = _bounds()
    train_x = torch.tensor([[row["lr"], row["weight_decay"]] for row in rows], dtype=torch.double)
    train_y = -torch.tensor([[row["val_bpb"]] for row in rows], dtype=torch.double)
    model = SingleTaskGP(train_x, train_y, input_transform=Normalize(d=2, bounds=bounds),
                         outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))
    acquisition = UpperConfidenceBound(model, beta=UCB_BETA)
    candidate, _ = optimize_acqf(acquisition, bounds=bounds, q=1, num_restarts=10,
                                 raw_samples=128)
    return float(candidate[0, 0]), float(candidate[0, 1])


def matches_requested(result: dict, lr: float, weight_decay: float) -> bool:
    return (math.isclose(result["lr"], lr, rel_tol=1e-9, abs_tol=1e-12)
            and math.isclose(result["weight_decay"], weight_decay, rel_tol=1e-9, abs_tol=1e-12))


def main() -> int:
    fingerprint = get_fingerprint()
    prior = [row for row in load_successful_runs() if row["fingerprint"] == fingerprint]
    cold_start = len(prior) < MIN_PRIOR_OBS
    lr, weight_decay = sobol_candidate() if cold_start else bo_candidate(prior)
    mode = "Sobol cold start" if cold_start else "GP-UCB warm start"
    print(f"{mode}; {len(prior)} prior observation(s), fingerprint={fingerprint}")
    print(f"Requested lr={lr:.10g}, weight_decay={weight_decay:.10g}")

    returncode, result, elapsed = run_training(["--lr", repr(lr), "--weight-decay", repr(weight_decay)])
    if result is not None and not matches_requested(result, lr, weight_decay):
        raise RuntimeError(
            "Protected CLI contract violation: train.py reported "
            f"lr={result['lr']}, weight_decay={result['weight_decay']}; requested "
            f"lr={lr}, weight_decay={weight_decay}. No tool row was logged."
        )
    if returncode != 0 or result is None:
        row = append_run(fingerprint=fingerprint, source="tool", wall_clock_seconds=elapsed, result=None,
                         requested_lr=lr, requested_weight_decay=weight_decay,
                         n_prior_obs=len(prior), cold_start=cold_start)
        print(f"Training failed; logged incomplete tool run {row['iteration']}.")
        return returncode if returncode != 0 else 1

    row = append_run(fingerprint=fingerprint, source="tool", wall_clock_seconds=elapsed, result=result,
                     requested_lr=lr, requested_weight_decay=weight_decay,
                     n_prior_obs=len(prior), cold_start=cold_start)
    previous_best = min((observation["val_bpb"] for observation in prior), default=None)
    improved = previous_best is None or result["val_bpb"] < previous_best
    status = "improved" if improved else "did not improve"
    previous = "n/a" if previous_best is None else f"{previous_best:.6f}"
    print(f"Logged tool run {row['iteration']}: val_bpb={result['val_bpb']:.6f}; "
          f"best prior={previous}; {status}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
