import gevent  # isort:skip
from gevent import monkey  # isort:skip

monkey.patch_all()  # isort:skip

import argparse
import json
import logging
import os
import signal
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import scipy
import torch
from botorch import fit_gpytorch_mll
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models import ModelListGP, SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions import Hartmann
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

PROJECT_DIR = Path(__file__).resolve().parents[2]
SCHED_DIR = Path(__file__).resolve().parents[0]
sys.path.append(str(PROJECT_DIR))
sys.path.append(str(SCHED_DIR))

from bo_utils import (  # noqa: E402
    from_x_to_resource_config,
    sample_cost_parallel,
    sample_duration_parallel,
)
from manager import WORKFLOW_CONFIG  # noqa: E402
from utils.config import NUM_RESOURCES  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

neg_hartmann6 = Hartmann(negate=True)


def obj_function(X: torch.Tensor) -> torch.Tensor:
    # Objective is MAXIMIZED by BoTorch. We want to MINIMIZE cost => maximize (-cost).
    return sample_cost_parallel(X=X) * (-1.0)


def outcome_constraint(X: torch.Tensor) -> torch.Tensor:
    """
    Constraint is satisfied when <= 0.
    Here: duration - qos <= 0  <=> duration <= qos.
    """
    return sample_duration_parallel(X=X) - WORKFLOW_CONFIG["qos"]


def obj_callable(Z: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
    # Z[..., 0] corresponds to objective GP output
    return Z[..., 0]


def constraint_callable(Z: torch.Tensor, X: torch.Tensor | None = None) -> torch.Tensor:
    # Z[..., 1] corresponds to constraint GP output
    return Z[..., 1]


def eval_obj_con(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate objective and constraint with no gradients; return shape (n, 1) tensors."""
    with torch.no_grad():
        obj = obj_function(X).unsqueeze(-1)
        con = outcome_constraint(X).unsqueeze(-1)
    return obj, con


def best_feasible(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: torch.Tensor,
    tol: float = 0.0,
) -> tuple[float, torch.Tensor, bool]:
    """
    Returns:
      best_obj_value: max objective among feasible points (objective = -cost); -inf if none feasible
      best_x: location of best feasible (or least-violating point if none feasible)
      has_feasible: whether at least one feasible point exists
    """
    feas = (train_con.squeeze(-1) <= tol)
    if feas.any():
        feas_obj = train_obj[feas].squeeze(-1)
        i = torch.argmax(feas_obj).item()
        return feas_obj[i].item(), train_x[feas][i], True

    # No feasible yet: pick least-violating (smallest positive constraint violation)
    vio = train_con.squeeze(-1).clamp_min(0.0)
    i = torch.argmin(vio).item()
    return float("-inf"), train_x[i], False


def compute_infeasible_cost_shift(train_obj: torch.Tensor) -> float:
    """
    For ConstrainedMCObjective with potentially negative objectives (e.g., objective = -cost),
    provide a shift so that (objective + shift) is nonnegative.
    """
    min_obj = float(train_obj.min().item())
    return max(0.0, -min_obj + 1e-6)


def generate_initial_data(n: int = 10):
    x_dim = len(WORKFLOW_CONFIG["functions"]) * NUM_RESOURCES
    train_x = torch.rand(n, x_dim, device=device, dtype=dtype)
    train_obj, train_con = eval_obj_con(train_x)

    best_obj, best_x, has_feasible = best_feasible(train_x, train_obj, train_con)
    return train_x, train_obj, train_con, best_obj


def initialize_model(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: torch.Tensor,
    state_dict=None,
):
    model_obj = SingleTaskGP(train_x, train_obj).to(train_x)
    model_con = SingleTaskGP(train_x, train_con).to(train_x)
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(
    acq_func,
    bounds: torch.Tensor,
    batch_size: int,
    num_restarts: int,
    raw_samples: int,
):
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={"batch_limit": 5, "maxiter": 200},
    )
    new_x = candidates.detach()
    new_obj, new_con = eval_obj_con(new_x)
    return new_x, new_obj, new_con


def update_random_observations(best_random: list[float], batch_size: int):
    """
    Random baseline: track best FEASIBLE objective (=-cost) discovered by random samples so far.
    """
    x_dim = len(WORKFLOW_CONFIG["functions"]) * NUM_RESOURCES
    rand_x = torch.rand(batch_size, x_dim, device=device, dtype=dtype)
    rand_obj, rand_con = eval_obj_con(rand_x)

    batch_best_obj, _, has_feasible = best_feasible(rand_x, rand_obj, rand_con)
    prev = best_random[-1]
    if not has_feasible:
        best_random.append(prev)
    else:
        best_random.append(max(prev, batch_best_obj))
    return best_random


def robust_outlier_filter_mad(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    train_con: torch.Tensor,
    z_cut: float = 8.0,
):
    """
    Conservative outlier filter on objective using robust MAD z-scores.
    It never removes the current best feasible point.
    """
    n = train_obj.shape[0]
    if n < 20:
        return train_x, train_obj, train_con

    y = train_obj.squeeze(-1)
    med = y.median()
    mad = (y - med).abs().median().clamp_min(1e-12)
    z = 0.6745 * (y - med) / mad

    keep = z.abs() <= z_cut

    # Always keep the best feasible point (if any)
    best_obj, best_x, has_feasible = best_feasible(train_x, train_obj, train_con)
    if has_feasible:
        # find matching index (exact match on tensor row)
        # fall back safely if not found
        matches = (train_x == best_x).all(dim=-1)
        if matches.any():
            keep[matches.nonzero(as_tuple=False).squeeze(-1)[0]] = True

    if keep.all():
        return train_x, train_obj, train_con

    return train_x[keep], train_obj[keep], train_con[keep]


def bo_loop(
    workflow_config,
    n_init: int = 10,
    n_batch: int = 20,
    mc_samples: int = 32,
    batch_size: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 32,
    infeasible_cost: float | None = None,  # if None, computed automatically from data
    anomaly_detection: bool = True,
    confidence: float = 0.95,  # retained for API compatibility (not used by MAD filter)
    verbose: bool = True,
):
    global WORKFLOW_CONFIG
    WORKFLOW_CONFIG = workflow_config

    n_stages = len(WORKFLOW_CONFIG["functions"])
    bounds = torch.tensor(
        [[0.0] * NUM_RESOURCES * n_stages, [1.0] * NUM_RESOURCES * n_stages],
        device=device,
        dtype=dtype,
    )

    best_observed_nei: list[float] = []
    best_random: list[float] = []

    train_x_nei, train_obj_nei, train_con_nei, init_best_obj = generate_initial_data(
        n=n_init
    )

    # Track best FEASIBLE objective; -inf if none feasible yet.
    best_obj, best_x, has_feasible = best_feasible(
        train_x_nei, train_obj_nei, train_con_nei
    )
    best_observed_nei.append(best_obj)

    # Random baseline starts at same initial budget; best feasible among initial points
    best_random.append(best_obj)

    mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)

    for iteration in range(1, n_batch + 1):
        t0 = time.monotonic()

        fit_gpytorch_mll(mll_nei)

        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))

        # Compute an infeasible-cost SHIFT appropriate for negative objectives (objective = -cost).
        if infeasible_cost is None:
            infeasible_cost_shift = compute_infeasible_cost_shift(train_obj_nei)
        else:
            infeasible_cost_shift = float(infeasible_cost)

        constrained_obj = ConstrainedMCObjective(
            objective=obj_callable,
            constraints=[constraint_callable],
            infeasible_cost=infeasible_cost_shift,
        )

        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
            objective=constrained_obj,
        )

        new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(
            acq_func=qNEI,
            bounds=bounds,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
        train_con_nei = torch.cat([train_con_nei, new_con_nei])

        if anomaly_detection:
            train_x_nei, train_obj_nei, train_con_nei = robust_outlier_filter_mad(
                train_x_nei, train_obj_nei, train_con_nei, z_cut=8.0
            )

        best_random = update_random_observations(best_random, batch_size)

        best_obj, best_x, has_feasible = best_feasible(
            train_x_nei, train_obj_nei, train_con_nei
        )
        best_observed_nei.append(best_obj)

        # Reinitialize with warm-start state dict
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            train_con_nei,
            model_nei.state_dict(),
        )

        t1 = time.monotonic()

        # Report best feasible costs (if any)
        rand_best_obj = best_random[-1]
        rand_best_cost = None if rand_best_obj == float("-inf") else -rand_best_obj
        nei_best_cost = None if best_obj == float("-inf") else -best_obj

        # Also report feasibility of the current best_x we are tracking
        best_con_val = float(outcome_constraint(best_x.unsqueeze(0)).item())

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_feasible_cost (random, qNEI) = "
                f"({rand_best_cost if rand_best_cost is not None else 'NA'}, "
                f"{nei_best_cost if nei_best_cost is not None else 'NA'}), "
                f"best_constraint = {best_con_val:>7.3f} (<=0 feasible), "
                f"time = {t1 - t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

    # Final selection: prefer best feasible; if none feasible, return least-violating point.
    final_best_obj, final_best_x, final_has_feasible = best_feasible(
        train_x_nei, train_obj_nei, train_con_nei
    )

    if final_has_feasible:
        best_cost = -final_best_obj
    else:
        # Fallback: least violating point
        vio = train_con_nei.squeeze(-1).clamp_min(0.0)
        i = torch.argmin(vio).item()
        best_cost = -float(train_obj_nei[i].item())
        final_best_x = train_x_nei[i]
        if verbose:
            print(
                "\nWARNING: No feasible solution found within the BO budget. "
                f"Returning least-violating solution with violation={float(vio[i].item()):.3f}."
            )

    resource_config = from_x_to_resource_config(x=final_best_x)
    return best_cost, resource_config
