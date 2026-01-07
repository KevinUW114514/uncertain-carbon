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
import pickle
from collections.abc import Mapping
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
    sample_cost_duration,
)
from manager import WORKFLOW_CONFIG  # noqa: E402
from utils.config import NUM_RESOURCES  # noqa: E402

from config import CONFIG  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

neg_hartmann6 = Hartmann(negate=True)

def log_sorted_samples(
    samples,
    log_path,
    feasible_only=True,
):
    """
    Log sorted samples to disk in a deterministic, human-readable format.

    This function is intentionally schema-agnostic: it does not assume
    any specific keys in the sample records and only formats what it receives.

    Args:
        samples (Iterable[Any]): Iterable of sample records (dict-like preferred).
        log_path (str | Path): File path to write logs to.
        feasible_only (bool): Whether the logged samples are feasibility-filtered.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    header = {
        "timestamp": datetime.utcnow().isoformat(),
        "num_samples": len(samples),
        "feasible_only": feasible_only,
    }

    with log_path.open("a", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("SORTED SAMPLE LOG\n")
        f.write(json.dumps(header, indent=2, sort_keys=True) + "\n")
        f.write("-" * 80 + "\n")

        for idx, sample in enumerate(samples):
            if isinstance(sample, Mapping):
                record = dict(sample)
            else:
                # Fallback: best-effort serialization for non-dict samples
                record = {"value": sample}

            # Always inject rank, but do not otherwise reshape the record
            record = {"rank": idx, **record}

            f.write(json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
            f.write("-" * 80 + "\n")


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
        # obj = obj_function(X).unsqueeze(-1)
        # con = outcome_constraint(X).unsqueeze(-1)
        obj, con, sample_data = sample_cost_duration(X=X)
        obj = obj * (-1.0)
        con = con - WORKFLOW_CONFIG["qos"]
        
    return obj, con, sample_data

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
    train_obj, train_con, sample_data = eval_obj_con(train_x)

    # best_obj, best_x, has_feasible = best_feasible(train_x, train_obj, train_con)
    best_obj = None
    return train_x, train_obj, train_con, best_obj, sample_data


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
    new_obj, new_con, sample_data = eval_obj_con(new_x)
    return new_x, new_obj, new_con, sample_data


def update_random_observations(best_random: list[float], batch_size: int):
    """
    Random baseline: track best FEASIBLE objective (=-cost) discovered by random samples so far.
    """
    x_dim = len(WORKFLOW_CONFIG["functions"]) * NUM_RESOURCES
    rand_x = torch.rand(batch_size, x_dim, device=device, dtype=dtype)
    rand_obj, rand_con, sample_data = eval_obj_con(rand_x)

    batch_best_obj, _, has_feasible = best_feasible(rand_x, rand_obj, rand_con)
    prev = best_random[-1]
    if not has_feasible:
        best_random.append(prev)
    else:
        best_random.append(max(prev, batch_best_obj))
    return best_random, sample_data


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
    suffix: str = "",
    n_init: int = 10,
    n_batch: int = 10,
    mc_samples: int = 64,
    batch_size: int = 3,
    num_restarts: int = 10,
    raw_samples: int = 100,
    infeasible_cost: float | None = None,  # if None, computed automatically from data
    anomaly_detection: bool = True,
    confidence: float = 0.95,  # retained for API compatibility (not used by MAD filter)
    verbose: bool = True,
    log_path: str = "bo_energy_log.log",
    *,
    # Checkpoint / resume controls
    save_model: bool = True,
    save_path: str = "botorch_energy_model.pt",
    resume_if_exists: bool = True,
):
    global WORKFLOW_CONFIG
    WORKFLOW_CONFIG = workflow_config

    n_stages = len(WORKFLOW_CONFIG["functions"])
    x_dim_expected = NUM_RESOURCES * n_stages

    bounds = torch.tensor(
        [[0.0] * x_dim_expected, [1.0] * x_dim_expected],
        device=device,
        dtype=dtype,
    )

    samples: list[dict] = []

    # ----------------------------
    # Resume from checkpoint if any
    # ----------------------------
    ckpt = None
    if resume_if_exists and os.path.exists(save_path):
        print("Resuming from existing checkpoint...")
        ckpt = torch.load(save_path, map_location="cpu")

        # Load training data
        train_x_nei = ckpt["train_x"].to(device=device, dtype=dtype)
        train_obj_nei = ckpt["train_obj"].to(device=device, dtype=dtype)
        train_con_nei = ckpt["train_con"].to(device=device, dtype=dtype)

        # Basic safety check: dimensionality must match current workflow
        if train_x_nei.shape[-1] != x_dim_expected:
            raise ValueError(
                f"Checkpoint x_dim={train_x_nei.shape[-1]} does not match "
                f"current expected x_dim={x_dim_expected}. "
                "This usually means WORKFLOW_CONFIG['functions'] or NUM_RESOURCES changed."
            )

        # Restore histories if present (optional)
        best_observed_nei: list[float] = ckpt.get("best_observed_nei", [])
        best_random: list[float] = ckpt.get("best_random", [])

        completed_batches: int = int(ckpt.get("completed_batches", 0))

        # Rebuild model and load state dict
        mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)
        if "model_state_dict" in ckpt:
            model_nei.load_state_dict(ckpt["model_state_dict"])

        if verbose:
            print(
                f"Resumed from checkpoint: {save_path}\n"
                f"  training_points = {train_x_nei.shape[0]}\n"
                f"  completed_batches = {completed_batches}\n"
                f"  continuing for additional_batches = {n_batch}"
            )

        start_iteration = completed_batches + 1
        end_iteration = completed_batches + n_batch

        # Ensure histories are initialized sensibly if missing/empty
        if not best_observed_nei:
            best_obj, _, _ = best_feasible(train_x_nei, train_obj_nei, train_con_nei)
            best_observed_nei = [best_obj]
        if not best_random:
            # You disabled update_random_observations in your loop anyway.
            best_random = [best_observed_nei[-1]]
            
        save_path = f"resume_{save_path}_{suffix}"
        
        # samples = 

    else:
        print("No checkpoint found, starting fresh BO run.")
        # ----------------------------
        # Fresh start: random initialization
        # ----------------------------
        best_observed_nei: list[float] = []
        best_random: list[float] = []
        completed_batches = 0
        
        start_time = time.time()

        train_x_nei, train_obj_nei, train_con_nei, _, sample_data = generate_initial_data(n=n_init)

        best_obj, best_x, has_feasible = best_feasible(
            train_x_nei, train_obj_nei, train_con_nei
        )
        best_observed_nei.append(best_obj)
        best_random.append(best_obj)

        mll_nei, model_nei = initialize_model(train_x_nei, train_obj_nei, train_con_nei)

        start_iteration = 1
        end_iteration = n_batch
        total_time = time.time() - start_time

        # for x, obj, con in zip(train_x_nei, train_obj_nei, train_con_nei):
        #     samples.append(
        #         {
        #             "cost": -float(obj.item()),
        #             "feasible": float(con.item()) <= 0.0,
        #             "constraint": float(con.item()),
        #             "resource_config": from_x_to_resource_config(x=x),
        #             "iteration": 0,
        #             # "timestamp": datetime.now().isoformat(),
        #         }
        #     )
        samples.extend(sample_data)


        s = f"Starting fresh (no checkpoint at {save_path}).\n" + \
            f"  initial_points = {n_init}\n" + \
            f"  running_batches = {n_batch}\n" + \
            f"  initial_best_feasible_cost = {-best_obj if has_feasible else 'NA'}\n" + \
            f"  initial_best_constraint = " + \
            (f"{float(outcome_constraint(best_x.unsqueeze(0)).item()):.3f}" if has_feasible else "NA") + "\n" + \
            f"  initial_best_resource_config = " + \
            (str(from_x_to_resource_config(x=best_x)) if has_feasible else "NA") + "\n" + \
            f"  time_taken_for_init = {total_time:.2f} seconds"
            
        if verbose:
            print(s)
        
        with open(log_path, "a") as f:
            f.write(s + "\n")
            f.write("=" * 80 + "\n")

    # ----------------------------
    # BO loop
    # ----------------------------
    for iteration in range(start_iteration, end_iteration + 1):
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

        new_x_nei, new_obj_nei, new_con_nei, sample_data = optimize_acqf_and_get_observation(
            acq_func=qNEI,
            bounds=bounds,
            batch_size=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
        train_con_nei = torch.cat([train_con_nei, new_con_nei])

        # Record newly evaluated samples
        # for x, obj, con in zip(new_x_nei, new_obj_nei, new_con_nei):
        #     cost = -float(obj.item())          # objective = -cost
        #     constraint = float(con.item())
        #     feasible = constraint <= 0.0

        #     samples.append(
        #         {
        #             "cost": cost,
        #             "feasible": feasible,
        #             "constraint": constraint,
        #             "resource_config": from_x_to_resource_config(x=x),
        #             "iteration": iteration,
        #             # "timestamp": datetime.now().isoformat(),
        #         }
        #     )
        samples.extend(sample_data)


        if anomaly_detection:
            train_x_nei, train_obj_nei, train_con_nei = robust_outlier_filter_mad(
                train_x_nei, train_obj_nei, train_con_nei, z_cut=8.0
            )

        best_obj, best_x, has_feasible = best_feasible(
            train_x_nei, train_obj_nei, train_con_nei
        )
        best_observed_nei.append(best_obj)

        # Warm-start reinit
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            train_con_nei,
            model_nei.state_dict(),
        )

        t1 = time.monotonic()

        # Reporting
        rand_best_obj = best_random[-1] if best_random else float("-inf")
        rand_best_cost = None if rand_best_obj == float("-inf") else -rand_best_obj
        nei_best_cost = None if best_obj == float("-inf") else -best_obj
        best_con_val = float(outcome_constraint(best_x.unsqueeze(0)).item())

        s = (
            f"\nBatch {iteration:>2}: best_feasible_cost (random, qNEI) = "
            f"({rand_best_cost if rand_best_cost is not None else 'NA'}, "
            f"{nei_best_cost if nei_best_cost is not None else 'NA'}), "
            f"best_constraint = {best_con_val:>7.3f} (<=0 feasible), "
            f"time = {t1 - t0:>4.2f}.\n"
        )
        
        if has_feasible:
            s += "Best feasible resource configuration: "
            s += str(from_x_to_resource_config(x=best_x))
            s += "\n"
        else:
            s += "No feasible solution found yet.\n"
            

        with open(log_path, "a") as f:
            f.write(s + "\n")
            f.write("=" * 80 + "\n")

        if verbose:
            print("=" * 80)
            print(s)
            print("=" * 80)
        else:
            print(".", end="")

        # ----------------------------
        # Save checkpoint after each batch (robust resume)
        # ----------------------------
        completed_batches = iteration
        if save_model:
            torch.save(
                {
                    "model_state_dict": model_nei.state_dict(),
                    "train_x": train_x_nei.detach().cpu(),
                    "train_obj": train_obj_nei.detach().cpu(),
                    "train_con": train_con_nei.detach().cpu(),
                    "best_observed_nei": best_observed_nei,
                    "best_random": best_random,
                    "completed_batches": completed_batches,
                    "dtype": str(dtype),
                    "device": str(device),
                    "num_resources": NUM_RESOURCES,
                    "functions": WORKFLOW_CONFIG.get("functions"),
                    "qos": WORKFLOW_CONFIG.get("qos"),
                    "workflow_config": WORKFLOW_CONFIG,  # remove if non-serializable
                    "saved_at": datetime.now().isoformat(),
                },
                save_path,
            )
            json.dump(samples, open(CONFIG.sample_path, "w"), indent=2)

    # ----------------------------
    # Final selection: same as your original logic
    # ----------------------------
    final_best_obj, final_best_x, final_has_feasible = best_feasible(
        train_x_nei, train_obj_nei, train_con_nei
    )

    if final_has_feasible:
        best_cost = -final_best_obj
    else:
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

    # Keep only feasible samples (recommended)
    feasible_samples = [s for s in samples if s["feasible"]]

    # Sort by cost ascending
    feasible_samples_sorted = sorted(feasible_samples, key=lambda s: s["cost"])
    log_sorted_samples(
        feasible_samples_sorted,
        log_path,
        feasible_only=True,
    )
    
    json.dump(samples, open(CONFIG.sample_path, "w"), indent=2)
    json.dump(feasible_samples_sorted, open(CONFIG.json_path, "w"), indent=2)

    return best_cost, resource_config

