import re
from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple


@dataclass(frozen=True)
class DecisionRecord:
    """
    One decision event parsed from a log line like:
    [t=76] hour=2985.0 decision=GB baseline=LT optimal=LT actual(algo)=7.03301e+07 actual(base)=6.06223e+07 increase=16.014%
    """
    t: int
    hour: float
    decision_algo: str
    decision_base: str
    decision_opt: str
    carbon_algo: float
    carbon_base: float


_DECISION_LINE_RE = re.compile(
    r"""
    \[t=(?P<t>\d+)\]\s+
    hour=(?P<hour>[-+]?\d*\.?\d+)\s+
    decision=(?P<decision_algo>\w+)\s+
    baseline=(?P<decision_base>\w+)\s+
    optimal=(?P<decision_opt>\w+)\s+
    actual\(algo\)=(?P<carbon_algo>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s+
    actual\(base\)=(?P<carbon_base>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)
    """,
    re.VERBOSE | re.IGNORECASE,
)


_TOTAL_RE = re.compile(
    r"TOTAL\s+actual\s+carbon:\s+algo=(?P<algo>[-+]?\d*\.?\d+(?:e[-+]?\d+)?),\s+baseline=(?P<base>[-+]?\d*\.?\d+(?:e[-+]?\d+)?)",
    re.IGNORECASE,
)


def parse_decision_records(log_text: str) -> List[DecisionRecord]:
    """
    Parse all decision records from raw log text.

    Returns:
        List[DecisionRecord]
    """
    records: List[DecisionRecord] = []
    for line in log_text.splitlines():
        line = line.strip()
        m = _DECISION_LINE_RE.search(line)
        if not m:
            continue
        records.append(
            DecisionRecord(
                t=int(m.group("t")),
                hour=float(m.group("hour")),
                decision_algo=m.group("decision_algo"),
                decision_base=m.group("decision_base"),
                decision_opt=m.group("decision_opt"),
                carbon_algo=float(m.group("carbon_algo")),
                carbon_base=float(m.group("carbon_base")),
            )
        )
    return records


def parse_total_line(log_text: str) -> Optional[Tuple[float, float]]:
    """
    Parse the TOTAL actual carbon summary line (if present).

    Returns:
        (total_algo, total_base) or None
    """
    m = _TOTAL_RE.search(log_text)
    if not m:
        return None
    return float(m.group("algo")), float(m.group("base"))


# -----------------------------
# Core helper computations
# -----------------------------

def _sum(values: Iterable[float]) -> float:
    s = 0.0
    for v in values:
        s += float(v)
    return s


def _filter(records: Iterable[DecisionRecord], predicate) -> List[DecisionRecord]:
    return [r for r in records if predicate(r)]


# -----------------------------
# Metrics
# -----------------------------

def net_savings_vs_baseline(records: List[DecisionRecord]) -> Dict[str, float]:
    """
    Net savings across the provided records.

    Note: This is over only the records you pass in. If your "TOTAL" includes
    many hours not in the records, use parse_total_line() for the full-run net.
    """
    base = _sum(r.carbon_base for r in records)
    algo = _sum(r.carbon_algo for r in records)
    avoided = base - algo
    pct = (avoided / base * 100.0) if base != 0 else 0.0
    return {
        "baseline_total": base,
        "algo_total": algo,
        "avoided_total": avoided,          # positive means improvement vs baseline
        "avoided_pct": pct,
        "n": float(len(records)),
    }


def conditional_savings_when_different(records: List[DecisionRecord]) -> Dict[str, float]:
    """
    Savings computed only on timesteps where algo decision != baseline decision.

    Returns:
        conditional_baseline_total
        conditional_algo_total
        conditional_avoided_total
        conditional_avoided_pct
        n_conditional
    """
    s = _filter(records, lambda r: r.decision_algo != r.decision_base)
    base = _sum(r.carbon_base for r in s)
    algo = _sum(r.carbon_algo for r in s)
    avoided = base - algo
    pct = (avoided / base * 100.0) if base != 0 else 0.0
    return {
        "conditional_baseline_total": base,
        "conditional_algo_total": algo,
        "conditional_avoided_total": avoided,
        "conditional_avoided_pct": pct,
        "n_conditional": float(len(s)),
    }
    
from typing import Dict, List


def optimal_match_counts(records: List[DecisionRecord]) -> Dict[str, float]:
    """
    Counts how often:
      - algorithm decision equals optimal
      - baseline decision equals optimal
    out of total records.

    Returns:
        total
        algo_optimal_count, algo_optimal_rate
        baseline_optimal_count, baseline_optimal_rate
        both_optimal_count, both_optimal_rate
        algo_only_optimal_count, baseline_only_optimal_count
        neither_optimal_count
    """
    total = len(records)
    if total == 0:
        return {
            "total": 0.0,
            "algo_optimal_count": 0.0,
            "algo_optimal_rate": 0.0,
            "baseline_optimal_count": 0.0,
            "baseline_optimal_rate": 0.0,
            "both_optimal_count": 0.0,
            "both_optimal_rate": 0.0,
            "algo_only_optimal_count": 0.0,
            "baseline_only_optimal_count": 0.0,
            "neither_optimal_count": 0.0,
        }

    algo_opt = 0
    base_opt = 0
    both_opt = 0
    algo_only = 0
    base_only = 0
    neither = 0

    for r in records:
        a = (r.decision_algo == r.decision_opt)
        b = (r.decision_base == r.decision_opt)

        if a:
            algo_opt += 1
        if b:
            base_opt += 1

        if a and b:
            both_opt += 1
        elif a and not b:
            algo_only += 1
        elif b and not a:
            base_only += 1
        else:
            neither += 1

    return {
        "total": float(total),

        "algo_optimal_count": float(algo_opt),
        "algo_optimal_rate": float(algo_opt) / total,

        "baseline_optimal_count": float(base_opt),
        "baseline_optimal_rate": float(base_opt) / total,

        "both_optimal_count": float(both_opt),
        "both_optimal_rate": float(both_opt) / total,

        "algo_only_optimal_count": float(algo_only),
        "baseline_only_optimal_count": float(base_only),
        "neither_optimal_count": float(neither),
    }


def optimal_match_breakdown(records: List[DecisionRecord]) -> Dict[str, Dict[str, float]]:
    """
    A slightly richer breakdown: counts/rates by optimal class.

    Example output keys:
      result["optimal=GB"]["algo_match_rate"]
      result["optimal=LT"]["baseline_match_count"]

    Useful if optimal is imbalanced (e.g., optimal is almost always LT).
    """
    totals: Dict[str, int] = {}
    algo_match: Dict[str, int] = {}
    base_match: Dict[str, int] = {}

    for r in records:
        opt = r.decision_opt
        totals[opt] = totals.get(opt, 0) + 1
        if r.decision_algo == opt:
            algo_match[opt] = algo_match.get(opt, 0) + 1
        if r.decision_base == opt:
            base_match[opt] = base_match.get(opt, 0) + 1

    out: Dict[str, Dict[str, float]] = {}
    for opt, n in totals.items():
        a = algo_match.get(opt, 0)
        b = base_match.get(opt, 0)
        out[f"optimal={opt}"] = {
            "total": float(n),
            "algo_match_count": float(a),
            "algo_match_rate": float(a) / n if n else 0.0,
            "baseline_match_count": float(b),
            "baseline_match_rate": float(b) / n if n else 0.0,
        }
    return out



def gross_savings_and_penalties(records: List[DecisionRecord]) -> Dict[str, float]:
    """
    Decompose into:
      - gross_savings: sum(max(base - algo, 0))
      - gross_penalty: sum(max(algo - base, 0))
      - net_avoided: gross_savings - gross_penalty
      - savings_to_penalty_ratio: gross_savings / gross_penalty (inf if penalty=0)
      - win_rate_count: wins / n
    """
    diffs = [(r.carbon_base - r.carbon_algo) for r in records]
    gross_savings = _sum(d for d in diffs if d > 0)
    gross_penalty = _sum(-d for d in diffs if d < 0)
    net = gross_savings - gross_penalty
    ratio = (gross_savings / gross_penalty) if gross_penalty != 0 else float("inf")
    wins = sum(1 for d in diffs if d > 0)
    return {
        "gross_savings": gross_savings,
        "gross_penalty": gross_penalty,
        "net_avoided": net,
        "savings_to_penalty_ratio": ratio,
        "win_rate_count": (wins / len(records)) if records else 0.0,
        "n": float(len(records)),
    }


def impact_weighted_win_share(records: List[DecisionRecord]) -> Dict[str, float]:
    """
    Impact-weighted win share:
        sum(wins magnitudes) / sum(|diff|)
    where diff = base - algo.

    Interpretation:
      0.5 => wins and losses cancel in magnitude.
      >0.5 => more magnitude on the win side.
    """
    diffs = [(r.carbon_base - r.carbon_algo) for r in records]
    win_mag = _sum(d for d in diffs if d > 0)
    total_mag = _sum(abs(d) for d in diffs)
    share = (win_mag / total_mag) if total_mag != 0 else 0.0
    return {
        "impact_weighted_win_share": share,
        "win_magnitude": win_mag,
        "total_magnitude": total_mag,
        "n": float(len(records)),
    }


def regret_vs_optimal(
    records: List[DecisionRecord],
    carbon_optimal_by_t: Dict[int, float],
) -> Dict[str, float]:
    """
    Compute regret vs optimal using a provided mapping of optimal carbon per t.

    You must supply carbon_optimal_by_t[t] = carbon if the optimal action had been taken at t.

    Returns:
        total_regret: sum(algo - opt)
        mean_regret
        median_regret
        mean_regret_ratio: mean( (algo-opt)/opt ) over t with opt>0
    """
    regrets: List[float] = []
    ratios: List[float] = []
    for r in records:
        if r.t not in carbon_optimal_by_t:
            continue
        opt = float(carbon_optimal_by_t[r.t])
        reg = r.carbon_algo - opt
        regrets.append(reg)
        if opt > 0:
            ratios.append(reg / opt)

    if not regrets:
        return {
            "total_regret": 0.0,
            "mean_regret": 0.0,
            "median_regret": 0.0,
            "mean_regret_ratio": 0.0,
            "n_with_opt": 0.0,
        }

    regrets_sorted = sorted(regrets)
    mid = len(regrets_sorted) // 2
    if len(regrets_sorted) % 2 == 1:
        median = regrets_sorted[mid]
    else:
        median = 0.5 * (regrets_sorted[mid - 1] + regrets_sorted[mid])

    return {
        "total_regret": _sum(regrets),
        "mean_regret": _sum(regrets) / len(regrets),
        "median_regret": float(median),
        "mean_regret_ratio": (_sum(ratios) / len(ratios)) if ratios else 0.0,
        "n_with_opt": float(len(regrets)),
    }


def tail_slice_savings_by_baseline(
    records: List[DecisionRecord],
    top_fraction: float = 0.10,
) -> Dict[str, float]:
    """
    Savings computed over the highest-baseline-carbon slice.

    Example: top_fraction=0.05 means top 5% timesteps by carbon_base.

    Returns:
        slice_baseline_total
        slice_algo_total
        slice_avoided_total
        slice_avoided_pct
        n_slice
        baseline_threshold (minimum baseline carbon in slice)
    """
    if not records:
        return {
            "slice_baseline_total": 0.0,
            "slice_algo_total": 0.0,
            "slice_avoided_total": 0.0,
            "slice_avoided_pct": 0.0,
            "n_slice": 0.0,
            "baseline_threshold": 0.0,
        }

    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")

    sorted_recs = sorted(records, key=lambda r: r.carbon_base, reverse=True)
    k = max(1, int(round(len(sorted_recs) * top_fraction)))
    s = sorted_recs[:k]

    base = _sum(r.carbon_base for r in s)
    algo = _sum(r.carbon_algo for r in s)
    avoided = base - algo
    pct = (avoided / base * 100.0) if base != 0 else 0.0
    threshold = min(r.carbon_base for r in s)

    return {
        "slice_baseline_total": base,
        "slice_algo_total": algo,
        "slice_avoided_total": avoided,
        "slice_avoided_pct": pct,
        "n_slice": float(len(s)),
        "baseline_threshold": float(threshold),
    }


def savings_per_switch(records: List[DecisionRecord]) -> Dict[str, float]:
    """
    Compute avoided carbon per 'switch' event (algo decision != baseline decision).
    """
    s = _filter(records, lambda r: r.decision_algo != r.decision_base)
    if not s:
        return {"avoided_per_switch": 0.0, "n_switches": 0.0, "total_avoided_on_switches": 0.0}

    avoided_total = _sum((r.carbon_base - r.carbon_algo) for r in s)
    return {
        "avoided_per_switch": avoided_total / len(s),
        "n_switches": float(len(s)),
        "total_avoided_on_switches": avoided_total,
    }


# -----------------------------
# Convenience: one-stop summary
# -----------------------------

def summarize_benefits(
    log_text: str,
    top_fraction: float = 0.10,
) -> Dict[str, Dict[str, float]]:
    """
    Parse records from log_text and return a compact suite of benefit metrics.

    This does NOT compute regret vs optimal (needs optimal-carbon inputs).
    """
    records = parse_decision_records(log_text)

    out = {
        "net_vs_baseline_on_parsed_records": net_savings_vs_baseline(records),
        "conditional_when_different": conditional_savings_when_different(records),
        "gross_savings_penalties": gross_savings_and_penalties(records),
        "impact_weighted_win_share": impact_weighted_win_share(records),
        "tail_slice_top_fraction": tail_slice_savings_by_baseline(records, top_fraction=top_fraction),
        "savings_per_switch": savings_per_switch(records),
    }

    totals = parse_total_line(log_text)
    if totals is not None:
        total_algo, total_base = totals
        avoided = total_base - total_algo
        pct = (avoided / total_base * 100.0) if total_base != 0 else 0.0
        out["net_vs_baseline_total_line"] = {
            "baseline_total": total_base,
            "algo_total": total_algo,
            "avoided_total": avoided,
            "avoided_pct": pct,
        }
        
    out["counts"] = optimal_match_counts(records)
    out["by_opt"] = optimal_match_breakdown(records)

    return out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    with open("result.log", "r") as f:
        sample_log = f.read()

    summary = summarize_benefits(sample_log, top_fraction=0.10)
    # Print a readable view
    import json
    print(json.dumps(summary, indent=2))
