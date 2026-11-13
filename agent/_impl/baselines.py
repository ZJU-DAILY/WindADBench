from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DIMS = ["accuracy", "earliness", "reliability", "cost"]
_M2B_CACHE: Dict[int, tuple] = {}


@dataclass(frozen=True)
class Profile:
    weights: Dict[str, float]
    no_gpu: bool = False
    max_fa_day: Optional[float] = None
    max_model_size: Optional[float] = None
    min_earliness: Optional[float] = None
    max_infer_time: Optional[float] = None
    risk_aversion: float = 0.5
    min_support: float = 4.0
    abstain_gap: float = 0.02


@dataclass
class KB:
    dim_pct: pd.DataFrame
    coords: pd.DataFrame
    registry: pd.DataFrame
    meta: pd.DataFrame
    wide: pd.DataFrame


def _normalise_weights(weights: Dict[str, float]) -> pd.Series:
    w = pd.Series({d: max(0.0, float(weights.get(d, 0.0))) for d in DIMS})
    if w.sum() <= 0:
        w[:] = 1.0 / len(DIMS)
    else:
        w /= w.sum()
    return w


def _support_weights(kb: KB, workloads: Sequence[str]) -> pd.Series:
    meta = kb.meta.set_index("workload").reindex(list(workloads))

    q = np.log1p(meta["n_events"].astype(float).clip(lower=1.0))
    return q / q.sum()


def utility_stats(kb: KB, weights: Dict[str, float], evidence: Sequence[str],
                  models: Optional[Sequence[str]] = None) -> pd.DataFrame:
    evidence = list(evidence)
    d = kb.dim_pct[kb.dim_pct["workload"].isin(evidence)]
    if models is not None:
        d = d[d["model"].isin(models)]
    q = _support_weights(kb, evidence)
    w = _normalise_weights(weights)
    model_names = sorted(d["model"].unique())
    if not model_names:
        return pd.DataFrame(columns=["utility_mean", "utility_se", "support",
                                     *[f"mean_{name}" for name in DIMS],
                                     "utility_lcb"])
    values = np.stack([
        d.pivot(index="model", columns="workload", values=dim)
        .reindex(index=model_names, columns=evidence).to_numpy(float)
        for dim in DIMS
    ], axis=2)
    valid = np.isfinite(values).all(axis=2)
    values = np.nan_to_num(values, nan=0.0)
    weight_matrix = np.where(valid, q.loc[evidence].to_numpy()[None, :], 0.0)
    weight_matrix /= np.maximum(weight_matrix.sum(axis=1, keepdims=True), 1e-12)
    means = np.sum(values * weight_matrix[:, :, None], axis=1)
    variances = np.sum(weight_matrix[:, :, None] *
                       (values - means[:, None, :]) ** 2, axis=1)
    n_eff = 1.0 / np.maximum(np.sum(weight_matrix ** 2, axis=1), 1e-12)
    ses = np.sqrt(variances / np.maximum(n_eff[:, None], 1.0))
    utilities = means @ w.to_numpy()
    uncertainties = np.sqrt(np.sum((ses * w.to_numpy()[None, :]) ** 2, axis=1))
    support_total = float(kb.meta.set_index("workload").loc[evidence,
                                                              "n_events"].sum())
    out = pd.DataFrame({"utility_mean": utilities, "utility_se": uncertainties,
                        "support": support_total}, index=model_names)
    for pos, dim in enumerate(DIMS):
        out[f"mean_{dim}"] = means[:, pos]
    out["utility_lcb"] = out["utility_mean"] - out["utility_se"]
    return out.sort_values("utility_lcb", ascending=False)


def utility(kb: KB, weights: Dict[str, float], evidence: Sequence[str],
            models: Optional[Sequence[str]] = None) -> pd.Series:
    return utility_stats(kb, weights, evidence, models)["utility_mean"].sort_values(
        ascending=False)


def feasible_models(kb: KB, profile: Profile, evidence: Sequence[str]) -> List[str]:
    reg = kb.registry.set_index("model")
    ev = kb.wide[kb.wide["workload"].isin(list(evidence))]
    fa = ev.groupby("model")["false_alarms_per_turbine_day"].mean()
    early = ev.groupby("model")["early_detection_rate"].mean()
    ok: List[str] = []
    for model in reg.index:
        if profile.no_gpu and bool(reg.loc[model, "needs_gpu"]):
            continue
        if profile.max_fa_day is not None and fa.get(model, np.inf) > profile.max_fa_day:
            continue
        if (profile.max_model_size is not None
                and reg.loc[model, "model_size"] > profile.max_model_size):
            continue
        if (profile.min_earliness is not None
                and early.get(model, -np.inf) < profile.min_earliness):
            continue
        if (profile.max_infer_time is not None
                and reg.loc[model, "infer_time"] > profile.max_infer_time):
            continue
        ok.append(model)
    return ok


def pareto_front(kb: KB, evidence: Sequence[str], models: Sequence[str]) -> List[str]:
    stats = utility_stats(kb, {d: 0.25 for d in DIMS}, evidence, models)
    cols = [f"mean_{d}" for d in DIMS]
    vals = stats[cols].to_numpy()
    front: List[str] = []
    for i, model in enumerate(stats.index):
        dominated = ((vals >= vals[i]).all(axis=1)
                     & (vals > vals[i]).any(axis=1)).any()
        if not dominated:
            front.append(model)
    return front


def failure_flags(kb: KB, model: str) -> List[str]:
    g = kb.wide[kb.wide["model"] == model]
    flags: List[str] = []
    if_g = g[g["track"] == "in-farm"]
    if (not if_g.empty and if_g["event_recall"].mean() >= 0.8
            and if_g["point_recall"].mean() <= 0.01
            and if_g["early_detection_rate"].mean() <= 0.05):
        flags.append("single-spike mirage")
    if not if_g.empty and if_g["false_alarms_per_turbine_day"].mean() >= 5:
        flags.append("alarm saturation")
    cf = g[g["track"] == "cross-farm"]
    if not cf.empty and (cf["acc"] < 0.5).any():
        flags.append("calibration collapse (cross-farm)")
    return flags


def recommend(kb: KB, profile: Profile, evidence: Sequence[str], k: int = 3,
              allow_abstain: bool = True) -> Dict:
    feasible = feasible_models(kb, profile, evidence)
    stats = utility_stats(kb, profile.weights, evidence, feasible)
    if stats.empty:
        return {"decision": "abstain", "primary": None, "top_k": [],
                "pareto_front": [], "n_feasible": 0, "excluded": sorted(
                    set(kb.registry["model"])), "flags": {},
                "confidence": "none", "margin": 0.0}
    stats["utility_lcb"] = stats["utility_mean"] - profile.risk_aversion * stats["utility_se"]
    stats = stats.sort_values(["utility_lcb", "utility_mean"], ascending=False)
    top = stats.index[:k].tolist()
    margin = float(stats.iloc[0]["utility_lcb"] - stats.iloc[1]["utility_lcb"]
                   if len(stats) > 1 else stats.iloc[0]["utility_se"])
    support = float(stats.iloc[0]["support"])
    evidence_meta = kb.meta[kb.meta["workload"].isin(list(evidence))]
    sparse_transfer = bool(((evidence_meta["track"] == "cross-farm")
                            & (evidence_meta["n_events"] <= 6)).any())
    confidence = ("medium" if sparse_transfer or stats.iloc[0]["utility_se"] > 0.06
                  else "high" if support >= 24 else "low")
    abstain = (allow_abstain and
               (support < profile.min_support or margin < profile.abstain_gap))
    d = stats[[f"mean_{dim}" for dim in DIMS]].rename(
        columns={f"mean_{dim}": dim for dim in DIMS})
    return {
        "decision": "abstain" if abstain else "recommend",
        "primary": top[0], "top_k": top,
        "pareto_front": pareto_front(kb, evidence, feasible),
        "low_cost_fallback": d["cost"].idxmax(),
        "high_sensitivity_fallback": d["earliness"].idxmax(),
        "n_feasible": len(feasible),
        "excluded": sorted(set(kb.registry["model"]) - set(feasible)),
        "flags": {model: failure_flags(kb, model) for model in top},
        "confidence": confidence, "margin": margin,
        "uncertainty": float(stats.iloc[0]["utility_se"]),
        "support": support,
    }


def _m2b_arrays(kb: KB, models: Sequence[str]):
    workloads = kb.meta["workload"].tolist()
    arrays = np.stack([
        kb.dim_pct[kb.dim_pct["model"].isin(list(models))]
        .pivot(index="model", columns="workload", values=dim)
        .reindex(index=list(models), columns=workloads).to_numpy(float)
        for dim in DIMS
    ], axis=2)
    return workloads, arrays


def _cv_completion_error(values: np.ndarray, references: Sequence[int],
                         selected: Sequence[int], k: int = 3) -> float:
    rest = [i for i in range(values.shape[1]) if i not in selected]
    if not rest:
        return 0.0
    ref = np.asarray(list(references), dtype=int)
    if selected:
        observed = values[ref][:, selected, :]
        distance = np.abs(observed[:, None, :, :] -
                          observed[None, :, :, :]).mean(axis=(2, 3))
    else:
        distance = np.zeros((len(ref), len(ref)))
    np.fill_diagonal(distance, np.inf)
    errors: List[float] = []
    for row, idx in enumerate(ref):
        neighbours = ref[np.argsort(distance[row])[:k]]
        prediction = values[neighbours][:, rest, :].mean(axis=0)
        errors.append(float(np.abs(prediction - values[idx, rest, :]).mean()))
    return float(np.mean(errors)) if errors else 0.0


def active_workload_order(kb: KB, models: Sequence[str], k: int = 3,
                         max_active: int = 6) -> List[str]:
    models = list(models)
    workloads, values = _m2b_arrays(kb, models)
    meta = kb.meta.set_index("workload").loc[workloads]
    costs = meta["cost_proxy"].astype(float).to_numpy()
    tracks = meta["track"].to_numpy()
    references = list(range(len(models)))
    selected: List[int] = []
    active_steps = min(len(workloads), max_active)
    while len(selected) < active_steps:
        baseline = (_cv_completion_error(values, references, selected, k)
                    if selected else None)
        seen_tracks = set(tracks[selected])
        candidates: List[tuple] = []
        for candidate in range(len(workloads)):
            if candidate in selected:
                continue
            if baseline is None:
                gain = -_cv_completion_error(values, references, [candidate], k)
            else:
                gain = baseline - _cv_completion_error(
                    values, references, selected + [candidate], k)

            coverage = 0.02 if tracks[candidate] not in seen_tracks else 0.0
            score = (gain / (costs[candidate] / np.median(costs) + 0.5)
                     + coverage)
            candidates.append((score, candidate))
        selected.append(max(candidates)[1])

    remaining = [i for i in range(len(workloads)) if i not in selected]
    selected.extend(sorted(remaining, key=lambda i: (costs[i], workloads[i])))
    return [workloads[i] for i in selected]


def conformal_radius(kb: KB, reference_models: Sequence[str],
                     revealed: Sequence[str], alpha: float = 0.1,
                     k: int = 3) -> float:
    revealed = list(revealed)
    all_workloads, values = _m2b_arrays(kb, list(reference_models))
    selected = [all_workloads.index(w) for w in revealed]
    rest = [i for i in range(len(all_workloads)) if i not in selected]
    if not rest or len(reference_models) <= k:
        return 0.0
    residuals: List[float] = []
    for idx in range(len(reference_models)):
        refs = [i for i in range(len(reference_models)) if i != idx]
        distance = np.abs(values[refs][:, selected, :] -
                          values[idx][None, selected, :]).mean(axis=(1, 2))
        neighbours = np.asarray(refs)[np.argsort(distance)[:k]]
        prediction = values[neighbours][:, rest, :].mean(axis=0)
        completed = values[idx].copy()
        completed[rest] = prediction
        residuals.extend(np.abs(completed.mean(axis=0) -
                                values[idx].mean(axis=0)).tolist())
    return float(np.quantile(residuals, 1.0 - alpha)) if residuals else 0.0

