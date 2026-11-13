from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evidence as kb
import d2m as d2m
import cost as m2b

SNAPSHOT = HERE
BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


def load_original(root: Path):
    snap = Path(root) / SNAPSHOT
    sys.path.insert(0, str(snap))
    import baselines as core
    assert Path(core.__file__).resolve().parent == snap.resolve()
    return core


def run_d2m(ev: kb.Evidence, core) -> tuple[pd.DataFrame, dict]:
    workloads = sorted(ev.meta.workload)
    models = ev.models
    scores = kb.dim_scores(ev.cells, models)
    R = ev.array(scores, models, workloads)
    orig = d2m.OriginalD2M(core, scores, ev.cells, ev.meta)
    grid = d2m.preference_grid()
    pool = [m for m in models if m in d2m.FIXED_POOL]
    eq = np.full(4, 0.25)
    rows, diag = [], {}
    for t in kb.FARMS:
        s = d2m.lofo_sets(t)
        ev_if = [w for w in s["evidence"] if w.startswith("IF-")]
        fh = d2m.FayHerriotD2M(R, workloads, ev.meta, s["evidence"])
        diag[t] = fh.diag
        gb = orig.utility(eq, s["evidence"]).index[0]
        ef1 = ev.cells[ev.cells.workload.isin(s["evidence"])].groupby("model").event_f1.mean().idxmax()
        st = orig.utility(eq, s["evidence"], pool).index[0]
        x_if = d2m.track_mean(R, workloads, ev_if, ev.meta)
        for w in grid:
            u_tr = d2m.truth_utility(R, workloads, ev.meta, s["truth"], w)
            snap = orig.utility(w, s["truth"]).reindex(models).to_numpy()
            assert np.allclose(snap, u_tr, atol=1e-12)
            best = u_tr.max()

            def regret(mname):
                return float(best - u_tr[models.index(mname)])

            ranked = {}
            u, lcb = fh.scores(w)
            o = d2m.order_by(lcb, u)
            ranked["v2_fay_herriot"] = ([models[i] for i in o], float(lcb[o[0]] - lcb[o[1]]))
            rec = orig.recommend(w, s["evidence"])
            ranked["original_agent"] = (list(rec["top_k"]), float(rec["margin"]))
            for name, vec in (("ablation_cf_only", fh.ybar @ w), ("ablation_regression_only", fh.reg @ w),
                              ("ablation_no_lcb", u), ("if_leaderboard", x_if @ w)):
                oo = np.lexsort((np.arange(len(models)), -vec))
                ranked[name] = ([models[i] for i in oo], np.nan)
            base = {"rows": []}
            for name, (rank, margin) in ranked.items():
                commit = bool(margin >= d2m.EPSILON) if np.isfinite(margin) else np.nan
                regs = [regret(m) for m in rank[:5]]
                row = dict(farm=t, weights=json.dumps([float(x) for x in w]), method=name, pick=rank[0],
                           regret=regs[0], hit1=bool(rank[0] == models[int(np.argmax(u_tr))]),
                           hit3=bool(models[int(np.argmax(u_tr))] in rank[:3]), margin=margin, commit=commit,
                           regret_abstain_pilot3=(regs[0] if commit is True else min(regs[:3])) if np.isfinite(margin) else np.nan)
                for k in range(1, 6):
                    row[f"regret_pilot{k}"] = min(regs[:k])
                rows.append(row)
            for name, pick in (("global_best", gb), ("event_f1", ef1), ("fixed_shortlist", st)):
                rows.append(dict(farm=t, weights=json.dumps([float(x) for x in w]), method=name, pick=pick,
                                 regret=regret(pick), hit1=bool(pick == models[int(np.argmax(u_tr))])))
            rows.append(dict(farm=t, weights=json.dumps([float(x) for x in w]), method="random",
                             regret=float(best - u_tr.mean()), hit1=1.0 / len(models)))
    df = pd.DataFrame(rows)
    summary = {"cases_per_method": int(df.groupby("method").size().iloc[0]), "methods": {}, "fay_herriot": diag}
    for name, g in df.groupby("method", sort=False):
        item = {"mean_regret": float(g.regret.mean()),
                "regret_by_farm": {f: float(x) for f, x in g.groupby("farm").regret.mean().items()},
                "hit1": float(g.hit1.astype(float).mean())}
        if g.get("hit3") is not None and g.hit3.notna().any():
            item["hit3"] = float(g.hit3.astype(float).mean())
        if "margin" in g and g.margin.notna().any():
            c = g.commit.astype(bool)
            item.update({"commit_rate": float(c.mean()),
                         "regret_when_commit": float(g.regret[c].mean()) if c.any() else None,
                         "regret_when_abstain": float(g.regret[~c].mean()) if (~c).any() else None,
                         "abstain_then_pilot3_regret": float(g.regret_abstain_pilot3.mean()),
                         "abstain_then_pilot3_mean_pilots": float(3 * (~c).mean()),
                         "always_pilot_regret": {k: float(g[f"regret_pilot{k}"].mean()) for k in range(1, 6)}})
        summary["methods"][name] = item
    return df, summary


def original_cost_proxy(cells: pd.DataFrame, workloads) -> pd.Series:
    inf = cells[cells.track == "in-farm"]
    per_source = {src: float(g.fit_s.median() + g.infer_s.median()) for src, g in inf.groupby("source")}
    meta = kb.workload_table().set_index("workload").loc[list(workloads)]
    return meta.source.map(per_source)


def fold_kb(core, scores: pd.DataFrame, ev: kb.Evidence, workloads, proxy):
    meta = ev.meta.set_index("workload").loc[list(workloads)].reset_index()
    meta["cost_proxy"] = proxy.loc[list(workloads)].to_numpy()
    dim_pct = scores[["model", "workload", "track", "source", "target"] + kb.DIMS].copy()
    core._M2B_CACHE.clear()
    return core.KB(dim_pct=dim_pct, coords=dim_pct[["model", "workload"]].copy(), registry=pd.DataFrame(),
                   meta=meta, wide=pd.DataFrame())


def matched_budget(folds: pd.DataFrame, budgets=BUDGETS) -> pd.DataFrame:
    out = []
    f = folds.assign(frac=folds.cost_actual / folds.cost_full)
    for (policy, model, rid), g in f.groupby(["policy", "model", "order_id"], sort=False):
        g = g.sort_values("K")
        for b in budgets:
            ok = g[g.frac <= b + 1e-12]
            row = ok.iloc[-1]
            out.append(dict(policy=policy, model=model, order_id=rid, budget=b, K=int(row.K), mae=row.mae,
                            rank_error=row.rank_error, pareto_ok=bool(row.pareto_pred == row.pareto_true)))
    return pd.DataFrame(out)


def d2m_extra(ev, cases: pd.DataFrame) -> dict:
    workloads = sorted(ev.meta.workload)
    R = ev.array(kb.dim_scores(ev.cells, ev.models), ev.models, workloads)
    cal = []
    for t in kb.FARMS:
        s = d2m.lofo_sets(t)
        fh = d2m.FayHerriotD2M(R, workloads, ev.meta, s["evidence"])
        truth = d2m.track_mean(R, workloads, s["truth"], ev.meta)
        qt = kb.support_weights(s["truth"], ev.meta).to_numpy()
        for d, dim in enumerate(kb.DIMS):
            psi_truth = float((qt ** 2).sum() * fh.diag[dim]["sigma2_direction"])
            z = np.abs(truth[:, d] - fh.theta[:, d]) / np.sqrt(fh.V[:, d] + psi_truth)
            for m, zz in zip(ev.models, z):
                cal.append(dict(farm=t, dim=dim, model=m, z=float(zz)))
    cal = pd.DataFrame(cal)
    out = {"interval_coverage_90": float((cal.z <= 1.645).mean()),
           "interval_coverage_90_by_dim": {k: float((g.z <= 1.645).mean()) for k, g in cal.groupby("dim")},
           "interval_coverage_90_by_farm": {k: float((g.z <= 1.645).mean()) for k, g in cal.groupby("farm")},
           "risk_coverage": {}}
    for name in ("v2_fay_herriot", "original_agent"):
        g = cases[cases.method == name].sort_values("margin", ascending=False, kind="mergesort")
        risk = np.cumsum(g.regret.to_numpy()) / np.arange(1, len(g) + 1)
        best = np.cumsum(np.sort(g.regret.to_numpy())) / np.arange(1, len(g) + 1)
        out["risk_coverage"][name] = {"aurc": float(risk.mean()), "aurc_random_order": float(g.regret.mean()),
                                      "aurc_oracle_order": float(best.mean()),
                                      "risk_at_coverage": {str(c): float(risk[int(np.ceil(c * len(g))) - 1])
                                                           for c in (0.25, 0.5, 0.6, 0.75, 1.0)}}
    return out

def long_scores(models, workloads, arr):
    meta = kb.workload_table().set_index("workload")
    rows = []
    for i, m in enumerate(models):
        for j, w in enumerate(workloads):
            r = dict(model=m, workload=w, **meta.loc[w].to_dict())
            r.update({d: float(arr[i, j, k]) for k, d in enumerate(kb.DIMS)})
            rows.append(r)
    return pd.DataFrame(rows)
