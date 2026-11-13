from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import final as af
import evidence as kb
import d2m as d2m
import cost as m2
import ppca as m3

DIMS = kb.DIMS


def svd_ppca_complete(Y, rank, y, obs):
    mu = Y.mean(0)
    _, s, Vt = np.linalg.svd(Y - mu, full_matrices=True)
    ev = np.zeros(Y.shape[1])
    ev[:len(s)] = s ** 2 / len(Y)
    s2 = max(ev[rank:].mean(), 1e-10)
    Wm = Vt[:rank].T * np.sqrt(np.clip(ev[:rank] - s2, 0, None))
    Wo = Wm[obs]
    z = np.linalg.solve(np.eye(rank) + Wo.T @ Wo / s2, Wo.T @ (y[obs] - mu[obs]) / s2)
    out = mu + Wm @ z
    out[obs] = y[obs]
    return out


def pandas_scores(cells):
    out = []
    for wl, g in cells.groupby("workload"):
        g = g.set_index("model")
        r = pd.DataFrame(index=g.index)
        parts = {"accuracy": [g.event_f1.rank(pct=True), g.vus_pr.rank(pct=True)],
                 "earliness": [g.edr.rank(pct=True), g.delay_h.fillna(np.inf).rank(pct=True, ascending=False)],
                 "reliability": [g.fa_day.rank(pct=True, ascending=False), g.mtbfa_h.rank(pct=True),
                                 g.normal_fpr.rank(pct=True, ascending=False)],
                 "cost": [g.fit_s.rank(pct=True, ascending=False), g.infer_s.rank(pct=True, ascending=False),
                          g.gpu_mb.fillna(-np.inf).rank(pct=True, ascending=False)]}
        for d in DIMS:
            r[d] = pd.concat(parts[d], axis=1).mean(axis=1, skipna=True)
        r["workload"] = wl
        out.append(r.reset_index())
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--evidence", type=Path, default=af.DEFAULT_EVIDENCE)
    ap.add_argument("--out", default="outputs/agent")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if Path(a.out).is_absolute() else root / a.out
    ev = af.load_evidence(root, a.evidence)
    M, W = ev.models, sorted(ev.meta.workload)
    chk = {}
    mine = pandas_scores(ev.cells)
    saved = kb.dim_scores(ev.cells, M)
    m = saved.merge(mine, on=["model", "workload"], suffixes=("", "_p"))
    chk["scores_max_abs_diff"] = float(max(np.abs(m[d] - m[d + "_p"]).max() for d in DIMS))
    assert chk["scores_max_abs_diff"] < 1e-12

    piv = {d: mine.pivot(index="model", columns="workload", values=d).loc[M] for d in DIMS}
    n_ev = ev.meta.set_index("workload").n_events
    q = lambda ws: (np.log1p(n_ev[ws].clip(lower=1)) / np.log1p(n_ev[ws].clip(lower=1)).sum()).values
    regs = []
    for t in "ABC":
        x, y = [f for f in "ABC" if f != t]
        cf, ifs, cts, tr = [f"CF-{x}>{y}", f"CF-{y}>{x}"], [f"IF-{x}", f"IF-{y}"], [f"CT-{x}", f"CT-{y}"], [f"CF-{x}>{t}", f"CF-{y}>{t}"]
        th, V = np.zeros((36, 4)), np.zeros((36, 4))
        for k, d in enumerate(DIMS):
            P = piv[d]
            yb = (P[cf] * q(cf)).sum(axis=1).to_numpy()
            X = np.column_stack([np.ones(36), (P[ifs] * q(ifs)).sum(axis=1), (P[cts] * q(cts)).sum(axis=1)])
            b = np.linalg.solve(X.T @ X, X.T @ yb)
            res = yb - X @ b
            psi = (q(cf) ** 2).sum() * np.mean((P[cf[0]] - P[cf[1]]).to_numpy() ** 2) / 2
            tau2 = max(0.0, res @ res / 33 - psi)
            gam = tau2 / (tau2 + psi)
            h = np.einsum("ij,jk,ik->i", X, np.linalg.inv(X.T @ X), X)
            th[:, k], V[:, k] = X @ b + gam * res, gam * psi + (1 - gam) ** 2 * h * (tau2 + psi)
        truth = np.stack([(piv[d][tr] * q(tr)).sum(axis=1).to_numpy() for d in DIMS], axis=1)
        for w in [np.array(v) for v in itertools.product([0, .25, .5, .75, 1], repeat=4) if abs(sum(v) - 1) < 1e-9]:
            u = th @ w
            l = u - 0.5 * np.sqrt(V @ w ** 2)
            pick = sorted(range(36), key=lambda i: (-l[i], -u[i], i))[0]
            regs.append((truth @ w).max() - (truth @ w)[pick])
    d2s = json.loads((out / "d2m_summary.json").read_text())
    chk["d2m_regret"] = [float(np.mean(regs)), d2s["methods"]["v2_fay_herriot"]["mean_regret"]]
    assert abs(chk["d2m_regret"][0] - chk["d2m_regret"][1]) < 1e-12

    R = ev.array(kb.dim_scores(ev.cells, M), M, W)
    s = d2m.lofo_sets("A")
    fh = d2m.FayHerriotD2M(R, W, ev.meta, s["evidence"])
    w = np.full(4, .25)
    r0 = af.recommend(fh, M, w, {m_: ["test"] for m_ in M})
    r1 = af.recommend(fh, M, w, {m_: ["test"] for m_ in M[1:]})
    r2 = af.recommend(fh, M, w, {m_: ["test"] for m_ in M[2:]})
    assert r0["decision"] == "no_feasible_detector" and r1["primary"] == M[0] and r2["primary"] in M[:2]
    chk["boundary_tests"] = [r0["decision"], r1["decision"], r2["decision"]]

    folds = pd.read_csv(out / "m2b_folds.csv")
    stops = pd.read_csv(out / "m2b_stops.csv")
    qw = kb.support_weights(W, ev.meta).to_numpy()
    tracks = kb.workload_table().set_index("workload").loc[W, "track"].to_numpy()
    cost = m2.CostModel(ev.cells, W)
    for f in (8, 30):
        cand = M[f]
        refs = [x for x in M if x != cand]
        inner = []
        for ref in refs:
            pop = [x for x in refs if x != ref]
            Yp = ev.array(kb.dim_scores(ev.cells, pop), pop, W).reshape(len(pop), -1)
            inner.append((Yp, m3.select_rank(Yp), ev.array(kb.dim_scores(ev.cells, pop, [ref]), [ref], W).reshape(-1)))

        def resid(sel):
            obs = np.array([w_ * 4 + d for w_ in sorted(sel) for d in range(4)], dtype=int)
            return np.array([np.abs(((Yp.mean(0) if len(obs) == 0 else svd_ppca_complete(Yp, r, y, obs)) - y).reshape(12, 4).T @ qw)
                             for Yp, r, y in inner])
        ef, ei = cost.components(refs)
        st = np.array([cost.marginal(ef, ei, [], i) for i in range(12)])
        base = resid([]).mean()
        sc = [((base - resid([i]).mean()) / (st[i] / np.median(st) + 0.5) + 0.02, -st[i], -i) for i in range(12)
              if base - resid([i]).mean() > 0]
        first = W[-max(sc)[2]]
        assert first == folds[(folds.policy == "v4") & (folds.model == cand) & (folds.K == 1)].workloads.iloc[0]
        srow = stops[(stops.model == cand) & (stops.rule == "inner_replay") & (stops.order == "v4")].iloc[0]
        order = folds[(folds.policy == "v4") & (folds.model == cand) & (folds.K == int(srow.stop_K))].workloads.iloc[0].split("|")
        r = np.sort(resid([W.index(x) for x in order]).ravel())
        rad = float(r[int(np.ceil(0.9 * (len(r) + 1))) - 1])
        assert abs(rad - srow.radius) < 1e-9
        chk[f"fold{f}"] = {"first": first, "radius": rad}
    (out / "verification.json").write_text(json.dumps({"status": "pass", **chk}, indent=2))
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
