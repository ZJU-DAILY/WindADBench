from __future__ import annotations

import argparse
import hashlib
import itertools
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
import cost as m2
import comparisons as rp2
import ppca as m3
import comparisons as rp3
import planning as a4

BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


def auroc(score, label) -> float:
    s, y = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def d2m_confidence(ev, cases):
    workloads = sorted(ev.meta.workload)
    R = ev.array(kb.dim_scores(ev.cells, ev.models), ev.models, workloads)
    rows = []
    for t in kb.FARMS:
        s = d2m.lofo_sets(t)
        fh = d2m.FayHerriotD2M(R, workloads, ev.meta, s["evidence"])
        for w in d2m.preference_grid():
            rec = a4.recommend(fh, ev.models, w)
            ut = d2m.truth_utility(R, workloads, ev.meta, s["truth"], w)
            rows.append(dict(farm=t, weights=json.dumps([float(x) for x in w]), primary=rec["primary"],
                             pcs=rec["confidence_pcs"], hit1=rec["primary"] == ev.models[int(np.argmax(ut))],
                             commit=rec["decision"] == "recommend", n_front=len(rec["pareto_front"])))
    df = pd.DataFrame(rows)
    ref = cases[cases.method == "v2_fay_herriot"].reset_index(drop=True)
    assert (ref.pick.to_numpy() == df.primary.to_numpy()).all(), "v4 D2M must reproduce the v2 ranking"
    return df, {"pcs_mean": float(df.pcs.mean()), "hit1_rate": float(df.hit1.mean()),
                "auroc_pcs_for_hit1": auroc(df.pcs, df.hit1),
                "pcs_when_commit": float(df.pcs[df.commit].mean()), "pcs_when_abstain": float(df.pcs[~df.commit].mean()),
                "hit1_when_commit": float(df.hit1[df.commit].mean()), "hit1_when_abstain": float(df.hit1[~df.commit].mean())}


def run_m2b(ev, core, seed=42, n_random=5, sens_seeds=range(100)):
    models, workloads = ev.models, sorted(ev.meta.workload)
    W, n = len(workloads), len(models)
    raw = m3.RawTensor(ev.cells, models, workloads)
    q = kb.support_weights(workloads, ev.meta).to_numpy()
    A = m3.card_matrix(q)
    omega2 = m2.preference_second_moment(d2m.preference_grid())
    cost = m2.CostModel(ev.cells, workloads)
    proxy = rp2.original_cost_proxy(ev.cells, workloads)
    tracks = kb.workload_table().set_index("workload").loc[workloads, "track"].to_numpy()
    rng = np.random.default_rng(seed)
    rows, stops, sens, folds_obj = [], [], [], {}
    for fold, cand in enumerate(models):
        refs = [i for i in range(n) if i != fold]
        S_ref, S_c = raw.scores(refs, refs), raw.scores(refs, [fold])
        Y_ref, y_c = S_ref.reshape(len(refs), -1), S_c.reshape(-1)
        main = m3.PPCA(Y_ref)
        inner = a4.InnerReplay(raw, refs, A)
        ref_cards, true_card = Y_ref @ A.T, A @ y_c
        cards_true = np.vstack([ref_cards, true_card])
        t_rank, t_par = m3.rank_of(cards_true, len(refs)), m3.pareto_member(cards_true, len(refs))
        rnames = [models[i] for i in refs]
        est_fit, est_inf = cost.components(rnames)
        act_fit, act_inf = cost.components([cand])
        full = cost.total(act_fit, act_inf, range(W))
        marg = lambda ch, i, ef=est_fit, ei=est_inf: cost.marginal(ef, ei, ch, i)
        standalone = np.array([marg([], i) for i in range(W)])
        kbf = rp2.fold_kb(core, rp3.long_scores(rnames + [cand], workloads, np.concatenate([S_ref, S_c])), ev, workloads, proxy)
        orders = {"v4": a4.plan_order(inner, tracks, marg, standalone),
                  "v4_uniform_cost": a4.plan_order(inner, tracks, lambda ch, i: 1.0, np.ones(W)),
                  "v3": m3.plan_order(main, A, omega2, marg),
                  "cheapest": m2.cheapest_order(cost, est_fit, est_inf, W),
                  "original": [workloads.index(x) for x in core.active_workload_order(kbf, rnames)]}
        rand = [[workloads.index(x) for x in rng.permutation(workloads)] for _ in range(n_random)]
        comp = {"ppca": lambda o: main.complete(m3.cells_of(o), y_c[m3.cells_of(o)]),
                "knn": lambda o: m3.knn_complete(Y_ref, y_c, m3.cells_of(o))}

        def record(policy, order, rid, completer):
            for K in range(W + 1):
                pred = A @ comp[completer](order[:K])
                cards = np.vstack([ref_cards, pred])
                rows.append(dict(model=cand, policy=policy, order_id=rid, K=K,
                                 workloads="|".join(workloads[i] for i in order[:K]),
                                 mae=float(np.abs(pred - true_card).mean()),
                                 rank_error=abs(m3.rank_of(cards, len(refs)) - t_rank),
                                 pareto_pred=m3.pareto_member(cards, len(refs)), pareto_true=t_par,
                                 cost_actual=cost.total(act_fit, act_inf, order[:K]), cost_full=full,
                                 **{f"pred_{d}": float(v) for d, v in zip(kb.DIMS, pred)}))

        record("v4", orders["v4"], 0, "ppca")
        record("v4_uniform_cost", orders["v4_uniform_cost"], 0, "ppca")
        record("v3", orders["v3"], 0, "ppca")
        record("cheapest_ppca", orders["cheapest"], 0, "ppca")
        record("original", orders["original"], 0, "knn")
        record("ablation_v4_order_knn", orders["v4"], 0, "knn")
        for rid, o in enumerate(rand):
            record("random_ppca", o, rid, "ppca")
            record("random_knn", o, rid, "knn")

        def stop_row(rule, name, rid, o, radii, completer):
            stop = next((K for K, r in zip(range(1, W + 1), radii) if r <= a4.STOP_RADIUS), W)
            pred = A @ comp[completer](o[:stop])
            err = np.abs(pred - true_card)
            cards = np.vstack([ref_cards, pred])
            stops.append(dict(model=cand, rule=rule, order=name, order_id=rid, stop_K=stop, radius=radii[stop - 1],
                              mae=float(err.mean()), covered=int((err <= radii[stop - 1]).sum()),
                              pareto_ok=bool(m3.pareto_member(cards, len(refs)) == t_par),
                              rank_error=abs(m3.rank_of(cards, len(refs)) - t_rank),
                              cost=cost.total(act_fit, act_inf, o[:stop]), cost_full=full))

        for name, o, rid in [(k, v, 0) for k, v in orders.items()] + [("random", o, r) for r, o in enumerate(rand)]:
            stop_row("inner_replay", name, rid, o, [inner.radius(o[:K]) for K in range(1, W + 1)], "ppca")
        o = orders["original"]
        stop_row("original_rule", "original", 0, o,
                 [core.conformal_radius(kbf, rnames, [workloads[i] for i in o[:K]]) for K in range(1, W + 1)], "knn")
        folds_obj[cand] = dict(inner=inner, main=main, y_c=y_c, true_card=true_card, ref_cards=ref_cards, A=A,
                               marg=marg, standalone=standalone, act=(act_fit, act_inf), full=full, t_par=t_par)
    for s_ in sens_seeds:
        r2 = np.random.default_rng(s_)
        for cand in models:
            f = folds_obj[cand]
            for rid in range(n_random):
                o = [workloads.index(x) for x in r2.permutation(workloads)]
                for K in range(W + 1):
                    obs = m3.cells_of(o[:K])
                    sens.append((s_, cand, rid, K, float(np.abs(A @ f["main"].complete(obs, f["y_c"][obs]) - f["true_card"]).mean()),
                                 cost.total(*f["act"], o[:K]) / f["full"]))
    sens = pd.DataFrame(sens, columns=["seed", "model", "order_id", "K", "mae", "cost_fraction"])
    return pd.DataFrame(rows), pd.DataFrame(stops), sens, folds_obj, (cost, tracks, workloads)


def constant_sensitivity(folds_obj, ctx) -> dict:
    cost, tracks, workloads = ctx
    W = len(workloads)
    out = {}
    for rho, smooth, act in itertools.product((0.0, 0.02, 0.05), (0.25, 0.5, 1.0), (6, 12)):
        mae10, saves, maes, cov = [], [], [], []
        for cand, f in folds_obj.items():
            o = a4.plan_order(f["inner"], tracks, f["marg"], f["standalone"], rho=rho, smooth=smooth, max_active=act)
            fr = [cost.total(*f["act"], o[:K]) / f["full"] for K in range(W + 1)]
            kb10 = max(K for K in range(W + 1) if fr[K] <= 0.10 + 1e-12)
            obs = m3.cells_of(o[:kb10])
            mae10.append(np.abs(f["A"] @ f["main"].complete(obs, f["y_c"][obs]) - f["true_card"]).mean())
            radii = [f["inner"].radius(o[:K]) for K in range(1, W + 1)]
            st = next((K for K, r in zip(range(1, W + 1), radii) if r <= a4.STOP_RADIUS), W)
            obs = m3.cells_of(o[:st])
            err = np.abs(f["A"] @ f["main"].complete(obs, f["y_c"][obs]) - f["true_card"])
            saves.append(1 - fr[st]); maes.append(err.mean()); cov.append((err <= radii[st - 1]).mean())
        out[f"rho={rho},smooth={smooth},active={act}"] = {"mae_at_10pct": float(np.mean(mae10)), "saving_at_stop": float(np.mean(saves)),
                                                         "mae_at_stop": float(np.mean(maes)), "coverage": float(np.mean(cov))}
    return out


def bootstrap(folds, stops, n_boot=2000, seed=7) -> dict:
    rng = np.random.default_rng(seed)
    mb = rp2.matched_budget(folds)
    per = mb.groupby(["policy", "model", "budget"]).mae.mean().unstack("policy")
    kk = folds.groupby(["policy", "model", "K"]).mae.mean().unstack("policy")
    st = stops.assign(save=1 - stops.cost / stops.cost_full)
    s_v4 = st[(st.rule == "inner_replay") & (st.order == "v4")].set_index("model").save
    s_or = st[(st.rule == "original_rule")].set_index("model").save
    s_rd = st[(st.rule == "inner_replay") & (st.order == "random")].groupby("model").save.mean()
    diffs = {f"mae_budget{b}_v4_minus_random": (per.xs(b, level="budget").v4 - per.xs(b, level="budget").random_ppca) for b in (0.05, 0.10, 0.20)}
    diffs["mae_budget0.1_v4_minus_original"] = per.xs(0.10, level="budget").v4 - per.xs(0.10, level="budget").original
    diffs["mae_budget0.1_v4_minus_cheapest"] = per.xs(0.10, level="budget").v4 - per.xs(0.10, level="budget").cheapest_ppca
    for K in (2, 4):
        diffs[f"mae_K{K}_v4_minus_random"] = kk.xs(K, level="K").v4 - kk.xs(K, level="K").random_ppca
        diffs[f"mae_K{K}_v4_uniform_minus_random"] = kk.xs(K, level="K").v4_uniform_cost - kk.xs(K, level="K").random_ppca
    diffs["saving_at_stop_v4_minus_original"] = s_v4 - s_or
    diffs["saving_at_stop_v4_minus_random"] = s_v4 - s_rd
    out = {}
    for name, d in diffs.items():
        d = d.to_numpy(float)
        bs = np.array([rng.choice(d, len(d)).mean() for _ in range(n_boot)])
        out[name] = {"mean": float(d.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                     "folds_better": int((d < 0).sum()) if "mae" in name else int((d > 0).sum()), "n": int(len(d))}
    return out


def summarize(folds, stops, sens) -> dict:
    s = {"policies": {}}
    for p, g in folds.groupby("policy", sort=False):
        k = g.groupby("K")
        s["policies"][p] = {"mae_by_K": {int(a): float(b) for a, b in k.mae.mean().items()},
                            "rank_error_by_K": {int(a): float(b) for a, b in k.rank_error.mean().items()},
                            "pareto_agreement_by_K": {int(a): float((d.pareto_pred == d.pareto_true).mean()) for a, d in k},
                            "cost_fraction_by_K": {int(a): float((d.cost_actual / d.cost_full).mean()) for a, d in k}}
    mb = rp2.matched_budget(folds)
    s["matched_budget"] = {p: {str(b): {"mae": float(x.mae.mean()), "mean_K": float(x.K.mean()),
                                         "rank_error": float(x.rank_error.mean()), "pareto_agreement": float(x.pareto_ok.mean())}
                               for b, x in g.groupby("budget")} for p, g in mb.groupby("policy", sort=False)}
    s["stopping"] = {}
    for (rule, order), g in stops.groupby(["rule", "order"], sort=False):
        s["stopping"][f"{rule}:{order}"] = {
            "mean_stop_K": float(g.stop_K.mean()),
            "stop_K_distribution": {int(a): int(b) for a, b in g.stop_K.value_counts().sort_index().items()},
            "saving": float(1 - (g.cost / g.cost_full).mean()), "median_cost_h": float(g.cost.median() / 3600),
            "mae": float(g.mae.mean()), "coverage": float(g.covered.sum() / (4 * len(g))),
            "pareto_agreement": float(g.pareto_ok.mean()), "rank_error": float(g.rank_error.mean())}
    by_seed = sens.groupby(["seed", "K"]).mae.mean().unstack("K")
    s["seed_sensitivity_by_K"] = {}
    for p in ("v4", "v4_uniform_cost"):
        act = folds[folds.policy == p].groupby("K").mae.mean()
        s["seed_sensitivity_by_K"][p] = {int(K): {"policy": float(act[K]), "random_mean": float(by_seed[K].mean()),
                                                  "better_seeds": int((act[K] < by_seed[K] - 1e-12).sum())} for K in range(1, 12)}
    rows = []
    for (sd, m, rid), g in sens.groupby(["seed", "model", "order_id"], sort=False):
        g = g.sort_values("K")
        for b in BUDGETS:
            rows.append((sd, b, float(g[g.cost_fraction <= b + 1e-12].iloc[-1].mae)))
    sb = pd.DataFrame(rows, columns=["seed", "budget", "mae"]).groupby(["seed", "budget"]).mae.mean().unstack()
    v4b = mb[mb.policy == "v4"].groupby("budget").mae.mean()
    s["seed_sensitivity_budget"] = {str(b): {"v4": float(v4b[b]), "random_mean": float(sb[b].mean()),
                                             "v4_better_seeds": int((v4b[b] < sb[b] - 1e-12).sum())} for b in BUDGETS}
    return s
