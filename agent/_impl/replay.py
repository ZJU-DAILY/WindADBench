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
import final as af
import evidence as kb
import d2m as d2m
import cost as m2
import ppca as m3
import planning as a4
import comparisons as rp2
import comparisons as rp3
import evaluation as rp4

PROFILES = {"cpu_only": dict(no_gpu=True), "no_alarm_saturation": dict(max_normal_fpr=0.10)}
SCENARIOS = {"limited_review": (np.array([.2, .1, .6, .1]), dict(max_fa_day=2.0)),
             "fault_safety": (np.array([.3, .5, .1, .1]), {}),
             "edge": (np.array([.2, .2, .2, .4]), dict(no_gpu=True)),
             "cold_start": (np.full(4, .25), {})}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def d2m_final(ev, core):
    cases, summ = rp2.run_d2m(ev, core)
    models, workloads = ev.models, sorted(ev.meta.workload)
    R = ev.array(kb.dim_scores(ev.cells, models), models, workloads)
    orig = d2m.OriginalD2M(core, kb.dim_scores(ev.cells, models), ev.cells, ev.meta)
    extra, cons, scen, cf_margin = [], [], [], []
    for t in kb.FARMS:
        s = d2m.lofo_sets(t)
        fh = d2m.FayHerriotD2M(R, workloads, ev.meta, s["evidence"])
        prof_all = d2m.track_mean(R, workloads, s["evidence"], ev.meta)
        truth_fpr = ev.cells[ev.cells.workload.isin(s["truth"])].groupby("model").normal_fpr.mean()
        truth_fa = ev.cells[ev.cells.workload.isin(s["truth"])].groupby("model").fa_day.mean()
        excl = {p: af.exclusion_reasons(ev.cells, models, s["evidence"], **kw) for p, kw in PROFILES.items()}
        for w in d2m.preference_grid():
            ut = d2m.truth_utility(R, workloads, ev.meta, s["truth"], w)
            wid = json.dumps([float(x) for x in w])
            orc = models[int(np.argmax(ut))]
            for name, score in (("weighted_sum_all", prof_all @ w), ("topsis_all", af.topsis(prof_all, w))):
                o = np.lexsort((np.arange(len(models)), -score))
                extra.append(dict(farm=t, weights=wid, method=name, pick=models[o[0]], regret=float(ut.max() - ut[o[0]]),
                                  hit1=bool(models[o[0]] == orc), hit3=bool(orc in [models[i] for i in o[:3]])))
            v = fh.ybar @ w
            o = np.lexsort((np.arange(len(models)), -v))
            cf_margin.append(dict(farm=t, weights=wid, margin=float(v[o[0]] - v[o[1]]), regret=float(ut.max() - ut[o[0]])))
            for p, ex in excl.items():
                feas = [i for i, m in enumerate(models) if m not in ex]
                best = max(ut[i] for i in feas)
                rec = af.recommend(fh, models, w, ex)
                st = core.utility_stats(orig.kb, dict(zip(kb.DIMS, w)), s["evidence"], [models[i] for i in feas])
                st["lcb"] = st.utility_mean - a4.LAMBDA * st.utility_se
                picks = {"v_final": rec["primary"],
                         "original_agent": st.sort_values(["lcb", "utility_mean"], ascending=False).index[0],
                         "cf_only": models[max(feas, key=lambda i: (v[i], -i))],
                         "weighted_sum_all": models[max(feas, key=lambda i: ((prof_all @ w)[i], -i))],
                         "topsis_all": models[max(feas, key=lambda i: (af.topsis(prof_all[feas], w)[feas.index(i)], -i))],
                         "global_best_eq": models[max(feas, key=lambda i: ((prof_all @ np.full(4, .25))[i], -i))]}
                for name, pk in picks.items():
                    cons.append(dict(farm=t, weights=wid, profile=p, n_feasible=len(feas), method=name, pick=pk,
                                     regret=float(best - ut[models.index(pk)])))
                cons.append(dict(farm=t, weights=wid, profile=p, n_feasible=len(feas), method="random",
                                 regret=float(best - np.mean([ut[i] for i in feas]))))
        for name, (w, kw) in SCENARIOS.items():
            ex = af.exclusion_reasons(ev.cells, models, s["evidence"], **kw)
            rec = af.recommend(fh, models, w, ex)
            ut = d2m.truth_utility(R, workloads, ev.meta, s["truth"], w)
            feas = [i for i, m in enumerate(models) if m not in ex]
            scen.append(dict(scenario=name, farm=t, decision=rec["decision"], primary=rec["primary"], top3="|".join(rec["top_k"]),
                             n_feasible=len(feas), regret_in_feasible=float(max(ut[i] for i in feas) - ut[models.index(rec["primary"])]),
                             pick_target_normal_fpr=float(truth_fpr[rec["primary"]]), pick_target_fa_day=float(truth_fa[rec["primary"]])))
    allc = pd.concat([cases, pd.DataFrame(extra)], ignore_index=True)
    for name in ("weighted_sum_all", "topsis_all"):
        g = allc[allc.method == name]
        summ["methods"][name] = {"mean_regret": float(g.regret.mean()), "hit1": float(g.hit1.mean()), "hit3": float(g.hit3.mean()),
                                 "regret_by_farm": {f: float(x) for f, x in g.groupby("farm").regret.mean().items()}}
    summ["v3_evaluation"] = rp3.d2m_extra(ev, cases)
    cm = pd.DataFrame(cf_margin).sort_values("margin", ascending=False, kind="mergesort")
    risk = np.cumsum(cm.regret.to_numpy()) / np.arange(1, len(cm) + 1)
    summ["v3_evaluation"]["risk_coverage"]["ablation_cf_only"] = {"aurc": float(risk.mean()), "aurc_random_order": float(cm.regret.mean())}
    cons = pd.DataFrame(cons)
    summ["constrained"] = {p: {m: float(x) for m, x in g.groupby("method").regret.mean().items()} | {"mean_feasible": float(g.n_feasible.mean())}
                           for p, g in cons.groupby("profile")}
    summ["scenarios"] = scen
    return allc, cons, summ


def d2m_guarded(ev, core, cases_unguarded: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    models, workloads = ev.models, sorted(ev.meta.workload)
    sc = kb.dim_scores(ev.cells, models)
    R = ev.array(sc, models, workloads)
    orig = d2m.OriginalD2M(core, sc, ev.cells, ev.meta)
    rows = []
    for t in kb.FARMS:
        s = d2m.lofo_sets(t)
        fh = d2m.FayHerriotD2M(R, workloads, ev.meta, s["evidence"])
        prof_all = d2m.track_mean(R, workloads, s["evidence"], ev.meta)
        x_if = d2m.track_mean(R, workloads, [w for w in s["evidence"] if w.startswith("IF-")], ev.meta)
        tfpr = ev.cells[ev.cells.workload.isin(s["truth"])].groupby("model").normal_fpr.mean()
        ex = af.exclusion_reasons(ev.cells, models, s["evidence"], max_normal_fpr=0.10)
        feas = [i for i, m in enumerate(models) if m not in ex]
        for w in d2m.preference_grid():
            ut = d2m.truth_utility(R, workloads, ev.meta, s["truth"], w)
            best = max(ut[i] for i in feas)
            orc = max(feas, key=lambda i: (ut[i], -i))
            u, l = fh.scores(w)
            st = core.utility_stats(orig.kb, dict(zip(kb.DIMS, w)), s["evidence"], [models[i] for i in feas])
            st["lcb"] = st.utility_mean - a4.LAMBDA * st.utility_se
            st = st.sort_values(["lcb", "utility_mean"], ascending=False)
            v, ws, tp = fh.ybar @ w, prof_all @ w, af.topsis(prof_all[feas], w)
            ranked = {"v_final": (sorted(feas, key=lambda i: (-l[i], -u[i], i)), None),
                      "original_agent": ([models.index(m) for m in st.index], float(st.lcb.iloc[0] - st.lcb.iloc[1])),
                      "cf_only": (sorted(feas, key=lambda i: (-v[i], i)), None),
                      "weighted_sum_all": (sorted(feas, key=lambda i: (-ws[i], i)), None),
                      "topsis_all": (sorted(feas, key=lambda i: (-tp[feas.index(i)], i)), None),
                      "if_leaderboard": (sorted(feas, key=lambda i: (-(x_if @ w)[i], i)), None),
                      "global_best_eq": (sorted(feas, key=lambda i: (-(prof_all @ np.full(4, .25))[i], i)), None)}
            o = ranked["v_final"][0]
            ranked["v_final"] = (o, float(l[o[0]] - l[o[1]]))
            o = ranked["cf_only"][0]
            ranked["cf_only"] = (o, float(v[o[0]] - v[o[1]]))
            for name, (order, margin) in ranked.items():
                regs = [best - ut[i] for i in order[:3]]
                commit = (margin >= a4.EPSILON) if margin is not None else None
                rows.append(dict(farm=t, weights=json.dumps([float(x) for x in w]), method=name, n_feasible=len(feas),
                                 pick=models[order[0]], regret=regs[0], hit1=order[0] == orc, hit3=orc in order[:3],
                                 target_fpr=float(tfpr[models[order[0]]]), margin=margin, commit=commit,
                                 pilot3=(regs[0] if commit else min(regs)) if margin is not None else None))
            rows.append(dict(farm=t, weights=json.dumps([float(x) for x in w]), method="random", n_feasible=len(feas),
                             regret=float(best - np.mean([ut[i] for i in feas])), hit1=1 / len(feas),
                             target_fpr=float(np.mean([tfpr[models[i]] for i in feas]))))
    df = pd.DataFrame(rows)
    out = {}
    for name, g in df.groupby("method", sort=False):
        item = {"mean_regret": float(g.regret.mean()), "regret_by_farm": {f: float(x) for f, x in g.groupby("farm").regret.mean().items()},
                "hit1": float(g.hit1.astype(float).mean()), "saturated_pick_rate": float((g.target_fpr > 0.10).mean()) if name != "random" else None}
        if g.hit3.notna().any():
            item["hit3"] = float(g.hit3.astype(float).mean())
        if g.margin.notna().any():
            gg = g.sort_values("margin", ascending=False, kind="mergesort")
            risk = np.cumsum(gg.regret.to_numpy()) / np.arange(1, len(gg) + 1)
            item.update({"aurc": float(risk.mean()), "commit_rate": float(gg.commit.astype(bool).mean()),
                         "abstain_then_pilot3": float(gg.pilot3.mean()), "mean_pilots": float(3 * (1 - gg.commit.astype(bool).mean()))})
        out[name] = item
    tf = {t: ev.cells[ev.cells.workload.isin(d2m.lofo_sets(t)["truth"])].groupby("model").normal_fpr.mean() for t in kb.FARMS}
    un = cases_unguarded[cases_unguarded.method.isin(["v2_fay_herriot", "original_agent", "ablation_cf_only", "weighted_sum_all", "topsis_all"])]
    out["unguarded_saturated_pick_rate"] = {m: float(np.mean([tf[r.farm][r.pick] > 0.10 for r in g.itertuples()])) for m, g in un.groupby("method")}
    out["mean_feasible"] = float(df.groupby("farm").n_feasible.first().mean())
    return df, out


def track_round_robin(fobj, ctx, seed=123, n_orders=5):
    cost, tracks, workloads = ctx
    W = len(workloads)
    rng = np.random.default_rng(seed)
    rows, stops = [], []
    for cand, f in fobj.items():
        for rid in range(n_orders):
            pools = {t: list(rng.permutation([i for i in range(W) if tracks[i] == t])) for t in ("in-farm", "cross-turbine", "cross-farm")}
            seq = list(rng.permutation(["in-farm", "cross-turbine", "cross-farm"]))
            o = []
            while len(o) < W:
                for t in seq:
                    if pools[t]:
                        o.append(int(pools[t].pop(0)))
            for K in range(W + 1):
                obs = m3.cells_of(o[:K])
                pred = f["A"] @ f["main"].complete(obs, f["y_c"][obs])
                rows.append(dict(model=cand, policy="track_round_robin", order_id=rid, K=K,
                                 mae=float(np.abs(pred - f["true_card"]).mean()),
                                 cost_actual=cost.total(*f["act"], o[:K]), cost_full=f["full"]))
            radii = [f["inner"].radius(o[:K]) for K in range(1, W + 1)]
            st = next((K for K, r in zip(range(1, W + 1), radii) if r <= a4.STOP_RADIUS), W)
            obs = m3.cells_of(o[:st])
            err = np.abs(f["A"] @ f["main"].complete(obs, f["y_c"][obs]) - f["true_card"])
            stops.append(dict(model=cand, order_id=rid, stop_K=st, mae=float(err.mean()), covered=int((err <= radii[st - 1]).sum()),
                              cost=cost.total(*f["act"], o[:st]), cost_full=f["full"]))
    return pd.DataFrame(rows), pd.DataFrame(stops)


def family_holdout(ev, seed=42, n_random=5):
    models, workloads = ev.models, sorted(ev.meta.workload)
    W, n = len(workloads), len(models)
    fam = ev.cells.drop_duplicates("model").set_index("model").family.reindex(models).to_numpy()
    raw = m3.RawTensor(ev.cells, models, workloads)
    A = m3.card_matrix(kb.support_weights(workloads, ev.meta).to_numpy())
    cost = m2.CostModel(ev.cells, workloads)
    tracks = kb.workload_table().set_index("workload").loc[workloads, "track"].to_numpy()
    rng = np.random.default_rng(seed)
    rows, stops = [], []
    for fold, cand in enumerate(models):
        refs = [i for i in range(n) if fam[i] != fam[fold]]
        Y_ref = raw.scores(refs, refs).reshape(len(refs), -1)
        y_c = raw.scores(refs, [fold]).reshape(-1)
        main = m3.PPCA(Y_ref)
        inner = a4.InnerReplay(raw, refs, A)
        true_card = A @ y_c
        est_fit, est_inf = cost.components([models[i] for i in refs])
        act_fit, act_inf = cost.components([cand])
        full = cost.total(act_fit, act_inf, range(W))
        marg = lambda ch, i, ef=est_fit, ei=est_inf: cost.marginal(ef, ei, ch, i)
        orders = {"v_final": a4.plan_order(inner, tracks, marg, np.array([marg([], i) for i in range(W)])),
                  "cheapest_ppca": m2.cheapest_order(cost, est_fit, est_inf, W)}
        rand = [[workloads.index(x) for x in rng.permutation(workloads)] for _ in range(n_random)]
        for name, o, rid in [(k, v, 0) for k, v in orders.items()] + [("random_ppca", o, r) for r, o in enumerate(rand)]:
            for K in range(W + 1):
                obs = m3.cells_of(o[:K])
                pred = A @ main.complete(obs, y_c[obs])
                rows.append(dict(model=cand, family=fam[fold], n_refs=len(refs), policy=name, order_id=rid, K=K,
                                 mae=float(np.abs(pred - true_card).mean()), cost_actual=cost.total(act_fit, act_inf, o[:K]),
                                 cost_full=full))
            radii = [inner.radius(o[:K]) for K in range(1, W + 1)]
            st = next((K for K, r in zip(range(1, W + 1), radii) if r <= a4.STOP_RADIUS), W)
            obs = m3.cells_of(o[:st])
            err = np.abs(A @ main.complete(obs, y_c[obs]) - true_card)
            stops.append(dict(model=cand, family=fam[fold], policy=name, order_id=rid, stop_K=st, mae=float(err.mean()),
                              covered=int((err <= radii[st - 1]).sum()), cost=cost.total(act_fit, act_inf, o[:st]), cost_full=full))
    return pd.DataFrame(rows), pd.DataFrame(stops)


def budget_mae(df, b=0.10):
    f = df.assign(frac=df.cost_actual / df.cost_full)
    return f.groupby(["policy", "model", "order_id"]).apply(lambda g: g[g.frac <= b + 1e-12].sort_values("K").iloc[-1].mae).groupby("policy").mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE.parents[1]))
    ap.add_argument("--evidence", type=Path, default=af.DEFAULT_EVIDENCE)
    ap.add_argument("--out", default="outputs/agent")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if Path(a.out).is_absolute() else root / a.out
    evidence_csv = root / a.evidence
    if evidence_csv.resolve().parent == out.resolve():
        raise ValueError("Output directory must differ from the input snapshot directory")
    ev = af.load_evidence(root, a.evidence)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    core = rp2.load_original(root)
    ev.cells.to_csv(out / "evidence_cells.csv", index=False)
    cases, cons, d2m_sum = d2m_final(ev, core)
    conf, d2m_sum["v4_confidence"] = rp4.d2m_confidence(ev, cases)
    guarded, d2m_sum["guarded_primary"] = d2m_guarded(ev, core, cases)
    guarded.to_csv(out / "d2m_guarded_cases.csv", index=False)
    cases.to_csv(out / "d2m_cases.csv", index=False)
    cons.to_csv(out / "d2m_constrained.csv", index=False)
    (out / "d2m_summary.json").write_text(json.dumps(d2m_sum, indent=2, default=float))
    folds, stops, sens, fobj, ctx = rp4.run_m2b(ev, core)
    rr_rows, rr_stops = track_round_robin(fobj, ctx)
    fam_rows, fam_stops = family_holdout(ev)
    folds.to_csv(out / "m2b_folds.csv", index=False)
    stops.to_csv(out / "m2b_stops.csv", index=False)
    rr_rows.to_csv(out / "m2b_track_round_robin.csv", index=False)
    fam_rows.to_csv(out / "m2b_family_holdout.csv", index=False)
    fam_stops.to_csv(out / "m2b_family_holdout_stops.csv", index=False)
    sens.to_csv(out / "m2b_random_seed_sensitivity.csv.gz", index=False, compression="gzip")
    summ = rp4.summarize(folds, stops, sens)
    summ["bootstrap_over_detectors"] = rp4.bootstrap(folds, stops)
    summ["constant_sensitivity_posthoc"] = rp4.constant_sensitivity(fobj, ctx)
    g = stops[(stops.rule == "inner_replay") & (stops.order == "v4")].merge(
        ev.cells.drop_duplicates("model")[["model", "family"]], on="model")
    summ["v4_stopping_details"] = {
        "mean_of_per_detector_saving": float(1 - (g.cost / g.cost_full).mean()),
        "aggregate_saving": float(1 - g.cost.sum() / g.cost_full.sum()),
        "component_coverage": float(g.covered.sum() / (4 * len(g))), "whole_card_coverage": float((g.covered == 4).mean()),
        "saving_by_family": {f: float(1 - (x.cost / x.cost_full).mean()) for f, x in g.groupby("family")},
        "first3_distinct_orders": int(folds[(folds.policy == "v4") & (folds.K == 3)].workloads.nunique())}
    rrs = rr_rows.assign(frac=rr_rows.cost_actual / rr_rows.cost_full)
    summ["track_round_robin"] = {"mae_by_K": {int(k): float(x) for k, x in rr_rows.groupby("K").mae.mean().items()},
                                 "mae_at_10pct": float(budget_mae(rr_rows).iloc[0]),
                                 "stop_K": float(rr_stops.stop_K.mean()), "saving_at_stop": float(1 - (rr_stops.cost / rr_stops.cost_full).mean()),
                                 "mae_at_stop": float(rr_stops.mae.mean())}
    fb = budget_mae(fam_rows)
    summ["family_holdout"] = {
        "mae_by_K": {p: {int(k): float(x) for k, x in d.groupby("K").mae.mean().items()} for p, d in fam_rows.groupby("policy")},
        "mae_at_10pct": {p: float(x) for p, x in fb.items()},
        "stopping": {p: {"stop_K": float(d.stop_K.mean()), "saving": float(1 - (d.cost / d.cost_full).mean()),
                         "mae": float(d.mae.mean()), "component_coverage": float(d.covered.sum() / (4 * len(d))),
                         "whole_card_coverage": float((d.covered == 4).mean())} for p, d in fam_stops.groupby("policy")},
        "v_final_mae_K4_by_family": {f: float(d.mae.mean()) for f, d in fam_rows[(fam_rows.policy == "v_final") & (fam_rows.K == 4)].groupby("family")},
        "random_mae_K4_by_family": {f: float(d.mae.mean()) for f, d in fam_rows[(fam_rows.policy == "random_ppca") & (fam_rows.K == 4)].groupby("family")}}
    (out / "m2b_summary.json").write_text(json.dumps(summ, indent=2))
    code = [HERE / name for name in ('final.py', 'evidence.py', 'd2m.py', 'cost.py', 'ppca.py', 'planning.py', 'baselines.py', 'comparisons.py', 'evaluation.py', 'replay.py', 'checks.py')]
    (out / "run_manifest.json").write_text(json.dumps({"seconds": time.time() - t0,
        "evidence": {"path": str(evidence_csv.resolve()), "sha256": sha(evidence_csv)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "reliability_metrics": kb.DIM_METRICS["reliability"], "code_sha256": {p.name: sha(p) for p in code}}, indent=2, default=str))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
