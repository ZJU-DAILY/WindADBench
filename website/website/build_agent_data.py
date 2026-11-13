from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'agent/_impl'))
import final as af
import evidence as kb
import d2m
import cost as m2
import ppca as m3
import planning as a4


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def coefficient(model):
    return dict(mu=model.mu, W=model.W, sigma2=model.sigma2, rank=model.rank)


def build(evidence, results_root, verification_output=None):
    ev = af.load_evidence(ROOT, evidence)
    models, workloads = ev.models, sorted(ev.meta.workload)
    refs = list(range(len(models)))
    raw = m3.RawTensor(ev.cells, models, workloads)
    Y = raw.scores(refs, refs).reshape(len(refs), -1)
    q = kb.support_weights(workloads, ev.meta).to_numpy()
    A = m3.card_matrix(q)
    ppca = m3.PPCA(Y)
    inner = a4.InnerReplay(raw, refs, A)
    cost = m2.CostModel(ev.cells, workloads)
    fit, infer = cost.components(models)
    tracks = kb.workload_table().set_index('workload').loc[workloads, 'track'].to_numpy()
    marginal = lambda chosen, i: cost.marginal(fit, infer, chosen, i)
    order = a4.plan_order(inner, tracks, marginal, np.array([marginal([], i) for i in refs[:len(workloads)]]))
    scores = ev.array(kb.dim_scores(ev.cells, models), models, workloads)
    fh_models, farms = {}, {}
    for target in kb.FARMS:
        evidence_w = d2m.lofo_sets(target)['evidence']
        fh = d2m.FayHerriotD2M(scores, workloads, ev.meta, evidence_w)
        fh_models[target] = fh
        selected = ev.cells[ev.cells.workload.isin(evidence_w)]
        fa = selected.groupby('model').fa_day.mean().reindex(models)
        fpr = selected[selected.track.eq('cross-farm')].groupby('model').normal_fpr.mean().reindex(models)
        farms[target] = dict(theta=fh.theta, V=fh.V, evidence=evidence_w, fa_day=fa.to_numpy(), normal_fpr=fpr.to_numpy())
    d_summary = json.loads((results_root / 'd2m_summary.json').read_text(encoding='utf-8'))
    m_summary = json.loads((results_root / 'm2b_summary.json').read_text(encoding='utf-8'))
    folds = pd.read_csv(results_root / 'm2b_folds.csv', float_precision='round_trip')
    random = folds[folds.policy.eq('random_ppca')].groupby('K').mae.quantile([.25, .5, .75]).unstack()
    replay = dict(d2m=d_summary['guarded_primary'], m2b=m_summary['policies']['v4'],
                  random=m_summary['policies']['random_ppca'],
                  random_curve=[dict(K=int(k), p25=row[.25], p50=row[.5], p75=row[.75]) for k, row in random.iterrows()])
    payload = dict(version='agent_final', models=models, workloads=workloads, dims=kb.DIMS,
        metrics=[dict(dimension=d, key=key, higher=higher, missing=rule) for d, key, higher, rule in m3.METRICS],
        raw=raw.X, q=q, reference_cards=Y @ A.T,
        cpu_only=ev.cells.groupby('model').cpu_only.first().reindex(models).to_numpy(),
        d2m=farms, ppca=coefficient(ppca), inner=[dict(**coefficient(mdl), y=y) for mdl, y in inner.items],
        plan=dict(order=[workloads[i] for i in order], radius=[inner.radius(order[:k]) for k in range(1, len(order)+1)],
                  marginal_seconds=[marginal(order[:k], i) for k, i in enumerate(order)]),
        policy=dict(risk_aversion=a4.LAMBDA, abstain_gap=a4.EPSILON, stop_radius=a4.STOP_RADIUS,
                    quantile=a4.QUANTILE, max_active=a4.MAX_ACTIVE), replay=replay,
        evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest())
    if verification_output:
        examples = dict(d2m=[], m2b=[])
        rng = np.random.default_rng(2026)
        for target in kb.FARMS:
            for weights in d2m.preference_grid() + [rng.dirichlet(np.ones(4)) for _ in range(15)]:
                for cpu, cap, fpr in [(False, None, .1), (True, 1., .2), (False, None, None)]:
                    excluded = af.exclusion_reasons(ev.cells, models, farms[target]['evidence'], cpu, cap, fpr)
                    rec = af.recommend(fh_models[target], models, weights, excluded)
                    utility, lcb = fh_models[target].scores(weights)
                    examples['d2m'].append(dict(target=target, weights=weights, no_gpu=cpu, max_fa_day=cap,
                        max_normal_fpr=fpr, decision=rec['decision'], primary=rec['primary'], top_k=rec['top_k'],
                        margin=rec['margin'], utility=utility, lcb=lcb))
        choices = [[], list(range(len(workloads)))] + [list(map(int, rng.choice(len(workloads), k, replace=False))) for k in range(1, 12)]
        choices += [order[:k] for k in range(1, len(order)+1)]
        for j, chosen in enumerate(choices):
            values = raw.X[j % len(models)]
            parts = np.empty_like(values)
            for w in range(len(workloads)):
                for k, (_, _, higher, rule) in enumerate(m3.METRICS):
                    parts[w, k] = kb._insert_percentile([values[w, k]], higher, rule, raw.X[:, w, k])[0]
            dim = np.stack([np.nanmean(parts[:, raw.dim_of == d], axis=1) for d in range(4)], axis=1)
            obs = m3.cells_of(chosen)
            predicted = ppca.complete(obs, dim.reshape(-1)[obs])
            examples['m2b'].append(dict(chosen=[workloads[i] for i in chosen], raw=values,
                dimensions=dim, completed=predicted, card=A @ predicted, radius=inner.radius(chosen)))
        verification_output.write_text(json.dumps(clean(examples), allow_nan=False), encoding='utf-8')
    return clean(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evidence', type=Path, default=ROOT / 'agent/evidence_cells.csv')
    parser.add_argument('--results-root', type=Path, default=ROOT / 'outputs/agent')
    parser.add_argument('--output', type=Path, default=HERE / 'agent_data.js')
    parser.add_argument('--verification-output', type=Path)
    args = parser.parse_args()
    payload = build(args.evidence.resolve(), args.results_root.resolve(), args.verification_output)
    args.output.write_text('window.WINDAD_AGENT=' + json.dumps(payload, allow_nan=False, separators=(',', ':')) + ';\n', encoding='utf-8')
    print(args.output)


if __name__ == '__main__':
    main()
