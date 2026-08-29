/* WindADBench — decision policies, ported 1:1 from analysis/agent/agent_core.py.
   Everything here is deterministic and runs entirely in the browser: no model
   call is involved in producing a decision. */
"use strict";

const POLICY = D.policy;
const DIMKEYS = ["a", "e", "r", "c"];

/* ---------------------------------------------------------------- shared -- */

/** log-support weights over an evidence set (agent_core._support_weights). */
function supportWeights(evidence) {
  const q = evidence.map(w => Math.log1p(Math.max(1, WMETA[w].n_events)));
  const s = q.reduce((a, b) => a + b, 0) || 1;
  return q.map(x => x / s);
}

/** non-negative, sum-to-one preference weights (agent_core._normalise_weights). */
function normWeights(weights) {
  const w = DIMS.map(d => Math.max(0, +weights[d] || 0));
  const s = w.reduce((a, b) => a + b, 0);
  return s <= 0 ? DIMS.map(() => 1 / DIMS.length) : w.map(x => x / s);
}

/* ------------------------------------------------------------------- D2M -- */

/**
 * Support-weighted mean, standard error, and utility per model.
 * Port of agent_core.utility_stats: a workload only contributes when all four
 * dimensions are finite, and the per-model weights are renormalised over the
 * workloads that survive that mask.
 */
function utilityStats(weights, evidence, models) {
  const q = supportWeights(evidence), w = normWeights(weights);
  const support = evidence.reduce((s, wl) => s + WMETA[wl].n_events, 0);
  return models.map(m => {
    const cells = evidence.map((wl, i) => {
      const d = dimMap[m]?.[wl];
      const v = d ? DIMKEYS.map(k => d[k]) : null;
      const ok = v && v.every(x => x != null && isFinite(x));
      return { v: ok ? v : [0, 0, 0, 0], q: ok ? q[i] : 0 };
    });
    const total = Math.max(cells.reduce((s, c) => s + c.q, 0), 1e-12);
    const wm = cells.map(c => c.q / total);
    const means = DIMS.map((_, j) => cells.reduce((s, c, i) => s + c.v[j] * wm[i], 0));
    const vars = DIMS.map((_, j) =>
      cells.reduce((s, c, i) => s + wm[i] * (c.v[j] - means[j]) ** 2, 0));
    const nEff = 1 / Math.max(wm.reduce((s, x) => s + x * x, 0), 1e-12);
    const ses = vars.map(v => Math.sqrt(v / Math.max(nEff, 1)));
    const utilityMean = means.reduce((s, v, j) => s + v * w[j], 0);
    const utilitySe = Math.sqrt(ses.reduce((s, v, j) => s + (v * w[j]) ** 2, 0));
    const dims = Object.fromEntries(DIMS.map((d, j) => [d, means[j]]));
    return { model: m, utilityMean, utilitySe, dims, support };
  });
}

/** Hard deployment constraints, applied before any preference-based ranking. */
function feasibleModels(profile, evidence) {
  const feasible = [], excluded = [];
  for (const m of MODELS) {
    const reg = REG[m];
    if (profile.no_gpu && reg.needs_gpu) { excluded.push([m, "needs GPU"]); continue; }
    if (profile.max_fa_day != null) {
      const fa = avgMetric(m, evidence, "false_alarms_per_turbine_day");
      if (fa != null && fa > profile.max_fa_day) {
        excluded.push([m, `FA/day ${fa.toFixed(2)} > ${profile.max_fa_day}`]); continue;
      }
    }
    if (profile.max_model_size != null && reg.model_size != null
        && reg.model_size > profile.max_model_size) {
      excluded.push([m, `size ${reg.model_size.toFixed(1)} MB > ${profile.max_model_size}`]); continue;
    }
    if (profile.min_earliness != null) {
      const e = avgMetric(m, evidence, "early_detection_rate");
      if (e != null && e < profile.min_earliness) {
        excluded.push([m, `early rate ${e.toFixed(3)} < ${profile.min_earliness}`]); continue;
      }
    }
    if (profile.max_infer_time != null && reg.infer_time != null
        && reg.infer_time > profile.max_infer_time) {
      excluded.push([m, `infer ${reg.infer_time.toFixed(3)} s > ${profile.max_infer_time}`]); continue;
    }
    feasible.push(m);
  }
  return { feasible, excluded };
}

/** Non-dominated models on equal-weight, support-weighted dimension means. */
function paretoFront(evidence, models) {
  const stats = utilityStats(Object.fromEntries(DIMS.map(d => [d, 0.25])), evidence, models);
  const vals = stats.map(s => DIMS.map(d => s.dims[d]));
  return stats.filter((_, i) => !vals.some((o, j) => j !== i
    && o.every((x, k) => x >= vals[i][k]) && o.some((x, k) => x > vals[i][k]))).map(s => s.model);
}

/** Conservative, rule-based warnings (precomputed in data.js by failure_flags). */
const modelFlags = (m) => REG[m].flags || [];

/**
 * D2M policy: filter, risk-adjust, and return an auditable portfolio.
 * Port of agent_core.recommend.
 */
function recommend(profile, evidence) {
  const { feasible, excluded } = feasibleModels(profile, evidence);
  if (!feasible.length) {
    return { decision: "abstain", reason: "no feasible model", primary: null, topK: [],
             pareto: [], nFeasible: 0, excluded, evidence, confidence: "none", margin: 0 };
  }
  const stats = utilityStats(profile.weights, evidence, feasible)
    .map(s => ({ ...s, lcb: s.utilityMean - POLICY.risk_aversion * s.utilitySe }))
    .sort((a, b) => (b.lcb - a.lcb) || (b.utilityMean - a.utilityMean));
  const top = stats.slice(0, POLICY.top_k);
  const margin = stats.length > 1 ? stats[0].lcb - stats[1].lcb : stats[0].utilitySe;
  const support = stats[0].support;
  const sparseTransfer = evidence.some(w =>
    WMETA[w].track === "cross-farm" && WMETA[w].n_events <= 6);
  const confidence = (sparseTransfer || stats[0].utilitySe > 0.06) ? "medium"
    : support >= 24 ? "high" : "low";
  const abstain = support < POLICY.min_support || margin < POLICY.abstain_gap;
  const byDim = (d) => stats.reduce((best, s) => s.dims[d] > best.dims[d] ? s : best, stats[0]).model;
  return {
    decision: abstain ? "abstain" : "recommend",
    reason: abstain
      ? (support < POLICY.min_support
        ? `evidence support ${support} events < ${POLICY.min_support}`
        : `top-2 margin ${margin.toFixed(3)} < ${POLICY.abstain_gap}`)
      : null,
    primary: top[0].model, topK: top, stats,
    pareto: paretoFront(evidence, feasible),
    lowCostFallback: byDim("cost"), highSensitivityFallback: byDim("earliness"),
    nFeasible: feasible.length, excluded, evidence,
    confidence, margin, uncertainty: stats[0].utilitySe, support,
  };
}

/** Closest benchmark farm by log-scale size and site type. */
function nearestFarm(turbines, features, type) {
  let best = null, bd = Infinity;
  for (const [farm, m] of Object.entries(D.farms)) {
    const d = Math.abs(Math.log(turbines / m.turbines))
      + Math.abs(Math.log(features / m.features)) + (type === m.type ? 0 : 0.35);
    if (d < bd) { bd = d; best = farm; }
  }
  return best;
}

/* ------------------------------------------------------------------- M2B -- */

/**
 * Percentile of `v` among the reference column, using pandas' average-tie
 * ranking so a new detector is scored exactly as if it had been added to the
 * knowledge base (rank among 37, not "fraction of 36 beaten").
 */
function rankPct(refs, v, higherBetter) {
  const pool = refs.filter(x => x != null && isFinite(x));
  if (!pool.length || v == null || !isFinite(v)) return null;
  let lower = 0, equal = 1;  // `equal` starts at 1 for v itself
  for (const x of pool) {
    if (x === v) equal++;
    else if (higherBetter ? x < v : x > v) lower++;
  }
  return (lower + (equal + 1) / 2) / (pool.length + 1);
}

/* dimension recipe, mirroring build_kb.build_dim_pct including its NA rules */
const DIM_SPEC = {
  a: [["event_f1", true], ["vus_pr", true]],
  e: [["early_detection_rate", true], ["mean_detection_delay", false, "worst"]],
  r: [["false_alarms_per_turbine_day", false], ["mtbfa", true, "best"]],
  c: [["infer_time", false], ["infer_gpu_mem", false, "zero"], ["model_size", false, "zero"]],
};

/** Turn one workload's raw measurements into the four dimension percentiles. */
function userDimPct(w, vals) {
  const out = {};
  for (const [dim, specs] of Object.entries(DIM_SPEC)) {
    const parts = specs.map(([k, higher, na]) => {
      const raw = MODELS.map(m => metMap[m][w]?.[k]);
      const pool = raw.filter(x => x != null && isFinite(x));
      if (!pool.length) return null;
      // build_kb fills the reference column BEFORE ranking, so the pool a new
      // detector is ranked against has to be filled the same way.
      const fill = na == null ? null
        : na === "zero" ? 0
        : na === "best" ? (higher ? Math.max(...pool) + 1 : Math.min(...pool) - 1)
        : (higher ? Math.min(...pool) - 1 : Math.max(...pool) + 1);
      const refs = fill == null ? raw
        : raw.map(x => (x == null || !isFinite(x)) ? fill : x);
      // A blank input means "not measured", which is not the same as the
      // benchmark's "no qualifying alarm", so it is skipped rather than filled.
      return rankPct(refs, vals[k], higher);
    });
    const ok = parts.filter(x => x != null);
    out[dim] = ok.length ? mean(ok) : null;
  }
  return out;
}

/**
 * Complete the unseen cells of a partial card with the k nearest reference
 * models in revealed-percentile space (agent_core.completed_card).
 */
function completedCard(revealed) {
  const ws = Object.keys(revealed);
  const dist = MODELS.map(m => {
    const ds = [];
    for (const w of ws) for (const k of DIMKEYS) {
      const u = revealed[w][k], r = dimMap[m][w]?.[k];
      if (u != null && r != null) ds.push(Math.abs(u - r));
    }
    return [m, ds.length ? mean(ds) : Infinity];
  }).sort((a, b) => a[1] - b[1]);
  const neighbours = dist.slice(0, POLICY.knn_k).map(x => x[0]);
  const cell = {};
  for (const w of WLS) {
    cell[w] = {};
    for (const k of DIMKEYS) {
      const u = revealed[w]?.[k];
      cell[w][k] = u != null ? u : meanOf(neighbours.map(n => dimMap[n][w]?.[k]));
    }
  }
  const card = {};
  for (const d of DIMS) card[d] = meanOf(WLS.map(w => cell[w][DIMK[d]]));
  card.generalization = meanOf(WLS.filter(w => WMETA[w].track !== "in-farm").map(w => cell[w].a));
  return { card, neighbours, cell };
}

/**
 * 90% reference-residual radius for a card built from `revealed` workloads
 * (agent_core.conformal_radius). Depends only on which workloads were run,
 * so it is available before any measurement is entered.
 */
function conformalRadius(revealed, alpha = 0.1) {
  const sel = WLS.map((w, i) => [w, i]).filter(([w]) => revealed.includes(w)).map(x => x[1]);
  const rest = WLS.map((_, i) => i).filter(i => !sel.includes(i));
  if (!rest.length || MODELS.length <= POLICY.knn_k) return 0;
  const V = MODELS.map(m => WLS.map(w => DIMKEYS.map(k => dimMap[m][w][k])));
  const residuals = [];
  for (let i = 0; i < MODELS.length; i++) {
    const dists = [];
    for (let j = 0; j < MODELS.length; j++) {
      if (j === i) continue;
      let s = 0, n = 0;
      for (const wi of sel) for (let k = 0; k < 4; k++) { s += Math.abs(V[j][wi][k] - V[i][wi][k]); n++; }
      dists.push([n ? s / n : 0, j]);
    }
    dists.sort((a, b) => a[0] - b[0]);
    const nb = dists.slice(0, POLICY.knn_k).map(x => x[1]);
    for (let k = 0; k < 4; k++) {
      let done = 0, truth = 0;
      for (let wi = 0; wi < WLS.length; wi++) {
        done += rest.includes(wi) ? mean(nb.map(j => V[j][wi][k])) : V[i][wi][k];
        truth += V[i][wi][k];
      }
      residuals.push(Math.abs(done / WLS.length - truth / WLS.length));
    }
  }
  residuals.sort((a, b) => a - b);
  return quantile(residuals, 1 - alpha);
}

/** Equal-weight aggregate rank of a completed card among the 36 published models. */
function cardRank(card) {
  const mine = meanOf(DIMS.map(d => card[d]));
  const better = MODELS.filter(m => meanOf(DIMS.map(d => D.cards[m][d])) > mine).length;
  return { rank: better + 1, of: MODELS.length + 1, aggregate: mine };
}
