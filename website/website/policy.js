"use strict";

const F = window.WINDAD_AGENT;
const POLICY = F.policy;
const modelFlags = m => REG[m].flags || [];
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

function normWeights(weights) {
  const values = F.dims.map(d => Math.max(0, Number(weights[d]) || 0));
  const total = values.reduce((a, b) => a + b, 0);
  return total ? values.map(v => v / total) : values.map(() => .25);
}

function recommend(profile, target) {
  const source = F.d2m[target], weights = normWeights(profile.weights);
  const stats = [], excluded = [];
  F.models.forEach((model, i) => {
    const reasons = [];
    if (profile.no_gpu && !F.cpu_only[i]) reasons.push("requires a GPU");
    if (profile.max_fa_day != null && source.fa_day[i] > profile.max_fa_day)
      reasons.push("mean FA/day " + source.fa_day[i].toFixed(2) + " > " + profile.max_fa_day);
    if (profile.max_normal_fpr != null && source.normal_fpr[i] > profile.max_normal_fpr)
      reasons.push("cross-farm normal FPR " + source.normal_fpr[i].toFixed(3) + " > " + profile.max_normal_fpr);
    if (reasons.length) { excluded.push([model, reasons.join("; ")]); return; }
    const utilityMean = dot(source.theta[i], weights);
    const utilitySe = Math.sqrt(dot(source.V[i], weights.map(x => x * x)));
    stats.push({ model, index: i, utilityMean, utilitySe, lcb: utilityMean - POLICY.risk_aversion * utilitySe,
      dims: Object.fromEntries(F.dims.map((d, j) => [d, source.theta[i][j]])) });
  });
  stats.sort((a, b) => b.lcb - a.lcb || b.utilityMean - a.utilityMean || a.index - b.index);
  const base = { evidence: source.evidence, excluded, nFeasible: stats.length, target, topK: stats.slice(0, 3) };
  if (!stats.length) return { ...base, decision: "no_feasible_detector", primary: null, margin: null };
  const margin = stats.length === 1 ? Infinity : stats[0].lcb - stats[1].lcb;
  const pareto = stats.filter(s => !stats.some(t => F.dims.every(d => t.dims[d] >= s.dims[d])
    && F.dims.some(d => t.dims[d] > s.dims[d]))).map(s => s.model);
  const byDimension = d => [...stats].sort((a, b) => b.dims[d] - a.dims[d] || a.index - b.index)[0].model;
  return { ...base, primary: stats[0].model, margin, pareto,
    decision: stats.length === 1 ? "recommend_single_feasible" : margin >= POLICY.abstain_gap ? "recommend" : "abstain",
    lowCostFallback: byDimension("cost"), highSensitivityFallback: byDimension("earliness") };
}

function insertedPercentile(value, pool, spec) {
  const sign = spec.higher ? 1 : -1;
  const key = v => v == null ? (spec.missing === "worst" ? -Infinity : spec.missing === "best" ? Infinity : null) : sign * v;
  const x = key(value);
  if (x == null) return null;
  const population = pool.map(key).filter(v => v != null);
  let worse = 0, ties = 0;
  for (const v of population) { if (v < x) worse++; else if (v === x) ties++; }
  return (1 + worse + .5 * ties) / (population.length + 1);
}

function userDimPct(workload, values) {
  const w = F.workloads.indexOf(workload);
  if (w < 0) throw new Error("Unknown workload");
  const parts = F.metrics.map((spec, k) => {
    const value = values[spec.key];
    if (spec.missing === "error" && (value == null || !Number.isFinite(value)))
      throw new Error(workload + ": enter " + spec.key);
    if (value != null && (!Number.isFinite(value) || value < 0)) throw new Error(workload + ": invalid " + spec.key);
    if (["event_f1", "vus_pr", "edr", "normal_fpr"].includes(spec.key) && value > 1)
      throw new Error(workload + ": " + spec.key + " must be between 0 and 1");
    return insertedPercentile(value, F.raw.map(model => model[w][k]), spec);
  });
  return Object.fromEntries(F.dims.map((d, i) => {
    const valid = parts.filter((v, k) => F.metrics[k].dimension === i && v != null);
    if (!valid.length) throw new Error(workload + ": no " + d + " measurement");
    return [d, valid.reduce((a, b) => a + b, 0) / valid.length];
  }));
}

function solve(matrix, vector) {
  const n = vector.length, a = matrix.map((row, i) => [...row, vector[i]]);
  for (let c = 0; c < n; c++) {
    let pivot = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[pivot][c])) pivot = r;
    [a[c], a[pivot]] = [a[pivot], a[c]];
    if (Math.abs(a[c][c]) < 1e-15) throw new Error("Singular PPCA posterior");
    for (let r = c + 1; r < n; r++) {
      const factor = a[r][c] / a[c][c];
      for (let j = c; j <= n; j++) a[r][j] -= factor * a[c][j];
    }
  }
  const answer = Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let value = a[r][n];
    for (let j = r + 1; j < n; j++) value -= a[r][j] * answer[j];
    answer[r] = value / a[r][r];
  }
  return answer;
}

function ppcaComplete(model, obs, values) {
  if (!obs.length) return [...model.mu];
  const matrix = Array.from({ length: model.rank }, (_, i) => Array.from({ length: model.rank }, (_, j) => i === j ? 1 : 0));
  const rhs = Array(model.rank).fill(0);
  obs.forEach((k, index) => {
    const row = model.W[k], residual = (values[index] - model.mu[k]) / model.sigma2;
    for (let i = 0; i < model.rank; i++) {
      rhs[i] += row[i] * residual;
      for (let j = 0; j < model.rank; j++) matrix[i][j] += row[i] * row[j] / model.sigma2;
    }
  });
  const z = solve(matrix, rhs), result = model.mu.map((v, i) => v + dot(model.W[i], z));
  obs.forEach((k, i) => result[k] = values[i]);
  return result;
}

function performanceCard(values) {
  return F.dims.map((_, d) => F.q.reduce((s, q, w) => s + q * values[w * 4 + d], 0));
}

function completedCard(revealed) {
  const obs = [], values = [];
  F.workloads.forEach((w, i) => {
    if (!revealed[w]) return;
    F.dims.forEach((d, j) => { obs.push(i * 4 + j); values.push(revealed[w][d]); });
  });
  const completed = ppcaComplete(F.ppca, obs, values), card = performanceCard(completed);
  return { card: Object.fromEntries(F.dims.map((d, i) => [d, card[i]])), completed };
}

const radiusCache = new Map();
function conformalRadius(workloads) {
  const selected = F.workloads.map((w, i) => workloads.includes(w) ? i : -1).filter(i => i >= 0);
  const key = selected.join(",");
  if (radiusCache.has(key)) return radiusCache.get(key);
  const obs = selected.flatMap(w => [0, 1, 2, 3].map(d => w * 4 + d));
  const residuals = [];
  for (const item of F.inner) {
    const completed = ppcaComplete(item, obs, obs.map(i => item.y[i]));
    const delta = performanceCard(completed.map((v, i) => v - item.y[i]));
    residuals.push(...delta.map(Math.abs));
  }
  residuals.sort((a, b) => a - b);
  const k = Math.ceil(POLICY.quantile * (residuals.length + 1));
  const radius = k <= residuals.length ? residuals[k - 1] : Infinity;
  radiusCache.set(key, radius);
  return radius;
}

function cardRank(card) {
  const mine = F.dims.reduce((sum, d) => sum + card[d], 0) / F.dims.length;
  const values = F.reference_cards.map(row => row.reduce((a, b) => a + b, 0) / row.length);
  const better = values.filter(v => v > mine).length, tied = values.filter(v => v === mine).length;
  return { rank: 1 + better + tied / 2, of: F.models.length + 1, aggregate: mine };
}
