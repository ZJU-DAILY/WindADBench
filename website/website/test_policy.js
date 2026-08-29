/* Numerical cross-check: the browser policy port (core.js + policy.js) must
   reproduce the Python reference values baked into data.js by build_data.py.

   Run:  node test_policy.js
*/
"use strict";
const fs = require("fs");
const vm = require("vm");

const sandbox = {
  console,
  window: { WINDAD: null, addEventListener() {} },
  document: { querySelector: () => null, querySelectorAll: () => [] },
  localStorage: { getItem: () => null, setItem() {}, removeItem() {} },
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

for (const file of ["data.js", "core.js", "policy.js"]) {
  vm.runInContext(fs.readFileSync(file, "utf8"), sandbox, { filename: file });
}

let failures = 0;
function check(name, actual, expected, tol) {
  const ok = Math.abs(actual - expected) <= tol;
  if (!ok) failures++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}: ${actual.toFixed(6)} vs ${expected} (tol ${tol})`);
}

const D = sandbox.window.WINDAD;

/* 1. conformalRadius must match the Python conformal_radius for every prefix
      of the published evaluation plan. */
console.log("— conformal radius along the evaluation plan —");
for (const { K, radius } of D.modeB.radius) {
  const mine = sandbox.conformalRadius(D.modeB.order.slice(0, K));
  check(`radius@K=${K}`, mine, radius, 5e-4);
}

/* 2. A published model's own dimension percentiles must be recovered when its
      raw metrics are fed back through the user-input path.

      Where the published model has a genuine NA, the two paths are meant to
      differ: build_kb reads NA as "no qualifying alarm" and assigns the
      worst/best percentile, whereas a blank form field only means "not
      measured" and is skipped. Those dimensions are reported as SKIP. */
console.log("\n— userDimPct round-trip on published models —");
const DIM_METRICS = {
  accuracy: ["event_f1", "vus_pr"],
  earliness: ["early_detection_rate", "mean_detection_delay"],
  reliability: ["false_alarms_per_turbine_day", "mtbfa"],
  cost: ["infer_time", "infer_gpu_mem", "model_size"],
};
const row = (m, w) => D.metrics.find(r => r.m === m && r.w === w);
for (const [m, w] of [["cblof", "IF-A"], ["gdn", "IF-B"], ["moment", "CF-C>B"], ["usad", "CT-C"]]) {
  const t = row(m, w);
  const got = sandbox.userDimPct(w, t);
  const ref = D.dims.find(d => d.m === m && d.w === w);
  for (const [k, label] of [["a", "accuracy"], ["e", "earliness"], ["r", "reliability"], ["c", "cost"]]) {
    if (DIM_METRICS[label].some(metric => t[metric] == null)) {
      console.log(`SKIP  ${m}/${w} ${label}: reference has an NA under benchmark semantics`);
      continue;
    }
    // The new detector is ranked among 37 rather than 36, so a one-slot shift
    // (about 1/37) is expected; anything larger means the recipe diverged.
    check(`${m}/${w} ${label}`, got[k], ref[k], 0.04);
  }
}

/* 3. D2M must return a well-formed, deterministic decision. */
console.log("\n— D2M smoke —");
const weights = { accuracy: .25, earliness: .25, reliability: .25, cost: .25 };
for (const [label, evidence] of [
  ["cold start into C", D.workloads.filter(w => w.target === "C" && w.track === "cross-farm").map(w => w.workload)],
  ["labels on farm A", ["IF-A", "CT-A"]],
]) {
  const rec = sandbox.recommend({ weights, no_gpu: false }, evidence);
  const again = sandbox.recommend({ weights, no_gpu: false }, evidence);
  const stable = rec.primary === again.primary;
  if (!stable) failures++;
  console.log(`${stable ? "PASS" : "FAIL"}  ${label}: ${rec.decision} -> ${rec.primary}`
    + ` (confidence ${rec.confidence}, margin ${rec.margin.toFixed(3)},`
    + ` support ${rec.support}, feasible ${rec.nFeasible}, pareto ${rec.pareto.length})`);
}

/* 4. Hard constraints must actually remove models. */
const cpu = sandbox.recommend({ weights, no_gpu: true }, ["IF-A", "CT-A"]);
const all = sandbox.recommend({ weights, no_gpu: false }, ["IF-A", "CT-A"]);
const filtered = cpu.nFeasible < all.nFeasible && cpu.excluded.length > 0;
if (!filtered) failures++;
console.log(`${filtered ? "PASS" : "FAIL"}  CPU-only filter: ${all.nFeasible} -> ${cpu.nFeasible} feasible`);

console.log(`\n${failures ? failures + " FAILURE(S)" : "all checks passed"}`);
process.exit(failures ? 1 : 0);
