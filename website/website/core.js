/* WindADBench — shared indices, metric registry, formatting, and routing.
   Loaded first; every other module builds on the names defined here. */
"use strict";

const D = window.WINDAD;
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

/* ---------------- data indices ---------------- */
const REG = Object.fromEntries(D.registry.map(r => [r.id, r]));
const MODELS = Object.keys(REG).sort();
const WLS = D.workloads.map(w => w.workload);
const WMETA = Object.fromEntries(D.workloads.map(w => [w.workload, w]));
const dimMap = {}; for (const d of D.dims) (dimMap[d.m] ??= {})[d.w] = d;
const metMap = {}; for (const m of D.metrics) (metMap[m.m] ??= {})[m.w] = m;
const DIMS = ["accuracy", "earliness", "reliability", "cost"];
const DIMK = { accuracy: "a", earliness: "e", reliability: "r", cost: "c" };
const CARD_DIMS = ["accuracy", "earliness", "reliability", "generalization", "cost"];
const FAMS = [...new Set(D.registry.map(r => r.family))].sort();
const FAM_FULL = {
  NL: "Non-Learning", ML: "Machine Learning", DL: "Deep Learning",
  LLM: "LLM-based", TSP: "Time-Series Pretrained", DLLM: "Domain-adapted LLM",
};
const famOptions = () => FAMS.map(f => `<option value="${f}">${f} · ${FAM_FULL[f] ?? f}</option>`).join("");

/* metric registry: label, direction (+1 higher better), decimals */
const M = {
  acc: ["Accuracy", 1, 3], point_precision: ["Point-P", 1, 3], point_recall: ["Point-R", 1, 3],
  point_f1: ["Point-F1", 1, 3], event_precision: ["Event-P", 1, 3], event_recall: ["Event-R", 1, 3],
  event_f1: ["Event-F1", 1, 3], event_affiliation_f1: ["Ev-Aff-F1", 1, 3],
  range_precision: ["Range-P", 1, 3], range_recall: ["Range-R", 1, 3], range_f1: ["Range-F1", 1, 3],
  affiliation_precision: ["Aff-P", 1, 3], affiliation_recall: ["Aff-R", 1, 3], affiliation_f1: ["Aff-F1", 1, 3],
  auc_pr: ["AUC-PR", 1, 3], auc_roc: ["AUC-ROC", 1, 3], range_auc_pr: ["R-AUC-PR", 1, 3],
  range_auc_roc: ["R-AUC-ROC", 1, 3], vus_pr: ["VUS-PR", 1, 3], vus_roc: ["VUS-ROC", 1, 3],
  mean_lead_time: ["Lead", 1, 1], mean_detection_delay: ["Delay", -1, 1],
  early_detection_rate: ["Early rate", 1, 3], false_alarms_per_turbine_day: ["FA/day", -1, 3],
  mtbfa: ["MTBFA", 1, 1],
  fit_time: ["Fit (s)", -1, 2], infer_time: ["Infer (s)", -1, 3],
  infer_gpu_mem: ["GPU (MB)", -1, 0], model_size: ["Size (MB)", -1, 1],
};
const GROUPS = {
  "Overview": ["event_f1", "vus_pr", "early_detection_rate", "mean_detection_delay",
               "false_alarms_per_turbine_day", "mtbfa", "infer_time"],
  "Detection": ["acc", "point_f1", "event_f1", "event_recall", "range_f1", "affiliation_f1", "event_affiliation_f1"],
  "Score-based": ["auc_pr", "auc_roc", "range_auc_pr", "range_auc_roc", "vus_pr", "vus_roc"],
  "Operational": ["mean_lead_time", "mean_detection_delay", "early_detection_rate",
                  "false_alarms_per_turbine_day", "mtbfa"],
  "Cost": ["fit_time", "infer_time", "infer_gpu_mem", "model_size"],
};

/* ---------------- formatting helpers ---------------- */
const esc = (s) => String(s).replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, k) => v == null ? "—" : v.toFixed(M[k][2]);
const wlLabel = (w) => w.replace(">", "→");
const famChip = (f) => `<span class="chip f-${f}" title="${FAM_FULL[f] ?? f}">${f}</span>`;
const flagHtml = (m) => (REG[m].flags || [])
  .map(f => `<span class="flag" title="rule-based failure flag">⚠ ${esc(f)}</span>`).join("");
const pctText = (v) => v == null ? "—" : Math.round(100 * v);

const mean = (a) => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
const meanOf = (a) => mean(a.filter(v => v != null && isFinite(v)));

/** Linear-interpolation quantile over an ascending array (numpy default). */
function quantile(sorted, p) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * p, lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function avgMetric(m, ws, k) {
  return meanOf(ws.map(w => metMap[m][w]?.[k]));
}
function meanDims(m, ws) {
  const o = {};
  for (const d of DIMS) o[d] = meanOf(ws.map(w => dimMap[m][w]?.[DIMK[d]])) ?? 0;
  return o;
}
function bars(vals, labels) {
  return `<div class="dimbars">` + labels.map((d, i) => `
    <div class="dimbar"><span>${d}</span><span class="bar"><i style="width:${Math.round(100 * (vals[i] ?? 0))}%"></i></span><span>${pctText(vals[i])}</span></div>`).join("") + `</div>`;
}

/* ---------------- router ---------------- */
const VIEWS = ["overview", "leaderboard", "models", "workloads", "agent"];
function route() {
  let h = (location.hash || "#overview").slice(1).split("/")[0];
  if (h === "newmodel" || h === "newdataset") h = "agent";  // legacy anchors
  const v = VIEWS.includes(h) ? h : "overview";
  VIEWS.forEach(x => $(`#view-${x}`).classList.toggle("active", x === v));
  $$("#nav a").forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + v));
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);
