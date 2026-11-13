"use strict";

const D2M_PRESETS = {
  "Balanced": { accuracy: .25, earliness: .25, reliability: .25, cost: .25 },
  "Limited review capacity": { accuracy: .2, earliness: .1, reliability: .6, cost: .1 },
  "Fault-safety first": { accuracy: .3, earliness: .5, reliability: .1, cost: .1 },
  "Efficiency first": { accuracy: .2, earliness: .2, reliability: .2, cost: .4 },
};

function d2mWeights() {
  const values = Object.fromEntries(F.dims.map(d => [d, +$(`#ndw-${d}`).value]));
  const weights = normWeights(values);
  F.dims.forEach((d, i) => $(`#ndv-${d}`).textContent = Math.round(100 * weights[i]) + "%");
  return Object.fromEntries(F.dims.map((d, i) => [d, weights[i]]));
}

function d2mProfile() {
  const number = id => $(id).value === "" ? null : +$(id).value;
  return { weights: d2mWeights(), no_gpu: $("#nd-cpu").checked,
    max_fa_day: number("#nd-fa"), max_normal_fpr: number("#nd-fpr") };
}

function runD2M() {
  if (["#nd-fa", "#nd-fpr"].some(id => $(id).validity.badInput)) {
    $("#nd-out").innerHTML = '<p class="error">Enter a complete numeric constraint.</p>';
    return;
  }
  const target = $("#nd-target").value, profile = d2mProfile();
  if ((profile.max_fa_day != null && profile.max_fa_day < 0) ||
      (profile.max_normal_fpr != null && (profile.max_normal_fpr < 0 || profile.max_normal_fpr > 1))) {
    $("#nd-out").innerHTML = '<p class="error">Enter a nonnegative FA/day limit and a normal FPR between 0 and 1.</p>';
    return;
  }
  renderD2M(recommend(profile, target), target, profile);
}

function renderD2M(rec, target, profile) {
  const host = $("#nd-out");
  if (!rec.primary) {
    host.innerHTML = `<div class="verdict warn"><span class="klabel">No feasible detector</span>
      <div class="name">Constraints exclude all reference models</div>
      <p class="small muted">Adjust the FA/day, normal FPR, or CPU-only constraint.</p></div>`;
    return;
  }
  const abstain = rec.decision === "abstain";
  const chips = model => `${famChip(REG[model].family)}${rec.pareto.includes(model)
    ? '<span class="star" title="Four-dimension Pareto front">★</span>' : ""}`;
  const margin = Number.isFinite(rec.margin) ? rec.margin.toFixed(3) : "single feasible model";
  const replay = F.replay.d2m;
  host.innerHTML = `<div class="verdict ${abstain ? "warn" : "ok"}">
    <div class="verdict-head"><span class="klabel">${abstain ? "Shortlist for pilot evaluation" : "Recommended detector"}</span>
      <span class="conf ${abstain ? "conf-medium" : "conf-high"}">LCB gap ${margin}</span></div>
    <div class="name">${rec.primary} ${chips(rec.primary)}</div>
    <p class="verdict-sub">Target Farm <b>${target}</b> · evidence ${rec.evidence.map(wlLabel).join(", ")}</p>
    ${abstain ? `<p class="verdict-note">The top-two gap is below ${POLICY.abstain_gap}; evaluate the shortlist before selecting one detector.</p>` : ""}
    ${bars(F.dims.map(d => rec.topK[0].dims[d]), F.dims)}</div>
    ${explainBlock("Explain this recommendation")}
    <details class="fold"><summary>Shortlist and evidence</summary>
      <ul class="rec-list">${rec.topK.map((s, i) => `<li><span>${i + 1}. <b>${s.model}</b> ${chips(s.model)}</span>
        <span>LCB ${s.lcb.toFixed(3)}</span></li>`).join("")}
        <li><span>Lowest-cost fallback</span><span>${rec.lowCostFallback}</span></li>
        <li><span>Earliest-warning fallback</span><span>${rec.highSensitivityFallback}</span></li></ul>
      <p class="small muted">${rec.nFeasible} feasible models · ${rec.excluded.length} excluded.
        Fay–Herriot estimates use only the other two farms, with risk aversion ${POLICY.risk_aversion}.</p>
      ${rec.excluded.length ? `<p class="small muted">${rec.excluded.map(([m, why]) => `${esc(m)}: ${esc(why)}`).join("<br>")}</p>` : ""}
      <p class="fineprint">Leave-one-farm-out replay with the 0.10 normal-FPR guard:
        mean regret ${replay.v_final.mean_regret.toFixed(4)}, compared with ${replay.global_best_eq.mean_regret.toFixed(4)}
        for the global-best baseline and ${replay.random.mean_regret.toFixed(4)} for random choice.</p></details>`;
  host.querySelector(".explain-btn").onclick = () => runExplanation(host.querySelector(".explain-out"),
    "Explain the detector shortlist and its evidence for the selected target farm.", d2mFacts(rec, target, profile.weights));
}

const M2B_FIELDS = [
  ["event_f1", "Event-F1"], ["vus_pr", "VUS-PR"], ["edr", "Early detection rate"],
  ["delay_h", "Delay (h)", "Blank when no event was detected"], ["fa_day", "FA/turbine-day"],
  ["mtbfa_h", "MTBFA (h)", "Blank when undefined"], ["normal_fpr", "Normal FPR"],
  ["fit_s", "Fit time (s)"], ["infer_s", "Inference time (s)"],
  ["gpu_mb", "GPU memory (MB)", "Blank when no GPU measurement applies"],
];

function m2bRow(i) {
  return `<div class="nm-row" data-i="${i}">
    <select class="nm-w" aria-label="Workload">${F.workloads.map(w => `<option value="${w}">${wlLabel(w)}</option>`).join("")}</select>
    <div class="nm-fields">${M2B_FIELDS.map(([key, label, hint]) => `<label>${label}
      <input type="number" min="0" step="any" class="nm-${key}" placeholder="${hint ? "Optional" : "Required"}" title="${hint || label}"></label>`).join("")}</div>
    <button class="btn ghost small nm-remove" type="button">Remove</button></div>`;
}

function renderPlan() {
  const replay = F.replay.m2b, random = F.replay.random;
  $("#nm-lede").innerHTML = `The evaluation plan uses the published reference models before your measurements are entered.
    At <b>K = 4</b>, leave-one-model-out replay gives card MAE <b>${replay.mae_by_K[4].toFixed(3)}</b>,
    compared with ${random.mae_by_K[4].toFixed(3)} for random order, using
    ${(100 * replay.cost_fraction_by_K[4]).toFixed(1)}% of the full evaluation cost.`;
  $("#nm-order").innerHTML = F.plan.order.map((w, i) => `<li class="${i < POLICY.max_active ? "active-step" : ""}">
    <span class="ord-n">${i + 1}</span><span class="ord-w">${wlLabel(w)}</span>
    <span class="ord-c">${F.plan.marginal_seconds[i].toFixed(0)}s</span>
    <span class="ord-r ${F.plan.radius[i] <= POLICY.stop_radius ? "met" : ""}">±${F.plan.radius[i].toFixed(3)}</span></li>`).join("");
  drawBudgetChart("#nm-chart");
}

function drawBudgetChart(selector) {
  const active = F.replay.m2b.mae_by_K, random = F.replay.random.mae_by_K;
  const keys = Object.keys(active).map(Number).sort((a, b) => a - b), width = 520, height = 230;
  const max = Math.ceil(20 * Math.max(...Object.values(active), ...Object.values(random))) / 20;
  const x = k => 46 + k / keys.at(-1) * 455, y = value => 190 - value / max * 168;
  const line = values => keys.map((k, i) => `${i ? "L" : "M"}${x(k).toFixed(2)},${y(values[k]).toFixed(2)}`).join("");
  $(selector).innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Mean card error against evaluated workloads">
    ${[0, max / 2, max].map(v => `<line x1="46" y1="${y(v)}" x2="501" y2="${y(v)}" stroke="#e5e9ed"/>
      <text x="38" y="${y(v) + 4}" text-anchor="end" font-size="11">${v.toFixed(2)}</text>`).join("")}
    <path d="${line(random)}" stroke="#8a99a8" stroke-dasharray="5 4" stroke-width="2" fill="none"/>
    <path d="${line(active)}" stroke="#2e7d5b" stroke-width="2" fill="none"/>
    ${[0, 4, 8, 12].map(k => `<text x="${x(k)}" y="207" text-anchor="middle" font-size="11">${k}</text>`).join("")}
    <text x="273" y="225" text-anchor="middle" font-size="11">Evaluated workloads K · card MAE</text>
    <text x="270" y="25" fill="#2e7d5b" font-size="11">Active plan</text>
    <text x="370" y="25" fill="#8a99a8" font-size="11">Random order</text></svg>`;
}

function runM2B() {
  const host = $("#nm-out"), revealed = {};
  try {
    $$(".nm-row").forEach(row => {
      const w = row.querySelector(".nm-w").value, values = {};
      let touched = false;
      for (const [key] of M2B_FIELDS) {
        const input = row.querySelector(`.nm-${key}`);
        if (input.validity.badInput) throw new Error(`Enter a complete numeric value for ${key}.`);
        const raw = input.value;
        values[key] = raw === "" ? null : Number(raw);
        touched ||= raw !== "";
      }
      if (!touched) return;
      if (revealed[w]) throw new Error(`Remove the duplicate ${w} workload.`);
      revealed[w] = userDimPct(w, values);
    });
    const workloads = Object.keys(revealed);
    if (!workloads.length) throw new Error("Enter the required measurements for at least one workload.");
    const result = completedCard(revealed);
    renderM2B({ ...result, radius: conformalRadius(workloads), rank: cardRank(result.card),
      nextWorkload: F.plan.order.find(w => !workloads.includes(w)) || null, revealedWorkloads: workloads });
  } catch (error) {
    host.innerHTML = `<p class="error">${esc(error.message)}</p>`;
  }
}

function renderM2B(result) {
  const host = $("#nm-out"), K = result.revealedWorkloads.length, met = result.radius <= POLICY.stop_radius;
  const replay = F.replay.m2b;
  host.innerHTML = `<div class="verdict ${met ? "ok" : "warn"}">
    <div class="verdict-head"><span class="klabel">Estimated positioning from ${K} workloads</span>
      <span class="conf ${met ? "conf-high" : "conf-medium"}">±${result.radius.toFixed(3)} ${met ? "≤" : ">"} ${POLICY.stop_radius}</span></div>
    <div class="name">Estimated rank ${Number(result.rank.rank.toFixed(1))} of ${result.rank.of}</div>
    <p class="verdict-sub">${met ? "The reference-residual radius has reached the stopping target."
      : `Next in the evaluation plan: <b>${result.nextWorkload ? wlLabel(result.nextWorkload) : "Complete"}</b>.`}</p>
    ${bars(F.dims.map(d => result.card[d]), F.dims)}</div>
    ${explainBlock("Explain this positioning")}
    <details class="fold"><summary>Card completion and replay</summary><ul class="rec-list">
      <li><span>Workloads evaluated</span><span>${result.revealedWorkloads.map(wlLabel).join(", ")}</span></li>
      <li><span>Completion model</span><span>Probabilistic PCA</span></li>
      <li><span>90% reference-residual radius</span><span>${result.radius.toFixed(3)}</span></li>
      <li><span>Active-plan replay card MAE at K = ${K}</span><span>${replay.mae_by_K[K].toFixed(3)}</span></li>
      <li><span>Active-plan replay rank error at K = ${K}</span><span>${replay.rank_error_by_K[K].toFixed(2)}</span></li></ul>
      <p class="fineprint">The card uses support-weighted accuracy, earliness, reliability and cost. Observed workload values are retained during completion.</p></details>`;
  host.querySelector(".explain-btn").onclick = () => runExplanation(host.querySelector(".explain-out"),
    "Explain the estimated performance card and whether the stopping target was reached.", m2bFacts(result));
}

function initAgent() {
  $$("#da-tabs .da-tab").forEach(button => button.onclick = () => {
    $$("#da-tabs .da-tab").forEach(b => b.classList.toggle("active", b === button));
    $$(".da-pane").forEach(p => p.classList.toggle("active", p.id === "da-" + button.dataset.t));
  });
  $("#nd-presets").innerHTML = Object.keys(D2M_PRESETS).map(p => `<button class="preset" data-p="${p}">${p}</button>`).join("");
  $("#nd-sliders").innerHTML = F.dims.map(d => `<div class="slider-row"><span>${d}</span>
    <input type="range" min="0" max="100" value="25" id="ndw-${d}"><span id="ndv-${d}">25%</span></div>`).join("");
  $$("#nd-presets .preset").forEach(button => button.onclick = () => {
    $$("#nd-presets .preset").forEach(b => b.classList.toggle("active", b === button));
    F.dims.forEach(d => $(`#ndw-${d}`).value = Math.round(100 * D2M_PRESETS[button.dataset.p][d]));
    runD2M();
  });
  ["#nd-target", "#nd-cpu", "#nd-fa", "#nd-fpr"].forEach(id => $(id).onchange = runD2M);
  ["#nd-fa", "#nd-fpr"].forEach(id => $(id).oninput = runD2M);
  F.dims.forEach(d => $(`#ndw-${d}`).oninput = runD2M);
  renderPlan();
  $("#nm-rows").innerHTML = m2bRow(0);
  $(".nm-w").value = F.plan.order[0];
  $("#nm-add").onclick = () => {
    const next = F.plan.order.find(w => !$$(".nm-w").some(select => select.value === w));
    if (!next) return;
    $("#nm-rows").insertAdjacentHTML("beforeend", m2bRow($$(".nm-row").length));
    $$(".nm-w").at(-1).value = next;
  };
  $("#nm-rows").onclick = event => {
    if (event.target.classList.contains("nm-remove")) event.target.closest(".nm-row").remove();
  };
  $("#nm-run").onclick = runM2B;
  $$("#nd-presets .preset")[0].click();
}
