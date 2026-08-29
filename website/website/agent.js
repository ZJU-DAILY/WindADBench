/* WindADBench — Decision Agent page: D2M portfolio selection and M2B
   evaluation planning. Rendering only; every number comes from policy.js. */
"use strict";

/* ================================================================== D2M === */

const D2M_PRESETS = {
  "Balanced": { accuracy: .25, earliness: .25, reliability: .25, cost: .25 },
  "Limited review capacity": { accuracy: .2, earliness: .1, reliability: .6, cost: .1 },
  "Fault-safety first": { accuracy: .3, earliness: .5, reliability: .1, cost: .1 },
  "Efficiency first": { accuracy: .2, earliness: .2, reliability: .2, cost: .4 },
};

function d2mWeights() {
  const raw = DIMS.map(d => +$(`#ndw-${d}`).value);
  const s = raw.reduce((a, b) => a + b, 0) || 1;
  DIMS.forEach((d, i) => $(`#ndv-${d}`).textContent = Math.round(100 * raw[i] / s) + "%");
  return Object.fromEntries(DIMS.map((d, i) => [d, raw[i] / s]));
}

function d2mProfile() {
  const num = (id) => $(id).value === "" ? null : +$(id).value;
  return {
    weights: d2mWeights(),
    no_gpu: $("#nd-cpu").checked,
    max_fa_day: num("#nd-fa"), max_model_size: num("#nd-size"),
    min_earliness: num("#nd-early"), max_infer_time: num("#nd-infer"),
  };
}

function runD2M() {
  const site = {
    turbines: +$("#nd-turbines").value || 10,
    channels: +$("#nd-features").value || 100,
    type: $("#nd-type").value,
    fault_labels: $("#nd-labels").value === "yes",
  };
  const farm = nearestFarm(site.turbines, site.channels, site.type);
  // With target labels the closest farm's own workloads apply; without them the
  // only admissible evidence is transfer INTO that farm, so target labels are
  // never used to pick the model that will be deployed on them.
  const evidence = site.fault_labels
    ? [`IF-${farm}`, `CT-${farm}`]
    : WLS.filter(w => w.endsWith(`>${farm}`));
  const profile = d2mProfile();
  const rec = recommend(profile, evidence);
  renderD2M(rec, site, farm, profile);
}

function renderD2M(rec, site, farm, profile) {
  const host = $("#nd-out");
  if (!rec.primary) {
    host.innerHTML = `<div class="verdict warn"><span class="klabel">No feasible detector</span>
      <div class="name">Constraints exclude all ${MODELS.length} models</div>
      <p class="small muted">Relax a hard constraint under “Advanced”.</p></div>`;
    return;
  }
  const fm = D.farms[farm];
  const abstain = rec.decision === "abstain";
  const chips = (m) => `${famChip(REG[m].family)}${rec.pareto.includes(m)
    ? '<span class="star" title="on the four-dimension Pareto front">★</span>' : ""}${flagHtml(m)}`;
  const checklist = site.fault_labels ? [
    "Map your SCADA channels onto the aligned feature space (wind speed, active power, rotor speed).",
    "Build farm–turbine–event workloads with the three leakage-control rules before any training.",
    `Pilot the shortlist on one in-farm workload first; reference median effort ≈ ${WMETA[`IF-${farm}`].cost_proxy.toFixed(1)} s per model.`,
  ] : [
    "Collect at least one month of normal-operation data per turbine for score scaling.",
    "Calibrate each threshold to the 1% FPR budget on YOUR normal data (99th percentile of normal scores).",
    "Run the shortlist in shadow mode and watch FA/turbine-day for 2–4 weeks before trusting alarms.",
  ];

  host.innerHTML = `
    <div class="verdict ${abstain ? "warn" : "ok"}">
      <div class="verdict-head">
        <span class="klabel">${abstain ? "Abstain — shortlist only" : "Recommended detector"}</span>
        <span class="conf conf-${rec.confidence}">confidence ${rec.confidence}</span>
      </div>
      <div class="name">${rec.primary} ${chips(rec.primary)}</div>
      <p class="verdict-sub">Closest benchmark farm <b>${farm}</b>
        (${fm.turbines} turbines · ${fm.features} channels · ${fm.type}) ·
        evidence ${rec.evidence.map(wlLabel).join(", ")} ·
        ${rec.support} anomaly events · top-2 margin ${rec.margin.toFixed(3)}
        ${site.fault_labels ? "" : "· cold start, target fault labels never used"}</p>
      ${abstain ? `<p class="verdict-note">The policy abstains from a single-model claim:
        ${esc(rec.reason)}. Treat the entries below as a shortlist to pilot, not a winner.</p>` : ""}
      ${bars(DIMS.map(d => rec.topK[0].dims[d]), DIMS)}
    </div>
    ${explainBlock("Explain this recommendation")}
    <details class="fold">
      <summary>Portfolio, fallbacks, and audit trail</summary>
      <h4>Risk-adjusted portfolio</h4>
      <ul class="rec-list">
        ${rec.topK.map((s, i) => `<li><span>${i + 1}. <b>${s.model}</b> ${chips(s.model)}</span>
          <span class="muted">U<sub>lcb</sub> = ${s.lcb.toFixed(3)}</span></li>`).join("")}
        <li><span>Lowest-cost fallback <b>${rec.lowCostFallback}</b> ·
          most sensitive <b>${rec.highSensitivityFallback}</b></span></li>
      </ul>
      <h4>Adaptation checklist</h4>
      <ol class="checklist">${checklist.map(c => `<li>${c}</li>`).join("")}</ol>
      <h4>Filtering</h4>
      <p class="small muted">${rec.nFeasible} feasible · ${rec.excluded.length} excluded by hard constraints
        · weights ${DIMS.map(d => `${d} ${Math.round(100 * profile.weights[d])}%`).join(" · ")}</p>
      ${rec.excluded.length ? `<p class="small muted excl">${rec.excluded
        .map(([m, why]) => `${m} — ${esc(why)}`).join("<br>")}</p>` : ""}
      <p class="fineprint">Leave-one-farm-out replay: agent mean regret
        ${D.modeA.mean_regret.agent.toFixed(3)} vs. ${D.modeA.mean_regret.global_best.toFixed(3)}
        for a single global best and ${D.modeA.mean_regret.random.toFixed(3)} for random choice;
        piloting the top-3 lowers it to ${D.modeA.pilot_regret["3"].toFixed(3)}.</p>
    </details>`;

  host.querySelector(".explain-btn").onclick = () => runExplanation(
    host.querySelector(".explain-out"),
    "Explain this deployment recommendation to a wind-farm engineer who must decide "
    + "whether to trust and pilot it.",
    d2mFacts(rec, site, farm, profile.weights));
}

/* ================================================================== M2B === */

const M2B_FIELDS = [
  ["event_f1", "Event-F1"], ["vus_pr", "VUS-PR"], ["early_detection_rate", "Early rate"],
  ["mean_detection_delay", "Delay"], ["false_alarms_per_turbine_day", "FA/day"],
  ["mtbfa", "MTBFA"],
];

function m2bRow(i) {
  return `<div class="nm-row" data-i="${i}">
    <select class="nm-w">${WLS.map(w => `<option value="${w}">${wlLabel(w)}</option>`).join("")}</select>
    ${M2B_FIELDS.map(([k, l]) =>
      `<input type="number" step="any" class="nm-${k}" placeholder="${l}" title="${l}">`).join("")}
  </div>`;
}

/** Nearest replayed budget for which summary statistics were published. */
function nearestBudget(K) {
  const budgets = POLICY.replay_budgets;
  return String(budgets.reduce((best, b) =>
    Math.abs(b - K) < Math.abs(best - K) ? b : best, budgets[0]));
}

function renderPlan() {
  const active = POLICY.max_active;
  const radius = Object.fromEntries(D.modeB.radius.map(r => [r.K, r.radius]));
  const stop = D.modeB.summary.stop_radius;
  $("#nm-lede").innerHTML = `The order below is chosen by nested cross-validation over the
    36 published detectors, so it is fixed before your detector is measured. After
    <b>K = 4</b> workloads the replayed card error is
    <b>${D.modeB.summary.mae_at_K["4"]}</b> against ${D.modeB.summary.rand_mae_at_K["4"]}
    for a random order, while ${Math.round(100 * D.modeB.summary.cost_saving_at_K["4"])}% of the
    evaluation cost is saved.`;
  $("#nm-order").innerHTML = D.modeB.order.map((w, i) => `
    <li class="${i < active ? "active-step" : ""}">
      <span class="ord-n">${i + 1}</span><span class="ord-w">${wlLabel(w)}</span>
      <span class="ord-c">${WMETA[w].cost_proxy.toFixed(0)}s</span>
      <span class="ord-r ${radius[i + 1] <= stop ? "met" : ""}">±${radius[i + 1].toFixed(3)}</span>
    </li>`).join("");
  drawBudgetChart("#nm-chart");
}

function drawBudgetChart(sel) {
  const curve = D.modeB.curve, rand = D.modeB.random;
  const W = 520, H = 240, L = 46, B = 34, T = 12, R = 10;
  const ymax = Math.ceil(20 * Math.max(...curve.map(p => p.active_mae),
    ...rand.map(p => p.rand_p75))) / 20;
  const n = curve.length;
  const x = (k) => L + (k - 1) / (n - 1) * (W - L - R);
  const y = (v) => T + (1 - v / ymax) * (H - T - B);
  const path = (pts, key) => pts.map((p, i) =>
    `${i ? "L" : "M"}${x(p.K).toFixed(1)},${y(p[key]).toFixed(1)}`).join("");
  const band = rand.map(p => `${x(p.K).toFixed(1)},${y(p.rand_p75).toFixed(1)}`).join(" ")
    + " " + [...rand].reverse().map(p => `${x(p.K).toFixed(1)},${y(p.rand_p25).toFixed(1)}`).join(" ");
  const ticks = [0, ymax / 3, 2 * ymax / 3, ymax];
  $(sel).innerHTML = `<svg viewBox="0 0 ${W} ${H}" font-size="10" fill="#5b6b7b" role="img"
      aria-label="card error against number of evaluated workloads">
    ${ticks.map(v => `<line x1="${L}" y1="${y(v)}" x2="${W - R}" y2="${y(v)}" stroke="#eef1f4"/>
      <text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${v.toFixed(2)}</text>`).join("")}
    ${[1, 4, 8, n].map(k => `<text x="${x(k)}" y="${H - B + 15}" text-anchor="middle">${k}</text>`).join("")}
    <polygon points="${band}" fill="#d7dde3" opacity=".55"/>
    <path d="${path(rand, "rand_p50")}" stroke="#a9b4bf" stroke-dasharray="5 4" fill="none" stroke-width="1.3"/>
    <path d="${path(curve, "active_mae")}" stroke="#2e7d5b" fill="none" stroke-width="2"/>
    ${curve.map(p => `<circle cx="${x(p.K)}" cy="${y(p.active_mae)}" r="2.4" fill="#2e7d5b"/>`).join("")}
    <text x="${(L + W) / 2}" y="${H - 4}" text-anchor="middle">evaluated workloads K · card MAE</text>
    <g transform="translate(${L + 12},${T + 6})">
      <line x1="0" y1="0" x2="16" y2="0" stroke="#2e7d5b" stroke-width="2"/>
      <text x="21" y="3">active plan</text>
      <rect x="96" y="-4" width="16" height="8" fill="#d7dde3"/>
      <text x="117" y="3">random orders (IQR)</text>
    </g></svg>`;
}

function runM2B() {
  const cost = {
    infer_time: $("#nm-infer").value === "" ? null : +$("#nm-infer").value,
    infer_gpu_mem: $("#nm-gpu").value === "" ? null : +$("#nm-gpu").value,
    model_size: $("#nm-size").value === "" ? null : +$("#nm-size").value,
  };
  const revealed = {};
  $$(".nm-row").forEach(row => {
    const w = row.querySelector(".nm-w").value, vals = { ...cost };
    let any = false;
    for (const [k] of M2B_FIELDS) {
      const raw = row.querySelector(`.nm-${k}`).value;
      vals[k] = raw === "" ? null : +raw;
      if (raw !== "") any = true;
    }
    if (any) revealed[w] = userDimPct(w, vals);
  });
  const host = $("#nm-out");
  const ws = Object.keys(revealed);
  if (!ws.length) {
    host.innerHTML = `<p class="muted small">Enter at least one measured value in step 2.</p>`;
    return;
  }
  const { card, neighbours } = completedCard(revealed);
  const radius = conformalRadius(ws);
  const rank = cardRank(card);
  const nextWorkload = D.modeB.order.find(w => !ws.includes(w)) ?? null;
  renderM2B({ card, neighbours, radius, rank, nextWorkload, revealedWorkloads: ws });
}

function renderM2B(result) {
  const host = $("#nm-out");
  const stop = D.modeB.summary.stop_radius;
  const K = result.revealedWorkloads.length;
  const met = result.radius <= stop;
  const budget = nearestBudget(K);
  const mae = D.modeB.summary.mae_at_K[budget];
  const flags = [];
  if (result.card.reliability != null && result.card.reliability < 0.25)
    flags.push("high false-alarm burden relative to the cohort");
  if (result.card.accuracy > 0.6 && result.card.earliness < 0.2)
    flags.push("detects well but warns late — single-spike risk");

  host.innerHTML = `
    <div class="verdict ${met ? "ok" : "warn"}">
      <div class="verdict-head">
        <span class="klabel">Estimated positioning from ${K} workload${K > 1 ? "s" : ""}</span>
        <span class="conf ${met ? "conf-high" : "conf-medium"}">
          ±${result.radius.toFixed(3)} ${met ? "≤" : ">"} ${stop} stop target</span>
      </div>
      <div class="name">≈ rank ${result.rank.rank} of ${result.rank.of}
        ${flags.map(f => `<span class="flag">⚠ ${esc(f)}</span>`).join("")}</div>
      <p class="verdict-sub">${met
        ? "The 90% reference-residual radius has reached the stopping target; the remaining workloads are unlikely to move the card."
        : `Keep going: the next workload in the plan is <b>${result.nextWorkload ? wlLabel(result.nextWorkload) : "—"}</b>.`}
        Nearest published detectors: ${result.neighbours.map(n => `<b>${n}</b>`).join(", ")}.</p>
      ${bars(CARD_DIMS.map(d => result.card[d]), CARD_DIMS)}
    </div>
    ${explainBlock("Explain this positioning")}
    <details class="fold">
      <summary>How this estimate was produced</summary>
      <ul class="rec-list">
        <li><span>Workloads evaluated</span><span>${result.revealedWorkloads.map(wlLabel).join(", ")}</span></li>
        <li><span>Unseen cells completed from</span><span>${result.neighbours.join(", ")} (k = ${POLICY.knn_k})</span></li>
        <li><span>Replayed card error at this budget</span><span>≈ ${(100 * mae).toFixed(1)} percentile points (K = ${budget})</span></li>
        <li><span>Rank error at this budget</span><span>≈ ${D.modeB.summary.rank_err_at_K[budget]} of ${MODELS.length}</span></li>
      </ul>
      <p class="fineprint">Percentiles place your detector among the ${MODELS.length} published
        models using the same ranking rule as the knowledge base. Retrospective
        leave-one-model-out protocol; run all ${WLS.length} workloads for a leaderboard entry.</p>
    </details>`;

  host.querySelector(".explain-btn").onclick = () => runExplanation(
    host.querySelector(".explain-out"),
    "Explain what this partial evaluation says about the new detector, and whether the "
    + "engineer should run more workloads.",
    m2bFacts(result));
}

/* ================================================================= init === */

function initAgent() {
  $$("#da-tabs .da-tab").forEach(b => b.onclick = () => {
    $$("#da-tabs .da-tab").forEach(x => x.classList.toggle("active", x === b));
    $$(".da-pane").forEach(p => p.classList.toggle("active", p.id === "da-" + b.dataset.t));
  });

  $("#nd-presets").innerHTML = Object.keys(D2M_PRESETS)
    .map(p => `<button class="preset" data-p="${p}">${p}</button>`).join("");
  $("#nd-sliders").innerHTML = DIMS.map(d => `
    <div class="slider-row"><span>${d}</span>
    <input type="range" min="0" max="100" value="25" id="ndw-${d}">
    <span id="ndv-${d}">25%</span></div>`).join("");
  $$("#nd-presets .preset").forEach(b => b.onclick = () => {
    $$("#nd-presets .preset").forEach(x => x.classList.remove("active"));
    b.classList.add("active");
    const p = D2M_PRESETS[b.dataset.p];
    DIMS.forEach(d => $(`#ndw-${d}`).value = Math.round(100 * p[d]));
    runD2M();
  });
  ["#nd-turbines", "#nd-features", "#nd-type", "#nd-labels", "#nd-cpu",
   "#nd-fa", "#nd-size", "#nd-early", "#nd-infer"].forEach(id => $(id).onchange = runD2M);
  DIMS.forEach(d => $(`#ndw-${d}`).oninput = () => {
    $$("#nd-presets .preset").forEach(x => x.classList.remove("active"));
    runD2M();
  });

  renderPlan();
  $("#nm-rows").innerHTML = m2bRow(0);
  $(".nm-w").value = D.modeB.order[0];
  $("#nm-add").onclick = () => {
    const next = D.modeB.order.find(w => !$$(".nm-w").some(s => s.value === w));
    $("#nm-rows").insertAdjacentHTML("beforeend", m2bRow($$(".nm-row").length));
    if (next) $$(".nm-w").at(-1).value = next;
  };
  $("#nm-run").onclick = runM2B;
  ["#nm-infer", "#nm-gpu", "#nm-size"].forEach(id => $(id).onchange = () => {
    if ($("#nm-out").children.length) runM2B();
  });

  $$("#nd-presets .preset")[0].click();
}
