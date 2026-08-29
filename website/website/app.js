/* WindADBench system UI — entry point.
   Module order: core.js -> views.js -> policy.js -> explain.js -> agent.js -> app.js */
"use strict";

function initControls() {
  $("#lb-track").onchange = () => { fillLb(); renderLb(); };
  ["#lb-workload", "#lb-family"].forEach(id => $(id).onchange = renderLb);
  $("#lb-search").oninput = renderLb;

  $("#md-family").onchange = renderModels;
  $("#md-cpu").onchange = renderModels;
  $("#md-search").oninput = renderModels;
}

function init() {
  const legend = FAMS.map(f => `<span class="leg">${famChip(f)} ${FAM_FULL[f] ?? f}</span>`).join("");
  $("#lb-legend").innerHTML = legend;
  $("#md-legend").innerHTML = legend;

  fillLb(); renderLb();
  fillModels(); renderModels();
  renderWorkloads();
  initControls();

  initLlmDialog();
  initAgent();
  route();
}

init();
