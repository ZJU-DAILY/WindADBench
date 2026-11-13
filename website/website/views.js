"use strict";

let lbGroup = "Overview", lbSort = { k: "event_f1", asc: false };
function lbWorkloads() {
  return BENCHMARK_WLS.filter(w => w.track === $("#lb-track").value).map(w => w.workload);
}
function fillLb() {
  const normal = $("#lb-track").value === "normal-reliability";
  if (normal && !["Normal operation", "Cost"].includes(lbGroup)) lbGroup = "Normal operation";
  if (!normal && lbGroup === "Normal operation") lbGroup = "Overview";
  if (!GROUPS[lbGroup].includes(lbSort.k)) lbSort = { k: GROUPS[lbGroup][0], asc: M[GROUPS[lbGroup][0]][1] < 0 };
  $("#lb-workload").innerHTML = `<option value="__avg">Macro average</option>` +
    lbWorkloads().map(w => `<option>${w}</option>`).join("");
  $("#lb-family").innerHTML = `<option value="">All families</option>` + famOptions();
  const groups = normal ? ["Normal operation", "Cost"] : Object.keys(GROUPS).filter(g => g !== "Normal operation");
  $("#lb-groups").innerHTML = groups.map(g =>
    `<button class="gtab ${g === lbGroup ? "active" : ""}" data-g="${g}">${g}</button>`).join("");
  $$("#lb-groups .gtab").forEach(b => b.onclick = () => {
    lbGroup = b.dataset.g;
    if (!GROUPS[lbGroup].includes(lbSort.k)) {
      lbSort = { k: GROUPS[lbGroup][0], asc: M[GROUPS[lbGroup][0]][1] < 0 };
    }
    fillLb(); renderLb();
  });
}
function renderLb() {
  const cols = GROUPS[lbGroup];
  const track = $("#lb-track").value;
  const macro = $("#lb-workload").value === "__avg";
  const ws = $("#lb-workload").value === "__avg" ? lbWorkloads() : [$("#lb-workload").value];
  const fam = $("#lb-family").value, q = $("#lb-search").value.toLowerCase();
  const rows = MODELS.filter(m => (!fam || REG[m].family === fam) && m.includes(q))
    .map(m => ({ model: m, family: REG[m].family,
      ...Object.fromEntries(cols.map(k => [k, macro ? (D.macroMetrics[track]?.[m]?.[k] ?? null) : avgMetric(m, ws, k)])) }));
  $("#lb-note").textContent = macro && lbGroup === "Cost" && track.startsWith("cross-")
    ? "Select a transfer direction to view its resource measurements." : "";
  rows.sort((x, y) => {
    const a = x[lbSort.k], b = y[lbSort.k];
    if (a == null) return 1; if (b == null) return -1;
    return lbSort.asc ? a - b : b - a;
  });
  const best = Object.fromEntries(cols.map(k => {
    const vs = rows.map(r => r[k]).filter(v => v != null);
    return [k, vs.length ? (M[k][1] > 0 ? Math.max(...vs) : Math.min(...vs)) : null];
  }));
  $("#lb-table").innerHTML =
    `<thead><tr><th>#</th><th>Model</th><th>Family</th>` +
    cols.map(k => `<th data-k="${k}" class="${lbSort.k === k ? "sorted" : ""}" title="${M[k][1] > 0 ? "higher" : "lower"} is better">${M[k][0]}${lbSort.k === k ? (lbSort.asc ? " ↑" : " ↓") : ""}</th>`).join("") +
    `</tr></thead><tbody>` +
    rows.map((r, i) => `<tr class="${i < 3 ? "top" + (i + 1) : ""}">
      <td class="rank">${i + 1}</td><td>${r.model}</td><td>${famChip(r.family)}</td>` +
      cols.map(k => `<td class="${r[k] != null && r[k] === best[k] ? "best" : ""}">${fmt(r[k], k)}</td>`).join("") +
      `</tr>`).join("") + `</tbody>`;
  $("#lb-table thead").onclick = (e) => {
    const k = e.target.dataset?.k; if (!k) return;
    if (lbSort.k === k) lbSort.asc = !lbSort.asc; else lbSort = { k, asc: M[k][1] < 0 };
    renderLb();
  };
}

function fillModels() {
  $("#md-family").innerHTML = `<option value="">All families</option>` + famOptions();
}
function renderModels() {
  const fam = $("#md-family").value, cpu = $("#md-cpu").checked, q = $("#md-search").value.toLowerCase();
  const rows = MODELS.filter(m => (!fam || REG[m].family === fam) && (!cpu || !REG[m].needs_gpu) && m.includes(q));
  $("#md-table").innerHTML =
    `<thead><tr><th>Model</th><th>Family</th><th>GPU</th><th>Size (MB)</th><th>Infer (s)</th>
     <th>Acc</th><th>Early</th><th>Reliab</th><th>Cost</th><th>Flags</th></tr></thead><tbody>` +
    rows.map(m => { const c = D.cards[m]; return `<tr data-m="${m}" class="click">
      <td><b>${m}</b></td><td>${famChip(REG[m].family)}</td>
      <td>${REG[m].needs_gpu ? "●" : "—"}</td>
      <td>${REG[m].model_size == null ? "—" : REG[m].model_size.toFixed(1)}</td>
      <td>${REG[m].infer_time == null ? "—" : REG[m].infer_time.toFixed(3)}</td>
      ${CARD_DIMS.map(d => `<td><span class="pct">${pctText(c[d])}</span></td>`).join("")}
      <td>${(REG[m].flags || []).length ? "⚠ " + REG[m].flags.length : ""}</td></tr>`; }).join("") + `</tbody>`;
  $$("#md-table tr.click").forEach(tr => tr.onclick = () => renderModelDetail(tr.dataset.m));
}
function renderModelDetail(m) {
  const c = D.cards[m];
  const rows = BENCHMARK_WLS.map(info => {
    const w = info.workload;
    const t = metMap[m][w] || {}, d = dimMap[m][w] || {};
    return `<tr><td>${wlLabel(w)}</td><td>${info.track}</td>
      <td>${fmt(t.event_f1, "event_f1")}</td><td>${fmt(t.early_detection_rate, "early_detection_rate")}</td>
      <td>${fmt(t.mean_detection_delay, "mean_detection_delay")}</td>
      <td>${fmt(t.false_alarms_per_turbine_day, "false_alarms_per_turbine_day")}</td>
      <td>${pctText(d.a)}</td><td>${pctText(d.e)}</td><td>${pctText(d.r)}</td></tr>`;
  }).join("");
  $("#md-detail").innerHTML = `<div class="panel detail">
    <div class="detail-head">
      <h3>${m} ${famChip(REG[m].family)} ${REG[m].needs_gpu ? '<span class="tag">GPU</span>' : '<span class="tag ok">CPU-only</span>'} ${flagHtml(m)}</h3>
      <button class="btn ghost small" id="md-explain" data-m="${m}">Explain this model</button>
    </div>
    <div class="detail-grid">
      <div><h4>Four-dimension card (percentile vs. 36 models)</h4>
        ${bars(CARD_DIMS.map(d => c[d]), CARD_DIMS)}
        <h4>Capability</h4>
        <p class="small muted">model size ${REG[m].model_size == null ? "—" : REG[m].model_size.toFixed(1)} MB ·
        mean fit ${REG[m].fit_time == null ? "—" : REG[m].fit_time.toFixed(2)} s ·
        mean infer ${REG[m].infer_time == null ? "—" : REG[m].infer_time.toFixed(3)} s</p>
        <div id="md-explain-out" class="explain-out"></div></div>
      <div><h4>Per-workload results</h4>
        <div class="tablewrap"><table class="mini"><thead><tr><th>Workload</th><th>Track</th><th>Ev-F1</th>
        <th>Early</th><th>Delay (h)</th><th>FA/day</th><th>A</th><th>E</th><th>R</th></tr></thead>
        <tbody>${rows}</tbody></table></div></div>
    </div></div>`;
  $("#md-explain").onclick = () => explainModel(m);
  $("#md-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function topModels(w, n) {
  const key = BMETA[w].track === "normal-reliability" ? "normal_point_fpr" : "event_f1";
  return MODELS.map(m => [m, metMap[m][w]?.[key]]).filter(x => x[1] != null)
    .sort((a, b) => M[key][1] * (b[1] - a[1])).slice(0, n);
}
function renderWorkloads() {
  $("#wl-table").innerHTML =
    `<thead><tr><th>Workload</th><th>Track</th><th>Source→Target</th>
     <th>Evaluation cost (s)</th><th>Ranking metric</th><th>Top-3 models</th></tr></thead><tbody>` +
    BENCHMARK_WLS.map(w => `<tr data-w="${w.workload}" class="click">
      <td><b>${wlLabel(w.workload)}</b></td><td>${w.track}</td>
      <td>${w.source}→${w.target}</td><td>${w.cost_proxy == null ? "—" : w.cost_proxy.toFixed(2)}</td>
      <td>${w.track === "normal-reliability" ? "Normal FPR" : "Event-F1"}</td>
      <td class="small">${topModels(w.workload, 3).map(x => x[0]).join(", ")}</td></tr>`).join("") + `</tbody>`;
  $$("#wl-table tr.click").forEach(tr => tr.onclick = () => renderWlDetail(tr.dataset.w));
}
function renderWlDetail(w) {
  const normal = BMETA[w].track === "normal-reliability";
  const rows = MODELS.map(m => {
    const t = metMap[m][w] || {}, d = dimMap[m][w] || {};
    const u = meanOf([d.a, d.e, d.r, d.c]) ?? 0;
    return { m, u, t };
  }).sort((a, b) => normal ? (a.t.normal_point_fpr ?? Infinity) - (b.t.normal_point_fpr ?? Infinity) : b.u - a.u).slice(0, 10);
  $("#wl-detail").innerHTML = `<div class="panel detail">
    <h3>${wlLabel(w)} <span class="tag">${BMETA[w].track}</span></h3>
    <h4>Top-10 models (${normal ? "normal-point FPR" : "equal-weight utility over four dimensions"})</h4>
    <div class="tablewrap"><table class="mini"><thead><tr><th>#</th><th>Model</th><th>Family</th><th>U</th>
      <th>${normal ? "Accuracy" : "Ev-F1"}</th><th>${normal ? "Normal FPR" : "Early"}</th><th>FA/day</th></tr></thead><tbody>` +
    rows.map((r, i) => `<tr><td>${i + 1}</td><td><b>${r.m}</b></td><td>${famChip(REG[r.m].family)}</td>
      <td>${normal ? "—" : r.u.toFixed(3)}</td><td>${normal ? fmt(r.t.acc, "acc") : fmt(r.t.event_f1, "event_f1")}</td>
      <td>${normal ? fmt(r.t.normal_point_fpr, "normal_point_fpr") : fmt(r.t.early_detection_rate, "early_detection_rate")}</td>
      <td>${fmt(r.t.false_alarms_per_turbine_day, "false_alarms_per_turbine_day")}</td></tr>`).join("") +
    `</tbody></table></div></div>`;
  $("#wl-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
