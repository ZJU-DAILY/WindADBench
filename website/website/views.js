/* WindADBench — leaderboard, model profiles, and workload profiles. */
"use strict";

/* ================ LEADERBOARD ================ */
let lbGroup = "Overview", lbSort = { k: "event_f1", asc: false };
function lbWorkloads() {
  return D.workloads.filter(w => w.track === $("#lb-track").value).map(w => w.workload);
}
function fillLb() {
  $("#lb-workload").innerHTML = `<option value="__avg">Macro average</option>` +
    lbWorkloads().map(w => `<option>${w}</option>`).join("");
  $("#lb-family").innerHTML = `<option value="">All families</option>` + famOptions();
  $("#lb-groups").innerHTML = Object.keys(GROUPS).map(g =>
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
  const ws = $("#lb-workload").value === "__avg" ? lbWorkloads() : [$("#lb-workload").value];
  const fam = $("#lb-family").value, q = $("#lb-search").value.toLowerCase();
  const rows = MODELS.filter(m => (!fam || REG[m].family === fam) && m.includes(q))
    .map(m => ({ model: m, family: REG[m].family,
      ...Object.fromEntries(cols.map(k => [k, avgMetric(m, ws, k)])) }));
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

/* ================ MODELS ================ */
function fillModels() {
  $("#md-family").innerHTML = `<option value="">All families</option>` + famOptions();
}
function renderModels() {
  const fam = $("#md-family").value, cpu = $("#md-cpu").checked, q = $("#md-search").value.toLowerCase();
  const rows = MODELS.filter(m => (!fam || REG[m].family === fam) && (!cpu || !REG[m].needs_gpu) && m.includes(q));
  $("#md-table").innerHTML =
    `<thead><tr><th>Model</th><th>Family</th><th>GPU</th><th>Size (MB)</th><th>Infer (s)</th>
     <th>Acc</th><th>Early</th><th>Reliab</th><th>General</th><th>Cost</th><th>Flags</th></tr></thead><tbody>` +
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
  const rows = WLS.map(w => {
    const t = metMap[m][w] || {}, d = dimMap[m][w] || {};
    return `<tr><td>${wlLabel(w)}</td><td>${WMETA[w].track}</td>
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
      <div><h4>Five-dimension card (percentile vs. 36 models)</h4>
        ${bars(CARD_DIMS.map(d => c[d]), CARD_DIMS)}
        <h4>Capability</h4>
        <p class="small muted">model size ${REG[m].model_size == null ? "—" : REG[m].model_size.toFixed(1)} MB ·
        median fit ${REG[m].fit_time == null ? "—" : REG[m].fit_time.toFixed(2)} s ·
        median infer ${REG[m].infer_time == null ? "—" : REG[m].infer_time.toFixed(3)} s</p>
        <div id="md-explain-out" class="explain-out"></div></div>
      <div><h4>Per-workload results</h4>
        <div class="tablewrap"><table class="mini"><thead><tr><th>Workload</th><th>Track</th><th>Ev-F1</th>
        <th>Early</th><th>Delay</th><th>FA/day</th><th>A</th><th>E</th><th>R</th></tr></thead>
        <tbody>${rows}</tbody></table></div></div>
    </div></div>`;
  $("#md-explain").onclick = () => explainModel(m);
  $("#md-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/* ================ WORKLOADS ================ */
function topModels(w, n) {
  return MODELS.map(m => [m, metMap[m][w]?.event_f1]).filter(x => x[1] != null)
    .sort((a, b) => b[1] - a[1]).slice(0, n);
}
function renderWorkloads() {
  $("#wl-table").innerHTML =
    `<thead><tr><th>Workload</th><th>Track</th><th>Source→Target</th><th>Events</th>
     <th title="single-workload reduction in reference-model completion error">Diagnostic value</th>
     <th title="position in the M2B active evaluation plan">Plan #</th>
     <th>Cost proxy (s)</th><th>Top-3 by Event-F1</th></tr></thead><tbody>` +
    D.workloads.map(w => `<tr data-w="${w.workload}" class="click">
      <td><b>${wlLabel(w.workload)}</b></td><td>${w.track}</td>
      <td>${w.source}→${w.target}</td><td>${w.n_events}</td>
      <td><span class="bar wide"><i style="width:${Math.round(100 * w.disc)}%"></i></span></td>
      <td>${w.order_pos}</td><td>${w.cost_proxy.toFixed(2)}</td>
      <td class="small">${topModels(w.workload, 3).map(x => x[0]).join(", ")}</td></tr>`).join("") + `</tbody>`;
  $$("#wl-table tr.click").forEach(tr => tr.onclick = () => renderWlDetail(tr.dataset.w));
}
function renderWlDetail(w) {
  const rows = MODELS.map(m => {
    const t = metMap[m][w] || {}, d = dimMap[m][w] || {};
    const u = meanOf([d.a, d.e, d.r, d.c]) ?? 0;
    return { m, u, t };
  }).sort((a, b) => b.u - a.u).slice(0, 10);
  $("#wl-detail").innerHTML = `<div class="panel detail">
    <h3>${wlLabel(w)} <span class="tag">${WMETA[w].track}</span>
      <span class="muted small">· ${WMETA[w].n_events} anomaly events · evaluation-plan position ${WMETA[w].order_pos}/${WLS.length}</span></h3>
    <h4>Top-10 models on this workload (equal-weight utility over 4 dimensions)</h4>
    <div class="tablewrap"><table class="mini"><thead><tr><th>#</th><th>Model</th><th>Family</th><th>U</th>
      <th>Ev-F1</th><th>Early</th><th>FA/day</th></tr></thead><tbody>` +
    rows.map((r, i) => `<tr><td>${i + 1}</td><td><b>${r.m}</b></td><td>${famChip(REG[r.m].family)}</td>
      <td>${r.u.toFixed(3)}</td><td>${fmt(r.t.event_f1, "event_f1")}</td>
      <td>${fmt(r.t.early_detection_rate, "early_detection_rate")}</td>
      <td>${fmt(r.t.false_alarms_per_turbine_day, "false_alarms_per_turbine_day")}</td></tr>`).join("") +
    `</tbody></table></div></div>`;
  $("#wl-detail").scrollIntoView({ behavior: "smooth", block: "nearest" });
}
