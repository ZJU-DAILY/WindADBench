/* WindADBench — optional written explanations via a user-supplied,
   OpenAI-compatible endpoint.

   The decision is always computed by policy.js first; the model receives the
   finished decision as structured facts and is instructed to describe it, never
   to revise it. Credentials live in this browser's localStorage and are sent
   only to the endpoint the user names. */
"use strict";

const LLM = {
  STORE: "windad.llm",

  /** localStorage is unavailable on a file:// origin and in private modes, so
      every access degrades to an in-memory store for the current page. */
  memory: null,

  cfg() {
    if (this.memory) return this.memory;
    try { return JSON.parse(localStorage.getItem(this.STORE)) || {}; }
    catch { return {}; }
  },
  save(cfg) {
    this.memory = cfg;
    try { localStorage.setItem(this.STORE, JSON.stringify(cfg)); return true; }
    catch { return false; }
  },
  clear() {
    this.memory = null;
    try { localStorage.removeItem(this.STORE); } catch { /* nothing to clear */ }
  },
  ready() { const c = this.cfg(); return Boolean(c.url && c.key && c.model); },

  /** Resolve a base URL to the chat-completions endpoint. */
  endpoint(url) {
    const base = String(url || "").trim().replace(/\/+$/, "");
    return /\/chat\/completions$/.test(base) ? base : base + "/chat/completions";
  },

  /**
   * Stream one completion. `onDelta` receives the accumulated text so far.
   * Falls back to a non-streaming request if the endpoint rejects `stream`.
   */
  async chat(system, user, onDelta, signal) {
    const c = this.cfg();
    if (!this.ready()) throw new Error("No endpoint configured.");
    const body = {
      model: c.model, temperature: 0.2, stream: true,
      messages: [{ role: "system", content: system }, { role: "user", content: user }],
    };
    let res;
    try {
      res = await fetch(this.endpoint(c.url), {
        method: "POST", signal,
        headers: { "content-type": "application/json", authorization: `Bearer ${c.key}` },
        body: JSON.stringify(body),
      });
    } catch (err) {
      if (err.name === "AbortError") throw err;
      throw new Error("Could not reach the endpoint. A browser call needs the server to "
        + "allow cross-origin requests (CORS); check the base URL and the provider's "
        + "browser-access policy. Original error: " + err.message);
    }
    if (!res.ok) {
      const detail = (await res.text().catch(() => "")).slice(0, 300);
      throw new Error(`${res.status} ${res.statusText}${detail ? " — " + detail : ""}`);
    }
    if (!res.body) {
      const json = await res.json();
      const text = json.choices?.[0]?.message?.content ?? "";
      onDelta?.(text);
      return text;
    }
    const reader = res.body.getReader(), decoder = new TextDecoder();
    let buffer = "", out = "", sawDelta = false;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          const json = JSON.parse(payload);
          if (json.error) throw new Error(json.error.message || "endpoint returned an error");
          const delta = json.choices?.[0]?.delta?.content
            ?? json.choices?.[0]?.message?.content ?? "";
          if (delta) { sawDelta = true; out += delta; onDelta?.(out); }
        } catch (err) {
          if (err instanceof SyntaxError) continue;  // keep-alive fragment
          throw err;
        }
      }
    }
    if (!sawDelta && !out) throw new Error("The endpoint returned an empty response.");
    return out;
  },
};

const EXPLAIN_SYSTEM = `You are the writing layer of the WindADBench Decision Agent, a \
benchmark of anomaly detectors for wind-turbine SCADA data.

A deterministic policy has ALREADY produced the result given to you. Your only job is to \
explain that result in clear English to an engineer who has to act on it.

Rules:
- Never change, re-rank, override, or second-guess the decision. Describe it as settled.
- Use only the numbers provided. Never invent a metric, model, or figure.
- Say plainly what the evidence does not cover and where the main risk lies.
- Percentiles are positions among the 36 benchmarked detectors, where 100 is best.
- Write 120-180 words as plain prose. No headings, no bullet lists, no markdown.`;

/** Render a stream of explanation text into a container element. */
async function runExplanation(host, task, facts) {
  const button = host.querySelector(".explain-btn");
  let out = host.querySelector(".explain-text");
  if (!out) {
    out = document.createElement("p");
    out.className = "explain-text";
    host.appendChild(out);
  }
  if (!LLM.ready()) {
    out.className = "explain-text hint";
    out.textContent = "Add an OpenAI-compatible endpoint under “Explanations…” to generate "
      + "a written explanation. The recommendation above is already complete without it.";
    return;
  }
  if (button) button.disabled = true;
  out.className = "explain-text";
  out.textContent = "Writing…";
  try {
    await LLM.chat(EXPLAIN_SYSTEM, `${task}\n\nResult:\n${JSON.stringify(facts, null, 1)}`,
      (text) => { out.textContent = text; });
  } catch (err) {
    out.className = "explain-text error";
    out.textContent = "Explanation failed: " + err.message;
  } finally {
    if (button) button.disabled = false;
  }
}

/** Button + output slot, wired by the caller through `onExplain`. */
function explainBlock(label) {
  return `<div class="explain-out">
    <button class="btn ghost small explain-btn">${label}</button>
  </div>`;
}

/* ---------------- fact builders ---------------- */

const round3 = (v) => v == null ? null : Math.round(1000 * v) / 1000;
const dimsOf = (obj) => Object.fromEntries(DIMS.map(d => [d, round3(obj[d])]));

function d2mFacts(rec, site, farm, weights) {
  return {
    site, closest_benchmark_farm: { id: farm, ...D.farms[farm] },
    evidence_workloads: rec.evidence.map(w => ({
      workload: w, track: WMETA[w].track, anomaly_events: WMETA[w].n_events })),
    priority_weights: dimsOf(weights),
    decision: rec.decision, abstain_reason: rec.reason,
    confidence: rec.confidence,
    top2_margin: round3(rec.margin), evidence_support_events: rec.support,
    models_feasible: rec.nFeasible, models_excluded: rec.excluded.length,
    portfolio: rec.topK.map(s => ({
      model: s.model, family: FAM_FULL[REG[s.model].family],
      risk_adjusted_utility: round3(s.lcb),
      dimension_percentiles: Object.fromEntries(
        DIMS.map(d => [d, Math.round(100 * s.dims[d])])),
      on_pareto_front: rec.pareto.includes(s.model),
      needs_gpu: REG[s.model].needs_gpu, model_size_mb: REG[s.model].model_size,
      failure_flags: modelFlags(s.model),
    })),
    fallbacks: { lowest_cost: rec.lowCostFallback, most_sensitive: rec.highSensitivityFallback },
    sample_exclusions: rec.excluded.slice(0, 6).map(([m, why]) => `${m}: ${why}`),
    replay_evidence: { d2m_mean_regret: D.modeA.mean_regret, top3_hit_rate: D.modeA.hit3 },
  };
}

function m2bFacts(result) {
  return {
    workloads_evaluated: result.revealedWorkloads,
    workloads_remaining: WLS.filter(w => !result.revealedWorkloads.includes(w)),
    estimated_card_percentiles: Object.fromEntries(
      CARD_DIMS.map(d => [d, pctText(result.card[d])])),
    estimated_rank: `${result.rank.rank} of ${result.rank.of}`,
    nearest_reference_models: result.neighbours,
    residual_radius_90pct: round3(result.radius),
    stopping_target: D.modeB.summary.stop_radius,
    stopping_reached: result.radius <= D.modeB.summary.stop_radius,
    next_workload_in_plan: result.nextWorkload,
    replay_evidence: {
      card_mae_at_K: D.modeB.summary.mae_at_K,
      random_order_mae_at_K: D.modeB.summary.rand_mae_at_K,
      rank_error_at_K: D.modeB.summary.rank_err_at_K,
      pareto_agreement_at_K: D.modeB.summary.pareto_agree_at_K,
      evaluation_cost_saved_at_K: D.modeB.summary.cost_saving_at_K,
    },
  };
}

function modelFacts(m) {
  const card = D.cards[m], reg = REG[m];
  return {
    model: m, family: FAM_FULL[reg.family],
    card_percentiles: Object.fromEntries(CARD_DIMS.map(d => [d, pctText(card[d])])),
    needs_gpu: reg.needs_gpu, model_size_mb: reg.model_size,
    median_fit_seconds: reg.fit_time, median_inference_seconds: reg.infer_time,
    failure_flags: modelFlags(m),
    per_track_event_f1: Object.fromEntries(
      ["in-farm", "cross-turbine", "cross-farm"].map(t => [t, round3(
        avgMetric(m, WLS.filter(w => WMETA[w].track === t), "event_f1"))])),
    per_track_early_detection_rate: Object.fromEntries(
      ["in-farm", "cross-turbine", "cross-farm"].map(t => [t, round3(
        avgMetric(m, WLS.filter(w => WMETA[w].track === t), "early_detection_rate"))])),
    mean_false_alarms_per_turbine_day: round3(avgMetric(m, WLS, "false_alarms_per_turbine_day")),
  };
}

function explainModel(m) {
  const host = $("#md-explain-out");
  host.innerHTML = explainBlock("Regenerate");
  host.querySelector(".explain-btn").onclick = () => explainModel(m);
  runExplanation(host,
    `Explain the capability profile of detector "${m}" to an engineer deciding whether to `
    + `shortlist it. Cover what it is good at, what it is weak at, and the operating caveats.`,
    modelFacts(m));
}

/* ---------------- settings dialog ---------------- */

function initLlmDialog() {
  const dlg = $("#llm-dialog"), status = $("#llm-status"), resolved = $("#llm-resolved");
  const fields = { url: $("#llm-url"), key: $("#llm-key"), model: $("#llm-model") };
  const read = () => ({ url: fields.url.value.trim(), key: fields.key.value.trim(),
                        model: fields.model.value.trim() });
  const showResolved = () => {
    const url = fields.url.value.trim();
    resolved.textContent = url ? "Requests go to " + LLM.endpoint(url) : "";
  };
  const refresh = () => {
    const c = LLM.cfg();
    fields.url.value = c.url ?? ""; fields.key.value = c.key ?? "";
    fields.model.value = c.model ?? "";
    status.textContent = ""; status.className = "llm-status";
    showResolved();
    $("#llm-open").textContent = LLM.ready() ? "Explanations: on" : "Explanations…";
  };

  fields.url.oninput = showResolved;
  $("#llm-open").onclick = () => { refresh(); dlg.showModal(); };
  $("#llm-save").onclick = () => {
    if (!LLM.save(read())) {
      status.className = "llm-status error";
      status.textContent = "Kept for this page only — this browser blocks storage on a "
        + "file:// page. Serve the site over http to remember the settings.";
    }
  };
  $("#llm-clear").onclick = () => {
    LLM.clear(); refresh();
    status.textContent = "Cleared from this browser.";
  };
  $("#llm-test").onclick = async () => {
    LLM.save(read());
    status.className = "llm-status"; status.textContent = "Testing…";
    try {
      await LLM.chat("You are a connectivity probe.", "Reply with the single word: ready.",
        () => {});
      status.className = "llm-status ok"; status.textContent = "Connection works.";
    } catch (err) {
      status.className = "llm-status error"; status.textContent = err.message;
    }
  };
  dlg.addEventListener("close", refresh);
  refresh();
}
