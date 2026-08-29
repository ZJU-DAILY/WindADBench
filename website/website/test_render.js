/* DOM smoke test: loads index.html with all scripts and exercises the views.

   Needs jsdom, which is not vendored with this static site:
     npm install --no-save jsdom && node test_render.js
*/
const { JSDOM, VirtualConsole } = require("jsdom");
const SITE = require("path").resolve(__dirname);
const errors = [];
const vc = new VirtualConsole()
  .on("jsdomError", e => errors.push("jsdomError: " + e.message + "\n  " + (e.stack||"").split("\n").slice(1,4).join("\n  ")))
  .on("error", (...a) => errors.push("console.error: " + a.join(" ")));

JSDOM.fromFile(SITE + "/index.html", {
  runScripts: "dangerously",
  resources: "usable",
  virtualConsole: vc,
  beforeParse(window) {
    window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
    window.HTMLDialogElement.prototype.close = function () { this.open = false; };
    window.scrollTo = () => {};
    window.Element.prototype.scrollIntoView = () => {};
  },
}).then(async (dom) => {
  const { window } = dom;
  await new Promise(r => setTimeout(r, 900));
  const $ = (s) => window.document.querySelector(s);
  const checks = [];
  const has = (name, sel, min) => {
    const el = $(sel);
    const n = el ? el.querySelectorAll("*").length : -1;
    checks.push([name, n >= min, `${n} nodes`]);
  };
  has("leaderboard table", "#lb-table", 200);
  has("models table", "#md-table", 200);
  has("workloads table", "#wl-table", 60);
  has("family legend", "#lb-legend", 6);
  has("evaluation plan list", "#nm-order", 12);
  has("budget chart", "#nm-chart", 10);
  has("D2M presets", "#nd-presets", 4);
  has("D2M sliders", "#nd-sliders", 8);
  has("D2M result", "#nd-out", 10);
  has("M2B input row", "#nm-rows", 6);

  try {
    $("#lb-track").value = "cross-farm";
    $("#lb-track").dispatchEvent(new window.Event("change"));
    checks.push(["track switch", $("#lb-workload").innerHTML.includes("CF"), ""]);

    const preset = window.document.querySelectorAll("#nd-presets .preset")[1];
    preset.click();
    const verdict = $("#nd-out").innerHTML;
    checks.push(["preset -> verdict", verdict.includes("verdict"), verdict.slice(0, 80)]);
    checks.push(["explain button", verdict.includes("explain-btn"), ""]);

    const row = $("#nm-rows .nm-row");
    row.querySelector(".nm-event_f1").value = "0.82";
    row.querySelector(".nm-early_detection_rate").value = "0.31";
    $("#nm-run").click();
    const out = $("#nm-out").innerHTML;
    checks.push(["M2B run -> rank", /rank \d+ of \d+/.test(out), out.slice(0, 100)]);

    $("#nm-add").click();
    checks.push(["M2B add row", window.document.querySelectorAll(".nm-row").length === 2, ""]);

    const md = window.document.querySelector("#md-table tr.click");
    md.click();
    checks.push(["model detail", $("#md-detail").innerHTML.includes("Per-workload results"), ""]);

    window.document.querySelector("#wl-table tr.click").click();
    checks.push(["workload detail", $("#wl-detail").innerHTML.includes("Top-10"), ""]);

    $("#llm-open").click();
    checks.push(["llm dialog opens", $("#llm-dialog").open === true, ""]);
    // explanation without a configured endpoint must fall back to a hint
    $("#nd-out .explain-btn").click();
    await new Promise(r => setTimeout(r, 60));
    const hint = $("#nd-out .explain-text");
    checks.push(["explain hint w/o key", hint && hint.className.includes("hint"), hint ? hint.className : "missing"]);
  } catch (e) {
    errors.push("interaction: " + e.message + "\n  " + (e.stack||"").split("\n")[1]);
  }

  let bad = 0;
  for (const [name, ok, info] of checks) { if (!ok) bad++; console.log(`${ok ? "PASS" : "FAIL"}  ${name}${ok ? "" : " — " + info}`); }
  if (errors.length) { console.log("\nERRORS:"); errors.forEach(e => console.log("  " + e)); }
  console.log(`\n${bad || errors.length ? "FAILED" : "all render checks passed"}`);
  process.exit(bad || errors.length ? 1 : 0);
});
