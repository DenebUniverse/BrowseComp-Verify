/* Local aggregate results only. No dependencies, telemetry, or model calls. */
"use strict";

const isChinese = document.documentElement.lang.startsWith("zh");
const tr = (en, zh) => isChinese ? zh : en;
const $ = (selector) => document.querySelector(selector);
const fmt = (value) => Number(value).toFixed(2);
const escapeHTML = (text) => String(text).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const table = (headers, rows) => `<table><thead><tr>${headers.map((h) => `<th scope="col">${escapeHTML(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell, i) => i === 0 ? `<th scope="row">${escapeHTML(cell)}</th>` : `<td>${escapeHTML(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
const option = (value, label) => {const el = document.createElement("option"); el.value = value; el.textContent = label; return el;};

function initScatter(points) {
  const select = $("#scatter-model");
  points.forEach((p) => select.append(option(p.id, p.model)));
  // Fixed offsets keep the complete model labels readable at desktop width.
  const label = {
    gptoss_120b_agent: [0,-17,"middle"], tongyi_30b_agent: [12,18,"start"],
    miro_v1_5_30b_agent: [12,6,"start"], glm_v4_7_358b_agent: [-10,22,"end"],
    deepseek_v3_2_agent: [0,24,"middle"], gptoss_20b_agent: [12,-14,"start"],
    gpt_v5_2_agent: [-12,-15,"end"], deepseek_v4_flash_max_agent: [14,20,"start"],
    deepseek_v4_pro_max_agent: [12,-17,"start"], claude_v4_6_opus_high_agent: [0,-19,"middle"]
  };
  const colors = ["#425bb0","#8b6e9d","#b56a25","#ad6670","#287864","#425bb0","#425bb0","#61919c","#287864","#b56a25"];
  const x = (v) => 70 + (v-20)/80*555;
  const y = (v) => 425 - (v-20)/80*385;
  let grid = "";
  for (let v=20; v<=100; v+=20) {
    grid += `<line x1="70" y1="${y(v)}" x2="625" y2="${y(v)}" stroke="#e8e9e7"/><line x1="${x(v)}" y1="40" x2="${x(v)}" y2="425" stroke="#e8e9e7"/><text class="scatter-tick" x="58" y="${y(v)+4}" text-anchor="end">${v}%</text><text class="scatter-tick" x="${x(v)}" y="446" text-anchor="middle">${v}%</text>`;
  }
  $("#scatter").innerHTML = `<svg viewBox="0 0 700 490" role="group" aria-label="${tr("Interactive solver versus verifier accuracy scatter plot. Each point is keyboard accessible.","求解与验证准确率交互散点图，每个点均支持键盘操作。")}"><path d="M70 40H625L70 425Z" fill="#f0f3f9"/>${grid}<line x1="70" y1="425" x2="625" y2="40" stroke="#8993a8" stroke-dasharray="5 5"/><text x="84" y="62" class="scatter-tick" fill="#425bb0">${tr("Verifier > Solver","验证 > 求解")}</text><text class="scatter-axis" x="348" y="481" text-anchor="middle">${tr("Solver accuracy (%)","求解准确率（%）")}</text><text class="scatter-axis" transform="translate(17 232) rotate(-90)" text-anchor="middle">${tr("Verifier accuracy (%)","验证准确率（%）")}</text>${points.map((p,i) => {
    const [dx,dy,anchor] = label[p.id];
    return `<g class="scatter-point" data-id="${escapeHTML(p.id)}" tabindex="0" role="button" aria-pressed="false" aria-label="${escapeHTML(p.model)}: ${tr("solver","求解")} ${fmt(p.solver_accuracy)}%, ${tr("verifier","验证")} ${fmt(p.verifier_accuracy)}%"><title>${escapeHTML(p.model)} · ${tr("Solver","求解")} ${fmt(p.solver_accuracy)}% · ${tr("Verifier","验证")} ${fmt(p.verifier_accuracy)}%</title><circle cx="${x(p.solver_accuracy)}" cy="${y(p.verifier_accuracy)}" r="7" fill="${colors[i]}" stroke="white" stroke-width="2"/><text x="${x(p.solver_accuracy)+dx}" y="${y(p.verifier_accuracy)+dy}" text-anchor="${anchor}">${escapeHTML(p.model)}</text></g>`;
  }).join("")}</svg>`;
  function highlight(id) {
    document.querySelectorAll(".scatter-point").forEach((el) => {
      el.classList.toggle("dim", id !== "all" && el.dataset.id !== id);
      el.classList.toggle("selected", el.dataset.id === id);
      el.setAttribute("aria-pressed", String(el.dataset.id === id));
    });
    const p = points.find((p) => p.id === id);
    $("#scatter-detail").textContent = p ? `${p.model} · ${tr("Solver","求解")}: ${fmt(p.solver_accuracy)}% · ${tr("Verifier","验证")}: ${fmt(p.verifier_accuracy)}%` : tr("Choose a model or select a point to inspect its two accuracies.","选择模型或点击图中的点，查看它在两种角色下的准确率。");
    select.value = id;
  }
  select.addEventListener("change", () => highlight(select.value));
  document.querySelectorAll(".scatter-point").forEach((el) => {
    const activate = () => highlight(select.value === el.dataset.id ? "all" : el.dataset.id);
    el.addEventListener("click", activate);
    el.addEventListener("keydown", (event) => {if (event.key === "Enter" || event.key === " ") {event.preventDefault(); activate();}});
  });
  $("#scatter-table").innerHTML = table([tr("Model","模型"),tr("Solver","求解"),tr("Verifier","验证")], points.map((p) => [p.model,`${fmt(p.solver_accuracy)}%`,`${fmt(p.verifier_accuracy)}%`]));
}

function initMainResults(rows) {
  const names = {correct:tr("Correct candidate","正确候选"),incorrect:tr("Incorrect candidate","错误候选"),overall:tr("Overall","总体")};
  function render() {
    const order = $("#result-order").value;
    const sorted = [...rows].sort((a,b) => order === "gap" ? Math.abs(b.correct-b.incorrect)-Math.abs(a.correct-a.incorrect) : (order === "asc" ? a.overall-b.overall : b.overall-a.overall));
    $("#main-chart").innerHTML = sorted.map((r) => `<div class="result-row"><div class="result-name">${escapeHTML(r.model)}</div><div class="bar-group">${["correct","incorrect","overall"].map((key) => `<div class="bar-line ${key}" aria-label="${names[key]}: ${fmt(r[key])}${tr(" percent","%") }"><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${r[key]}%"></div></div><span class="bar-number">${fmt(r[key])}</span></div>`).join("")}</div></div>`).join("") + '<div class="axis-labels" aria-hidden="true"><span>0</span><span>50</span><span>100%</span></div>';
    $("#main-table").innerHTML = table([tr("Agent","智能体"),tr("Overall","总体"),tr("Correct","正确候选"),tr("Incorrect","错误候选"),tr("Gap (pp)","差距（百分点）")], sorted.map((r) => [r.model,...["overall","correct","incorrect"].map((k) => fmt(r[k])+"%"),fmt(Math.abs(r.correct-r.incorrect))]));
  }
  $("#result-order").addEventListener("change", render); render();
}

function initAblation(rows) {
  const select = $("#ablation-model");
  rows.forEach((r,i) => select.append(option(String(i),r.model)));
  select.value = "1";
  let prompt = "strict";
  function render() {
    const r = rows[Number(select.value)], v = r[prompt];
    const labels = {overall:tr("Overall (%)","总体（%）"),correct:tr("Correct (%)","正确候选（%）"),incorrect:tr("Incorrect (%)","错误候选（%）"),gap:tr("Class gap (pp)","类别差距（百分点）")};
    $("#ablation-chart").innerHTML = `<div class="ablation-stats">${["overall","correct","incorrect","gap"].map((k) => `<div class="ablation-stat ${k}"><b>${fmt(v[k])}</b><span>${labels[k]}</span></div>`).join("")}</div><span class="figure-kicker">${tr("Overall accuracy: both prompts","两种提示词的总体准确率")}</span>${["strict","relaxed"].map((p) => `<div class="comparison-line ${p}"><span>${p === "strict" ? tr("Strict","严格") : tr("Relaxed","宽松")}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="width:${r[p].overall}%"></div></div><strong>${fmt(r[p].overall)}</strong></div>`).join("")}`;
    const change = r.relaxed.overall-r.strict.overall;
    $("#ablation-insight").textContent = isChinese ? `严格 → 宽松：类别差距 ${fmt(r.strict.gap)} → ${fmt(r.relaxed.gap)} 个百分点；总体准确率 ${fmt(r.strict.overall)}% → ${fmt(r.relaxed.overall)}%（${change >= 0 ? "+" : "−"}${fmt(Math.abs(change))} 个百分点）。${change < 0 ? "类别差距缩小，但总体准确率下降。" : "这里的总体准确率提高了，但两类错误之间仍存在权衡。"}` : `Strict → relaxed: class gap ${fmt(r.strict.gap)} → ${fmt(r.relaxed.gap)} pp. Overall ${fmt(r.strict.overall)}% → ${fmt(r.relaxed.overall)}% (${change >= 0 ? "+" : "−"}${fmt(Math.abs(change))} pp). ${change < 0 ? "The gap shrinks while overall accuracy falls." : "Here, overall accuracy improves, but the error trade-off remains."}`;
    document.querySelectorAll("[data-prompt]").forEach((b) => b.setAttribute("aria-pressed",String(b.dataset.prompt === prompt)));
  }
  document.querySelectorAll("[data-prompt]").forEach((b) => b.addEventListener("click", () => {prompt=b.dataset.prompt; render();}));
  select.addEventListener("change",render);
  $("#ablation-table").innerHTML = table([tr("Model / prompt","模型 / 提示词"),tr("Overall","总体"),tr("Correct","正确候选"),tr("Incorrect","错误候选"),tr("Gap (pp)","差距（百分点）")],rows.flatMap((r) => ["strict","relaxed"].map((p) => [`${r.model} / ${p === "strict" ? tr("strict","严格") : tr("relaxed","宽松")}`,...["overall","correct","incorrect"].map((k) => fmt(r[p][k])+"%"),fmt(r[p].gap)])));
  render();
}

function initExample() {
  const stages = isChinese ? [
    '<div class="example-stage"><strong>两个条件必须同时成立。</strong><ul><li>Alex Rowan 参与撰写了 <em>Project Cedar</em>。</li><li>Alex Rowan 获得了 <em>2024 Horizon Prize</em>。</li></ul><p>智能体从问题本身推导出这两项检查。</p></div>',
    '<div class="example-stage"><p><span class="status">支持</span>项目页面将 Alex Rowan 列为共同作者。</p><p><span class="status missing">未找到</span>已检索页面无法证实获奖这一条件。</p><p>没有找到支持证据，并不意味着获奖陈述为假。</p></div>',
    '<div class="example-stage"><strong>按照严格规则，候选答案会被拒绝。</strong><p>一项必要约束仍缺乏支持。如果候选答案实际上正确，这就是证据不完整造成的误拒，而不是已经证实的矛盾。</p></div>'
  ] : [
    '<div class="example-stage"><strong>Two claims must both hold.</strong><ul><li>Alex Rowan coauthored <em>Project Cedar</em>.</li><li>Alex Rowan received the <em>2024 Horizon Prize</em>.</li></ul><p>The agent derives these checks from the question itself.</p></div>',
    '<div class="example-stage"><p><span class="status">SUPPORTED</span>A project page lists Alex Rowan as a coauthor.</p><p><span class="status missing">NOT FOUND</span>The retrieved pages do not establish the award claim.</p><p>Not finding support does not establish that the award claim is false.</p></div>',
    '<div class="example-stage"><strong>Under the strict rule, the candidate is rejected.</strong><p>One necessary constraint remains unsupported. If the candidate is actually correct, this is a false rejection caused by incomplete evidence—not a demonstrated contradiction.</p></div>'
  ];
  function render(step) {$("#example-step").innerHTML=stages[step]; document.querySelectorAll("[data-step]").forEach((b) => b.setAttribute("aria-pressed",String(Number(b.dataset.step)===step)));}
  document.querySelectorAll("[data-step]").forEach((b) => b.addEventListener("click",() => render(Number(b.dataset.step)))); render(0);
}

function initNavigation() {
  const links=[...document.querySelectorAll(".toc a")];
  if (!("IntersectionObserver" in window)) return;
  const observer=new IntersectionObserver((entries) => {entries.forEach((e) => {if(e.isIntersecting){links.forEach((a) => {const active=a.hash===`#${e.target.id}`;a.classList.toggle("active",active);if(active)a.setAttribute("aria-current","location");else a.removeAttribute("aria-current");});}});},{rootMargin:"-10% 0px -65% 0px",threshold:0});
  document.querySelectorAll("article>section").forEach((s) => observer.observe(s));
}

async function init() {
  initExample(); initNavigation();
  $("#copy-citation").addEventListener("click",async () => {
    try {await navigator.clipboard.writeText($("#bibtex").textContent);$("#copy-status").textContent=tr("BibTeX copied.","已复制 BibTeX。");}
    catch {const range=document.createRange();range.selectNodeContents($("#bibtex"));const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);$("#copy-status").textContent=tr("Copy unavailable here. The citation is selected; use your browser’s Copy command.","当前环境无法自动复制。已选中引用内容，请使用浏览器的复制命令。");}
  });
  try {
    const response=await fetch(new URL("../data/results.json",document.querySelector('script[src$="assets/main.js"]').src),{cache:"no-cache"});
    if(!response.ok) throw new Error(`HTTP ${response.status}`);
    const data=await response.json(); initScatter(data.scatter);initMainResults(data.main);initAblation(data.ablation);
  } catch(error) {
    for(const id of ["#scatter","#main-chart","#ablation-chart"]) $(id).textContent=tr("The figure could not load. Please refresh or use the published paper linked above.","图表加载失败，请刷新页面或查看上方链接中的论文。");
    console.error("Unable to load local aggregate results:",error);
  }
}
init();
