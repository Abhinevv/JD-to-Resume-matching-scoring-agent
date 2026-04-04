const state = { results: null, selectedCandidateIndex: 0, dbRows: [] };
const byId = (id) => document.getElementById(id);

const elements = {
  statusBox: byId("statusBox"),
  healthBadge: byId("healthBadge"),
  resultsSection: byId("resultsSection"),
  summaryGrid: byId("summaryGrid"),
  rankingsList: byId("rankingsList"),
  candidateSelect: byId("candidateSelect"),
  candidateDetail: byId("candidateDetail"),
  scoreChart: byId("scoreChart"),
  skillChart: byId("skillChart"),
  roleChart: byId("roleChart"),
  skillGapChart: byId("skillGapChart"),
  itemsetsTable: byId("itemsetsTable"),
  rulesTable: byId("rulesTable"),
  clusterProfiles: byId("clusterProfiles"),
  databaseSummary: byId("databaseSummary"),
  databaseTable: byId("databaseTable"),
  resumeFileList: byId("resumeFileList"),
  weightWarning: byId("weightWarning"),
};

function setStatus(message, kind = "idle") {
  elements.statusBox.textContent = message;
  elements.statusBox.className = `status-box ${kind}`;
}

function formatPct(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function scoreClass(score) {
  if (score >= 0.7) return "strong";
  if (score >= 0.45) return "moderate";
  return "weak";
}

function updateRangeLabel(inputId, labelId) {
  const input = byId(inputId);
  const label = byId(labelId);
  label.textContent = Number(input.value).toFixed(input.step === "1" ? 0 : 2);
}

function syncWeightWarning() {
  const total =
    Number(byId("semanticWeight").value) +
    Number(byId("skillWeight").value) +
    Number(byId("experienceWeight").value);
  elements.weightWarning.textContent = `Weight total: ${total.toFixed(2)}. Ideal is 1.00.`;
  elements.weightWarning.className = `weight-note ${Math.abs(total - 1) > 0.05 ? "warn" : ""}`;
}

function attachRangeUpdates() {
  [
    ["nClusters", "nClustersValue"],
    ["semanticWeight", "semanticWeightValue"],
    ["skillWeight", "skillWeightValue"],
    ["experienceWeight", "experienceWeightValue"],
    ["minSupport", "minSupportValue"],
  ].forEach(([inputId, labelId]) => {
    const input = byId(inputId);
    input.addEventListener("input", () => {
      updateRangeLabel(inputId, labelId);
      syncWeightWarning();
    });
    updateRangeLabel(inputId, labelId);
  });
  syncWeightWarning();
}

function getSettings() {
  return {
    n_clusters: byId("nClusters").value,
    min_support: byId("minSupport").value,
    sem_weight: byId("semanticWeight").value,
    skill_weight: byId("skillWeight").value,
    exp_weight: byId("experienceWeight").value,
  };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof data === "string" ? data : data.detail || JSON.stringify(data);
    throw new Error(message);
  }
  return data;
}

function renderSummary(ranked) {
  const strong = ranked.filter((item) => item.match_score >= 0.7).length;
  const moderate = ranked.filter((item) => item.match_score >= 0.45 && item.match_score < 0.7).length;
  const topScore = ranked[0]?.match_score || 0;
  const cards = [
    ["Total Candidates", ranked.length],
    ["Strong Matches", strong],
    ["Moderate", moderate],
    ["Top Score", formatPct(topScore)],
  ];
  elements.summaryGrid.innerHTML = cards.map(([label, value]) => `
    <div class="summary-card">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");
}

function renderRankings(ranked) {
  elements.rankingsList.innerHTML = ranked.map((item, index) => `
    <article class="candidate-card ${index === state.selectedCandidateIndex ? "active" : ""}" data-candidate-index="${index}">
      <div class="candidate-row">
        <div class="candidate-row">
          <span class="rank-badge">#${index + 1}</span>
          <div>
            <strong>${item.name}</strong>
            <div class="muted">${item.predicted_role || "Unknown role"}</div>
          </div>
        </div>
        <span class="score-badge ${scoreClass(item.match_score)}">${formatPct(item.match_score)}</span>
      </div>
      <div class="pill-row" style="margin-top:14px">
        ${(item.matched_skills || []).slice(0, 5).map((skill) => `<span class="tag match">${skill}</span>`).join("") || `<span class="muted">No matched skills</span>`}
      </div>
    </article>
  `).join("");

  elements.rankingsList.querySelectorAll("[data-candidate-index]").forEach((node) => {
    node.addEventListener("click", () => {
      state.selectedCandidateIndex = Number(node.dataset.candidateIndex);
      syncCandidateSelector();
      renderRankings(ranked);
      renderCandidateDetail();
    });
  });
}

function syncCandidateSelector() {
  elements.candidateSelect.innerHTML = state.results.ranked_candidates.map((item, index) => `
    <option value="${index}" ${index === state.selectedCandidateIndex ? "selected" : ""}>${item.name}</option>
  `).join("");
}

function renderBarList(target, rows) {
  if (!rows.length) {
    target.innerHTML = `<p class="muted">No data available.</p>`;
    return;
  }
  target.innerHTML = `<div class="bar-list">${
    rows.map((row) => `
      <div class="bar-row">
        <div class="bar-head">
          <span>${row.label}</span>
          <strong>${row.valueLabel}</strong>
        </div>
        <div class="bar-track">
          <div class="bar-fill ${row.tone || ""}" style="width:${Math.max(3, row.value)}%"></div>
        </div>
      </div>
    `).join("")
  }</div>`;
}

function renderCandidateDetail() {
  const candidate = state.results.ranked_candidates[state.selectedCandidateIndex];
  if (!candidate) return;
  const detailRows = [
    ["Match Score", formatPct(candidate.match_score)],
    ["Semantic", formatPct(candidate.semantic_similarity)],
    ["Skill Match", formatPct(candidate.skill_overlap)],
    ["Experience Match", formatPct(candidate.experience_score)],
    ["Predicted Role", candidate.predicted_role || "Unknown"],
    ["Experience", `${(candidate.experience_found || 0).toFixed(1)} / ${(candidate.experience_required || 0).toFixed(1)} years`],
  ];

  elements.candidateDetail.innerHTML = `
    <h3>${candidate.name}</h3>
    <p class="muted">${candidate.recommendation || ""}</p>
    <div class="detail-grid">
      ${detailRows.map(([label, value]) => `<div><span class="muted">${label}</span><strong>${value}</strong></div>`).join("")}
    </div>
    <h4>Matched skills</h4>
    <div class="pill-row">${(candidate.matched_skills || []).map((skill) => `<span class="tag match">${skill}</span>`).join("") || `<span class="muted">None</span>`}</div>
    <h4>Missing skills</h4>
    <div class="pill-row">${(candidate.missing_skills || []).map((skill) => `<span class="tag missing">${skill}</span>`).join("") || `<span class="muted">None</span>`}</div>
    <h4>Extra skills</h4>
    <div class="pill-row">${(candidate.extra_skills || []).map((skill) => `<span class="tag extra">${skill}</span>`).join("") || `<span class="muted">None</span>`}</div>
  `;
}

function renderAnalytics() {
  const ranked = state.results.ranked_candidates || [];
  renderBarList(elements.scoreChart, ranked.slice(0, 10).map((item) => ({
    label: item.name,
    value: item.match_score * 100,
    valueLabel: formatPct(item.match_score),
    tone: scoreClass(item.match_score) === "weak" ? "danger" : scoreClass(item.match_score) === "moderate" ? "warn" : "",
  })));

  renderBarList(elements.skillChart, (state.results.skill_frequencies || []).slice(0, 10).map((item) => ({
    label: item.skill,
    value: Math.min(100, Number(item.count || 0) * 10),
    valueLabel: `${item.count}`,
  })));

  const roleCounts = ranked.reduce((acc, item) => {
    const role = item.predicted_role || "Unknown";
    acc[role] = (acc[role] || 0) + 1;
    return acc;
  }, {});
  renderBarList(elements.roleChart, Object.entries(roleCounts).map(([label, count]) => ({
    label,
    value: (count / Math.max(ranked.length, 1)) * 100,
    valueLabel: `${count}`,
  })));

  renderBarList(elements.skillGapChart, (state.results.skill_gap || []).slice(0, 10).map((item) => ({
    label: item.skill,
    value: Number(item.coverage_pct || 0) * 100,
    valueLabel: formatPct(item.coverage_pct || 0),
    tone: Number(item.coverage_pct || 0) < 0.4 ? "danger" : Number(item.coverage_pct || 0) < 0.7 ? "warn" : "",
  })));
}

function renderTable(target, columns, rows) {
  if (!rows.length) {
    target.innerHTML = `<p class="muted">No data available.</p>`;
    return;
  }
  target.innerHTML = `
    <div class="table-scroll">
      <table>
        <thead><tr>${columns.map((column) => `<th>${column.label}</th>`).join("")}</tr></thead>
        <tbody>
          ${rows.map((row) => `<tr>${columns.map((column) => `<td class="${column.codeish ? "codeish" : ""}">${row[column.key] ?? ""}</td>`).join("")}</tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderMining() {
  renderTable(elements.itemsetsTable,
    [{ key: "itemsets", label: "Itemset" }, { key: "support", label: "Support" }],
    (state.results.apriori_itemsets || []).slice(0, 20).map((item) => ({
      itemsets: Array.isArray(item.itemsets) ? item.itemsets.join(", ") : "",
      support: Number(item.support || 0).toFixed(3),
    })),
  );

  renderTable(elements.rulesTable,
    [
      { key: "antecedents", label: "Antecedents" },
      { key: "consequents", label: "Consequents" },
      { key: "confidence", label: "Confidence" },
      { key: "lift", label: "Lift" },
    ],
    (state.results.apriori_rules || []).slice(0, 15).map((item) => ({
      antecedents: Array.isArray(item.antecedents) ? item.antecedents.join(", ") : "",
      consequents: Array.isArray(item.consequents) ? item.consequents.join(", ") : "",
      confidence: Number(item.confidence || 0).toFixed(3),
      lift: Number(item.lift || 0).toFixed(3),
    })),
  );

  const profiles = state.results.cluster_profiles || {};
  elements.clusterProfiles.innerHTML = Object.entries(profiles).length
    ? Object.entries(profiles).map(([clusterId, profile]) => `
        <article class="cluster-card">
          <div class="candidate-row">
            <strong>Cluster ${clusterId}</strong>
            <span class="score-badge">${profile.size} resume(s)</span>
          </div>
          <p class="muted">Dominant role: ${profile.dominant_role || "Unknown"}</p>
          <p>Average experience: ${(profile.avg_experience || 0).toFixed(1)} years</p>
          <div class="pill-row">${(profile.top_skills || []).map((skill) => `<span class="tag extra">${skill}</span>`).join("") || `<span class="muted">No top skills</span>`}</div>
        </article>
      `).join("")
    : `<p class="muted">No cluster profiles available.</p>`;
}

function renderDatabaseSummary(meta) {
  const cards = [
    ["SQLite Records", meta.resume_count ?? state.dbRows.length],
    ["FAISS Vectors", meta.faiss_vectors ?? 0],
    ["Loaded In View", state.dbRows.length],
  ];
  elements.databaseSummary.innerHTML = cards.map(([label, value]) => `
    <div class="summary-card">
      <span>${label}</span>
      <strong>${value}</strong>
    </div>
  `).join("");
}

function renderDatabaseTable() {
  renderTable(elements.databaseTable,
    [
      { key: "resume_id", label: "Resume ID", codeish: true },
      { key: "name", label: "Name" },
      { key: "experience", label: "Experience" },
      { key: "education", label: "Education" },
      { key: "predicted_role", label: "Predicted Role" },
      { key: "match_score", label: "Match Score" },
    ],
    state.dbRows.map((row) => ({
      resume_id: row.resume_id || "",
      name: row.name || "",
      experience: Number(row.experience || 0).toFixed(1),
      education: row.education || "",
      predicted_role: row.predicted_role || "",
      match_score: formatPct(row.match_score || 0),
    })),
  );
}

async function refreshDatabaseView() {
  const [meta, resumesResponse] = await Promise.all([fetchJson("/dashboard/meta"), fetchJson("/resumes")]);
  state.dbRows = resumesResponse.resumes || [];
  renderDatabaseSummary(meta);
  renderDatabaseTable();
}

function renderAllResults() {
  const ranked = state.results.ranked_candidates || [];
  if (!ranked.length) {
    setStatus("Run completed, but no ranked candidates were returned.", "error");
    return;
  }
  elements.resultsSection.classList.remove("hidden");
  renderSummary(ranked);
  syncCandidateSelector();
  renderRankings(ranked);
  renderCandidateDetail();
  renderAnalytics();
  renderMining();
}

async function runSample() {
  const settings = getSettings();
  const form = new FormData();
  form.append("role", byId("sampleRole").value);
  form.append("n_clusters", settings.n_clusters);
  form.append("min_support", settings.min_support);
  form.append("sem_weight", settings.sem_weight);
  form.append("skill_weight", settings.skill_weight);
  form.append("exp_weight", settings.exp_weight);
  setStatus("Running sample pipeline...", "loading");
  state.results = await fetchJson("/sample", { method: "POST", body: form });
  state.selectedCandidateIndex = 0;
  renderAllResults();
  await refreshDatabaseView();
  setStatus("Sample pipeline completed.", "success");
}

async function runUpload() {
  const jdText = byId("jdText").value.trim();
  const files = byId("resumeFiles").files;
  if (!jdText || !files.length) {
    setStatus("Please add a job description and at least one resume file.", "error");
    return;
  }

  const settings = getSettings();
  const form = new FormData();
  form.append("jd_text", jdText);
  form.append("n_clusters", settings.n_clusters);
  form.append("min_support", settings.min_support);
  form.append("sem_weight", settings.sem_weight);
  form.append("skill_weight", settings.skill_weight);
  form.append("exp_weight", settings.exp_weight);
  Array.from(files).forEach((file) => form.append("resumes", file));

  setStatus("Uploading files and running pipeline...", "loading");
  state.results = await fetchJson("/match", { method: "POST", body: form });
  state.selectedCandidateIndex = 0;
  renderAllResults();
  await refreshDatabaseView();
  setStatus("Upload pipeline completed.", "success");
}

function downloadCsv() {
  if (!state.dbRows.length) {
    setStatus("No database rows available to export yet.", "error");
    return;
  }
  const columns = ["resume_id", "name", "experience", "education", "predicted_role", "match_score"];
  const lines = [columns.join(","), ...state.dbRows.map((row) => columns.map((column) => JSON.stringify(row[column] ?? "")).join(","))];
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "resume_results.csv";
  link.click();
  URL.revokeObjectURL(url);
}

function setupTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".tab-button").forEach((node) => node.classList.remove("active"));
      document.querySelectorAll(".tab-pane").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      byId(button.dataset.tab).classList.add("active");
    });
  });

  document.querySelectorAll(".results-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".results-tab").forEach((node) => node.classList.remove("active"));
      document.querySelectorAll(".results-pane").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      byId(button.dataset.resultsTab).classList.add("active");
    });
  });
}

function setupFileList() {
  byId("resumeFiles").addEventListener("change", (event) => {
    const files = Array.from(event.target.files || []);
    elements.resumeFileList.innerHTML = files.length
      ? files.map((file) => `<div>${file.name} <span class="muted">(${(file.size / 1024).toFixed(1)} KB)</span></div>`).join("")
      : "No files selected yet.";
  });
}

async function checkHealth() {
  try {
    const data = await fetchJson("/health");
    elements.healthBadge.textContent = data.status;
  } catch (_error) {
    elements.healthBadge.textContent = "Unavailable";
  }
}

function bindEvents() {
  byId("runSampleButton").addEventListener("click", () => runSample().catch((error) => setStatus(`Sample run failed: ${error.message}`, "error")));
  byId("runUploadButton").addEventListener("click", () => runUpload().catch((error) => setStatus(`Upload run failed: ${error.message}`, "error")));
  byId("candidateSelect").addEventListener("change", (event) => {
    state.selectedCandidateIndex = Number(event.target.value);
    renderRankings(state.results.ranked_candidates || []);
    renderCandidateDetail();
  });
  byId("refreshDatabaseButton").addEventListener("click", () => refreshDatabaseView().catch((error) => setStatus(`Database refresh failed: ${error.message}`, "error")));
  byId("downloadCsvButton").addEventListener("click", downloadCsv);
}

function init() {
  attachRangeUpdates();
  setupTabs();
  setupFileList();
  bindEvents();
  checkHealth();
  refreshDatabaseView().catch((error) => setStatus(`Database refresh failed: ${error.message}`, "error"));
}

init();
