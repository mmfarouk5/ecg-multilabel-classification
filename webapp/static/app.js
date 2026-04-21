/**
 * ECG Diagnosis AI — Frontend Application
 */
(function () {
  "use strict";

  // ── DOM refs ──────────────────────────────────────────
  const $dropzone = document.getElementById("dropzone");
  const $fileInput = document.getElementById("file-input");
  const $fileInfo = document.getElementById("file-info");
  const $fileName = document.getElementById("file-name");
  const $btnSample = document.getElementById("btn-sample");
  const $btnUpload = document.getElementById("btn-upload");
  const $ecgSection = document.getElementById("ecg-section");
  const $ecgGrid = document.getElementById("ecg-grid");

  const $resultsSection = document.getElementById("results-section");
  const $resultsSummary = document.getElementById("results-summary");
  const $resultsGrid = document.getElementById("results-grid");
  const $groundTruth = document.getElementById("ground-truth");
  const $loader = document.getElementById("loader");
  const $loaderText = document.getElementById("loader-text");
  const $toast = document.getElementById("toast");

  const LEAD_NAMES = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"];
  let currentFile = null;

  // ── Utilities ─────────────────────────────────────────
  function showLoader(msg) {
    $loaderText.textContent = msg || "Processing...";
    $loader.classList.add("active");
  }
  function hideLoader() { $loader.classList.remove("active"); }

  function showToast(msg, type) {
    $toast.textContent = msg;
    $toast.className = "toast visible " + (type || "error");
    setTimeout(() => { $toast.classList.remove("visible"); }, 4000);
  }

  // ── Drag & Drop ───────────────────────────────────────
  $dropzone.addEventListener("click", () => $fileInput.click());

  $dropzone.addEventListener("dragover", (e) => {
    e.preventDefault(); $dropzone.classList.add("drag-over");
  });
  $dropzone.addEventListener("dragleave", () => $dropzone.classList.remove("drag-over"));
  $dropzone.addEventListener("drop", (e) => {
    e.preventDefault(); $dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  });

  $fileInput.addEventListener("change", () => {
    if ($fileInput.files.length) handleFile($fileInput.files[0]);
  });

  function handleFile(file) {
    if (!file.name.endsWith(".csv")) {
      showToast("Please upload a CSV file.", "error"); return;
    }
    currentFile = file;
    $fileName.textContent = file.name + " (" + (file.size / 1024).toFixed(1) + " KB)";
    $fileInfo.classList.add("visible");
    $btnUpload.style.display = "inline-flex";
  }

  // ── Upload predict ────────────────────────────────────
  $btnUpload.addEventListener("click", async () => {
    if (!currentFile) return;
    showLoader("Preprocessing & analyzing ECG signal...");
    try {
      const form = new FormData();
      form.append("file", currentFile);
      const res = await fetch("/api/predict", { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Prediction failed");
      }
      const data = await res.json();
      renderResults(data, null);
    } catch (e) {
      showToast(e.message, "error");
    } finally { hideLoader(); }
  });

  // ── Sample predict ────────────────────────────────────
  $btnSample.addEventListener("click", async () => {
    showLoader("Loading random PTB-XL sample & running inference...");
    try {
      const res = await fetch("/api/predict-sample", { method: "POST" });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Sample prediction failed");
      }
      const data = await res.json();
      renderResults(data, data.ground_truth);
      showToast("Sample #" + data.sample_index + " loaded successfully", "success");
    } catch (e) {
      showToast(e.message, "error");
    } finally { hideLoader(); }
  });

  // ── Render Results ────────────────────────────────────
  function renderResults(data, groundTruth) {
    // Show sections
    $ecgSection.classList.add("visible");
    $resultsSection.classList.add("visible");

    // Render ECG
    if (data.signal) renderECG(data.signal);

    // ── Build interpretable summary ──────────────────────
    const nPred = data.num_predicted;
    const predicted = data.predicted_classes;
    const detected = data.classes.filter((c) => c.predicted);
    const hasCritical = detected.some((c) => c.severity === "critical");
    const hasWarning  = detected.some((c) => c.severity === "warning");
    const isNormal = nPred === 0 || (nPred === 1 && predicted[0] === "Normal ECG");

    // Determine top-level verdict
    let summaryColor, summaryIcon, summaryTitle, summarySubtitle;
    if (isNormal) {
      summaryColor = "var(--success)";
      summaryIcon = "✅";
      summaryTitle = "Normal ECG";
      summarySubtitle = "No significant cardiac abnormalities were detected by the model.";
    } else if (hasCritical) {
      summaryColor = "var(--danger)";
      summaryIcon = "🚨";
      summaryTitle = "Critical Finding";
      summarySubtitle = "The model detected at least one critical cardiac condition that warrants urgent clinical review.";
    } else {
      summaryColor = "var(--warning)";
      summaryIcon = "⚠️";
      summaryTitle = "Abnormalities Detected";
      summarySubtitle = "The model detected cardiac abnormalities that may require further clinical evaluation.";
    }

    // Build detected-findings breakdown
    let findingsHTML = "";
    if (!isNormal) {
      const findingItems = detected
        .filter((c) => c.name !== "NORM")
        .sort((a, b) => b.probability - a.probability)
        .map((c) => {
          const pct = (c.probability * 100).toFixed(1);
          let sevLabel = "Warning";
          let sevClass = "badge-warning";
          if (c.severity === "critical") { sevLabel = "Critical"; sevClass = "badge-danger"; }
          return `<div class="summary-finding">
            <span class="summary-finding-dot" style="background:${c.color}"></span>
            <span class="summary-finding-name">${c.full_name}</span>
            <span class="badge ${sevClass}" style="font-size:0.6rem;padding:2px 8px">${sevLabel}</span>
            <span class="summary-finding-prob" style="color:${c.color}">${pct}%</span>
          </div>`;
        })
        .join("");

      findingsHTML = `<div class="summary-findings">
        <div class="summary-findings-title">Detected Conditions (probability ≥ 50%)</div>
        ${findingItems}
      </div>`;
    }

    // Clinical interpretation
    let interpretationHTML = "";
    if (isNormal) {
      interpretationHTML = `<div class="summary-interpretation">
        <strong>Interpretation:</strong> The ECG signal appears within normal limits. All five diagnostic categories scored below the detection threshold, or only the Normal class was activated.
      </div>`;
    } else {
      const names = detected.filter((c) => c.name !== "NORM").map((c) => c.full_name);
      const highest = detected.filter((c) => c.name !== "NORM").sort((a, b) => b.probability - a.probability)[0];
      let interp = `The model identified <strong>${names.join(", ")}</strong> in this ECG recording.`;
      interp += ` The highest-confidence finding is <strong>${highest.full_name}</strong> at <strong>${(highest.probability * 100).toFixed(1)}%</strong>.`;
      if (hasCritical) {
        interp += ` <span style="color:var(--danger)">Critical findings such as Myocardial Infarction require immediate clinical attention.</span>`;
      }
      interpretationHTML = `<div class="summary-interpretation">
        <strong>Interpretation:</strong> ${interp}
      </div>`;
    }

    $resultsSummary.innerHTML = `
      <div class="summary-header">
        <div class="summary-icon" style="background:${summaryColor}22;color:${summaryColor}">
          ${summaryIcon}
        </div>
        <div class="summary-text">
          <h3 style="color:${summaryColor}">${summaryTitle}</h3>
          <p>${summarySubtitle}</p>
        </div>
      </div>
      ${findingsHTML}
      ${interpretationHTML}
      <div class="summary-disclaimer">
        ⚕️ This is an AI-assisted screening tool for research purposes only. Results must be validated by a qualified clinician before any clinical decision is made.
      </div>`;

    // Cards
    $resultsGrid.innerHTML = "";
    data.classes.forEach((cls) => {
      const pct = (cls.probability * 100).toFixed(1);
      const circumference = 2 * Math.PI * 34;
      const offset = circumference - (cls.probability * circumference);
      const isPred = cls.predicted;

      const card = document.createElement("div");
      card.className = "result-card glass" + (isPred ? " predicted" : "");
      card.style.borderColor = isPred ? cls.color + "44" : "";

      let badgeClass = "badge-accent";
      if (cls.severity === "critical") badgeClass = "badge-danger";
      else if (cls.severity === "warning") badgeClass = "badge-warning";
      else if (cls.severity === "normal") badgeClass = "badge-success";

      card.innerHTML = `
        ${isPred ? '<div class="predicted-badge"><span class="badge ' + badgeClass + '">Detected</span></div>' : ''}
        <div class="card-header">
          <span class="card-name" style="color:${cls.color}">${cls.name}</span>
          <span class="badge ${badgeClass}" style="font-size:0.65rem">${cls.confidence}</span>
        </div>
        <div class="card-full-name">${cls.full_name}</div>
        <div class="prob-ring">
          <svg viewBox="0 0 76 76">
            <circle class="prob-ring-bg" cx="38" cy="38" r="34"/>
            <circle class="prob-ring-fill" cx="38" cy="38" r="34"
              stroke="${cls.color}" stroke-dasharray="${circumference}"
              stroke-dashoffset="${offset}"/>
          </svg>
          <div class="prob-value" style="color:${cls.color}">${pct}%</div>
        </div>
        <div class="card-desc">${cls.description}</div>`;

      $resultsGrid.appendChild(card);
    });

    // Ground truth
    if (groundTruth && groundTruth.length) {
      $groundTruth.style.display = "block";
      let tags = groundTruth.map((g) =>
        `<span class="gt-tag ${g.present ? 'present' : 'absent'}">${g.name}${g.present ? ' ✓' : ''}</span>`
      ).join("");
      $groundTruth.innerHTML = `<h4>📋 Ground Truth Labels</h4><div class="gt-tags">${tags}</div>`;
    } else {
      $groundTruth.style.display = "none";
    }

    // Scroll to ECG section
    $ecgSection.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  // ── ECG Rendering ─────────────────────────────────────
  function renderECG(signal) {
    // signal: array of arrays [1000][12]
    $ecgGrid.innerHTML = "";

    LEAD_NAMES.forEach((name, leadIdx) => {
      const div = document.createElement("div");
      div.className = "ecg-lead";
      div.innerHTML = `<div class="ecg-lead-label">${name}</div><canvas height="80"></canvas>`;
      $ecgGrid.appendChild(div);
      const canvas = div.querySelector("canvas");
      drawLeadSignal(canvas, signal, leadIdx, "rgba(0,212,255,0.85)");
    });
  }

  function drawLeadSignal(canvas, signal, leadIdx, color) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 24;
    const h = parseInt(canvas.getAttribute("height")) || 80;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    canvas.style.width = w + "px";
    canvas.style.height = h + "px";

    const ctx = canvas.getContext("2d");
    ctx.scale(dpr, dpr);

    // Extract lead data
    const data = signal.map((row) => row[leadIdx]);
    const len = data.length;

    // Compute range
    let mn = Infinity, mx = -Infinity;
    for (let i = 0; i < len; i++) {
      if (data[i] < mn) mn = data[i];
      if (data[i] > mx) mx = data[i];
    }
    const range = mx - mn || 1;
    const padY = 6;

    // Draw subtle grid
    ctx.strokeStyle = "rgba(0,212,255,0.06)";
    ctx.lineWidth = 0.5;
    for (let y = 0; y < h; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    for (let x = 0; x < w; x += 25) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }

    // Draw signal
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.lineJoin = "round";
    ctx.beginPath();
    for (let i = 0; i < len; i++) {
      const x = (i / (len - 1)) * w;
      const y = padY + ((mx - data[i]) / range) * (h - 2 * padY);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Glow effect
    ctx.strokeStyle = color.replace("0.85", "0.15");
    ctx.lineWidth = 4;
    ctx.beginPath();
    for (let i = 0; i < len; i++) {
      const x = (i / (len - 1)) * w;
      const y = padY + ((mx - data[i]) / range) * (h - 2 * padY);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // ── Resize handler — re-render ECG on window resize ───
  let _lastSignal = null;
  let resizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (_lastSignal) renderECG(_lastSignal);
    }, 250);
  });

  // Store signal whenever renderResults is called
  const _origRenderResults = renderResults;
  renderResults = function (data, gt) {
    if (data.signal) _lastSignal = data.signal;
    return _origRenderResults(data, gt);
  };

})();
