const fileInput = document.getElementById("file-input");
const uploadForm = document.getElementById("upload-form");
const resultsPanel = document.getElementById("results-panel");
const previewImg = document.getElementById("preview");
const predictionList = document.getElementById("prediction-list");
const mainBtn = document.querySelector(".btn-main");
const mainBtnText = mainBtn.innerHTML;

// Trigger preview and submission immediately upon file selection
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) {
    previewImg.src = URL.createObjectURL(file);
    resultsPanel.style.display = "block";
    requestAnimationFrame(() => {
      handleSubmit(file);
    });
  }
});

// Handle submission logic
async function handleSubmit(file) {
  if (!file) return;

  const data = new FormData();
  data.append("file", file);

  mainBtn.disabled = true;
  mainBtn.innerHTML = `<svg class="animate-spin" style="animation:spin 1s linear infinite; width:20px; height:20px;" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" stroke-dasharray="30 60"></circle></svg> Analyzing...`;
  predictionList.innerHTML = "";

  try {
    const resp = await fetch("/predict", { method: "POST", body: data });
    const json = await resp.json();

    if (!resp.ok) throw new Error(json.error || "Error");

    let html = "";
    json.results.forEach((r, index) => {
      const isTop = index === 0;
      const colorStyle = isTop
        ? "border-color: var(--accent-blue); color: var(--accent-blue);"
        : "";

      html += `
        <div class="tag-row" style="${
          isTop
            ? "border: 2px solid var(--accent-blue); background: #eff6ff;"
            : ""
        }">
          <span class="tag-label" style="${colorStyle}">${r.label}</span>
          <span class="tag-score">${(r.confidence * 100).toFixed(1)}%</span>
        </div>
      `;
    });
    predictionList.innerHTML = html;
  } catch (err) {
    predictionList.innerHTML = `
      <div style="color:#ef4444; background:#fef2f2; padding:16px; border-radius:12px; text-align:center; border:1px solid #fee2e2;">
        <strong>Connection Failed</strong><br>
        <span style="font-size:0.9em; opacity:0.8;">${err.message}</span>
      </div>
    `;
  } finally {
    mainBtn.disabled = false;
    mainBtn.innerHTML = mainBtnText;
  }
}

uploadForm.addEventListener("submit", (e) => e.preventDefault());
