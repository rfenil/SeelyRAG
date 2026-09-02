const state = {
  lastQueryId: null,
  docs: [],
};

const $ = (selector) => document.querySelector(selector);

function cleanText(value) {
  return String(value)
    .replaceAll("Â·", "|")
    .replaceAll("·", "|")
    .replaceAll("–", "-")
    .replaceAll("—", "-")
    .replaceAll("‑", "-")
    .replaceAll(" ", " ")
    .replaceAll("“", '"')
    .replaceAll("”", '"')
    .replaceAll("‘", "'")
    .replaceAll("’", "'");
}

function text(value) {
  return value === null || value === undefined || value === "" ? "n/a" : cleanText(value);
}

function escapeHtml(value) {
  return cleanText(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setBusy(form, busy) {
  form.querySelectorAll("button, input, select, textarea").forEach((node) => {
    node.disabled = busy;
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.detail || `${response.status} ${response.statusText}`);
  }
  return body;
}

function showNotice(target, message, cls = "") {
  target.innerHTML = `<div class="notice ${cls}">${escapeHtml(message)}</div>`;
}

function link(url, label) {
  if (!url) return "";
  return `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">${escapeHtml(label)}</a>`;
}

async function refreshHealth() {
  const target = $("#health");
  try {
    const body = await api("/health");
    target.className = body.status === "ok" ? "ok" : "error";
    // Which model vendor answered and which reranker ran are ours to know,
    // not the installer's. Whether the service is up is theirs.
    target.textContent = body.status === "ok" ? "Ready" : `Service ${body.status}`;
  } catch (error) {
    target.className = "error";
    target.textContent = error.message;
  }
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `${name}-panel`);
  });
}

async function ask(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const target = $("#answer");
  const query = $("#ask-query").value.trim();
  if (!query) {
    showNotice(target, "Enter a question first.", "error");
    return;
  }

  const payload = {
    query,
    top_k: Number($("#ask-top-k").value || 8),
    stream: false,
  };
  const family = $("#ask-family").value;
  if (family) payload.product_hint = family;

  setBusy(form, true);
  showNotice(target, "Asking...");
  try {
    const body = await api("/ask", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.lastQueryId = body.query_id;
    renderAnswer(body);
  } catch (error) {
    showNotice(target, error.message, "error");
  } finally {
    setBusy(form, false);
    refreshHealth();
  }
}

function formatAnswer(answer) {
  const lines = cleanText(answer).split(/\r?\n/);
  return lines
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const safe = escapeHtml(line);
      if (/^\d+\.\s+/.test(line)) {
        return `<p class="answer-step">${safe}</p>`;
      }
      if (/^-\s+/.test(line)) {
        return `<p class="answer-bullet">${safe.slice(2)}</p>`;
      }
      if (line.endsWith(":")) {
        return `<h3 class="answer-section">${safe}</h3>`;
      }
      return `<p>${safe}</p>`;
    })
    .join("");
}

function lowConfidenceNotice(confidence) {
  // The rest of the metadata bar -- query id, latency, family, the word
  // "confidence" on every answer -- is diagnostic noise to an installer. A LOW
  // confidence is not: assemble() forces it when an answer carries no citation
  // at all, so it is the one case the reader has to see.
  if (confidence !== "low") return "";
  return '<p class="answer-warning">Low confidence - the manuals only touch on '
    + 'this. Check the sources before acting on it.</p>';
}

function renderAnswer(body) {
  const citations = body.citations || [];
  $("#answer").innerHTML = `
    <article class="answer-card">
      ${lowConfidenceNotice(body.confidence)}
      <div class="answer-text">${formatAnswer(body.answer || "")}</div>
      <div class="feedback">
        <button type="button" data-rating="up">Useful</button>
        <button type="button" data-rating="down">Wrong</button>
        <input id="feedback-comment" placeholder="Optional feedback">
      </div>
    </article>
    ${citations.map(renderCitation).join("")}
  `;
  document.querySelectorAll("[data-rating]").forEach((button) => {
    button.addEventListener("click", sendFeedback);
  });
}

function renderCitation(citation) {
  const title = `[${citation.n}] ${citation.title || "Untitled"}`;
  const page = citation.page_label ? `p.${citation.page_label}` : "no printed page";
  const image = citation.page_url
    ? `<img class="page-image" src="${escapeHtml(citation.page_url)}" alt="${escapeHtml(title)}">`
    : '<div class="page-image hidden"></div>';
  return `
    <article class="citation">
      <div class="citation-body">
        <h3>${escapeHtml(title)}</h3>
        <p class="meta">${escapeHtml(page)}</p>
        <p>${escapeHtml(citation.snippet || "")}</p>
        <div class="links">
          ${link(citation.article_url, "Article")}
          ${link(citation.doc_url, "Source")}
          ${link(citation.page_url, "Page image")}
        </div>
      </div>
      ${image}
    </article>
  `;
}

async function sendFeedback(event) {
  if (!state.lastQueryId) return;
  const button = event.currentTarget;
  const payload = {
    query_id: state.lastQueryId,
    rating: button.dataset.rating,
    comment: $("#feedback-comment").value.trim() || null,
  };
  try {
    await api("/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    document.querySelectorAll("[data-rating]").forEach((node) => {
      node.classList.toggle("selected", node === button);
    });
  } catch (error) {
    showNotice($("#answer"), error.message, "error");
  }
}

async function search(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const target = $("#search-results");
  const query = $("#search-query").value.trim();
  if (!query) {
    showNotice(target, "Enter a search first.", "error");
    return;
  }

  const payload = {
    query,
    top_k: Number($("#search-top-k").value || 8),
  };
  const family = $("#search-family").value;
  const docType = $("#search-doc-type").value;
  const table = $("#search-table").value;
  if (family) payload.product_family = family;
  if (docType) payload.doc_type = docType;
  if (table) payload.is_table = table === "true";

  setBusy(form, true);
  showNotice(target, "Searching...");
  try {
    const body = await api("/search", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    renderSearch(body);
  } catch (error) {
    showNotice(target, error.message, "error");
  } finally {
    setBusy(form, false);
  }
}

function renderSearch(body) {
  const hits = body.hits || [];
  if (!hits.length) {
    showNotice($("#search-results"), "No hits.");
    return;
  }
  $("#search-results").innerHTML = hits
    .map(
      (hit, index) => `
    <article class="hit">
      <div class="hit-head">
        <h3>${index + 1}. ${escapeHtml(hit.title || "Untitled")}</h3>
        <span class="meta">${text(hit.product_family)} | ${text(hit.kind)} | p.${text(hit.page_label)}</span>
      </div>
      <p>${escapeHtml((hit.text || "").slice(0, 700))}</p>
      <div class="links">
        ${link(hit.article_url, "Article")}
        ${link(hit.page_url, "Page image")}
      </div>
    </article>
  `
    )
    .join("");
}

async function loadDocs() {
  const target = $("#docs-results");
  showNotice(target, "Loading inventory...");
  try {
    const body = await api("/docs-inventory");
    state.docs = body.documents || [];
    renderDocs();
  } catch (error) {
    showNotice(target, error.message, "error");
  }
}

function renderDocs() {
  const filter = $("#doc-filter").value.trim().toLowerCase();
  const docs = state.docs.filter((doc) => {
    const textBlob =
      `${doc.title} ${doc.product_family} ${doc.doc_type} ${doc.category}`.toLowerCase();
    return textBlob.includes(filter);
  });
  if (!docs.length) {
    showNotice($("#docs-results"), "No documents.");
    return;
  }
  $("#docs-results").innerHTML = docs
    .map(
      (doc) => `
    <article class="doc">
      <div class="doc-head">
        <h3>${escapeHtml(doc.title || "Untitled")}</h3>
        <span class="meta">${text(doc.product_family)} | ${text(doc.doc_type)}</span>
      </div>
      <p class="meta">${text(doc.chunks)} chunks | ${text(doc.pages)} pages</p>
      <div class="links">${link(doc.article_url, "Article")}</div>
    </article>
  `
    )
    .join("");
}

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });
  $("#ask-form").addEventListener("submit", ask);
  $("#search-form").addEventListener("submit", search);
  $("#refresh-docs").addEventListener("click", loadDocs);
  $("#doc-filter").addEventListener("input", renderDocs);
  refreshHealth();
});
