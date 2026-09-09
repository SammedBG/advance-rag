// ==========================================================================
// ADVANCED RAG PLATFORM — CLIENT APPLICATION CONTROLLER
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    initChips();
    initQueryExecution();
    initIngestion();
    initMCPTools();
    initMetricsScraper();
    initModal();
    checkHealth();
});

// State
let lastRetrievedChunks = [];

// Navigation / Tabs
function initTabs() {
    const navItems = document.querySelectorAll(".nav-item");
    const panes = document.querySelectorAll(".tab-pane");
    const tabTitle = document.getElementById("current-tab-title");
    const tabDesc = document.getElementById("current-tab-desc");

    const tabDescriptions = {
        playground: {
            title: "RAG & Agent Playground",
            desc: "Interactive multi-strategy retrieval, grounded generation, and agent reasoning."
        },
        knowledge: {
            title: "Knowledge & Indexer Hub",
            desc: "Structure-aware document ingestion, chunk hierarchies, and vector management."
        },
        mcp: {
            title: "MCP & Tools Diagnostics",
            desc: "Model Context Protocol (MCP 2.x) tools execution for external cluster inspection."
        },
        evaluation: {
            title: "Evaluation Scorecard",
            desc: "Gold-standard IR metrics (MRR, HitRate, nDCG) and faithfulness benchmarks."
        },
        metrics: {
            title: "Telemetry & Prometheus Metrics",
            desc: "Real-time scrapable metrics, latency distributions, and cache analytics."
        }
    };

    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const tab = item.getAttribute("data-tab");

            navItems.forEach(n => n.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            item.classList.add("active");
            const targetPane = document.getElementById(`pane-${tab}`);
            if (targetPane) targetPane.classList.add("active");

            if (tabDescriptions[tab]) {
                tabTitle.textContent = tabDescriptions[tab].title;
                tabDesc.textContent = tabDescriptions[tab].desc;
            }

            if (tab === "metrics") {
                scrapeMetrics();
            }
        });
    });

    // Pill group for query mode
    const pills = document.querySelectorAll(".mode-pill");
    pills.forEach(p => {
        p.addEventListener("click", () => {
            pills.forEach(x => x.classList.remove("active"));
            p.classList.add("active");
        });
    });
}

// Quick query chips
function initChips() {
    const chips = document.querySelectorAll(".chip");
    const queryInput = document.getElementById("query-input");

    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            const q = chip.getAttribute("data-query");
            queryInput.value = q;
            queryInput.focus();
        });
    });
}

// Query Execution
function initQueryExecution() {
    const btnSubmit = document.getElementById("btn-submit-query");
    const queryInput = document.getElementById("query-input");
    const techFilter = document.getElementById("technology-filter");

    btnSubmit.addEventListener("click", async () => {
        const query = queryInput.value.trim();
        if (!query) return;

        const selectedMode = document.querySelector(".mode-pill.active").getAttribute("data-mode");
        const technology = techFilter.value || null;

        btnSubmit.disabled = true;
        btnSubmit.innerHTML = `<span class="btn-icon">⏳</span><span>Executing...</span>`;

        const startTime = performance.now();

        try {
            if (selectedMode === "agent") {
                await executeAgentQuery(query, technology, startTime);
            } else if (selectedMode === "stream") {
                await executeStreamingQuery(query, technology, startTime);
            } else {
                await executeSearchQuery(query, technology, startTime);
            }
        } catch (err) {
            alert(`Query Execution Error: ${err.message}`);
        } finally {
            btnSubmit.disabled = false;
            btnSubmit.innerHTML = `<span class="btn-icon">⚡</span><span>Execute Query</span>`;
        }
    });
}

async function executeStreamingQuery(query, technology, startTime) {
    const respContainer = document.getElementById("response-container");
    respContainer.classList.remove("hidden");
    const answerEl = document.getElementById("answer-text");
    answerEl.innerHTML = `<span style="color: var(--text-muted); font-style: italic;">Connecting to real-time stream...</span>`;
    document.getElementById("route-text").textContent = "Route: STREAMING SSE";
    document.getElementById("stat-latency").textContent = "📡 Connecting...";
    document.getElementById("stat-tokens").textContent = "🎟️ Stream active";

    const response = await fetch("/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: query,
            filters: technology ? { technology: technology } : null
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let fullAnswer = "";
    answerEl.innerHTML = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
            if (line.startsWith("data: ")) {
                try {
                    const payload = JSON.parse(line.substring(6));
                    if (payload.event === "token") {
                        fullAnswer += payload.token;
                        answerEl.innerHTML = fullAnswer.replace(/\[(\d+)\]/g, (match, p1) => {
                            return `<span class="citation-ref" onclick="highlightChunk(${parseInt(p1) - 1})">[${p1}]</span>`;
                        });
                    } else if (payload.event === "status") {
                        document.getElementById("stat-latency").textContent = `📡 ${payload.message}`;
                    } else if (payload.event === "done") {
                        const durationSec = ((performance.now() - startTime) / 1000).toFixed(2);
                        document.getElementById("stat-latency").textContent = `⏱️ ${durationSec}s`;
                        if (payload.grounding_score !== undefined) {
                            const score = payload.grounding_score;
                            document.getElementById("grounding-score-val").textContent = score.toFixed(2);
                            document.getElementById("grounding-fill").style.width = `${Math.round(score * 100)}%`;
                        }
                    }
                } catch (e) {
                    console.warn("SSE chunk parse error:", e);
                }
            }
        }
    }
}

async function executeAgentQuery(query, technology, startTime) {
    const response = await fetch("/agent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    const durationSec = ((performance.now() - startTime) / 1000).toFixed(2);

    renderResponse(data, durationSec, "agent");
}

async function executeSearchQuery(query, technology, startTime) {
    const response = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            query: query,
            technology: technology,
            limit: 5,
            retrieval_limit: 20
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    const durationSec = ((performance.now() - startTime) / 1000).toFixed(2);

    renderResponse(data, durationSec, "search");
}

function renderResponse(data, durationSec, mode) {
    const respContainer = document.getElementById("response-container");
    respContainer.classList.remove("hidden");

    // Route Badge
    const routeText = document.getElementById("route-text");
    const route = data.route || (mode === "agent" ? "RAG Agent" : "Hybrid Search");
    routeText.textContent = `Route: ${route.toUpperCase()}`;

    // Stats
    document.getElementById("stat-latency").textContent = `⏱️ ${durationSec}s`;
    
    const gen = data.generation || {};
    const tokens = gen.total_tokens || (data.context_stats ? data.context_stats.compressed_tokens : 0);
    document.getElementById("stat-tokens").textContent = `🎟️ ${tokens} tokens`;

    // Answer with formatted citation links
    const answerEl = document.getElementById("answer-text");
    let answerText = data.answer || "No answer generated.";
    
    // Replace citations [1], [2] with interactive citation buttons
    answerText = answerText.replace(/\[(\d+)\]/g, (match, p1) => {
        return `<span class="citation-ref" onclick="highlightChunk(${parseInt(p1) - 1})">[${p1}]</span>`;
    });
    answerEl.innerHTML = answerText;

    // Grounding Meter
    const grounding = data.grounding || {};
    const score = typeof grounding.score === "number" ? grounding.score : (grounding.grounded ? 1.0 : 0.0);
    const scorePct = Math.round(score * 100);
    
    document.getElementById("grounding-score-val").textContent = score.toFixed(2);
    document.getElementById("grounding-fill").style.width = `${scorePct}%`;

    // Citation Status
    const citVal = data.citation_validation || {};
    const citBadge = document.getElementById("citation-badge");
    const citStatus = document.getElementById("cit-status-text");

    if (citVal.valid) {
        citBadge.style.color = "var(--accent-green)";
        citStatus.textContent = `Citations Valid (${data.citations ? data.citations.join(", ") : "All Passed"})`;
    } else {
        citBadge.style.color = "var(--accent-amber)";
        citStatus.textContent = `Citations: ${citVal.missing_citations ? "Missing" : "Review"}`;
    }

    // Agent Reasoning Trace
    const traceTimeline = document.getElementById("trace-timeline");
    traceTimeline.innerHTML = "";
    const trace = data.reasoning_trace || [
        `Executed query '${data.query}'`,
        `Synthesized grounded response.`
    ];
    trace.forEach(step => {
        const li = document.createElement("li");
        li.className = "trace-step";
        li.textContent = step;
        traceTimeline.appendChild(li);
    });

    // Render Chunks Panel
    renderChunks(data.results || data.compressed_contexts || []);
}

function renderChunks(chunks) {
    lastRetrievedChunks = chunks;
    const list = document.getElementById("chunks-list");
    const countBadge = document.getElementById("chunk-count-badge");
    countBadge.textContent = `${chunks.length} Contexts`;

    if (!chunks || chunks.length === 0) {
        list.innerHTML = `
            <div class="empty-state">
                <span class="empty-icon">ℹ️</span>
                <p>No direct knowledge chunks retrieved for this query route.</p>
            </div>
        `;
        return;
    }

    list.innerHTML = "";
    chunks.forEach((chunk, idx) => {
        const item = document.createElement("div");
        item.className = "chunk-item";
        item.id = `chunk-card-${idx}`;
        
        const title = chunk.title || `Chunk [${idx + 1}]`;
        const score = typeof chunk.score === "number" ? chunk.score.toFixed(4) : "RRF Match";
        const content = chunk.content || "";

        item.innerHTML = `
            <div class="chunk-header">
                <span class="chunk-rank">[${idx + 1}] ${title}</span>
                <span class="chunk-score">Score: ${score}</span>
            </div>
            <div class="chunk-snippet">${content}</div>
        `;

        item.addEventListener("click", () => {
            openChunkModal(chunk, idx);
        });

        list.appendChild(item);
    });
}

window.highlightChunk = function(index) {
    document.querySelectorAll(".chunk-item").forEach(c => c.classList.remove("highlighted"));
    const target = document.getElementById(`chunk-card-${index}`);
    if (target) {
        target.classList.add("highlighted");
        target.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
};

// Ingestion Form
function initIngestion() {
    const form = document.getElementById("ingest-form");
    const resultBox = document.getElementById("ingest-result-box");

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const title = document.getElementById("doc-title").value.trim();
        const tech = document.getElementById("doc-tech").value.trim();
        const scope = document.getElementById("doc-scope").value;
        const content = document.getElementById("doc-content").value.trim();

        const btn = document.getElementById("btn-ingest");
        btn.disabled = true;
        btn.innerHTML = `<span>⏳ Ingesting...</span>`;

        try {
            const resp = await fetch("/index", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    title: title,
                    technology: tech || "general",
                    content: content,
                    access_scope: scope
                })
            });

            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
            }

            const res = await resp.json();
            resultBox.className = "ingest-result tag-green";
            resultBox.innerHTML = `<strong>✓ Ingestion Successful!</strong> Document ID: <code>${res.document_id}</code>, Chunks: ${res.chunks_indexed || res.total_chunks || 'Indexed'}`;
            resultBox.classList.remove("hidden");
            form.reset();
        } catch (err) {
            resultBox.className = "ingest-result tag-red";
            resultBox.innerHTML = `<strong>✗ Ingestion Failed:</strong> ${err.message}`;
            resultBox.classList.remove("hidden");
        } finally {
            btn.disabled = false;
            btn.innerHTML = `<span>⚡ Ingest & Upsert Vectors</span>`;
        }
    });
}

// MCP Tools
function initMCPTools() {
    const toolSelect = document.getElementById("mcp-tool-select");
    const podNameInput = document.getElementById("mcp-pod-name");
    const btnInvoke = document.getElementById("btn-invoke-mcp");
    const outputBox = document.getElementById("mcp-output-box");
    const rawJson = document.getElementById("mcp-raw-json");

    btnInvoke.addEventListener("click", async () => {
        const tool = toolSelect.value;
        btnInvoke.disabled = true;
        btnInvoke.textContent = "Executing...";

        try {
            let queryText = "";
            if (tool === "get_pod_status_tool") {
                queryText = `get pod status for ${podNameInput.value}`;
            } else if (tool === "get_cluster_health_tool") {
                queryText = `check cluster health`;
            } else {
                queryText = `search knowledge for ${podNameInput.value}`;
            }

            const resp = await fetch("/agent", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query: queryText })
            });

            const data = await resp.json();
            outputBox.classList.remove("hidden");
            rawJson.textContent = JSON.stringify(data.mcp_result || data, null, 2);
        } catch (err) {
            outputBox.classList.remove("hidden");
            rawJson.textContent = `Error: ${err.message}`;
        } finally {
            btnInvoke.disabled = false;
            btnInvoke.textContent = "Execute MCP Tool";
        }
    });
}

// Metrics Scraper
function initMetricsScraper() {
    const btn = document.getElementById("btn-scrape-metrics");
    btn.addEventListener("click", scrapeMetrics);
}

async function scrapeMetrics() {
    const out = document.getElementById("metrics-raw-output");
    try {
        const res = await fetch("/metrics");
        const text = await res.text();
        out.textContent = text;
    } catch (err) {
        out.textContent = `Error scraping /metrics: ${err.message}`;
    }
}

// Health Checker
async function checkHealth() {
    try {
        const res = await fetch("/health");
        if (res.ok) {
            document.getElementById("status-qdrant").textContent = "Active";
            document.getElementById("status-db").textContent = "Connected";
        }
    } catch (err) {
        console.warn("Health check error:", err);
    }
}

// Modal
function initModal() {
    const modal = document.getElementById("chunk-modal");
    const closeBtn = document.getElementById("modal-close");
    const backdrop = document.getElementById("modal-backdrop");

    closeBtn.addEventListener("click", () => modal.classList.add("hidden"));
    backdrop.addEventListener("click", () => modal.classList.add("hidden"));
}

function openChunkModal(chunk, idx) {
    const modal = document.getElementById("chunk-modal");
    document.getElementById("modal-title").textContent = `Chunk [${idx + 1}] — ${chunk.title || 'Details'}`;
    
    const metaGrid = document.getElementById("modal-meta");
    metaGrid.innerHTML = `
        <div><strong>Chunk Type:</strong> ${chunk.chunk_type || 'child'}</div>
        <div><strong>Score:</strong> ${typeof chunk.score === 'number' ? chunk.score.toFixed(4) : 'N/A'}</div>
        <div><strong>Document ID:</strong> ${chunk.document_id || 'doc-1'}</div>
        <div><strong>Parent ID:</strong> ${chunk.parent_id || 'None'}</div>
    `;

    document.getElementById("modal-content").textContent = chunk.content || "";
    modal.classList.remove("hidden");
}
