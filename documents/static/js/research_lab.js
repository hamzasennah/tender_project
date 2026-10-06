(function () {
    const METHODS = {
        pe: {
            label: "Prompt Engineering",
            endpoint: "peUrl",
            loading: "Running Prompt Engineering...",
            metadataKey: "prompt_engineering_metadata",
            evidenceLabel: "Evidence / Sources",
        },
        rag: {
            label: "RAG",
            endpoint: "ragUrl",
            loading: "Running RAG analysis...",
            metadataKey: "rag_metadata",
            evidenceLabel: "Evidence",
        },
        raptor: {
            label: "RAPTOR",
            endpoint: "raptorUrl",
            loading: "Running RAPTOR...",
            metadataKey: "raptor_metadata",
            evidenceLabel: "Sources",
        },
    };

    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function excerpt(text) {
        const normalized = String(text || "").replace(/\s+/g, " ").trim();
        if (normalized.length <= 420) return normalized;
        return `${normalized.slice(0, 420).trim()}...`;
    }

    function friendlyResearchError(error) {
        const message = String(error?.message || "").toLowerCase();
        const status = error?.response?.status;
        if (message.includes("rate") || message.includes("timeout") || status >= 500) {
            return "Provider temporarily unavailable.";
        }
        if (message.includes("not found") || message.includes("missing") || message.includes("not ready")) {
            return "Required document artifacts are not ready for this method.";
        }
        return "This method could not complete the analysis.";
    }

    function selectedDocument(form) {
        return form.querySelector('input[name="document"]:checked');
    }

    function selectedMethod(form) {
        return form.querySelector('input[name="method"]:checked')?.value || "rag";
    }

    function endpointFor(documentInput, methodKey) {
        return documentInput?.dataset?.[METHODS[methodKey].endpoint];
    }

    function formatDuration(ms) {
        if (!Number.isFinite(ms)) return "Not available";
        if (ms < 1000) return `${Math.round(ms)} ms`;
        return `${(ms / 1000).toFixed(1)} s`;
    }

    function metadataFor(methodKey, payload) {
        return payload?.[METHODS[methodKey].metadataKey] || {};
    }

    function metricRows(methodKey, payload, clientLatencyMs) {
        const metadata = metadataFor(methodKey, payload);
        const context = metadata.context || metadata;
        const rows = [
            ["Latency", formatDuration(clientLatencyMs)],
        ];
        if (Number.isFinite(metadata.response_time_ms)) {
            rows.push(["Server time", formatDuration(metadata.response_time_ms)]);
        }
        if (Number.isFinite(context.context_char_count)) {
            rows.push(["Context chars", String(context.context_char_count)]);
        }
        const tokenCount = context.context_token_count || context.estimated_context_token_count;
        if (Number.isFinite(tokenCount)) {
            rows.push(["Context tokens", String(tokenCount)]);
        }
        const includedCount = context.included_count ?? metadata.retrieval?.result_count;
        if (Number.isFinite(includedCount)) {
            rows.push(["Passages", String(includedCount)]);
        }
        return rows;
    }

    function evidenceItems(methodKey, payload) {
        if (methodKey === "rag") {
            const evidence = Array.isArray(payload?.supporting_evidence) ? payload.supporting_evidence : [];
            return evidence.slice(0, 4).map((item) => ({
                title: item.page ? `Page ${item.page}` : item.section || item.source || "Document excerpt",
                detail: item.section || item.source || "Retrieved passage",
                text: excerpt(item.text),
            })).filter((item) => item.text);
        }
        if (methodKey === "raptor") {
            const sources = Array.isArray(payload?.sources) ? payload.sources : [];
            return sources.slice(0, 5).map((item) => ({
                title: item.source || "RAPTOR source",
                detail: [
                    item.node_type,
                    Number.isFinite(item.level) ? `level ${item.level}` : "",
                    Number.isFinite(item.chunk_index) ? `chunk ${item.chunk_index}` : "",
                ].filter(Boolean).join(" / "),
                text: "",
            }));
        }
        return [];
    }

    function renderEvidence(methodKey, payload) {
        const items = evidenceItems(methodKey, payload);
        if (!items.length) {
            return `<p class="muted-copy">No supporting passage exposed by this method.</p>`;
        }
        return `
            <div class="research-evidence-list">
                ${items.map((item) => `
                    <article class="research-evidence-item">
                        <strong>${escapeHtml(item.title)}</strong>
                        ${item.detail ? `<span>${escapeHtml(item.detail)}</span>` : ""}
                        ${item.text ? `<blockquote>${escapeHtml(item.text)}</blockquote>` : ""}
                    </article>
                `).join("")}
            </div>
        `;
    }

    function resultCard(methodKey, state) {
        const method = METHODS[methodKey];
        if (state.status === "loading") {
            return `
                <article class="research-result-card is-loading" data-method-result="${methodKey}">
                    <h3>${escapeHtml(method.label)}</h3>
                    <p class="muted-copy">${escapeHtml(method.loading)}</p>
                </article>
            `;
        }
        if (state.status === "error") {
            return `
                <article class="research-result-card is-error" data-method-result="${methodKey}">
                    <h3>${escapeHtml(method.label)}</h3>
                    <section>
                        <h4>Error</h4>
                        <p>${escapeHtml(friendlyResearchError(state.error))}</p>
                        <button class="text-button" type="button" data-retry-method="${methodKey}">Retry</button>
                    </section>
                </article>
            `;
        }
        const metrics = metricRows(methodKey, state.payload, state.clientLatencyMs);
        return `
            <article class="research-result-card" data-method-result="${methodKey}">
                <h3>${escapeHtml(method.label)}</h3>
                <section>
                    <h4>Answer</h4>
                    <div class="research-answer">${escapeHtml(state.payload?.answer || "No answer returned.")}</div>
                </section>
                <section>
                    <h4>${escapeHtml(method.evidenceLabel)}</h4>
                    ${renderEvidence(methodKey, state.payload)}
                </section>
                <section>
                    <h4>Execution</h4>
                    <dl class="research-metrics">
                        ${metrics.map(([label, value]) => `<div><dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}
                    </dl>
                </section>
            </article>
        `;
    }

    function showResults(results, states) {
        const placeholder = document.querySelector("[data-research-placeholder]");
        placeholder.hidden = true;
        results.hidden = false;
        results.innerHTML = Object.entries(states).map(([methodKey, state]) => resultCard(methodKey, state)).join("");
    }

    async function runMethod(form, methodKey) {
        const documentInput = selectedDocument(form);
        const question = form.elements.question.value.trim();
        const endpoint = endpointFor(documentInput, methodKey);
        if (!documentInput || !endpoint) {
            throw new Error("Select a ready document before running an analysis.");
        }
        const startedAt = performance.now();
        const payload = { question };
        if (methodKey !== "pe") payload.top_k = 5;
        const data = await TenderApp.request(endpoint, {
            method: "POST",
            json: payload,
            timeout: 180000,
        });
        return {
            status: "success",
            payload: data,
            clientLatencyMs: performance.now() - startedAt,
        };
    }

    function initResearchLab() {
        const form = document.querySelector("[data-research-form]");
        const results = document.querySelector("[data-research-results]");
        if (!form || !results) return;

        const runSingle = async (methodKey) => {
            const question = form.elements.question.value.trim();
            if (!question) {
                TenderApp.showToast("Enter a question before running analysis.", "error");
                return;
            }
            showResults(results, { [methodKey]: { status: "loading" } });
            try {
                const state = await runMethod(form, methodKey);
                showResults(results, { [methodKey]: state });
            } catch (error) {
                showResults(results, { [methodKey]: { status: "error", error } });
            }
        };

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            await runSingle(selectedMethod(form));
        });

        form.querySelector("[data-run-all]")?.addEventListener("click", async () => {
            const question = form.elements.question.value.trim();
            if (!question) {
                TenderApp.showToast("Enter a question before comparing methods.", "error");
                return;
            }
            const methods = ["pe", "rag", "raptor"];
            showResults(results, {
                pe: { status: "loading" },
                rag: { status: "loading" },
                raptor: { status: "loading" },
            });
            const settled = await Promise.allSettled(methods.map((methodKey) => runMethod(form, methodKey)));
            const states = {};
            settled.forEach((result, index) => {
                states[methods[index]] = result.status === "fulfilled"
                    ? result.value
                    : { status: "error", error: result.reason };
            });
            showResults(results, states);
        });

        results.addEventListener("click", async (event) => {
            const methodKey = event.target?.dataset?.retryMethod;
            if (!methodKey) return;
            await runSingle(methodKey);
        });
    }

    document.addEventListener("DOMContentLoaded", initResearchLab);
})();
