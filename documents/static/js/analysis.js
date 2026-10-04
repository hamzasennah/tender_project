(function () {
    function methodLabel(method) {
        if (method === "pe") return "Prompt Engineering";
        if (method === "rag") return "RAG";
        return "RAPTOR";
    }

    function endpointFor(workspace, method) {
        if (method === "pe") return workspace.dataset.peUrl;
        if (method === "rag") return workspace.dataset.ragUrl;
        return workspace.dataset.raptorUrl;
    }

    function renderLoading(panel) {
        panel.innerHTML = `
            <div class="empty-state compact">
                <h3>Analyzing document...</h3>
                <p>The selected method is preparing context and asking the configured provider.</p>
            </div>
        `;
    }

    function renderSources(sources) {
        if (!Array.isArray(sources) || !sources.length) {
            return `<div class="question-box"><strong>Sources / Evidence</strong><p class="muted">No retrieval sources were returned for this method.</p></div>`;
        }
        return `
            <div>
                <p class="eyebrow">Sources / Evidence</p>
                <div class="source-list">
                    ${sources.map((source, index) => `
                        <article class="source-item">
                            <strong>Source ${source.source_number || index + 1}</strong>
                            <p>${escapeHtml(source.text || source.preview || source.chunk_text || "Source metadata returned without preview text.")}</p>
                        </article>
                    `).join("")}
                </div>
            </div>
        `;
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function renderAnswer(panel, payload, question, method) {
        const metadata = payload.rag_metadata || payload.raptor_metadata || payload.prompt_engineering_metadata || {};
        panel.innerHTML = `
            <div class="answer-shell">
                <div class="question-box">
                    <p class="eyebrow">Question</p>
                    <strong>${escapeHtml(question)}</strong>
                </div>
                <div>
                    <p class="eyebrow">Answer - ${methodLabel(method)}</p>
                    <div class="answer-copy">${escapeHtml(payload.answer || "No answer returned.")}</div>
                </div>
                ${renderSources(payload.sources)}
                <details class="technical-details">
                    <summary>Technical details</summary>
                    <pre>${escapeHtml(JSON.stringify(metadata, null, 2))}</pre>
                </details>
            </div>
        `;
    }

    function initPipelineActions() {
        document.querySelectorAll("[data-pipeline-action]").forEach((button) => {
            button.addEventListener("click", async () => {
                button.disabled = true;
                const original = button.textContent;
                button.textContent = "Processing...";
                try {
                    await TenderApp.request(button.dataset.actionUrl, {
                        method: "POST",
                        timeout: 180000,
                    });
                    TenderApp.showToast("Pipeline step completed.");
                    window.location.reload();
                } catch (error) {
                    TenderApp.showToast(error.message || "Pipeline step failed.", "error");
                } finally {
                    button.disabled = false;
                    button.textContent = original;
                }
            });
        });
    }

    function initAnalysis() {
        const workspace = document.querySelector("[data-document-workspace]");
        const form = document.querySelector("[data-analysis-form]");
        const panel = document.querySelector("[data-answer-panel]");
        if (!workspace || !form || !panel) return;

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const question = form.elements.question.value.trim();
            const method = form.elements.method.value;
            const topK = Number(form.elements.top_k.value || 5);
            const submit = form.querySelector("[data-analysis-submit]");
            if (!question) {
                TenderApp.showToast("Enter a question before analyzing.", "error");
                return;
            }
            submit.disabled = true;
            renderLoading(panel);
            try {
                const payload = { question };
                if (method !== "pe") payload.top_k = topK;
                const data = await TenderApp.request(endpointFor(workspace, method), {
                    method: "POST",
                    json: payload,
                    timeout: 180000,
                });
                renderAnswer(panel, data, question, method);
                TenderApp.showToast("Analysis completed.");
            } catch (error) {
                panel.innerHTML = `
                    <div class="empty-state compact">
                        <h3>Analysis failed</h3>
                        <p>${escapeHtml(error.message || "The request could not be completed.")}</p>
                    </div>
                `;
                TenderApp.showToast(error.message || "Analysis failed.", "error");
            } finally {
                submit.disabled = false;
            }
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        initPipelineActions();
        initAnalysis();
    });
})();
