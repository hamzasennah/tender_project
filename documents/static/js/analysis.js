(function () {
    function escapeHtml(value) {
        return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function friendlyError(error) {
        const message = String(error?.message || "").toLowerCase();
        if (message.includes("rate") || message.includes("timeout") || message.includes("429") || message.includes("503")) {
            return "Temporary processing issue. We could not complete the answer right now. Please try again in a moment.";
        }
        if (message.includes("not found") || message.includes("missing") || message.includes("chunks") || message.includes("embedding")) {
            return "This document is still being prepared for questions. Please try again shortly.";
        }
        return "We could not complete the answer right now. Please try again.";
    }

    function renderLoading(panel) {
        panel.innerHTML = `
            <div class="answer-placeholder">
                <h2>Reading the document</h2>
                <p>Preparing an answer and checking the most relevant passages.</p>
            </div>
        `;
    }

    function excerpt(text) {
        const normalized = String(text || "").replace(/\s+/g, " ").trim();
        if (normalized.length <= 520) return normalized;
        return `${normalized.slice(0, 520).trim()}...`;
    }

    function evidenceFromSearch(searchPayload) {
        const results = Array.isArray(searchPayload?.results) ? searchPayload.results : [];
        return results.slice(0, 3).map((result, index) => ({
            label: `Passage ${index + 1}`,
            text: excerpt(result.text || ""),
        })).filter((item) => item.text);
    }

    function renderEvidence(items) {
        if (!items.length) {
            return `
                <section>
                    <p class="kicker">Evidence from the document</p>
                    <p class="muted-copy">No passage preview was returned for this answer.</p>
                </section>
            `;
        }
        return `
            <section>
                <p class="kicker">Evidence from the document</p>
                <div class="evidence-list">
                    ${items.map((item) => `
                        <article class="evidence-item">
                            <strong>${escapeHtml(item.label)}</strong>
                            <blockquote>${escapeHtml(item.text)}</blockquote>
                        </article>
                    `).join("")}
                </div>
            </section>
        `;
    }

    function renderAnswer(panel, answerPayload, searchPayload, question) {
        const evidence = evidenceFromSearch(searchPayload);
        panel.innerHTML = `
            <div class="answer-shell">
                <div class="question-box">${escapeHtml(question)}</div>
                <section>
                    <h2>Answer</h2>
                    <div class="answer-copy">${escapeHtml(answerPayload.answer || "No answer returned.")}</div>
                </section>
                ${renderEvidence(evidence)}
            </div>
        `;
    }

    function initSuggestedQuestions() {
        document.querySelectorAll("[data-suggested-question]").forEach((button) => {
            button.addEventListener("click", () => {
                const form = document.querySelector("[data-analysis-form]");
                const textarea = form?.elements?.question;
                if (!textarea || textarea.disabled) return;
                textarea.value = button.dataset.suggestedQuestion;
                textarea.focus();
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
            const submit = form.querySelector("[data-analysis-submit]");
            if (!question) {
                TenderApp.showToast("Enter a question before asking the document.", "error");
                return;
            }
            submit.disabled = true;
            renderLoading(panel);
            try {
                const searchPayload = await TenderApp.request(workspace.dataset.searchUrl, {
                    method: "POST",
                    json: { query: question, top_k: 3 },
                    timeout: 120000,
                });
                const answerPayload = await TenderApp.request(workspace.dataset.askUrl, {
                    method: "POST",
                    json: { question, top_k: 5 },
                    timeout: 180000,
                });
                renderAnswer(panel, answerPayload, searchPayload, question);
            } catch (error) {
                panel.innerHTML = `
                    <div class="notice-panel">
                        <strong>${escapeHtml(friendlyError(error))}</strong>
                        <p>Please retry in a moment. If the document was just uploaded, preparation may still be finishing.</p>
                    </div>
                `;
                TenderApp.showToast(friendlyError(error), "error");
            } finally {
                submit.disabled = false;
            }
        });
    }

    function methodEndpoint(workspace, method) {
        if (method === "pe") return workspace.dataset.peUrl;
        if (method === "rag") return workspace.dataset.ragUrl;
        return workspace.dataset.raptorUrl;
    }

    function initCompareMode() {
        const workspace = document.querySelector("[data-document-workspace]");
        const form = document.querySelector("[data-compare-form]");
        const results = document.querySelector("[data-compare-results]");
        if (!workspace || !form || !results) return;

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            const question = form.elements.question.value.trim();
            if (!question) return;
            results.innerHTML = `<p class="muted-copy">Comparing methods...</p>`;
            const methods = [
                ["pe", "Prompt Engineering"],
                ["rag", "RAG"],
                ["raptor", "RAPTOR"],
            ];
            const cards = [];
            for (const [key, label] of methods) {
                try {
                    const payload = { question };
                    if (key !== "pe") payload.top_k = 5;
                    const data = await TenderApp.request(methodEndpoint(workspace, key), {
                        method: "POST",
                        json: payload,
                        timeout: 180000,
                    });
                    cards.push(`<article class="compare-card"><h3>${label}</h3><p>${escapeHtml(data.answer || "No answer returned.")}</p></article>`);
                } catch (error) {
                    cards.push(`<article class="compare-card"><h3>${label}</h3><p>${escapeHtml(friendlyError(error))}</p></article>`);
                }
            }
            results.innerHTML = cards.join("");
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        initSuggestedQuestions();
        initAnalysis();
        initCompareMode();
    });
})();
