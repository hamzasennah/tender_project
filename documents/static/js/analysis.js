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
        const status = error?.response?.status;
        if (message.includes("rate") || message.includes("timeout") || status >= 500) {
            return "We couldn't answer from this document right now. Try again in a moment.";
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

    function evidenceFromAnswer(answerPayload) {
        const evidence = Array.isArray(answerPayload?.supporting_evidence)
            ? answerPayload.supporting_evidence
            : [];
        return evidence.slice(0, 4).map((item) => ({
            label: item.page ? `Page ${item.page}` : (item.section || item.source || "Document excerpt"),
            section: item.section || "Document excerpt",
            text: excerpt(item.text || ""),
            page: item.page,
            supportedClaims: Array.isArray(item.supported_claims) ? item.supported_claims : [],
        })).filter((item) => item.text);
    }

    function evidenceLink(documentUrl, item) {
        if (!item.page || !documentUrl || documentUrl === "#") return documentUrl || "#";
        return `${documentUrl}#page=${encodeURIComponent(item.page)}`;
    }

    function renderEvidence(items, documentUrl) {
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
                            <div class="evidence-meta">
                                <strong>${escapeHtml(item.label)}</strong>
                                <span>${escapeHtml(item.section)}</span>
                            </div>
                            ${item.supportedClaims.length ? `<p class="evidence-support">Supports: ${escapeHtml(item.supportedClaims.join(", "))}</p>` : ""}
                            <blockquote>${escapeHtml(item.text)}</blockquote>
                            <a class="quiet-link evidence-link" href="${escapeHtml(evidenceLink(documentUrl, item))}">Open in document</a>
                        </article>
                    `).join("")}
                </div>
            </section>
        `;
    }

    function renderAnswer(panel, answerPayload, question, documentUrl) {
        const evidence = evidenceFromAnswer(answerPayload);
        panel.innerHTML = `
            <div class="answer-shell">
                <div class="question-box">${escapeHtml(question)}</div>
                <section>
                    <h2>Answer</h2>
                    <div class="answer-copy">${escapeHtml(answerPayload.answer || "No answer returned.")}</div>
                </section>
                ${renderEvidence(evidence, documentUrl)}
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
                const answerPayload = await TenderApp.request(workspace.dataset.askUrl, {
                    method: "POST",
                    json: { question, top_k: 5 },
                    timeout: 180000,
                });
                renderAnswer(panel, answerPayload, question, workspace.dataset.documentUrl || "#");
            } catch (error) {
                panel.innerHTML = `
                    <div class="notice-panel">
                        <strong>${escapeHtml(friendlyError(error))}</strong>
                        <p>Please retry in a moment. If the document was just uploaded, preparation may still be finishing.</p>
                    </div>
                `;
            } finally {
                submit.disabled = false;
            }
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        initSuggestedQuestions();
        initAnalysis();
    });
})();
