(function () {
    function formatBytes(bytes) {
        if (!bytes) return "0 B";
        const units = ["B", "KB", "MB", "GB"];
        const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
    }

    function friendlyProcessingError(error) {
        const message = String(error?.message || "").toLowerCase();
        if (message.includes("rate") || message.includes("timeout") || message.includes("provider") || message.includes("429") || message.includes("503")) {
            return "Temporary processing issue. We could not finish preparing this document right now. Please try again in a moment.";
        }
        return "We could not finish preparing this document. Please try again in a moment.";
    }

    async function prepareDocument(documentId, state) {
        const steps = [
            { label: "Reading the PDF", url: `/api/documents/${documentId}/extraction/` },
            { label: "Organizing document passages", url: `/api/documents/${documentId}/chunks/` },
            { label: "Preparing document search", url: `/api/documents/${documentId}/embeddings/` },
            { label: "Finalizing research view", url: `/api/documents/${documentId}/raptor/`, optional: true },
        ];

        for (let index = 0; index < steps.length; index += 1) {
            const step = steps[index];
            state.title.textContent = "Preparing your document";
            state.message.textContent = step.label;
            state.progress.value = index;
            try {
                await TenderApp.request(step.url, { method: "POST", timeout: 180000 });
            } catch (error) {
                if (!step.optional) throw error;
            }
            state.progress.value = index + 1;
        }
    }

    function initUpload() {
        const form = document.querySelector("[data-upload-form]");
        if (!form) return;

        const input = form.querySelector("[data-file-input]");
        const dropzone = form.querySelector("[data-dropzone]");
        const selection = form.querySelector("[data-upload-selection]");
        const submit = form.querySelector("[data-upload-submit]");
        const preparation = form.querySelector("[data-preparation-state]");
        const prepTitle = form.querySelector("[data-preparation-title]");
        const prepMessage = form.querySelector("[data-preparation-message]");
        const prepProgress = form.querySelector("[data-preparation-progress]");

        const syncSelection = () => {
            const file = input.files?.[0];
            if (!file) {
                selection.hidden = true;
                selection.textContent = "";
                return;
            }
            selection.hidden = false;
            selection.textContent = `${file.name} - ${formatBytes(file.size)}`;
        };

        input.addEventListener("change", syncSelection);
        ["dragenter", "dragover"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.add("is-dragover");
            });
        });
        ["dragleave", "drop"].forEach((eventName) => {
            dropzone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropzone.classList.remove("is-dragover");
            });
        });
        dropzone.addEventListener("drop", (event) => {
            if (event.dataTransfer.files.length) {
                input.files = event.dataTransfer.files;
                syncSelection();
            }
        });

        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            if (!input.files.length) {
                TenderApp.showToast("Choose a PDF before importing.", "error");
                return;
            }
            submit.disabled = true;
            submit.textContent = "Uploading...";
            if (preparation) preparation.hidden = false;
            try {
                const data = await TenderApp.request(form.action, {
                    method: "POST",
                    body: new FormData(form),
                });
                if (form.dataset.prepareAfterUpload === "true") {
                    await prepareDocument(data.id, {
                        title: prepTitle,
                        message: prepMessage,
                        progress: prepProgress,
                    });
                }
                TenderApp.showToast("Document ready.");
                window.location.href = `/documents/${data.id}/`;
            } catch (error) {
                if (prepTitle && prepMessage) {
                    prepTitle.textContent = "Temporary processing issue";
                    prepMessage.textContent = friendlyProcessingError(error);
                }
                TenderApp.showToast(friendlyProcessingError(error), "error");
                submit.disabled = false;
                submit.textContent = "Upload and prepare";
            }
        });
    }

    function initFilter() {
        const input = document.querySelector("[data-document-filter]");
        if (!input) return;
        const rows = Array.from(document.querySelectorAll("[data-document-row]"));
        input.addEventListener("input", () => {
            const query = input.value.trim().toLowerCase();
            rows.forEach((row) => {
                row.hidden = query && !row.dataset.searchText.includes(query);
            });
        });
    }

    function initDelete() {
        document.querySelectorAll("[data-delete-document]").forEach((button) => {
            button.addEventListener("click", async () => {
                const ok = await TenderApp.confirmAction(
                    "Delete this document and its associated data? This action cannot be undone.",
                    "Delete document",
                );
                if (!ok) return;
                button.disabled = true;
                try {
                    await TenderApp.request(button.dataset.deleteUrl, { method: "DELETE" });
                    TenderApp.showToast("Document deleted.");
                    if (button.dataset.deleteRedirect) {
                        window.location.href = button.dataset.deleteRedirect;
                    } else {
                        button.closest("[data-document-row]")?.remove();
                    }
                } catch (error) {
                    TenderApp.showToast("We could not delete this document.", "error");
                } finally {
                    button.disabled = false;
                }
            });
        });
    }

    document.addEventListener("DOMContentLoaded", () => {
        initUpload();
        initFilter();
        initDelete();
    });
})();
