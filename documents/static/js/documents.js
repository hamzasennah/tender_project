(function () {
    function formatBytes(bytes) {
        if (!bytes) return "0 B";
        const units = ["B", "KB", "MB", "GB"];
        const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
        return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
    }

    function initUpload() {
        const form = document.querySelector("[data-upload-form]");
        if (!form) return;

        const input = form.querySelector("[data-file-input]");
        const dropzone = form.querySelector("[data-dropzone]");
        const selection = form.querySelector("[data-upload-selection]");
        const submit = form.querySelector("[data-upload-submit]");

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
            submit.textContent = "Importing document...";
            try {
                const data = await TenderApp.request(form.action, {
                    method: "POST",
                    body: new FormData(form),
                });
                TenderApp.showToast("Document imported.");
                window.location.href = `/documents/${data.id}/`;
            } catch (error) {
                TenderApp.showToast(error.message || "Upload failed.", "error");
            } finally {
                submit.disabled = false;
                submit.textContent = "Import document";
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
                    TenderApp.showToast(error.message || "Delete failed.", "error");
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
