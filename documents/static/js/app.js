(function () {
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(";").shift();
        }
        return "";
    }

    function showToast(message, type = "success") {
        const region = document.querySelector("[data-toast-region]");
        if (!region) return;

        const toast = document.createElement("div");
        toast.className = `toast ${type}`;
        toast.textContent = message;
        region.appendChild(toast);
        window.setTimeout(() => toast.remove(), 4200);
    }

    async function request(url, options = {}) {
        const controller = new AbortController();
        const timeout = window.setTimeout(() => controller.abort(), options.timeout || 90000);
        const headers = new Headers(options.headers || {});
        const method = (options.method || "GET").toUpperCase();

        if (!headers.has("X-CSRFToken") && method !== "GET") {
            headers.set("X-CSRFToken", getCookie("csrftoken"));
        }

        if (options.json && !headers.has("Content-Type")) {
            headers.set("Content-Type", "application/json");
        }

        try {
            const response = await fetch(url, {
                ...options,
                headers,
                signal: controller.signal,
                body: options.json ? JSON.stringify(options.json) : options.body,
            });
            const text = await response.text();
            let data = null;
            if (text) {
                try {
                    data = JSON.parse(text);
                } catch (error) {
                    throw new Error("The server returned an unreadable response.");
                }
            }
            if (!response.ok) {
                const message = data?.error_message || data?.detail || data?.file?.[0] || `Request failed with HTTP ${response.status}.`;
                const err = new Error(message);
                err.response = response;
                err.data = data;
                throw err;
            }
            return data;
        } finally {
            window.clearTimeout(timeout);
        }
    }

    function confirmAction(message, acceptLabel = "Delete") {
        const modal = document.querySelector("[data-confirm-modal]");
        if (!modal) return Promise.resolve(window.confirm(message));

        const text = modal.querySelector("[data-confirm-message]");
        const accept = modal.querySelector("[data-confirm-accept]");
        const cancel = modal.querySelector("[data-confirm-cancel]");
        text.textContent = message;
        accept.textContent = acceptLabel;
        modal.hidden = false;
        accept.focus();

        return new Promise((resolve) => {
            const cleanup = (value) => {
                modal.hidden = true;
                accept.removeEventListener("click", onAccept);
                cancel.removeEventListener("click", onCancel);
                modal.removeEventListener("click", onBackdrop);
                document.removeEventListener("keydown", onKey);
                resolve(value);
            };
            const onAccept = () => cleanup(true);
            const onCancel = () => cleanup(false);
            const onBackdrop = (event) => {
                if (event.target === modal) cleanup(false);
            };
            const onKey = (event) => {
                if (event.key === "Escape") cleanup(false);
            };
            accept.addEventListener("click", onAccept);
            cancel.addEventListener("click", onCancel);
            modal.addEventListener("click", onBackdrop);
            document.addEventListener("keydown", onKey);
        });
    }

    function initTabs() {
        document.querySelectorAll("[data-tab-trigger]").forEach((button) => {
            button.addEventListener("click", () => {
                const target = button.dataset.tabTrigger;
                document.querySelectorAll("[data-tab-trigger]").forEach((item) => {
                    item.classList.toggle("is-active", item === button);
                    item.setAttribute("aria-selected", item === button ? "true" : "false");
                });
                document.querySelectorAll("[data-tab-panel]").forEach((panel) => {
                    panel.classList.toggle("is-active", panel.dataset.tabPanel === target);
                });
            });
        });
    }

    window.TenderApp = {
        request,
        showToast,
        confirmAction,
        getCookie,
    };

    document.addEventListener("DOMContentLoaded", initTabs);
})();
