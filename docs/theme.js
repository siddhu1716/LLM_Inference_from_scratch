(() => {
  "use strict";
  const button = document.querySelector("[data-theme-toggle]");
  const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");
  const storageKey = "mini-infer-theme";
  let preference;
  try {
    preference = localStorage.getItem(storageKey);
  } catch {
    // The theme control also works in a storage-restricted preview.
  }

  function apply() {
    const theme = ["light", "dark"].includes(preference)
      ? preference
      : systemTheme.matches ? "dark" : "light";
    document.documentElement.dataset.theme = theme;
    if (button) {
      const label = theme === "dark" ? "Use light theme" : "Use dark theme";
      button.setAttribute("aria-label", label);
      button.title = label;
      button.querySelector("[data-theme-icon]").textContent = theme === "dark" ? "☀" : "◐";
    }
  }

  apply();
  systemTheme.addEventListener("change", apply);
  button?.addEventListener("click", () => {
    preference = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    try {
      localStorage.setItem(storageKey, preference);
    } catch {
      // Keep the selected mode for this page.
    }
    apply();
  });
})();
