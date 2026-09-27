(() => {
  "use strict";

  const root = document.documentElement;
  const body = document.body;
  const themeButton = document.querySelector("[data-theme-toggle]");
  const menuButton = document.querySelector("[data-menu-toggle]");
  const sidebar = document.querySelector(".sidebar");
  const siteSearch = document.querySelector("#site-search");
  const searchResults = document.querySelector("#search-results");
  const interviewSearch = document.querySelector("#interview-search");
  const filterButtons = [...document.querySelectorAll("[data-filter]")];
  const questionItems = [...document.querySelectorAll(".qa-item")];
  const questionGroups = [...document.querySelectorAll(".qa-group")];
  const questionCount = document.querySelector("#question-count");
  const emptyState = document.querySelector("#qa-empty");
  const expandButton = document.querySelector("#expand-visible");
  const backToTop = document.querySelector(".back-to-top");
  const searchableItems = [...document.querySelectorAll("[data-search-item]")];

  const normalize = (value) =>
    value
      .toLocaleLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "");

  const safeStorage = {
    get(key) {
      try {
        return window.localStorage.getItem(key);
      } catch {
        return null;
      }
    },
    set(key, value) {
      try {
        window.localStorage.setItem(key, value);
      } catch {
        // Persistence is optional; controls still work when storage is blocked.
      }
    },
  };

  const preferredTheme = () => {
    const saved = safeStorage.get("mini-infer-theme");
    if (saved === "light" || saved === "dark") {
      return saved;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  };

  const applyTheme = (theme) => {
    root.dataset.theme = theme;
    if (themeButton) {
      const nextTheme = theme === "dark" ? "light" : "dark";
      themeButton.setAttribute("aria-label", `Use ${nextTheme} theme`);
      themeButton.setAttribute("title", `Use ${nextTheme} theme`);
      themeButton.querySelector("[data-theme-icon]").textContent = theme === "dark" ? "☀" : "◐";
    }
  };

  applyTheme(preferredTheme());

  themeButton?.addEventListener("click", () => {
    const theme = root.dataset.theme === "dark" ? "light" : "dark";
    applyTheme(theme);
    safeStorage.set("mini-infer-theme", theme);
  });

  const setMenu = (open) => {
    body.classList.toggle("nav-open", open);
    menuButton?.setAttribute("aria-expanded", String(open));
  };

  const desktopNavigation = window.matchMedia("(min-width: 56.01rem)");
  const closeMenuOnDesktop = (event) => {
    if (event.matches) {
      setMenu(false);
    }
  };

  desktopNavigation.addEventListener("change", closeMenuOnDesktop);
  window.addEventListener("resize", () => closeMenuOnDesktop(desktopNavigation));
  closeMenuOnDesktop(desktopNavigation);

  menuButton?.addEventListener("click", () => setMenu(!body.classList.contains("nav-open")));
  sidebar?.addEventListener("click", (event) => {
    if (event.target.closest("a")) {
      setMenu(false);
    }
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      setMenu(false);
      searchResults.hidden = true;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
      event.preventDefault();
      siteSearch?.focus();
    }
  });

  const searchIndex = searchableItems.map((item) => {
    const heading = item.querySelector("h2, h3, h4, summary") || item;
    const label = heading.textContent.trim();
    const section = item.closest("section");
    return {
      element: item,
      href: `#${item.id || section?.id || ""}`,
      label,
      group: section?.dataset.searchGroup || "Guide",
      text: normalize(`${label} ${item.textContent} ${item.dataset.keywords || ""}`),
    };
  });

  const showSearchResults = () => {
    if (!siteSearch || !searchResults) {
      return;
    }
    const query = normalize(siteSearch.value.trim());
    searchResults.replaceChildren();
    if (query.length < 2) {
      searchResults.hidden = true;
      return;
    }

    const terms = query.split(/\s+/).filter(Boolean);
    const matches = searchIndex
      .filter((entry) => terms.every((term) => entry.text.includes(term)))
      .slice(0, 10);

    if (matches.length === 0) {
      const item = document.createElement("li");
      item.textContent = "No matching topics. Try a broader term.";
      item.className = "empty-state";
      searchResults.append(item);
    } else {
      for (const match of matches) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        const label = document.createElement("span");
        const group = document.createElement("small");
        link.href = match.href;
        label.textContent = match.label;
        group.textContent = match.group;
        link.append(label, group);
        item.append(link);
        searchResults.append(item);
      }
    }
    searchResults.hidden = false;
  };

  siteSearch?.addEventListener("input", showSearchResults);
  siteSearch?.addEventListener("focus", showSearchResults);
  searchResults?.addEventListener("click", () => {
    searchResults.hidden = true;
    siteSearch.value = "";
  });
  document.addEventListener("click", (event) => {
    if (!event.target.closest(".search-wrap")) {
      searchResults.hidden = true;
    }
  });

  let activeCategory = "all";

  const applyQuestionFilters = () => {
    const query = normalize(interviewSearch?.value.trim() || "");
    const terms = query.split(/\s+/).filter(Boolean);
    let visible = 0;

    for (const item of questionItems) {
      const categoryMatch = activeCategory === "all" || item.dataset.category === activeCategory;
      const text = normalize(`${item.textContent} ${item.dataset.keywords || ""}`);
      const queryMatch = terms.every((term) => text.includes(term));
      const show = categoryMatch && queryMatch;
      item.hidden = !show;
      if (show) {
        visible += 1;
      } else {
        item.open = false;
      }
    }

    for (const group of questionGroups) {
      group.hidden = !group.querySelector(".qa-item:not([hidden])");
    }

    if (questionCount) {
      questionCount.textContent = `${visible} of ${questionItems.length} questions`;
    }
    if (emptyState) {
      emptyState.hidden = visible !== 0;
    }
  };

  interviewSearch?.addEventListener("input", applyQuestionFilters);
  for (const button of filterButtons) {
    button.addEventListener("click", () => {
      activeCategory = button.dataset.filter;
      for (const peer of filterButtons) {
        peer.setAttribute("aria-pressed", String(peer === button));
      }
      applyQuestionFilters();
    });
  }

  expandButton?.addEventListener("click", () => {
    const visibleItems = questionItems.filter((item) => !item.hidden);
    const shouldOpen = visibleItems.some((item) => !item.open);
    for (const item of visibleItems) {
      item.open = shouldOpen;
    }
    expandButton.textContent = shouldOpen ? "Collapse visible" : "Expand visible";
  });

  applyQuestionFilters();

  const navLinks = [...document.querySelectorAll(".nav-list a[href^='#']")];
  const observedSections = navLinks
    .map((link) => document.querySelector(link.getAttribute("href")))
    .filter(Boolean);

  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((entry) => entry.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
        if (!visible) {
          return;
        }
        for (const link of navLinks) {
          link.toggleAttribute(
            "aria-current",
            link.getAttribute("href") === `#${visible.target.id}`,
          );
        }
      },
      { rootMargin: "-18% 0px -68% 0px", threshold: [0, 0.15, 0.5] },
    );
    observedSections.forEach((section) => observer.observe(section));
  }

  const updateScrollControls = () => {
    backToTop?.classList.toggle("visible", window.scrollY > 900);
  };
  window.addEventListener("scroll", updateScrollControls, { passive: true });
  updateScrollControls();

  backToTop?.addEventListener("click", (event) => {
    event.preventDefault();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
})();
