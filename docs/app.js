(() => {
  "use strict";

  const body = document.body;
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
  const guideBase = new URL(".", document.currentScript.src);

  const normalize = (value) =>
    value
      .toLocaleLowerCase()
      .normalize("NFKD")
      .replace(/[\u0300-\u036f]/g, "");

  const setMenu = (open) => {
    body.classList.toggle("nav-open", open);
    menuButton?.setAttribute("aria-expanded", String(open));
    menuButton?.setAttribute("aria-label", open ? "Close section navigation" : "Open section navigation");
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
      if (body.classList.contains("nav-open")) menuButton?.focus();
      if (searchResults?.contains(document.activeElement)) siteSearch?.focus();
      setMenu(false);
      if (searchResults) searchResults.hidden = true;
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLocaleLowerCase() === "k") {
      event.preventDefault();
      siteSearch?.focus();
    }
  });

  // These destinations join the local, full-text chapter index across static pages.
  const destinations = [
    ["Token laboratory", "index.html#request-replay", "Python model replay logits tokenizer sampled bytes greedy prefill decode"],
    ["Model architectures", "models/index.html", "dense decoder GQA MoE mixture experts DeepSeek V3 MLA R1 distilled"],
    ["Translate a KV address", "engines/memory.html#address-lab", "block table hash vllm sglang request index pool slot mapping req_to_token"],
    ["Prefix reuse and ownership", "engines/memory.html#prefix-reuse", "cache hash radix block boundary page reference count copy on write"],
    ["SGLang HiCache", "engines/memory.html#hicache", "hiecache hierarchical caching GPU device host DRAM L3 storage tier policy prefetch eviction"],
    ["Trace a distributed token", "systems/fabric.html#token-messages", "TP PP DP EP tensor pipeline data expert messages collectives routing all reduce"],
    ["Prefill/decode handoff", "systems/fabric.html#pd", "distributed PD connector ownership transfer heterogeneous prefill decode"],
    ["Attention–FFN disaggregation", "systems/fabric.html#afd", "AFD heterogeneous experts dispatch combine activations links network bandwidth"],
    ["Heterogeneous hardware", "systems/fabric.html#compatibility", "GPU fabric layout precision transfer compatibility bandwidth"],
    ["26-day curriculum", "curriculum.html", "beginner foundations study plan learning path"],
    ["Model mechanics", "notes/phase-1-mechanics.html", "tokens embeddings attention logits RoPE GQA KV cache"],
    ["Engine internals", "notes/phase-2-engines.html", "roofline runtime kernels model formats CUDA"],
    ["Optimization notes", "notes/phase-3-optimization.html", "speculative decoding quantization prefix caching parallelism GPTQ AWQ"],
    ["Production notes", "notes/phase-4-production.html", "routing autoscaling queueing metrics cost deployment cold start"],
    ["Compare vLLM and SGLang", "engines/index.html#request-lab", "side by side request lab end to end warm cold cache"],
    ["vLLM walkthrough", "engines/vllm.html#request-lab", "V1 paged attention EngineCore scheduler streaming request"],
    ["SGLang walkthrough", "engines/sglang.html#request-lab", "SRT radix attention tokenizer manager radixcache streaming request"],
    ["GPU memory capacity calculator", "systems/index.html#capacity", "KV cache budget precision concurrency heads size worksheet"],
    ["Inference optimization lab", "systems/index.html#optimization", "bottleneck diagnose latency throughput profiling tuning"],
    ["Distributed inference topologies", "systems/index.html#topologies", "tensor pipeline data expert context parallelism TP PP DP EP MoE GPUs"],
    ["Prefill and decode disaggregation", "systems/index.html#disaggregation", "KV transfer handoff split bandwidth routing"],
    ["Measure serving performance", "systems/index.html#measurement", "TTFT ITL goodput SLO p95 benchmark"],
    ["Annotated reading library", "reading/index.html", "100 days of inference NVIDIA research papers Medium articles resources"],
    ["Editable request diagram", "index.html#request-diagram", "Mermaid Excalidraw flowchart process block diagram"],
    ["Interview practice", "index.html#interview-bank", "questions answers interview bank"],
  ].map(([label, path, keywords]) => ({
    label, href: new URL(path, guideBase).href, group: "Field guide",
    title: normalize(label), keywords: normalize(keywords), text: normalize(label + " " + keywords),
  }));

  const localIndex = searchableItems.map((item) => {
    const heading = item.querySelector("h1, h2, h3, h4, summary, figcaption, a, code") || item;
    const label = heading.textContent.trim();
    const section = item.closest("[data-search-group]");
    return {
      href: item.matches("a[href]") ? item.href : `#${item.id || section?.id || ""}`,
      label,
      title: normalize(label),
      keywords: normalize(item.dataset.keywords || ""),
      group: section?.dataset.searchGroup || "Guide",
      text: normalize(`${label} ${item.textContent} ${item.dataset.keywords || ""}`),
    };
  });
  const searchIndex = [...localIndex, ...destinations];

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
    const score = (entry) => terms.reduce((sum, term) =>
      sum + (entry.title.includes(term) ? 4 : entry.keywords.includes(term) ? 2 : 0),
    entry.title === query ? 8 : 0);
    const seen = new Set();
    const matches = searchIndex
      .filter((entry) => terms.every((term) => entry.text.includes(term)))
      .map((entry) => ({entry, rank: score(entry)}))
      .sort((a, b) => b.rank - a.rank)
      .map((match) => match.entry)
      .filter((entry) => {
        const href = new URL(entry.href, location.href).href;
        if (seen.has(href)) return false;
        seen.add(href);
        return true;
      })
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
  siteSearch?.addEventListener("keydown", (event) => {
    if (!searchResults || searchResults.hidden) return;
    const first = searchResults.querySelector("a");
    if (first && ["ArrowDown", "Enter"].includes(event.key)) {
      event.preventDefault();
      if (event.key === "Enter") first.click();
      else first.focus();
    }
  });
  searchResults?.addEventListener("keydown", (event) => {
    if (!["ArrowDown", "ArrowUp"].includes(event.key)) return;
    event.preventDefault();
    const links = [...searchResults.querySelectorAll("a")];
    const next = links.indexOf(document.activeElement) + (event.key === "ArrowDown" ? 1 : -1);
    (next < 0 ? siteSearch : links[Math.min(next, links.length - 1)])?.focus();
  });
  searchResults?.addEventListener("click", (event) => {
    const link = event.target.closest("a");
    if (!link) return;
    const target = link.pathname === location.pathname ? document.getElementById(link.hash.slice(1)) : null;
    if (target?.matches(".qa-item")) {
      if (target.hidden) {
        interviewSearch.value = "";
        filterButtons.find((button) => button.dataset.filter === "all")?.click();
      }
      target.open = true;
    }
    searchResults.hidden = true;
    siteSearch.value = "";
  });
  document.addEventListener("click", (event) => {
    if (searchResults && !event.target.closest(".search-wrap")) {
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
    if (expandButton) {
      expandButton.disabled = visible === 0;
      expandButton.textContent = visible && questionItems.every((item) => item.hidden || item.open)
        ? "Collapse visible" : "Expand visible";
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
  function openQuestionFromHash() {
    const target = document.getElementById(location.hash.slice(1));
    if (target?.matches(".qa-item")) target.open = true;
  }
  openQuestionFromHash();
  window.addEventListener("hashchange", openQuestionFromHash);

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
          if (link.hash === `#${visible.target.id}`) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
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
