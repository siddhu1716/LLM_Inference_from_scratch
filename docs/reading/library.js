(() => {
  "use strict";

  const query = document.querySelector("#query");
  const format = document.querySelector("#format");
  const topic = document.querySelector("#topic");
  const count = document.querySelector("#count");
  const empty = document.querySelector("#empty");
  const reset = document.querySelector("#reset");
  const contents = document.querySelector(".contents");
  const normalize = (text) => text.toLowerCase().normalize("NFKD")
    .replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]+/g, " ").trim();

  const groups = Array.from(document.querySelectorAll(".collection"), (section) => ({
    section,
    jump: contents.querySelector(`a[href="#${section.id}"]`),
    items: Array.from(section.querySelectorAll(".entry"), (entry) => ({
      entry,
      searchable: normalize([
        section.querySelector("h2").textContent,
        entry.textContent,
        entry.dataset.keywords,
      ].join(" ")),
    })),
  }));

  const params = new URLSearchParams(location.search);
  query.value = params.get("q") || "";
  for (const select of [format, topic]) {
    const value = params.get(select.id);
    if (Array.from(select.options).some((option) => option.value === value)) select.value = value;
  }

  function update() {
    const terms = normalize(query.value).split(" ").filter(Boolean);
    let total = 0;
    for (const { section, jump, items } of groups) {
      const selectedTopic = !topic.value || section.id === topic.value;
      let matches = 0;
      for (const { entry, searchable } of items) {
        const selectedType = !format.value || entry.dataset.type === format.value;
        entry.hidden = !(selectedTopic && selectedType && terms.every((word) => searchable.includes(word)));
        if (!entry.hidden) matches++;
      }
      section.hidden = jump.hidden = matches === 0;
      total += matches;
    }
    count.textContent = total + (total === 1 ? " reading" : " readings");
    empty.hidden = total > 0;
    contents.hidden = total === 0;
    reset.hidden = !query.value && !format.value && !topic.value;

    // Keep filtered views shareable, including a direct link to all Medium articles.
    const url = new URL(location.href);
    for (const [key, value] of [["q", query.value], ["format", format.value], ["topic", topic.value]]) {
      if (value) url.searchParams.set(key, value);
      else url.searchParams.delete(key);
    }
    if (url.href !== location.href && /^https?:$/.test(location.protocol)) {
      history.replaceState(null, "", url);
    }
  }

  function clearFilters() {
    query.value = format.value = topic.value = "";
    update();
    query.focus();
  }

  query.addEventListener("input", update);
  format.addEventListener("change", update);
  topic.addEventListener("change", update);
  reset.addEventListener("click", clearFilters);
  document.querySelector("#reset-empty").addEventListener("click", clearFilters);
  document.querySelector("#filter-controls").hidden = false;
  update();
})();
