(() => {
  "use strict";

  // Keep a wide diagram, code sample, or table reachable with Tab and arrow keys.
  // When its content fits, leave it out of the keyboard sequence.
  const selector = [
    "main pre", "main table", ".svg-diagram", ".table-wrap", ".formula",
    ".equation", ".equation-card code", ".bit-strip", ".lc-viewport", "[data-scroll-region]",
  ].join(",");
  const regions = Array.from(document.querySelectorAll(selector))
    .filter((region) => !region.hasAttribute("tabindex"));

  function update(region) {
    const overflows = region.clientWidth > 0 && region.scrollWidth > region.clientWidth + 1;
    if (overflows && ["auto", "scroll"].includes(getComputedStyle(region).overflowX)) {
      region.tabIndex = 0;
    } else {
      region.removeAttribute("tabindex");
    }
  }

  regions.forEach(update);
  if ("ResizeObserver" in window) {
    const observer = new ResizeObserver((entries) => entries.forEach(({ target }) => update(target)));
    regions.forEach((region) => observer.observe(region));
  } else {
    window.addEventListener("resize", () => regions.forEach(update));
  }
})();
