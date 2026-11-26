(() => {
  "use strict";

  const boardURL = new URL("h100-board.svg", document.currentScript.src).href;
  const diagrams = {
    tile: {
      title: "Matrix tile",
      steps: [
        ["Stage the weights", "The loader prepares a weight tile in host DRAM. This copy happens during loading in our example.", "M30 135 H87", "host"],
        ["Copy to device memory", "A host-to-device copy crosses PCIe. The tile now has a device address in HBM3.", "M80 134 H130 L151 113 H182 V97", "hbm"],
        ["Fill the shared L2 cache", "A kernel load misses L2 in this example. The memory system fetches the line from HBM3.", "M182 97 V119 H204 L222 135 H306", "l2"],
        ["Stage a tile on one SM", "A thread block brings operands into its shared memory and registers. Other blocks can work on other tiles.", "M306 135 H284 V101 H277", "sm"],
        ["Issue matrix instructions", "Warps execute the kernel. Tensor Core instructions multiply and accumulate operand tiles.", "M273 85 H295 V113 H273 Z", "sm"],
        ["Store the output", "Threads store the accumulated result to global memory. Cached data can be written back to HBM later.", "M278 106 V135 H338 L365 162 H418 V182", "hbm2"]
      ]
    },
    global: {
      title: "Two SMs · global memory",
      steps: [
        ["Produce a partial result", "A block in kernel A computes a tile. Its registers and ordinary shared memory belong to that block's execution.", "M228 86 H252 V113 H228 Z", "producer"],
        ["Publish a global store", "The block stores the partial to global memory. The cache hierarchy may retain it in L2.", "M242 100 V135 H308", "l2"],
        ["Order the consumer kernel", "Kernel B follows A in the same CUDA stream. Completion of A orders B's access to the stored data.", "M245 135 H367", "l2"],
        ["Load the partial", "A block of B can run on another SM and read the global address. An L2 hit can avoid an HBM read.", "M308 135 H373 V173 H365", "consumer"]
      ]
    },
    cluster: {
      title: "Two SMs · cluster DSM",
      steps: [
        ["Place a block cluster", "Hopper can co-schedule a cluster of thread blocks in one GPU processing cluster (GPC).", "M268 79 H300 V116 H268 Z", "sm"],
        ["Initialize local storage", "Each participating block creates and initializes its own shared-memory segment. A cluster barrier makes the segments ready.", "M270 84 H279 V99 H270 Z", "local"],
        ["Access a peer block's segment", "A block maps a peer block's shared memory within this cluster. A remote load, store or atomic can use that address.", "M274 90 H290 V108", "remote"],
        ["Finish before block exit", "The cluster synchronizes after remote accesses. The owner can exit once nobody still uses its shared memory.", "M290 108 H274 V90", "sm"]
      ]
    },
    peer: {
      title: "Peer GPU · NVLink",
      steps: [
        ["Compute a local partial", "Each tensor-parallel rank computes a partial output using its local shard of the weights.", "M272 86 H295 V112 H272 Z", "sm"],
        ["Make the buffer ready", "The collective consumes device buffers after the producing work is ready, with ordering established by the stream or events.", "M278 100 V135 H368", "l2"],
        ["Exchange across GPUs", "In this HGX example, NVLink and NVSwitch carry collective traffic between participating H100s.", "M367 135 H428 L453 183 H478 L491 191 H534", "peer"],
        ["Use the reduced result", "For a sum all-reduce, each rank receives the complete sum. A dependent layer on this GPU can then consume it.", "M531 192 H490 L475 182 H452 L411 135 H285 V101", "sm"]
      ]
    }
  };
  const regions = {
    host: [17, 95, 85, 79], hbm: [166, 73, 35, 43], l2: [221, 124, 169, 22],
    sm: [264, 72, 40, 50], hbm2: [401, 159, 35, 43], producer: [221, 72, 40, 50],
    consumer: [350, 150, 40, 50], local: [268, 81, 13, 19], remote: [285, 99, 13, 19],
    peer: [467, 152, 76, 70]
  };

  function mount(figure) {
    const caption = figure.querySelector("figcaption");
    const originalLink = figure.querySelector("a");
    const board = figure.querySelector("img") || new Image();
    board.src = boardURL;
    board.alt = "Logical H100 SXM5 module with five HBM3 stacks around the GPU die, a host connection on the left and a peer connection on the right.";
    board.width = 560;
    board.height = 258;
    const content = document.createElement("div");
    content.className = "h100-console";
    content.innerHTML = `
      <div class="h100-screen">
        <svg viewBox="0 0 560 258" class="h100-overlay" aria-hidden="true">
          <rect class="h100-region" rx="4"></rect>
          <path class="h100-glow"></path><path class="h100-wire"></path>
          <circle class="h100-packet" r="4"></circle>
        </svg>
      </div>
      <div class="h100-workbench">
        <div class="h100-selector"><label>Trace <select aria-label="Choose a GPU data path"></select></label><span class="h100-count"></span></div>
        <div class="h100-explanation"><strong></strong><p></p></div>
        <div class="h100-controls">
          <button type="button" data-action="play"></button>
          <button type="button" data-action="previous" aria-label="Previous step" title="Previous step">‹</button>
          <button type="button" data-action="next" aria-label="Next step" title="Next step">›</button>
        </div>
      </div>`;
    figure.replaceChildren();
    if (caption) figure.append(caption);
    figure.append(content);
    const get = (selector) => content.querySelector(selector);
    get(".h100-screen").prepend(board);
    if (originalLink) get(".h100-controls").append(originalLink);
    const select = get("select");
    for (const [key, route] of Object.entries(diagrams)) select.add(new Option(route.title, key));
    const playButton = get('[data-action="play"]');
    const previous = get('[data-action="previous"]');
    const next = get('[data-action="next"]');
    const path = get(".h100-wire");
    const point = get(".h100-packet");
    const highlight = get(".h100-region");
    const note = get(".h100-explanation");
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let steps = diagrams.tile.steps;
    let index = 0;
    let playing = false;
    let frame = 0;
    let elapsed = 0;
    let lastTimestamp;
    let pathLength = 0;
    let autoplayAvailable = !reduced.matches;
    const duration = 4700;

    function placePacket(fraction) {
      const at = path.getPointAtLength(pathLength * Math.max(0, Math.min(1, fraction)));
      point.setAttribute("cx", at.x);
      point.setAttribute("cy", at.y);
    }

    function updateStep() {
      const [title, text, points, region] = steps[index];
      get(".h100-count").textContent = (index + 1) + " / " + steps.length;
      note.querySelector("strong").textContent = title;
      note.querySelector("p").textContent = text;
      path.setAttribute("d", points);
      get(".h100-glow").setAttribute("d", points);
      regions[region].forEach((value, i) => highlight.setAttribute(["x","y","width","height"][i], value));
      previous.disabled = index === 0;
      next.disabled = index === steps.length - 1;
      pathLength = path.getTotalLength();
      placePacket(playing && !reduced.matches ? 0 : 1);
      updatePlayLabel();
    }

    function updatePlayLabel() {
      playButton.textContent = playing ? "Pause" : index === steps.length - 1 ? "Replay" : "Play";
      playButton.setAttribute("aria-label", playButton.textContent + " GPU trace");
    }

    function pause() {
      playing = false;
      cancelAnimationFrame(frame);
      lastTimestamp = undefined;
      note.setAttribute("aria-live", "polite");
      updatePlayLabel();
    }

    function tick(timestamp) {
      if (!playing) return;
      if (lastTimestamp !== undefined) elapsed += timestamp - lastTimestamp;
      lastTimestamp = timestamp;
      if (elapsed >= duration) {
        if (index === steps.length - 1) { pause(); return; }
        index += 1;
        elapsed = 0;
        updateStep();
      }
      if (!reduced.matches) placePacket((elapsed / 2700) % 1);
      frame = requestAnimationFrame(tick);
    }

    function play() {
      autoplayAvailable = false;
      if (playing) return;
      if (index === steps.length - 1) { index = 0; elapsed = 0; }
      playing = true;
      note.setAttribute("aria-live", "off");
      updateStep();
      frame = requestAnimationFrame(tick);
    }

    function choose(step) {
      autoplayAvailable = false;
      pause();
      index = step;
      elapsed = 0;
      updateStep();
    }
    playButton.addEventListener("click", () => { if (playing) pause(); else play(); });
    previous.addEventListener("click", () => choose(index - 1));
    next.addEventListener("click", () => choose(index + 1));
    select.addEventListener("change", () => { steps = diagrams[select.value].steps; choose(0); });
    document.addEventListener("visibilitychange", () => { if (document.hidden) pause(); });
    reduced.addEventListener("change", () => { autoplayAvailable = false; pause(); updateStep(); });
    updateStep();
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver(([entry]) => {
        if (!entry.isIntersecting) pause();
        else if (autoplayAvailable && !document.hidden) play();
      }, { threshold: .15 });
      observer.observe(figure);
    }
  }
  document.querySelectorAll("[data-h100-lab]").forEach(mount);
})();
