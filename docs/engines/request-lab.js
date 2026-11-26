(() => {
  "use strict";

  // All counts are for one hypothetical request on a dense, causal model.
  // Each decode forward consumes the last sampled token to predict the next.
  const steps = [
    { id: "request", label: "Receive", title: "A chat message arrives", handoff: "Chat messages → generation request",
      text: "The client asks for a streamed reply with a three-token output limit. The model and tokenizer are already loaded. The service tracks a request ID so each piece of the reply returns to the right connection." },
    { id: "tokens", label: "Tokenize", title: "Convert the conversation to IDs", handoff: "Rendered conversation → 40 prompt IDs",
      text: "A chat template adds role markers and the assistant’s turn to the conversation. The tokenizer maps that rendered text to integers. Assume 40 input tokens including the system instruction and markers. Actual counts depend on the template and tokenizer." },
    { id: "match", label: "Schedule", title: "Find work the engine can reuse", handoff: "Waiting request → prefix lookup → batch",
      text: "The scheduler checks which leading token positions already have usable keys and values (KV), then budgets space and computation for the rest. Other requests may enter the same batch." },
    { id: "prefill", label: "Prefill", title: "Read the prompt and score the next token", handoff: "Prompt IDs + past KV → logits",
      text: "The model runs attention and its feed-forward layers over the uncached prompt positions. Attention can read earlier cached keys and values. The final prompt position produces logits, one score per possible next token." },
    { id: "first", label: "First token", title: "Deliver the first piece of text", handoff: "Logits → sampled ID → “Blue”",
      text: "Suppose the sampler selects the ID for “Blue”. The output path converts it into a text delta and writes it to the client. Time to first token (TTFT) includes the queue, prompt processing and delivery of this first visible piece." },
    { id: "decode1", label: "Decode 1", title: "Feed the first token back in", handoff: "“Blue” + 40 KV positions → “ sky”",
      text: "The next model pass processes the ID for “Blue”, appends its KV as position 41 and predicts a second token. The leading space in “ sky” is part of this example token. The client now sees “Blue sky”." },
    { id: "decode2", label: "Decode 2", title: "Repeat the loop once more", handoff: "“ sky” + 41 KV positions → “.”",
      text: "The same decode loop consumes “ sky”, writes position 42 and samples a period. The third output reaches the request’s limit. A fourth model pass would only be needed to predict a fourth token. Inter-token latency measures the gap between token arrivals." },
    { id: "finish", label: "Release", title: "Close the stream and release the request", handoff: "Three output IDs → length finish → [DONE]",
      text: "The stream reports a length finish and closes. The scheduler removes the request from the active batch and releases its KV references. Cache entries eligible for prefix reuse can survive until eviction. A model end token could have finished the request earlier." },
  ];
  const abort = { id: "abort", label: "Cancel", title: "The client disconnects after “Blue”", handoff: "Closed connection → abort → release resources",
    text: "The connection closes before the second token reaches the client. Cancellation propagates to the engine, which stops scheduling further work for this request and releases its resources. Work already in flight can finish. The disconnected client receives no final stream marker." };

  // Each entry links into the existing, commit-pinned source walkthrough.
  const engines = {
    vllm: {
      label: "vLLM",
      request: ["API server", "The OpenAI-compatible HTTP route validates the body and starts the chat handler.", "api-surface"],
      tokens: ["Renderer · AsyncLLM", "The frontend renders the chat, prepares the request and passes token IDs and sampling parameters to EngineCore.", "architecture"],
      match: ["EngineCore · Scheduler", "The scheduler looks up matching full prefix blocks and allocates slots for the remainder under its token budget. Here a block holds 16 positions. Under memory pressure, preempted requests can wait and later recompute missing KV.", "kv-cache"],
      prefill: ["Executor · GPU model runner", "The worker receives a block table mapping logical positions to physical KV blocks. Multiple prompt chunks may need separate scheduler steps for long inputs.", "model-runner"],
      first: ["Sampler · OutputProcessor", "The sampled token ID returns through EngineCore. The frontend’s OutputProcessor detokenizes it and the chat handler emits a server-sent event.", "streaming"],
      decode1: ["EngineCore · GPU worker", "The scheduler assigns a new token position. The worker reads the request’s existing KV through the block table and attends to the new position too.", "scheduler"],
      decode2: ["EngineCore · GPU worker", "The scheduler can batch surviving decode requests with newly admitted work. The model’s next-token dependency still orders successive tokens of this one request.", "prefill-decode"],
      finish: ["Scheduler · KVCacheManager", "The scheduler drops the request’s block references. A complete cached block can stay indexed by hash and be reused while it remains resident.", "kv-cache"],
      abort: ["AsyncLLM · EngineCore", "The frontend forwards the cancellation. The scheduler retires the request and releases block references as pending work is resolved.", "streaming"],
    },
    sglang: {
      label: "SGLang",
      request: ["HTTP server · chat adapter", "The HTTP handler translates the OpenAI-style chat into the runtime’s generate request.", "api-surface"],
      tokens: ["TokenizerManager", "The main process tracks the client, prepares token IDs and sampling parameters, and passes a tokenized request to the scheduler.", "architecture"],
      match: ["Scheduler · radix cache", "The scheduler follows the token sequence down a radix tree. A matching path gives the slot indices for reusable KV. If memory runs short, unlocked entries can be evicted and running requests may return to the waiting queue.", "radix-cache"],
      prefill: ["Scheduler · ModelRunner", "An EXTEND batch carries the uncached positions into the model runner. A request-to-token mapping connects the request to the physical KV slots it can read.", "batch-flow"],
      first: ["DetokenizerManager · TokenizerManager", "Sampled IDs pass to the detokenizer process. Text returns to the TokenizerManager, which wakes the waiting response generator for this client.", "streaming"],
      decode1: ["Scheduler · DECODE batch", "The request joins a running decode batch. The model runner reads its earlier KV and writes one new slot for the previous output token.", "memory-pools"],
      decode2: ["Scheduler · ModelRunner", "Each surviving request contributes its own next input. The scheduler can prepare another batch while GPU work completes when overlap scheduling is enabled.", "scheduler"],
      finish: ["Scheduler · radix cache", "The request row returns to the pool and its prefix locks are released. Eligible KV can remain in the tree and unlocked cache entries can be evicted for new work.", "radix-cache"],
      abort: ["TokenizerManager · Scheduler", "An abort message identifies the request. Its active batch or waiting-queue state is retired, allowing the request row and cache references to be reclaimed.", "scheduler"],
    },
  };

  const labMarkup = `
    <div class="lab-config">
      <label>Engine
        <select data-lab-engine><option value="vllm">vLLM</option><option value="sglang">SGLang</option></select>
      </label>
      <label>Prefix cache
        <select data-lab-cache><option value="cold">Cold cache</option><option value="warm">32-token prefix hit</option></select>
      </label>
      <label>Ending
        <select data-lab-ending><option value="limit">Three-token limit</option><option value="cancel">Client disconnects</option></select>
      </label>
    </div>
    <div class="lab-prompt"><span>Prompt</span><q>Describe the afternoon in two words.</q></div>
    <div class="lab-inspector">
      <div class="lab-steps" role="group" aria-label="Choose a request stage" data-lab-steps></div>
      <div class="lab-panel">
        <div class="lab-caption" data-lab-count></div>
        <h3 data-lab-title></h3>
        <p data-lab-text></p>
        <div class="lab-handoff" data-lab-handoff></div>
        <div class="lab-owner"><h4 data-lab-owner></h4><p data-lab-implementation></p><a data-lab-chapter>Read the implementation</a></div>
      </div>
    </div>
    <div class="lab-state">
      <figure class="lab-cache">
        <figcaption>Prompt KV positions</figcaption>
        <div class="lab-blocks">
          <span data-lab-block>01&ndash;16<small>pending</small></span>
          <span data-lab-block>17&ndash;32<small>pending</small></span>
          <span data-lab-block>33&ndash;40<small>pending</small></span>
        </div>
        <p data-lab-cache-note></p>
      </figure>
      <div class="lab-counter">
        <div><span>Positions reused</span><strong data-lab-reused>0</strong></div>
        <div><span>Positions computed this run</span><strong data-lab-computed>0</strong></div>
        <div><span>Client sees</span><strong data-lab-output>Waiting for text</strong></div>
      </div>
    </div>
    <div class="lab-playback"><button type="button" data-lab-back>Previous</button><button type="button" data-lab-next>Next step</button></div>
    <span class="lab-sr" role="status" aria-live="polite" data-lab-status></span>
  `;

  function setupLab(root) {
    root.innerHTML = labMarkup;
    const engine = root.querySelector("[data-lab-engine]");
    engine.value = root.dataset.requestLab;
    const cache = root.querySelector("[data-lab-cache]");
    const ending = root.querySelector("[data-lab-ending]");
    const stepsNav = root.querySelector("[data-lab-steps]");
    const back = root.querySelector("[data-lab-back]");
    const next = root.querySelector("[data-lab-next]");
    const put = (field, content) => { root.querySelector(`[data-lab-${field}]`).textContent = content; };
    let position = 0;
    let sequence = steps;
    let stepButtons = [];

    function buildSteps() {
      const previousStep = sequence[position].id;
      sequence = ending.value === "cancel" ? [...steps.slice(0, 5), abort] : steps;
      const sameStep = sequence.findIndex((step) => step.id === previousStep);
      position = sameStep < 0 ? sequence.length - 1 : sameStep;
      stepButtons = sequence.map((step, index) => {
        const button = document.createElement("button");
        button.type = "button";
        const ordinal = document.createElement("span");
        ordinal.textContent = String(index + 1).padStart(2, "0");
        button.append(ordinal, step.label);
        button.addEventListener("click", () => { position = index; render(); });
        return button;
      });
      stepsNav.replaceChildren(...stepButtons);
    }

    function render() {
      const step = sequence[position];
      const choice = engines[engine.value];
      const [owner, implementation, chapter] = choice[step.id];
      const warm = cache.value === "warm";
      const ended = step.id === "finish" || step.id === "abort";
      const prefillComplete = position >= 3;
      const matched = warm && position >= 2;
      const reused = matched ? 32 : 0;
      const produced = step.id === "abort" ? 1 : Math.min(3, Math.max(0, position - 3));
      // The last sampled output has not been consumed by a forward pass.
      const computed = prefillComplete ? 40 - reused + Math.max(0, produced - 1) : 0;
      stepButtons.forEach((button, index) => {
        if (index === position) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      });
      put("count", `Step ${position + 1} of ${sequence.length}`);
      put("title", step.title);
      put("text", step.text);
      put("owner", owner);
      put("implementation", implementation);
      put("handoff", step.handoff);
      put("output", ["Waiting for text", "Blue", "Blue sky", "Blue sky."][produced]);
      put("reused", String(reused));
      put("computed", String(computed));
      const read = root.querySelector("[data-lab-chapter]");
      read.href = `${engine.value}.html#${chapter}`;
      read.textContent = `Read the ${choice.label} implementation`;
      const cacheDescription = position < 2
        ? "The scheduler has not checked the prompt yet."
        : ended ? "The request no longer holds active KV references."
        : warm ? "The first 32 prompt positions are reused. Eight require fresh computation."
        : "All 40 prompt positions require fresh computation.";
      put("cache-note", cacheDescription);
      put("status", `${choice.label}, step ${position + 1} of ${sequence.length}. ${step.title}. ${cacheDescription}`);
      const blocks = [...root.querySelectorAll("[data-lab-block]")];
      blocks.forEach((block, index) => {
        let state = prefillComplete ? "computed" : "pending";
        if (matched && index < 2) state = "reused";
        if (ended) state = "released";
        block.dataset.state = state;
        block.querySelector("small").textContent = state;
      });
      back.disabled = position === 0;
      next.textContent = position === sequence.length - 1 ? "Start again" : "Next step";
    }

    engine.addEventListener("change", render);
    cache.addEventListener("change", render);
    ending.addEventListener("change", () => { buildSteps(); render(); });
    back.addEventListener("click", () => { position = Math.max(0, position - 1); render(); });
    next.addEventListener("click", () => { position = (position + 1) % sequence.length; render(); });
    buildSteps();
    render();
    root.hidden = false;
  }

  document.querySelectorAll("[data-request-lab]").forEach(setupLab);

  // Deep links, including search results inside a chapter, expose that chapter
  // in the otherwise compact navigation.
  function showLinkedChapter() {
    const section = document.getElementById(location.hash.slice(1))?.closest("section[id]");
    if (!section) return;
    for (const group of document.querySelectorAll(".engine-nav-group")) {
      if ([...group.querySelectorAll("a")].some((link) => link.hash === `#${section.id}`)) {
        group.open = true;
      }
    }
  }
  window.addEventListener("hashchange", showLinkedChapter);
  showLinkedChapter();
})();
