(() => {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const SPEEDS = { "0.5": 3200, "1": 1900, "2": 950 };
  const PALETTE = ["#3156d3", "#087e8b", "#e4572e", "#7b4cc2", "#19764b", "#9c5d00"];

  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = text;
    }
    return node;
  };

  // Renders `code` spans without accepting arbitrary HTML from data blocks.
  const richText = (text) => {
    const fragment = document.createDocumentFragment();
    String(text ?? "")
      .split(/(`[^`]+`)/g)
      .forEach((part) => {
        if (!part) {
          return;
        }
        if (part.length > 1 && part.startsWith("`") && part.endsWith("`")) {
          fragment.append(el("code", "", part.slice(1, -1)));
        } else {
          fragment.append(document.createTextNode(part));
        }
      });
    return fragment;
  };

  const svg = (tag, attributes = {}) => {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [name, value] of Object.entries(attributes)) {
      node.setAttribute(name, String(value));
    }
    return node;
  };

  class TokenLifecycle {
    constructor(root, data) {
      this.root = root;
      this.data = data;
      this.repo = String(data.repo || "").replace(/\/$/, "");
      this.commit = String(data.commit || "main");
      this.lanes = data.lanes || [];
      this.phases = data.phases || [];
      this.stages = data.stages || [];
      this.outputTokens = data.demo?.outputTokens?.length ? data.demo.outputTokens : ["token"];
      this.position = 0;
      this.playing = false;
      this.timer = null;
      this.cards = [];
      this.laneIndex = new Map(this.lanes.map((lane, index) => [lane.id, index]));
      this.phaseById = new Map(this.phases.map((phase) => [phase.id, phase]));
      this.timeline = this.buildTimeline();
      this.render();
      this.observe();
      this.go(Number(root.dataset.lifecycleStart || 0), { immediate: true });
    }

    buildTimeline() {
      const repeatPhase = this.phases.find((phase) => phase.repeat);
      const repeatIndexes = this.stages
        .map((stage, index) => (repeatPhase && stage.phase === repeatPhase.id ? index : -1))
        .filter((index) => index >= 0);
      this.repeatStart = repeatIndexes.length ? repeatIndexes[0] : -1;
      this.repeatEnd = repeatIndexes.length ? repeatIndexes[repeatIndexes.length - 1] : -1;

      const timeline = [];
      if (this.repeatStart < 0) {
        this.stages.forEach((_, index) => timeline.push({ index, iteration: 0 }));
        return timeline;
      }
      for (let index = 0; index < this.repeatStart; index += 1) {
        timeline.push({ index, iteration: 0 });
      }
      for (let iteration = 1; iteration <= this.outputTokens.length; iteration += 1) {
        for (let index = this.repeatStart; index <= this.repeatEnd; index += 1) {
          timeline.push({ index, iteration });
        }
      }
      for (let index = this.repeatEnd + 1; index < this.stages.length; index += 1) {
        timeline.push({ index, iteration: this.outputTokens.length + 1 });
      }
      return timeline;
    }

    laneColor(laneId) {
      const index = this.laneIndex.get(laneId) ?? 0;
      return this.lanes[index]?.color || PALETTE[index % PALETTE.length];
    }

    fileUrl(file) {
      if (file.url) {
        return file.url;
      }
      const base = `${this.repo}/blob/${this.commit}/${file.path}`;
      return file.line ? `${base}#L${file.line}` : base;
    }

    render() {
      const { data } = this;
      this.root.classList.add("lc");
      this.root.replaceChildren();
      this.root.setAttribute("tabindex", "-1");

      const header = el("div", "lc-header");
      const titleWrap = el("div", "lc-title");
      titleWrap.append(el("strong", "", data.title || `${data.engine} token lifecycle`));
      if (this.repo) {
        const pin = el("a", "lc-commit", `${data.engine} @ ${this.commit.slice(0, 7)}`);
        pin.href = `${this.repo}/tree/${this.commit}`;
        pin.target = "_blank";
        pin.rel = "noopener";
        pin.title = "All file links are pinned to this commit";
        titleWrap.append(pin);
      }
      header.append(titleWrap);

      const controls = el("div", "lc-controls");
      controls.setAttribute("role", "group");
      controls.setAttribute("aria-label", "Lifecycle playback controls");
      this.restartButton = this.button("⏮", "Restart lifecycle", () => this.go(0));
      this.prevButton = this.button("◀", "Previous step", () => this.previous());
      this.playButton = this.button("▶ Play", "Play lifecycle animation", () => this.toggle());
      this.playButton.classList.add("lc-play");
      this.playButton.setAttribute("aria-pressed", "false");
      this.nextButton = this.button("▶|", "Next step", () => this.next());
      const speedLabel = el("label", "lc-speed");
      speedLabel.append(document.createTextNode("Speed "));
      this.speed = el("select");
      for (const [value, label] of [["0.5", "0.5×"], ["1", "1×"], ["2", "2×"]]) {
        const option = el("option", "", label);
        option.value = value;
        option.selected = value === "1";
        this.speed.append(option);
      }
      this.speed.addEventListener("change", () => {
        if (this.playing) {
          this.schedule();
        }
      });
      speedLabel.append(this.speed);
      this.progress = el("span", "lc-progress");
      controls.append(
        this.restartButton,
        this.prevButton,
        this.playButton,
        this.nextButton,
        speedLabel,
        this.progress,
      );
      header.append(controls);
      this.root.append(header);

      const legend = el("div", "lc-legend");
      for (const phase of this.phases) {
        const firstStage = this.stages.findIndex((stage) => stage.phase === phase.id);
        if (firstStage < 0) {
          continue;
        }
        const chip = this.button(
          phase.repeat ? `↻ ${phase.label}` : phase.label,
          `Jump to ${phase.label}`,
          () => {
            this.pause();
            const target = this.timeline.findIndex((entry) => entry.index === firstStage);
            this.go(Math.max(0, target));
          },
        );
        chip.className = "lc-phase-chip";
        legend.append(chip);
      }
      this.root.append(legend);

      const body = el("div", "lc-body");
      this.viewport = el("div", "lc-viewport");
      this.board = el("div", "lc-board");
      this.board.style.setProperty("--lc-lanes", String(Math.max(1, this.lanes.length)));
      this.viewport.append(this.board);
      body.append(this.viewport);

      const side = el("div", "lc-side");
      this.stream = el("div", "lc-stream");
      const streamHead = el("div", "lc-stream-head");
      streamHead.append(el("strong", "", "Client sees"));
      this.iterationBadge = el("span", "lc-iteration");
      streamHead.append(this.iterationBadge);
      this.stream.append(streamHead);
      if (data.demo?.prompt) {
        const prompt = el("p", "lc-prompt");
        prompt.append(
          el("span", "lc-prompt-label", "prompt"),
          document.createTextNode(` ${data.demo.prompt}`),
        );
        this.stream.append(prompt);
      }
      this.output = el("p", "lc-output");
      this.output.setAttribute("aria-live", "polite");
      this.stream.append(this.output);
      side.append(this.stream);

      this.detail = el("section", "lc-detail");
      this.detail.setAttribute("aria-live", "polite");
      side.append(this.detail);

      this.trail = el("details", "lc-trail");
      this.trail.open = true;
      this.trailSummary = el("summary");
      this.trail.append(this.trailSummary);
      this.trailList = el("ol");
      this.trail.append(this.trailList);
      side.append(this.trail);
      body.append(side);
      this.root.append(body);

      this.buildBoard();
      this.root.addEventListener("keydown", (event) => this.onKey(event));
    }

    button(text, label, handler) {
      const button = el("button", "lc-button", text);
      button.type = "button";
      button.setAttribute("aria-label", label);
      button.title = label;
      button.addEventListener("click", handler);
      return button;
    }

    buildBoard() {
      let row = 1;
      let previousPhase = null;
      const placements = [];
      this.stages.forEach((stage, index) => {
        if (stage.phase !== previousPhase) {
          row += 1;
          placements.push({ type: "phase", phase: stage.phase, row });
          previousPhase = stage.phase;
        }
        row += 1;
        placements.push({ type: "stage", index, row });
      });
      const totalRows = row;
      this.board.style.gridTemplateRows = `repeat(${totalRows}, auto)`;

      this.lanes.forEach((lane, index) => {
        const background = el("div", "lc-lane-bg");
        background.style.gridColumn = String(index + 1);
        background.style.gridRow = `1 / ${totalRows + 1}`;
        background.style.setProperty("--lane", this.laneColor(lane.id));
        this.board.append(background);

        const head = el("div", "lc-lane-head");
        head.style.gridColumn = String(index + 1);
        head.style.gridRow = "1";
        head.style.setProperty("--lane", this.laneColor(lane.id));
        head.append(el("strong", "", lane.label));
        if (lane.sublabel) {
          head.append(el("small", "", lane.sublabel));
        }
        this.board.append(head);
      });

      this.markerId = `${this.root.id || "lc"}-arrow`;
      this.wires = svg("svg", { class: "lc-wires", "aria-hidden": "true", focusable: "false" });
      const defs = svg("defs");
      const marker = svg("marker", {
        id: this.markerId,
        viewBox: "0 0 10 10",
        refX: "8",
        refY: "5",
        markerWidth: "7",
        markerHeight: "7",
        orient: "auto-start-reverse",
      });
      marker.append(svg("path", { d: "M 0 0 L 10 5 L 0 10 z", class: "lc-arrowhead" }));
      defs.append(marker);
      this.wires.append(defs);
      this.board.append(this.wires);

      for (const placement of placements) {
        if (placement.type === "phase") {
          const phase = this.phaseById.get(placement.phase) || { label: placement.phase };
          const phaseRow = el("div", `lc-phase-row${phase.repeat ? " is-repeat" : ""}`);
          phaseRow.style.gridRow = String(placement.row);
          phaseRow.append(el("span", "", phase.repeat ? `↻ ${phase.label}` : phase.label));
          const note = phase.note || (phase.repeat ? "runs once per generated token; iteration 1 is prefill" : "");
          if (note) {
            phaseRow.append(el("small", "", note));
          }
          this.board.append(phaseRow);
          continue;
        }
        const stage = this.stages[placement.index];
        const laneNumber = (this.laneIndex.get(stage.lane) ?? 0) + 1;
        const card = el("button", "lc-stage");
        card.type = "button";
        card.style.gridColumn = String(laneNumber);
        card.style.gridRow = String(placement.row);
        card.style.setProperty("--lane", this.laneColor(stage.lane));
        card.dataset.stage = stage.id;
        const top = el("span", "lc-stage-top");
        top.append(el("span", "lc-stage-num", String(placement.index + 1)));
        top.append(el("span", "lc-stage-lane", this.lanes[laneNumber - 1]?.label || stage.lane));
        card.append(top);
        card.append(el("strong", "lc-stage-title", stage.title));
        if (stage.summary) {
          const summary = el("span", "lc-stage-summary");
          summary.append(richText(stage.summary));
          card.append(summary);
        }
        const count = (stage.files || []).length;
        if (count) {
          card.append(el("span", "lc-stage-files", `${count} file${count === 1 ? "" : "s"}`));
        }
        card.addEventListener("click", () => this.jumpToStage(placement.index));
        this.cards[placement.index] = card;
        this.board.append(card);
      }

      this.token = el("div", "lc-token");
      this.token.setAttribute("aria-hidden", "true");
      this.tokenLabel = el("span", "lc-token-label", "request");
      this.token.append(el("span", "lc-token-dot"), this.tokenLabel);
      this.board.append(this.token);
    }

    observe() {
      const relayout = () => {
        const compact = this.viewport.clientWidth < Math.max(1, this.lanes.length) * 165;
        this.root.classList.toggle("lc-compact", compact);
        this.drawWires();
        this.placeToken(true);
      };
      if ("ResizeObserver" in window) {
        new ResizeObserver(() => window.requestAnimationFrame(relayout)).observe(this.viewport);
      }
      window.addEventListener("load", relayout, { once: true });
      window.requestAnimationFrame(relayout);
    }

    anchor(card, side = "center") {
      const x = card.offsetLeft + card.offsetWidth / 2;
      const top = card.offsetTop;
      if (side === "top") {
        return { x, y: top };
      }
      if (side === "bottom") {
        return { x, y: top + card.offsetHeight };
      }
      if (side === "right") {
        return { x: card.offsetLeft + card.offsetWidth, y: top + card.offsetHeight / 2 };
      }
      return { x, y: top + card.offsetHeight / 2 };
    }

    drawWires() {
      for (const node of [...this.wires.querySelectorAll(".lc-wire, .lc-loop-text")]) {
        node.remove();
      }
      this.wires.setAttribute("width", String(this.board.scrollWidth));
      this.wires.setAttribute("height", String(this.board.scrollHeight));
      const markerUrl = `url(#${this.markerId})`;
      for (let index = 0; index < this.cards.length - 1; index += 1) {
        const from = this.cards[index];
        const to = this.cards[index + 1];
        if (!from || !to) {
          continue;
        }
        const start = this.anchor(from, "bottom");
        const end = this.anchor(to, "top");
        const bend = Math.max(18, (end.y - start.y) / 2);
        const path = svg("path", {
          class: `lc-wire${this.stages[index].transport ? " is-transport" : ""}`,
          d: `M ${start.x} ${start.y} C ${start.x} ${start.y + bend}, ${end.x} ${end.y - bend}, ${end.x} ${end.y - 2}`,
          "marker-end": markerUrl,
        });
        path.dataset.from = String(index);
        this.wires.append(path);
      }
      if (this.repeatStart >= 0 && this.repeatEnd > this.repeatStart) {
        const from = this.cards[this.repeatEnd];
        const to = this.cards[this.repeatStart];
        if (from && to) {
          // Route the loop through the board's reserved right gutter so it never crosses cards.
          const start = this.anchor(from, "right");
          const end = this.anchor(to, "right");
          const gutter = this.board.scrollWidth - 14;
          const radius = 12;
          this.wires.append(
            svg("path", {
              class: "lc-wire is-loop",
              d: [
                `M ${start.x} ${start.y}`,
                `H ${gutter - radius}`,
                `Q ${gutter} ${start.y} ${gutter} ${start.y - radius}`,
                `V ${end.y + radius}`,
                `Q ${gutter} ${end.y} ${gutter - radius} ${end.y}`,
                `H ${end.x + 3}`,
              ].join(" "),
              "marker-end": markerUrl,
            }),
          );
          const middle = (start.y + end.y) / 2;
          const label = svg("text", {
            x: String(gutter - 5),
            y: String(middle),
            class: "lc-loop-text",
            "text-anchor": "middle",
            transform: `rotate(-90 ${gutter - 5} ${middle})`,
          });
          label.textContent = "next token ↻";
          this.wires.append(label);
        }
      }
    }

    placeToken(immediate = false) {
      const entry = this.timeline[this.position];
      const card = entry && this.cards[entry.index];
      if (!card) {
        return;
      }
      const point = this.anchor(card, "top");
      this.token.classList.toggle("no-motion", immediate || reduceMotion.matches);
      this.token.style.transform = `translate(${point.x}px, ${point.y}px)`;
    }

    keepVisible(card) {
      const view = this.viewport;
      if (!card || view.scrollHeight <= view.clientHeight) {
        return;
      }
      const top = card.offsetTop - 56;
      const bottom = card.offsetTop + card.offsetHeight + 24;
      const behavior = reduceMotion.matches ? "auto" : "smooth";
      if (top < view.scrollTop) {
        view.scrollTo({ top, behavior });
      } else if (bottom > view.scrollTop + view.clientHeight) {
        view.scrollTo({ top: bottom - view.clientHeight, behavior });
      }
    }

    go(position, { immediate = false } = {}) {
      this.position = Math.min(Math.max(0, position), this.timeline.length - 1);
      const entry = this.timeline[this.position];
      const stage = this.stages[entry.index];
      const visited = new Set(this.timeline.slice(0, this.position).map((item) => item.index));

      this.cards.forEach((card, index) => {
        if (!card) {
          return;
        }
        const active = index === entry.index;
        card.classList.toggle("is-active", active);
        card.classList.toggle("is-visited", visited.has(index) && !active);
        if (active) {
          card.setAttribute("aria-current", "step");
        } else {
          card.removeAttribute("aria-current");
        }
      });
      for (const wire of this.wires.querySelectorAll("path.lc-wire")) {
        const isLoop = wire.classList.contains("is-loop");
        const looping = entry.index === this.repeatStart && entry.iteration > 1;
        wire.classList.toggle(
          "is-hot",
          isLoop ? looping : Number(wire.dataset.from) === entry.index - 1 && !looping,
        );
      }

      this.tokenLabel.textContent = this.tokenText(entry);
      this.placeToken(immediate);
      this.keepVisible(this.cards[entry.index]);
      this.renderStream(entry);
      this.renderDetail(stage, entry);
      this.renderTrail();
      this.progress.textContent = `Step ${this.position + 1} / ${this.timeline.length}`;
      this.prevButton.disabled = this.position === 0;
      this.nextButton.disabled = this.position === this.timeline.length - 1;
      if (this.position === this.timeline.length - 1) {
        this.pause();
      }
    }

    tokenText(entry) {
      if (entry.iteration === 0) {
        return "request";
      }
      if (entry.iteration > this.outputTokens.length) {
        return "done";
      }
      return entry.iteration === 1 ? "prefill → token 1" : `decode → token ${entry.iteration}`;
    }

    emittedCount() {
      let count = 0;
      for (let index = 0; index <= this.position; index += 1) {
        const entry = this.timeline[index];
        if (this.stages[entry.index].emits && entry.iteration > 0) {
          count = Math.max(count, entry.iteration);
        }
      }
      return Math.min(count, this.outputTokens.length);
    }

    renderStream(entry) {
      const count = this.emittedCount();
      this.output.replaceChildren();
      if (!count) {
        this.output.append(el("span", "lc-output-empty", "(nothing streamed yet)"));
      }
      this.outputTokens.slice(0, count).forEach((piece, index) => {
        const chip = el("span", `lc-chunk${index === count - 1 ? " is-new" : ""}`, piece);
        chip.title = `token ${index + 1}`;
        this.output.append(chip);
      });
      if (entry.iteration === 0) {
        this.iterationBadge.textContent = "admission";
      } else if (entry.iteration > this.outputTokens.length) {
        this.iterationBadge.textContent = "finished";
      } else {
        const kind = entry.iteration === 1 ? "prefill" : "decode";
        this.iterationBadge.textContent = `engine step ${entry.iteration} · ${kind}`;
      }
    }

    renderDetail(stage, entry) {
      const detail = this.detail;
      detail.replaceChildren();
      const kicker = el("div", "lc-detail-kicker");
      const lane = this.lanes[this.laneIndex.get(stage.lane) ?? 0];
      const laneTag = el("span", "lc-tag", lane?.label || stage.lane);
      laneTag.style.setProperty("--lane", this.laneColor(stage.lane));
      kicker.append(laneTag);
      const phase = this.phaseById.get(stage.phase);
      if (phase) {
        kicker.append(el("span", "lc-tag is-phase", phase.label));
      }
      detail.append(kicker);
      detail.append(el("h3", "", `${this.stages.indexOf(stage) + 1}. ${stage.title}`));

      if (stage.summary) {
        const summary = el("p", "lc-detail-summary");
        summary.append(richText(stage.summary));
        detail.append(summary);
      }
      const iterationNote =
        entry.iteration === 1 ? stage.prefill : entry.iteration > 1 ? stage.decode : null;
      if (iterationNote && entry.iteration <= this.outputTokens.length) {
        const note = el("p", "lc-iteration-note");
        note.append(
          el("strong", "", entry.iteration === 1 ? "In the prefill step: " : "In a decode step: "),
        );
        note.append(richText(iterationNote));
        detail.append(note);
      }
      for (const paragraph of [].concat(stage.detail || [])) {
        const p = el("p");
        p.append(richText(paragraph));
        detail.append(p);
      }
      if (stage.token) {
        detail.append(el("h4", "", "What the request/token looks like here"));
        detail.append(el("pre", "lc-shape", stage.token));
      }
      if (stage.files?.length) {
        detail.append(el("h4", "", "Files triggered at this step"));
        const list = el("ul", "lc-files");
        for (const file of stage.files) {
          const item = el("li");
          const link = el("a", "lc-file-path", file.path);
          link.href = this.fileUrl(file);
          link.target = "_blank";
          link.rel = "noopener";
          item.append(link);
          if (file.symbol) {
            item.append(el("code", "lc-file-symbol", file.symbol));
          }
          if (file.role) {
            const role = el("span", "lc-file-role");
            role.append(richText(file.role));
            item.append(role);
          }
          list.append(item);
        }
        detail.append(list);
      }
      if (stage.features?.length) {
        detail.append(el("h4", "", "Features involved"));
        const chips = el("div", "lc-features");
        for (const feature of stage.features) {
          const label = typeof feature === "string" ? feature : feature.label;
          const href = typeof feature === "string" ? null : feature.href;
          const chip = el(href ? "a" : "span", "lc-feature", label);
          if (href) {
            chip.href = href;
          }
          chips.append(chip);
        }
        detail.append(chips);
      }
      if (stage.transport) {
        const handoff = el("p", "lc-handoff");
        handoff.append(el("strong", "", "Hand-off: "));
        handoff.append(richText(stage.transport));
        detail.append(handoff);
      }
    }

    renderTrail() {
      const seen = new Map();
      for (let index = 0; index <= this.position; index += 1) {
        const stage = this.stages[this.timeline[index].index];
        for (const file of stage.files || []) {
          if (!seen.has(file.path)) {
            seen.set(file.path, { file, stage });
          }
        }
      }
      this.trailSummary.textContent = `Files touched so far: ${seen.size}`;
      this.trailList.replaceChildren();
      for (const { file, stage } of seen.values()) {
        const item = el("li");
        item.style.setProperty("--lane", this.laneColor(stage.lane));
        const link = el("a", "", file.path);
        link.href = this.fileUrl(file);
        link.target = "_blank";
        link.rel = "noopener";
        item.append(link);
        this.trailList.append(item);
      }
    }

    jumpToStage(index) {
      const current = this.timeline[this.position];
      let target = this.timeline.findIndex(
        (entry) => entry.index === index && entry.iteration === current.iteration,
      );
      if (target < 0) {
        target = this.timeline.findIndex((entry) => entry.index === index);
      }
      this.pause();
      this.go(target);
    }

    next() {
      if (this.position < this.timeline.length - 1) {
        this.go(this.position + 1);
      }
    }

    previous() {
      if (this.position > 0) {
        this.go(this.position - 1);
      }
    }

    schedule() {
      window.clearInterval(this.timer);
      this.timer = window.setInterval(() => {
        if (this.position >= this.timeline.length - 1) {
          this.pause();
          return;
        }
        this.next();
      }, SPEEDS[this.speed.value] || SPEEDS["1"]);
    }

    toggle() {
      if (this.playing) {
        this.pause();
        return;
      }
      if (this.position >= this.timeline.length - 1) {
        this.go(0);
      }
      this.playing = true;
      this.playButton.textContent = "❚❚ Pause";
      this.playButton.setAttribute("aria-pressed", "true");
      this.root.classList.add("is-playing");
      this.schedule();
    }

    pause() {
      this.playing = false;
      window.clearInterval(this.timer);
      if (this.playButton) {
        this.playButton.textContent = "▶ Play";
        this.playButton.setAttribute("aria-pressed", "false");
      }
      this.root.classList.remove("is-playing");
    }

    onKey(event) {
      if (event.target.closest("select, input, textarea")) {
        return;
      }
      const actions = {
        ArrowRight: () => this.next(),
        ArrowLeft: () => this.previous(),
        Home: () => this.go(0),
        End: () => this.go(this.timeline.length - 1),
      };
      const action = actions[event.key];
      if (action) {
        event.preventDefault();
        this.pause();
        action();
      }
    }
  }

  const start = () => {
    document.querySelectorAll("[data-lifecycle]").forEach((root) => {
      const source = document.getElementById(root.dataset.lifecycle);
      try {
        const data = JSON.parse(source?.textContent || "");
        root.lifecycle = new TokenLifecycle(root, data);
      } catch (error) {
        root.replaceChildren(
          el("p", "lc-error", `Lifecycle diagram could not be rendered: ${error.message}`),
        );
      }
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();
