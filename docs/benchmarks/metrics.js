(() => {
  "use strict";

  const byId = (id) => document.getElementById(id);
  const display = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
  const fmt = (value) => display.format(value);
  const ms = (value) => fmt(value) + " ms";
  const text = (id, value) => { byId(id).textContent = value; };

  const patterns = {
    steady: { queue: 50, prefill: 80, tail: 20, samples: [0,40,80,120], chunks: [1,1,1,1],
      note: "One output token arrives per content callback. Compare the regular callback gaps with the server decode window." },
    queue: { queue: 350, prefill: 80, tail: 20, samples: [0,40,80,120], chunks: [1,1,1,1],
      note: "With the preset durations, queueing adds 300 ms to TTFT versus the steady trace. The same four tokens still arrive at 40 ms intervals." },
    stall: { queue: 50, prefill: 80, tail: 20, samples: [0,40,240,280], chunks: [1,1,1,1],
      note: "The 200 ms gap is an actual server-side stall in this example. A single average smooths over the visible interruption." },
    bundle: { queue: 50, prefill: 80, tail: 20, samples: [0,25,50,75,100], chunks: [1,3,1],
      note: "The middle callback flushes three token IDs together. Its contents contribute three tokens to throughput and just one observed arrival." },
    firstBundle: { queue: 50, prefill: 80, tail: 20, samples: [0,25,50,75], chunks: [3,1],
      note: "The first flush waits for three tokens. Only one token arrives after that flush. Compare the content-window TPOT denominator with the post-first-delivery rate." },
    single: { queue: 50, prefill: 80, tail: 20, samples: [0], chunks: [1],
      note: "A one-token response has a measurable TTFT and completion time, and zero adjacent content gaps. Its post-first-token average is undefined." },
    tail: { queue: 50, prefill: 80, tail: 280, samples: [0,40,80,120], chunks: [1,1,1,1],
      note: "The completion marker arrives well after the final text at the preset tail. The full-response TPOT includes that wait, while content-window TPOT stops at last text." }
  };
  const loadProfiles = {
    calm: { attempts: 40, groups: [[36,128,8,100,30],[4,256,16,250,45]] },
    busy: { attempts: 100, groups: [[72,128,8,150,40],[16,256,16,300,50],[7,512,24,700,90],[1,512,24,1600,90]] },
    overload: { attempts: 160, groups: [[45,128,8,200,40],[45,256,16,650,65],[20,512,24,1200,100]] },
    failed: { attempts: 20, groups: [] }
  };

  function checkForm(form, errorBox) {
    const invalid = Array.from(form.elements).find((field) => field.willValidate && !field.validity.valid);
    errorBox.hidden = !invalid;
    errorBox.textContent = invalid ? invalid.labels[0].textContent.trim() + " — " + invalid.validationMessage : "";
    return !invalid;
  }

  function fillRows(body, rows, rowHeadings = false) {
    body.replaceChildren(...rows.map((values) => {
      const row = document.createElement("tr");
      values.forEach((value, i) => {
        const heading = rowHeadings && i === 0;
        const cell = document.createElement(heading ? "th" : "td");
        if (heading) cell.scope = "row";
        cell.textContent = value;
        row.append(cell);
      });
      return row;
    }));
  }

  function fillEquations(target, equations) {
    const cells = target.querySelectorAll("dd");
    equations.forEach((equation, index) => { cells[index].textContent = equation; });
  }

  const traceForm = byId("trace-form");
  function renderTrace() {
    const valid = checkForm(traceForm, byId("trace-error"));
    byId("trace-results").hidden = !valid;
    byId("latency-math").hidden = !valid;
    byId("latency-invalid").hidden = valid;
    if (!valid) return;
    const controls = traceForm.elements;
    const pattern = patterns[controls.pattern.value];
    const queue = controls.queue.valueAsNumber;
    const prefill = controls.prefill.valueAsNumber;
    const tail = controls.tail.valueAsNumber;
    const receive = 10;
    const scheduled = receive + queue;
    const firstSample = scheduled + prefill;
    const samples = pattern.samples.map((offset) => firstSample + offset);
    const lastSample = samples.at(-1);
    let emitted = 0;
    const arrivals = pattern.chunks.map((count) => {
      emitted += count;
      return samples[emitted - 1] + 10;
    });
    const first = arrivals[0];
    const last = arrivals.at(-1);
    const done = last + tail;
    const stream = last - first;
    const gaps = arrivals.slice(1).map((arrival, i) => arrival - arrivals[i]);
    const firstCount = pattern.chunks[0];

    text("first", ms(first));
    text("last", ms(last));
    text("done", ms(done));
    const labels = ["Ingress", "Queue", "Prefill", "First delivery", "Streaming", "Tail"];
    const durations = [receive, queue, prefill, first - firstSample, stream, tail];
    document.querySelectorAll(".phase-meter i").forEach((seg, i) => { seg.style.flexGrow = durations[i]; });
    byId("phase-legend").querySelectorAll("li").forEach((label, i) => {
      label.textContent = labels[i] + " " + ms(durations[i]);
    });
    const log = [["Send request","0 ms",0,"—"],["Role only","12 ms",0,"—"]];
    arrivals.forEach((arrival, i) => log.push(["Content " + (i + 1), ms(arrival), pattern.chunks[i], i ? ms(gaps[i-1]) : "—"]));
    log.push(["Done", ms(done), 0, "—"]);
    fillRows(byId("event-rows"), log, true);
    text("server-spans", "Server receives at " + ms(receive) + ", schedules prefill at " + ms(scheduled)
      + ", samples token 1 at " + ms(firstSample) + " and samples the last token at " + ms(lastSample)
      + ". The server's decode window after the first token is " + ms(lastSample - firstSample) + ".");
    text("trace-note", pattern.note);
    let gapEquation = "No pair of content callbacks. Mean gap is undefined.";
    let contentEquation = "N − 1 = 0. Content TPOT is undefined for one output token.";
    let fullEquation = "N − 1 = 0. Completion-window TPOT is undefined for one output token.";
    let userRate = "No post-first-delivery interval. This rate is undefined.";
    if (gaps.length) {
      gapEquation = "Gapᵢ = current − previous content arrival. Mean = (" + gaps.map(fmt).join(" + ") + ") / "
        + gaps.length + " = " + fmt(stream / gaps.length) + " ms/gap.";
    }
    if (emitted > 1) {
      contentEquation = "(Last content − first content) / (output tokens − 1) = (" + fmt(last) + " − "
        + fmt(first) + ") / (" + emitted + " − 1) = " + fmt(stream / (emitted - 1)) + " ms/token.";
      fullEquation = "(Done − first content) / (output tokens − 1) = (" + fmt(done) + " − "
        + fmt(first) + ") / (" + emitted + " − 1) = " + fmt((done - first) / (emitted - 1)) + " ms/token.";
    }
    if (stream > 0) {
      userRate = "Tokens delivered after first callback / streaming seconds = (" + emitted + " − " + firstCount
        + ") / " + (stream / 1000) + " = " + fmt(1000 * (emitted - firstCount) / stream) + " tok/s.";
    }
    fillEquations(byId("latency-math"), [
      "First content − send = " + fmt(first) + " − 0 = " + ms(first) + ".",
      "Last content − send = " + fmt(last) + " − 0 = " + ms(last) + ".",
      "Done − send = " + fmt(done) + " − 0 = " + ms(done) + ". Completion tail = " + fmt(done) + " − " + fmt(last) + " = " + ms(tail) + ".",
      gapEquation, contentEquation, fullEquation, userRate,
      "Output tokens / request seconds = " + emitted + " / " + (done / 1000) + " = " + fmt(1000 * emitted / done) + " tok/s.",
      "Queue = " + fmt(scheduled) + " − " + receive + " = " + ms(queue) + ". Prefill to first sample = "
        + fmt(firstSample) + " − " + fmt(scheduled) + " = " + ms(prefill) + ". Last sample − first sample = "
        + fmt(lastSample) + " − " + fmt(firstSample) + " = " + ms(lastSample - firstSample) + "."
    ]);
  }

  traceForm.addEventListener("input", (event) => {
    if (event.target.name === "pattern") {
      const pattern = patterns[event.target.value];
      for (const name of ["queue", "prefill", "tail"]) traceForm.elements[name].value = pattern[name];
    }
    renderTrace();
  });

  const loadForm = byId("load-form");
  function renderLoad() {
    const valid = checkForm(loadForm, byId("load-error"));
    byId("load-results").hidden = !valid;
    if (!valid) {
      byId("percentile-math").replaceChildren();
      const note = document.createElement("li");
      note.textContent = "Enter valid target values above to restore the load example.";
      byId("percentile-math").append(note);
      return;
    }
    const controls = loadForm.elements;
    const { attempts, groups } = loadProfiles[controls.load.value];
    const windowSeconds = 10;
    let good = 0;
    let successes = 0;
    let inputs = 0;
    let outputs = 0;
    const times = [];
    const rows = groups.map(([count, prompt, output, ttft, tpot]) => {
      const qualifies = ttft <= controls.ttft.valueAsNumber && tpot <= controls.tpot.valueAsNumber;
      successes += count;
      inputs += count * prompt;
      outputs += count * output;
      if (qualifies) good += count;
      times.push(...Array(count).fill(ttft));
      return [count, prompt + " / " + output, ms(ttft), fmt(tpot) + " ms/token", qualifies ? "Yes" : "No"];
    });
    const failures = attempts - successes;
    const goodRate = fmt(good / windowSeconds) + " req/s";
    const successRate = fmt(successes / windowSeconds) + " req/s";
    const attainment = fmt(100 * good / attempts) + "%";
    text("success-rate", successRate);
    text("good-rate", goodRate);
    text("attainment", attainment);
    fillRows(byId("cohorts"), rows.length ? rows : [["0","—","—","—","—"]]);
    text("load-note", attempts + " attempts, " + successes + " successes, " + failures + " failures. "
      + good + " successful requests meet both inclusive targets.");
    fillEquations(byId("rate-math"), [
      "Attempts / window = " + attempts + " / 10 = " + fmt(attempts / windowSeconds) + " req/s.",
      "Successes / window = " + successes + " / 10 = " + successRate + ".",
      "Successful input tokens / window = " + fmt(inputs) + " / 10 = " + fmt(inputs / windowSeconds) + " tok/s.",
      "Successful output tokens / window = " + fmt(outputs) + " / 10 = " + fmt(outputs / windowSeconds) + " tok/s.",
      "(Input + output tokens) / window = (" + fmt(inputs) + " + " + fmt(outputs) + ") / 10 = " + fmt((inputs + outputs) / windowSeconds) + " tok/s.",
      "Qualifying successes / window = " + good + " / 10 = " + goodRate + ". Qualifying successes / all attempts = "
        + good + " / " + attempts + " × 100% = " + attainment + ".",
      "Failures / all attempts = " + failures + " / " + attempts + " × 100% = " + fmt(100 * failures / attempts) + "%."
    ]);
    times.sort((a, b) => a - b);
    byId("percentile-math").replaceChildren(...[50,95,99].map((p) => {
      const li = document.createElement("li");
      const rank = Math.ceil((p / 100) * times.length);
      li.textContent = rank === 0 ? "p" + p + " is undefined. No successful latency observations."
        : "p" + p + " = sample ⌈" + (p / 100).toFixed(2) + " × " + times.length + "⌉ = sample " + rank + " = " + ms(times[rank - 1]) + ".";
      return li;
    }));
  }
  loadForm.addEventListener("input", renderLoad);
  [traceForm,loadForm].forEach((form) => {
    form.addEventListener("submit", (event) => event.preventDefault());
    form.hidden = false;
  });
  window.addEventListener("pageshow", () => { renderTrace(); renderLoad(); });
  renderTrace();
  renderLoad();
})();
