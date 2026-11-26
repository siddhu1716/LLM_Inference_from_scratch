(() => {
  "use strict";

  // Raw top-three logits after each forward pass of cli._tiny_model(seed=7).
  const recordings = {
    Hi: [
      [[104, 2.7758], [173, 2.6726], [74, 2.0050]],
      [[4, 2.9804], [218, 2.7157], [212, 2.6789]],
      [[173, 2.8398], [4, 2.8114], [104, 2.4840]],
    ],
    "π": [
      [[65, 3.1315], [94, 2.9075], [231, 2.8336]],
      [[142, 2.8501], [231, 2.3478], [65, 2.2316]],
      [[142, 3.0099], [172, 2.1505], [115, 1.9362]],
    ],
    GPU: [
      [[104, 2.6270], [182, 2.5520], [228, 2.1989]],
      [[188, 3.2445], [249, 2.8713], [226, 2.6892]],
      [[188, 4.3928], [19, 2.5780], [41, 2.3415]],
    ],
  };
  const prompt = document.querySelector("#sample-prompt");
  const buttons = [...document.querySelectorAll("[data-pass]")];
  let pass = 0;

  function render() {
    const text = prompt.value;
    const ids = [...new TextEncoder().encode(text)];
    const ranking = recordings[text][pass];
    const generated = recordings[text].slice(0, pass + 1).map((step) => step[0][0]);
    const chosen = generated[pass];
    const inputs = pass === 0 ? ids : [generated[pass - 1]];
    const before = pass === 0 ? 0 : ids.length + pass - 1;
    const after = before + inputs.length;

    document.querySelector("#replay-input").textContent = inputs.join(" · ");
    document.querySelector("#replay-cache").textContent = `${before} → ${after}`;
    document.querySelector("#replay-token").textContent = chosen;
    document.querySelector("#replay-generated").textContent = `[${generated.join(", ")}]`;
    document.querySelector("#replay-explain").textContent = pass === 0
      ? `The byte tokenizer maps “${text}” to ${ids.join(", ")}. Prefill processes all ${ids.length} positions. The last position scores all 256 possible next bytes and ID ${chosen} wins.`
      : `Decode takes the previous winner, ID ${inputs[0]}, and reuses ${before} cached positions. One new position is added and ID ${chosen} wins.${pass === 2 ? " The request has now generated its three requested tokens." : " That ID becomes the input for the next pass."}`;
    document.querySelector("#replay-scores").textContent =
      ranking.map(([id, score]) => `${id} (${score.toFixed(4)})`).join(" · ");
    document.querySelector("#replay-command").textContent =
      `mini-infer sample "${text}" --seed 7 --temperature 0 --max-new-tokens 3`;
    for (const button of buttons) {
      button.setAttribute("aria-pressed", String(Number(button.dataset.pass) === pass));
    }
  }

  for (const button of buttons) {
    button.addEventListener("click", () => {
      pass = Number(button.dataset.pass);
      render();
    });
  }
  prompt.addEventListener("change", () => {
    pass = 0;
    render();
  });
  render();
  document.querySelector(".replay-controls").hidden = false;
})();
