(() => {
  "use strict";
  const controls = document.querySelector("#model-inputs");
  const models = new Map(Array.from(document.querySelectorAll("tr[data-key]"), (row) => [row.dataset.key, row]));
  const number = new Intl.NumberFormat("en-US", { maximumFractionDigits: 3 });

  function showModel(slot, positions) {
    const row = models.get(document.getElementById("model-" + slot).value);
    const shape = row.dataset;
    const source = row.querySelector("a");
    const card = document.querySelector(`[data-result="${slot}"]`);
    const field = (name) => card.querySelector(`[data-field="${name}"]`);
    const width = shape.latent
      ? Number(shape.latent) + Number(shape.rotary)
      : 2 * Number(shape.kvHeads) * Number(shape.headDim);
    const perToken = Number(shape.layers) * width * 2;
    const totalMiB = perToken * positions / (1024 ** 2);
    const dimensions = shape.latent
      ? `(${shape.latent} latent + ${shape.rotary} rotary) values`
      : `2 tensors × ${shape.kvHeads} KV heads × ${shape.headDim} values`;
    field("family").textContent = shape.family;
    field("name").textContent = source.textContent;
    field("expression").textContent = `${shape.layers} layers × ${dimensions} × 2 B`;
    field("token").textContent = number.format(perToken / 1024) + " KiB";
    field("batch").textContent = number.format(totalMiB) + " MiB";
    field("config").href = source.href;
    return source.textContent + " uses " + number.format(totalMiB) + " MiB";
  }

  function update() {
    const context = Number(document.querySelector("#context").value);
    const sequences = Number(document.querySelector("#sequences").value);
    const positions = context * sequences;
    const a = showModel("a", positions);
    const b = showModel("b", positions);
    const load = `${number.format(context)} positions in each of ${sequences} ${sequences === 1 ? "sequence" : "sequences"}.`;
    document.querySelector("#comparison").textContent = `${load} ${a}. ${b}.`;
  }

  controls.addEventListener("change", update);
  update();
  controls.hidden = false;
})();
