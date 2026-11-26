(() => {
  "use strict";

  const lab = document.querySelector("#address-lab");
  const layout = lab.querySelector("#address-layout");
  const prefix = lab.querySelector("#address-prefix");
  const position = lab.querySelector("#address-position");
  const output = (id, text) => { document.getElementById(id).textContent = text; };
  const first = [..."ABCDEFGHIJ"];
  const prompts = { six: [..."ABCDEFXYZK"], same: [...first], none: [..."QBCDEFXYZK"] };
  const originalPages = [7, 2, 9];
  const freshPages = [6, 5, 3];
  const originalTokens = [31, 8, 91, 42, 17, 63, 71, 39, 90, 96];
  const freshTokens = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109];

  function render() {
    const isVllm = layout.value === "vllm";
    const size = layout.value === "sg-token" ? 1 : 4;
    const second = prompts[prefix.value];
    const divergence = first.findIndex((token, i) => token !== second[i]);
    const match = divergence < 0 ? first.length : divergence;
    // Ordinary generation needs logits at the final input position.
    const reusable = Math.floor(Math.min(match, first.length - 1) / size) * size;
    const p = Number(position.value);
    const logicalPage = Math.floor(p / size);
    const offset = p % size;
    const pages = originalPages.map((oldId, i) => i * size < reusable ? oldId : freshPages[i]);
    const row = second.map((_, i) => {
      if (size === 1) return i < reusable ? originalTokens[i] : freshTokens[i];
      return pages[Math.floor(i / size)] * size + i % size;
    });
    const loc = row[p];
    output("address-value", String(p));
    output("address-token", `ID ${second[p]} · position ${p}`);
    output("address-lookup", isVllm ? `table[B][${logicalPage}] = ${pages[logicalPage]}` : `req_to_token[8, ${p}] = ${loc}`);
    output("address-result", isVllm ? `${pages[logicalPage]} × 4 + ${offset} = ${loc}` : `K[layer][${loc}], V[layer][${loc}]`);
    output("address-explain", isVllm
      ? `Floor(${p} / 4) selects logical block ${logicalPage}. Its table entry is physical block ${pages[logicalPage]}. The remainder ${offset} selects the token slot within it.`
      : size === 1
        ? `Request B owns row 8. Its entry at column ${p} contains device slot ${loc}. The dense attention layer uses that location in its own key and value buffers.`
        : `The allocator chose physical page ${pages[logicalPage]}. It expanded this page to token locations before writing the row. Column ${p} already stores ${loc}, including offset ${offset}.`);
    output("address-reuse", `${match} matching IDs · ${reusable} reused positions · ${second.length - reusable} prompt positions to compute`);
    output("address-a", isVllm ? `A blocks [${originalPages.join(", ")}]` : size === 1
      ? `row 3 [${originalTokens.join(", ")}]`
      : `row 3 [${first.map((_, i) => originalPages[Math.floor(i / 4)] * 4 + i % 4).join(", ")}]`);
    output("address-b", isVllm ? `B blocks [${pages.join(", ")}]` : `row 8 [${row.join(", ")}]`);
    const strip = document.getElementById("address-strip");
    strip.replaceChildren(...second.map((token, i) => {
      const cell = document.createElement("span");
      cell.className = i < reusable ? "reused" : "fresh";
      if (i === p) cell.setAttribute("aria-current", "true");
      const mark = document.createElement("small");
      mark.textContent = String(i);
      cell.append(mark, token);
      return cell;
    }));
  }

  for (const control of [layout, prefix, position]) control.addEventListener("input", render);
  render();
  document.getElementById("address-controls").hidden = false;
})();
