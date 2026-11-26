"use strict";

const capacityForm = document.getElementById("capacity-form");
const budgetStatus = document.getElementById("capacity-status");
const memoryResults = document.getElementById("memory-results");
const gibibyte = 2 ** 30;

function formatGiB(value) {
  if (value > 0 && value < 0.01) return "<0.01 GiB";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }) + " GiB";
}

function updateCapacity() {
  const invalid = Array.from(capacityForm.elements).find(
    (field) => field.willValidate && !field.validity.valid
  );
  memoryResults.hidden = Boolean(invalid);
  if (invalid) {
    const label = invalid.labels[0].textContent.trim();
    budgetStatus.textContent = label + " — " + invalid.validationMessage;
    budgetStatus.toggleAttribute("data-warning", true);
    return;
  }

  const inputs = Object.fromEntries(new FormData(capacityForm));
  const weightGiB = inputs.parameters * 1e9 * inputs.weightBytes / gibibyte;
  const bytesPerToken = 2 * inputs.layers * inputs.heads * inputs.dimension * inputs.kvBytes;
  const sequenceGiB = bytesPerToken * inputs.tokens / gibibyte;
  const kvGiB = sequenceGiB * inputs.sequences;
  const reserveGiB = Number(inputs.reserve);
  const kvBudget = inputs.memory - reserveGiB - weightGiB;
  const remaining = kvBudget - kvGiB;
  const limit = Math.max(0, Math.floor(kvBudget / sequenceGiB));
  const fits = remaining >= 0;

  document.getElementById("out-weights").textContent = formatGiB(weightGiB);
  document.getElementById("out-kv").textContent = formatGiB(kvGiB);
  document.getElementById("out-reserve").textContent = formatGiB(reserveGiB);
  document.getElementById("out-gap").textContent = formatGiB(Math.abs(remaining));
  document.getElementById("gap-label").textContent = fits ? "Headroom" : "Shortfall";
  document.getElementById("bar-weights").style.flexGrow = weightGiB;
  document.getElementById("bar-kv").style.flexGrow = kvGiB;
  document.getElementById("bar-reserve").style.flexGrow = reserveGiB;
  document.getElementById("bar-free").style.flexGrow = Math.max(0, remaining);

  const status = fits ? "The estimate fits." : "The requested allocation exceeds device memory.";
  const kibPerToken = (bytesPerToken / 1024).toLocaleString("en-US", { maximumFractionDigits: 6 });
  budgetStatus.textContent = status + " At " + kibPerToken + " KiB per token and "
    + formatGiB(sequenceGiB) + " per sequence, the memory budget holds up to "
    + limit.toLocaleString("en-US") + (limit === 1 ? " such sequence." : " such sequences.");
  budgetStatus.toggleAttribute("data-warning", !fits);
}

capacityForm.addEventListener("input", updateCapacity);
capacityForm.addEventListener("submit", (event) => event.preventDefault());
capacityForm.addEventListener("reset", () => requestAnimationFrame(updateCapacity));
capacityForm.hidden = false;
updateCapacity();

const strategyControl = document.getElementById("strategy-control");
const strategies = document.querySelectorAll("[data-strategy]");
function showStrategy() {
  const selected = strategyControl.querySelector("input:checked").value;
  for (const strategy of strategies) {
    strategy.hidden = strategy.dataset.strategy !== selected;
  }
}
strategyControl.addEventListener("change", showStrategy);
strategyControl.hidden = false;
showStrategy();

window.addEventListener("pageshow", () => {
  updateCapacity();
  showStrategy();
});
