"use strict";

const wireForm = document.getElementById("wire-form");
const wireResults = document.getElementById("wire-results");
const wireStatus = document.getElementById("wire-status");
const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 5 });

function quantity(value) {
  return value > 0 && value < 0.00001 ? value.toExponential(2) : numberFormat.format(value);
}

function showLinkBudget() {
  const direct = wireForm.elements.gate.value === "direct";
  wireForm.elements.experts.disabled = !direct;
  const invalid = Array.from(wireForm.elements).find(
    (field) => field.willValidate && !field.validity.valid
  );
  wireResults.hidden = Boolean(invalid);
  wireStatus.toggleAttribute("data-warning", Boolean(invalid));
  if (invalid) {
    wireStatus.textContent = invalid.labels[0].textContent.trim() + " — " + invalid.validationMessage;
    return;
  }

  const inputs = Object.fromEntries(new FormData(wireForm));
  const pdMiB = inputs.prompt * inputs.kvKiB / 1024;
  const rows = inputs.tokens * (direct ? inputs.experts : 1);
  const dispatchKiB = rows * inputs.width * inputs.bytes / 1024;
  const afdMiB = 2 * inputs.layers * dispatchKiB / 1024;
  const pdGiB = pdMiB / 1024;
  const afdGiB = afdMiB / 1024;
  const pdTime = quantity(1000 * pdGiB / inputs.bandwidth);
  const afdTime = quantity(1000 * afdGiB / inputs.bandwidth);
  const rate = quantity(Number(inputs.bandwidth)) + " GiB/s.";

  document.getElementById("pd-amount").textContent = quantity(pdMiB) + " MiB";
  document.getElementById("pd-duration").textContent = pdTime + " ms of payload transfer at " + rate;
  document.getElementById("pd-rate").textContent = quantity(Number(inputs.handoffs))
    + " handoffs/s require " + quantity(inputs.handoffs * pdGiB) + " GiB/s.";
  document.getElementById("afd-amount").textContent = quantity(afdMiB) + " MiB";
  document.getElementById("afd-duration").textContent = afdTime + " ms of serialized payload transfer at " + rate;
  document.getElementById("afd-rate").textContent = quantity(Number(inputs.steps))
    + " steps/s require " + quantity(inputs.steps * afdGiB) + " GiB/s across both directions.";
  const rowDescription = direct ? "one row per expert assignment." : "one row per input token.";
  const layerText = Number(inputs.layers) === 1
    ? "One split layer dispatches "
    : quantity(Number(inputs.layers)) + " split layers each dispatch ";
  wireStatus.textContent = layerText + quantity(dispatchKiB) + " KiB outward and the same amount back, with "
    + rowDescription + " Payload times are " + pdTime + " ms for PD and " + afdTime + " ms for AFD.";
}

wireForm.addEventListener("input", showLinkBudget);
wireForm.addEventListener("reset", () => requestAnimationFrame(showLinkBudget));
wireForm.addEventListener("submit", (event) => event.preventDefault());
window.addEventListener("pageshow", showLinkBudget);
wireForm.hidden = false;
showLinkBudget();
