(() => {
  "use strict";
  const lab = document.querySelector(".score-lab");
  if (!lab) return;
  const picker = lab.querySelector("select");
  const panels = [...lab.querySelectorAll("[data-rule]")];
  const percent = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 });
  const pct = (part) => percent.format(part);

  function samplePassRate(n, c, k) {
    if (k > n - c) return 1;
    let allFail = 1;
    for (let draw = 0; draw < k; draw++) {
      allFail *= (n - c - draw) / (n - draw);
    }
    return 1 - allFail;
  }

  function calculate(rule, v) {
    switch (rule) {
      case "accuracy":
        if (v.correct > v.total) throw new Error("Passed tasks cannot exceed evaluated tasks.");
        return v.correct + " / " + v.total + " = " + pct(v.correct / v.total) + " of tasks passed.";
      case "pass":
        if (v.c > v.n || v.k > v.n) throw new Error("Passing samples and attempts must each fit within the sample count.");
        return "pass@" + v.k + " = " + pct(samplePassRate(v.n, v.c, v.k)) +
          ". A single draw from these samples passes " + pct(v.c / v.n) + " of the time.";
      case "f1": {
        if (v.match > Math.min(v.predicted, v.reference)) throw new Error("Overlap cannot exceed either token count.");
        if (!v.predicted || !v.reference) {
          return "F1 " + pct(Number(v.predicted === v.reference)) +
            ". Applying the SQuAD v2 empty-answer rule.";
        }
        return "Precision " + pct(v.match / v.predicted) + " · Recall " + pct(v.match / v.reference) +
          " · F1 " + pct(2 * v.match / (v.predicted + v.reference)) + ".";
      }
      case "bins": {
        if (v.shortPass > v.shortTotal || v.longPass > v.longTotal) throw new Error("Correct items cannot exceed the number of items in their group.");
        const short = v.shortPass / v.shortTotal;
        const long = v.longPass / v.longTotal;
        return "Short " + pct(short) + " · Long " + pct(long) +
          " · Pooled " + pct((v.shortPass + v.longPass) / (v.shortTotal + v.longTotal)) +
          " · Macro " + pct((short + long) / 2) + ".";
      }
    }
  }

  function update(panel) {
    const result = panel.querySelector(".score-result");
    try {
      const values = {};
      for (const field of panel.querySelectorAll("input")) {
        if (!field.value || !field.validity.valid || !Number.isInteger(field.valueAsNumber)) {
          throw new Error("For " + field.labels[0].textContent.toLowerCase() +
            ", enter a whole number between " + field.min + " and " + field.max + ".");
        }
        values[field.name] = field.valueAsNumber;
      }
      result.textContent = calculate(panel.dataset.rule, values);
      result.classList.remove("is-invalid");
    } catch (error) {
      result.textContent = error.message;
      result.classList.add("is-invalid");
    }
  }

  function selectRule() {
    for (const panel of panels) {
      panel.hidden = panel.dataset.rule !== picker.value;
      if (!panel.hidden) update(panel);
    }
  }

  lab.querySelector(".score-picker").hidden = false;
  for (const panel of panels) {
    panel.querySelector(".score-fields").hidden = false;
    panel.addEventListener("input", () => update(panel));
  }
  picker.addEventListener("change", selectRule);
  selectRule();
})();
