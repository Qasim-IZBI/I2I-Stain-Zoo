# Figures and numbers the WACV paper needs

> Written 2026-08-20. A handover **from the manuscript to this repository**, the reverse
> direction of `MANUSCRIPT_UPDATES.md`. It says what the paper actually asks for, which is
> less than this repository can produce and in two places is not quite what it currently
> produces.
>
> Manuscript workspace: `~/Desktop/Manuscript/Qasim/Uncertainty_decomposition/`.
> Drop finished files into its `figures/` directory. Nothing else in that workspace should
> be edited from here.

---

## 0. Read this first — the paper is much smaller than the pipeline

Between 2026-08-19 and 2026-08-20 the manuscript shed most of what this repository can
measure. Producing a figure for any of the following is wasted work:

| Gone from the paper | Consequence for plotting |
|---|---|
| **The bias term and the floor** | No `floor_sweep.png`, no `floor.png`, no variogram panels, no bias-vs-variance scatter |
| **All descriptors except CPA** | β₀, β₁, orientation dispersion and the three lumen terms are out. Any per-descriptor panel collapses to a single descriptor |
| **The lumen call and the brightness threshold** | No `white_thresh.png`, no lumen QC panel |
| **The kidney arm, from the main text** | One figure at most, and only in the supplementary |
| **The appearance-artefact control** | No perturbation-sensitivity curve |

The paper reports **one readout**, `task_specific_value` (CPA), on the **liver** cohort,
with the kidney arm in the supplementary. If a script emits seven descriptors, plot the
first and drop the rest rather than making a seven-panel figure nobody will use.

---

## 1. What is actually needed

Two figures in the main text, one in the supplementary, and a small set of numbers. That
is the whole request.

### Figure 1 (main) — reliability, one curve per uncertainty source

**Script:** `compare_uncertainty_sources.py` → `reliability_sources.png`
**Manuscript label:** `fig:reliability`
**Status:** produced already; check the presentation requirements in §2.

Four sources on one panel, scored on the same regions against the same target: ensemble
total σ, ensemble data-exposure σ, ensemble procedural σ, and cycle-reconstruction error.

The reference line is **0.80σ, not the diagonal**. For a symmetric error of scale σ,
E|e| = σ√(2/π) ≈ 0.80σ, so a perfectly calibrated ensemble sits on 0.80σ and drawing the
diagonal would label it 20% over-confident. If the current figure draws a diagonal, fix it.

Do **not** put a scale ratio on the cycle-error curve anywhere in the figure or its
annotations. Cycle error is in 0–255 intensity units and CPA error is a fraction, so a
ratio of the two is meaningless. Ranking is comparable across these sources; scale is not.

### Figure 2 (main) — risk–coverage, one curve per source, **plus a baseline that is missing**

**Script:** `compare_uncertainty_sources.py` → `risk_coverage.png`
**Manuscript label:** `fig:riskcoverage`
**Status:** produced already, but **not yet correct for the paper**.

The existing figure draws one curve per uncertainty source, the oracle ceiling, and the
flat random line. It does not draw the baseline that matters most.

> **The change needed.** Add a curve for **ranking regions by the ensemble's point
> prediction alone**, with no uncertainty involved. On this measure that baseline *beats* σ:
> −15.1% against −7.8% at 80% coverage pooled, and −15.9% against −11.5% within a slide.
> The manuscript volunteers those numbers in its own text, so the figure must show the
> curve. A risk–coverage plot without it says the opposite of what the paper says.

Keep the oracle and the flat random line; both are load-bearing. The random line is exactly
flat by construction rather than by simulation, and the caption says so, so do not replace
it with a Monte-Carlo estimate.

### Figure S1 (supplementary) — the pixel-scale replication

**Script:** `plot_pixel_reliability.py` → `reliability_pixel.png`
**Status:** produced already.

All components flat against cycle-reconstruction error, 44,106 tiles over 20 slides. This
is the replication of the earlier pixel-scale finding and it is deliberately *not* the
contrast; the main-text figures are the contrast. Label it so the two cannot be confused.

### Not requested

`calibration_phi.png` (working panel), `<wsi>_uncertainty.png` (σ and σ/μ heatmaps) and the
per-subset panels are not in the current manuscript. The heatmaps in particular were cut
when the paper dropped to a single readout. Do not spend time on them without asking.

---

## 2. Presentation requirements, and why each one is not cosmetic

The template is WACV 2027 in review mode. Three of these have already caused a defect.

**Column width is 3.281 in** (`\textwidth` 6.875 in, `\columnsep` 0.3125 in, two columns).
Size the figure for that, at that width, before exporting. Do not export large and rely on
`\includegraphics[width=\columnwidth]` to shrink it, because that shrinks the text with it.

**Text inside the figure must be legible at 3.281 in.** No LaTeX width change fixes small
baked-in text. Axis labels, tick labels and legend should read at roughly 7–8 pt *after*
placement, which means setting the figure size and font size together in matplotlib rather
than scaling afterwards.

**Nothing may exceed the column.** `wacv.sty` loads `lineno` with the `switch` option, which
puts line numbers in the outer margin of each column. Anything wider than `\columnwidth`
overflows into that margin and the line numbers print **on top of it**. This already
happened once with a table. In this template an overfull box is a layout bug, not a warning.
The manuscript's check is `grep -c 'Overfull .hbox' main.log` and it must return 0.

**Greyscale-separable.** Curves must be distinguishable when printed in black and white, so
vary line style and marker as well as colour. Four sources on one panel makes this real
rather than theoretical.

**Prefer PDF over PNG** for these line plots. They are vector figures; PDF keeps them crisp
at any zoom and usually smaller. `savefig('name.pdf')` and hand over the PDF. PNG is only
right for a rasterised image panel, of which the paper currently has none.

**No figure title inside the image.** The caption carries it. A baked-in title duplicates
the caption and wastes vertical space.

---

## 3. Numbers the manuscript is still missing

These are not in `MANUSCRIPT_UPDATES.md` and are currently `\TODO` markers in the paper.
Read them off the run; do not estimate them.

| Needed | Where it lives |
|---|---|
| **Normalised ECE, one value per uncertainty source** | `calibrate_phi.py` → `summary.json`, field `ece_normalised` |
| Per-subset ρ and scale ratio for CPA, five rows | the `--prediction fold` run, reported **per subset and never pooled** |

On the second: pooling across subsets induces a between-subset trend that is present in
none of them, so report the five rows separately or not at all.

---

## 4. One inconsistency to resolve before the figures are final

`MANUSCRIPT_UPDATES.md` §0 gives the procedural row as:

| source | ρ | 95% CI | slides +ve | p |
|---|---|---|---|---|
| ensemble procedural σ | +0.094 | **[+0.008, +0.185]** | 12/20 | **0.105** |

The interval **excludes zero** while p = 0.105. Those cannot both come from the same test.
The likeliest explanation is that the interval is a percentile bootstrap over slides while
the p comes from a different test with a different null, but the handover does not say.

This matters beyond tidiness: "procedural spread loses significance while data exposure
survives" is one of the paper's two supporting claims, and it currently rests on the single
number that contradicts its own interval. Check which of the two is the odd one out, and
report which test produced each. If the p is the anomaly, the cleanest fix at the manuscript
end is to drop the p column and let the intervals carry the inference.

---

## 5. Where to put the output

```
~/Desktop/Manuscript/Qasim/Uncertainty_decomposition/figures/
    reliability_sources.pdf
    risk_coverage.pdf
    reliability_pixel.pdf
```

Names must match those three; the manuscript's floats already reference them. Report back
the ECE values and the per-subset table as text rather than editing any `.tex` file.
