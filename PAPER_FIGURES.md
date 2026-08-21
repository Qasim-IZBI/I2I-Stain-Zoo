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

## 2a. Axis labels, legends and annotations — use the paper's symbols

The paper defines its quantities symbolically, and the figures must use the same symbols
rather than paraphrasing them in words. A reader moving between Table 2 and a figure should
not have to work out that "MAE" and $e$ are the same thing.

**The symbol register**, from the manuscript's §3 and §4:

| Symbol | Meaning | Never write |
|---|---|---|
| $\sigma$ | predictive standard deviation across the $M$ members | "uncertainty (SD)", "ensemble std" |
| $e$ | $\lvert \mu(x) - \varphi(y) \rvert$, the CPA absolute error | **"MAE"**, "MAE-CPA", "residual", "\|error\|" |
| $\mu$ | the ensemble's point prediction | "mean prediction", "predicted CPA" |
| $z$ | $e/\sigma$ | — |
| $\rho$ | Spearman rank correlation | "corr", "r" |
| $\mathbb{E}\lvert z\rvert/0.80$ | the scale summary | "calibration ratio" |

**"MAE" is the one to watch.** The manuscript deliberately does *not* use it: the per-region
quantity is a single absolute difference, not a mean, and the author ruled against borrowing
the earlier benchmark's naming. `make_risk_coverage_figure` currently labels its y-axis
`"change in MAE vs keeping all (%)"`, which contradicts the register. It must say $e$.

**The four sources, named as the paper names them.** Legend entries, in this order:

```
total σ            data-exposure σ            procedural σ            cycle-reconstruction residual
```

Note "residual", not "error" — the register fixes the term and the manuscript's Table 2 was
corrected to match on 2026-08-20.

### Figure 1 — reliability

| Element | Label |
|---|---|
| x axis | `σ` |
| y axis | `e` |
| reference line | `E|e| = 0.80σ` |
| per-curve annotations | `ρ` and `E|z|/0.80`, not spelled out in words |

Current code sets `xlabel = "ensemble σ"` and `ylabel = "|error| vs real tissue"`. Both need
changing: the first because one of the four curves is not an ensemble σ at all, the second
because the paper calls that quantity $e$.

### Figure 2 — risk--coverage

| Element | Label |
|---|---|
| x axis | `coverage (% of regions kept)` |
| y axis | `change in mean e (%)` |
| baselines | `oracle (rank by e)`, `random`, `μ alone` |

"μ alone" is the new baseline described in §1. Naming it with the symbol makes the point
without a sentence: it ranks on $\mu$, using no $\sigma$ at all.

### One thing that has to be decided before Figure 1 can be drawn

`compare_uncertainty_sources.py` puts the cycle-reconstruction residual on the same x-axis
as the ensemble components (`b["sd"] = b["regen"]`, then `x = mean_sd` in raw units with a
shared `set_xlim`). **Those are not the same units.** Ensemble σ is in CPA, a fraction of
order $10^{-2}$; the residual is in 0–255 intensity. On one raw axis the ensemble curves
collapse into the left edge.

That also makes a single x-axis label impossible, which is what surfaced this.

The fix that matches what the paper already says — *ranking is comparable across sources,
scale is not* — is **two panels sharing the y-axis**:

* left panel, the three ensemble components, x in CPA units, with the `E|e| = 0.80σ` line;
* right panel, the cycle-reconstruction residual, x in intensity units, **no reference
  line**, because a scale reference is meaningless where the units differ.

The y-axis is genuinely shared: the error is the same quantity for all four sources by
construction, since only σ moves between them. Do not normalise x to make one panel work —
raw units on both axes are what makes this comparison strong, and normalising is what the
ECE already does.

If you would rather solve it another way, say so before plotting; the manuscript caption
will need to match whatever is chosen.

---

## 2b. Regeneration requested 2026-08-21 — drop the residual as an uncertainty source

The manuscript no longer treats the cycle-reconstruction residual as an uncertainty. It
appears only as the *target* the conventional protocol validates against. The residual is a
single model's error magnitude, not a dispersion across members, so scoring it in the same
column as σ compared two different kinds of object; the empty scale column it always carried
was the symptom.

**Two delivered figures still carry a residual curve and need regenerating without it:**

* `within_slide_rho.pdf` — drop the fourth column.
* `risk_coverage.pdf` — drop the `cycle-recon. residual` curve. Keep `μ alone`, the oracle
  and the flat random line.

`reliability_residual.pdf` is no longer used at all and needs no successor.

`reliability_pixel.pdf` **moves into the main text**, where it now carries the conventional
protocol: pixel variance scored against the residual. No change to the figure itself.

### Optional, if a reviewer presses

The paper changes two things at once, the uncertainty and the target, and says so. The
experiment that would separate them is cheap and the data already exists: score **region σ
against the residual aggregated to the same regions**, as a target rather than as a source.
`regen_per_region(...)` in `compare_uncertainty_sources.py` already produces that
aggregation; it is a scoring change, not new computation. Not needed for the current
argument.

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
