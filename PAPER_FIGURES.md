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

**Revised 2026-08-22.** The manuscript is now **two documents**, submitted separately, and
three figures changed hands between them. Four figures are in use, one delivered figure is
not, and one is still outstanding.

| # | File | Appears in | Status |
|---|---|---|---|
| Fig 1 | `reliability_pixel` | **main paper**, §7 | delivered, in use |
| Fig 2 | `reliability_sources` | **main paper**, §7 | delivered, in use |
| Supp Fig 1 | `within_slide_rho` | **supplement**, §H | delivered but **needs regenerating**, see §2c |
| Supp Fig 2 | `risk_coverage` | **supplement**, §H | delivered, in use |
| Supp Fig 3 | `data_exposure_share` | **supplement**, §H | **not yet produced**, see §2d |
| — | `reliability_residual` | nowhere | superseded, no successor needed |
| — | `within_slide_rho_pixel` | nowhere | not requested |

### Figure 1 (main) — the conventional protocol

`reliability_pixel`, from `plot_pixel_reliability.py`. Per-pixel variance components summed
over the three colour channels, scored against the cycle-reconstruction residual at tile
scale, with the `E|e| = 0.46σ` line. This is the paper's **first** result, not a
supplementary one and not a replication: it is the same across-tile analysis the earlier
benchmark ran, with the point prediction partialled out, and the association does not
survive that control.

### Figure 2 (main) — the proposed protocol

`reliability_sources`, from `compare_uncertainty_sources.py`. The three ensemble components
of the CPA spread against the CPA absolute error, with the `E|e| = 0.80σ` line. The
cycle-reconstruction residual is **not** on this panel and must not be added: it is in
different units, and since 2026-08-21 it is never treated as an uncertainty source anywhere
in the paper.

Read together, these two are the paper's whole argument: the conventional protocol answers
nothing, the proposed one does.

### Supplement Figure 1 — the density control

`within_slide_rho`. See §2c: the delivered version is not yet what the manuscript describes.

### Supplement Figure 2 — selective prediction

`risk_coverage`, with the `μ alone` baseline, the oracle and the flat random line. Correct
as delivered.

### Not requested

`calibration_phi.png`, the `<wsi>_uncertainty.png` heatmaps, the per-subset panels, and any
figure for the floor, the topology descriptors, the lumen terms or the brightness threshold.
All of those analyses left the paper. Do not spend time on them without asking.

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

### Both reliability figures

| Element | Label |
|---|---|
| x axis | `σ` |
| y axis | `e` |
| reference line | `E|e| = 0.80σ` on the CPA figure, `E|e| = 0.46σ` on the pixel figure |
| per-curve annotations | **none — removed 2026-08-23, see §2e** |

Current code sets `xlabel = "ensemble σ"` and `ylabel = "|error| vs real tissue"`. Both need
changing: the first because one of the four curves is not an ensemble σ at all, the second
because the paper calls that quantity $e$.

### Risk--coverage

| Element | Label |
|---|---|
| x axis | `coverage (% of regions kept)` |
| y axis | `change in mean e (%)` |
| baselines | `oracle (rank by e)`, `random`, `μ alone` |

"μ alone" is the new baseline described in §1. Naming it with the symbol makes the point
without a sentence: it ranks on $\mu$, using no $\sigma$ at all.

### The units question, decided

`compare_uncertainty_sources.py` put the residual on the same raw x-axis as the ensemble
components, which cannot work: ensemble σ is in CPA, order $10^{-2}$, and the residual is
0–255 intensity. It was resolved by splitting them into two files, and the manuscript now
places the pixel-scale panel and the CPA-scale panel as its two main figures. Nothing
further is needed. Do not normalise either axis to bring them together; raw units on both
axes are what makes each panel readable, and the two reference lines differ by a factor of
$\sqrt{3}$ for a stated reason.

## 2b. Regeneration requested 2026-08-21 — **done**, kept for the record

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

## 2c. Regeneration requested 2026-08-22 — **still outstanding**

`within_slide_rho.pdf` still plots the **partialled** within-slide ρ only. **The manuscript
already describes the paired version**: the supplement's caption reads "before and after
partialling" and refers to "the gap between the two bars of a pair", and the text beside it
quotes both readings. Until this is regenerated the caption describes a figure that does not
exist. The manuscript carries a `\TODO` on that float.

The manuscript needs **both readings on the same panel**: for each of the three components, a
raw bar and a partialled bar side by side, each with its per-case scatter, mean, 95%
interval and positive-case count.

Two reasons, and the second is the one that makes it worth a rerun rather than a caption:

* The paragraph reporting the density confound quoted numbers with no figure behind them,
  which is the only such paragraph left in the Results section.
* The size of the correction is currently asserted in prose — "about half the apparent
  ranking ability survives" — where a paired bar shows it. The gap between the two bars of
  a pair *is* the part of the association that is collagen density rather than error, and
  that is the paper's answer to its strongest objection.

Raw values, for checking: total $+0.278$ (19/20), data exposure $+0.245$ (19/20),
procedural $+0.246$ (18/20). Partialled: $+0.150$ (14/20), $+0.143$ (16/20), $+0.094$
(12/20).

## 2d. One supplementary figure still needed — `data_exposure_share.pdf`

A distribution of the **data-exposure share of region-level CPA variance** across regions,
from the per-region CSV: a histogram or violin with the median and interquartile range
marked. Median $0.508$, IQR $0.44$–$0.57$. This moved out of the main text because it
quoted numbers with no plot; it needs one where it now lives.

**The filename is fixed as `data_exposure_share.pdf`** so that the manuscript can reference
it before it arrives. x-axis label: the share is a fraction of the total, so
`data-exposure share of Var` with the symbols of §2a, not "proportion of variance" in
words.

---

### Optional, if a reviewer presses

The paper changes two things at once, the uncertainty and the target, and says so. The
experiment that would separate them is cheap and the data already exists: score **region σ
against the residual aggregated to the same regions**, as a target rather than as a source.
`regen_per_region(...)` in `compare_uncertainty_sources.py` already produces that
aggregation; it is a scoring change, not new computation. Not needed for the current
argument.

---

## 2e. Annotations come off both reliability figures — requested 2026-08-23

`reliability_sources.pdf` annotates each curve with $\rho$ and $\mathbb{E}|z|/0.80$;
`reliability_pixel.pdf` annotates nothing. The manuscript first considered adding the same
block to the pixel figure for consistency and decided against it, because six annotations on
a five-curve plot crowd it and because the two figures' summary statistics are **not
comparable to each other**, so putting them in matching corners of adjacent plots invites
exactly the comparison the paper forbids.

**The values moved into a table instead.** The manuscript now carries Table 2, a single
head-to-head giving $\rho$, $\mathbb{E}|z|$ and the ECE for both protocols and all three
components. Two consequences for this repository:

**(1) Remove the per-curve annotations from `reliability_sources.pdf`.** The curves, the
error bars and the reference line stay; the text blocks go. §2a's annotation row is amended
to match. Do not add any to `reliability_pixel.pdf`.

**(2) Report the conventional arm's summary statistics as text.** Table 2 has **four blank
cells** and they are the only thing keeping it from being complete: $\mathbb{E}|z|$ and the
ECE for the pixel arm, per component. These have never reached the manuscript. §7 currently
says that arm fails on magnitude *qualitatively*, that the spread runs several times larger
than the errors, because that is all that can be read off a plot. With the numbers it becomes
a row in the table and a long-standing audit item closes.

**The divisor is 0.46, and this is the part to get right.** The two arms have different
calibrated lines. Pixel $\sigma$ sums three colour channels while the residual is a
per-channel mean, so the calibrated line there is
$\sigma\sqrt{2/\pi}/\sqrt{3} \approx 0.46\sigma$, against $0.80\sigma$ for the CPA arm. The
statistic must be normalised on each arm's **own** line so that one means calibrated in both.
Dividing the pixel arm by $0.80$ would report a perfectly calibrated ensemble as 74%
over-confident. **Report it as `E|z|/0.46` with the divisor written out**, so the divisor
travels with the number.

Report the ECE the same way it is computed for the CPA arm, from `ece_normalised` with both
axes min-max scaled, so the two arms' ECEs are at least constructed alike.

### Three decimal places on rho, wherever it is printed

The manuscript reports $\rho$ to three decimals, $+0.217$, $+0.169$, $+0.274$, because the
intervals beside them are quoted to three. Any $\rho$ handed over in text or drawn on a plot
should carry three, not the two the old annotations used.

## 3. Numbers the manuscript is still missing

These are not in `MANUSCRIPT_UPDATES.md` and are currently `\TODO` markers in the paper.
Read them off the run; do not estimate them.

| Needed | Where it lives |
|---|---|
| ~~Normalised ECE per component~~ | **Delivered** 2026-08-22 from `calibration_phi_04/summary.json`: total 0.28, procedural 0.24, data exposure 0.34 |
| Per-subset ρ and scale ratio for CPA, five rows | the `--prediction fold` run, reported **per subset and never pooled**. Still needed |

On the second: pooling across subsets induces a between-subset trend that is present in
none of them, so report the five rows separately or not at all.

---

## 4. An inconsistency in the source table — now only a supplementary risk

`MANUSCRIPT_UPDATES.md` §0 gives the procedural row as:

| source | ρ | 95% CI | slides +ve | p |
|---|---|---|---|---|
| ensemble procedural σ | +0.094 | **[+0.008, +0.185]** | 12/20 | **0.105** |

The interval **excludes zero** while p = 0.105. Those cannot both come from the same test.
The likeliest explanation is that the interval is a percentile bootstrap over slides while
the p comes from a different test with a different null, but the handover does not say.

**The manuscript resolved this by omission**: no $p$-value appears in either document, and
every inference is reported as a case-clustered interval. The risk is confined to the
per-subset table still owed under §3 — if that arrives with $p$-values, the contradiction
returns. Report intervals there too, and if a $p$ is unavoidable, say which test produced
it.

---

## 5. Where to put the output

```
~/Desktop/Manuscript/Qasim/Uncertainty_decomposition/figures/
    reliability_pixel.pdf      main paper, Figure 1
    reliability_sources.pdf    main paper, Figure 2
    within_slide_rho.pdf       supplement, Figure 1   <- regenerate, §2c
    risk_coverage.pdf          supplement, Figure 2
    data_exposure_share.pdf    supplement, Figure 3   <- still to produce, §2d
```

One directory serves both documents; which PDF a figure appears in is decided by the
manuscript, not by where the file sits. Names must match those above, since the floats
already reference them. Report the per-subset numbers as text rather than editing any
`.tex` file.
