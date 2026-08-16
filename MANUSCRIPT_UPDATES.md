# What the manuscript needs to say

> Written 2026-08-16, against the code at `38c84e0`. A handover from the codebase to
> the paper: what changed since the plan, what can be claimed, what cannot, and the
> conventions a reviewer will otherwise raise. Companion to `CLAUDE.md` (flags) and
> `PIPELINE_AFTER_INFERENCE.md` (how to run it).
>
> **Nothing here is a result yet.** Every number below is either from the floor
> sweep (real, on the liver cohort) or from synthetic validation of the tooling.
> The calibration itself has not been run on real data.

---

## 1. The headline changed

The study was built around a **bias** term: how far the virtual stain's structure
sits from the real tissue's, per region. That is closed on liver, and the paper
should say so rather than omit it.

**What replaced it:** the ensemble's *uncertainty* is the result, and the new claim
is that φ_struct — structural descriptors measured against real tissue — works as a
calibration target where cycle-reconstruction error does not.

So the contribution is now two things:

1. **A descriptor-space uncertainty decomposition** — procedural (seed) versus
   data-exposure (which slides were seen), separable only because the ensemble is a
   crossed 5 subsets × 10 seeds grid.
2. **φ_struct as a calibration target.** Regen error's failure to calibrate is the
   standing BMVC 2026 result and is **cited, not re-measured**.

---

## 2. The negative result, and how to frame it

The §7 floor pilot asks whether a bias estimate could clear the biological
variation between serial sections. Run on liver at three region sizes:

| region | CPA | β₀ | β₁ | dispersion | regions | variogram pairs / lag span |
|---|---|---|---|---|---|---|
| 0.75 mm | 1.06 | 1.11 | 1.00 | 1.23 | 1058 | 13,690 / 4.9× |
| 1.5 mm | 0.87 | 0.97 | 0.80 | 1.19 | 279 | 1,197 / 2.2× |
| 2.5 mm | 0.71 | 0.76 | 0.67 | 0.93 | 99 | 120 / 1.4× |

(floor SD ÷ between-region SD; usable < 0.5, marginal 0.5–0.9, floor-limited ≥ 0.9)

**Phrase it as a limit on the measurement, not a property of the model.** Not "the
model has no bias" — that is unsupported. The defensible sentence is: *with serial
sections and no second PSR level, inter-level biological variation is comparable to
the between-region spread at every region size this cohort supports, so a bias
estimate would be indistinguishable from the floor.*

Two qualifiers that keep it honest:

- **Design-specific, not universal.** A second real PSR level would measure the
  floor directly and might leave headroom. Say "cannot be measured with this
  design".
- **Liver only.** The kidney arm has not been run, and its segmenter is out of
  distribution besides.

Note the trend goes the right way (the floor averages out faster than the biology,
as §4.2 predicts) but the most favourable numbers rest on the weakest evidence — at
2.5 mm the sill spans 1.4× of lag over 120 pairs, where a flat variogram is the
absence of evidence rather than evidence of a plateau. Say why you stopped at
2.5 mm: at 4 mm there would be no variogram at all.

**Suggested placement:** 3–5 sentences in limitations, plus `floor_sweep.png` and a
short method paragraph in supplementary. It costs about a third of a page and buys
the answer to "how do you know low variance means correct?".

---

## 3. The descriptor vector (methods text)

Seven marginal statistics of a ~1–2 mm region, in two reference classes:

| # | descriptor | read from | referenced against |
|---|---|---|---|
| 1 | `task_specific_value` (CPA) | member's collagen mask | real SR |
| 2 | `beta0_per_mm2` | member's collagen mask | real SR |
| 3 | `beta1_per_mm2` | member's collagen mask | real SR |
| 4 | `regional_dispersion` | member's collagen mask | real SR |
| 5 | `lumen_fraction` | member's **generated SR** | real **H&E** |
| 6 | `beta0_lumen_per_mm2` | member's generated SR | real H&E |
| 7 | `beta1_lumen_per_mm2` | member's generated SR | real H&E |

Points the methods section has to make:

- **Mask, never intensity.** Every collagen term comes from the thresholded binary
  mask, so a global colour offset in the virtual stain cannot masquerade as
  structural error.
- **Counts are densities**, per mm² of tissue, so regions of differing size stay
  comparable. The **lumen** densities are per mm² of the *H&E footprint* — the only
  denominator available on both sides, since the real H&E has no collagen labels.
- **β₁ of the lumen space is the lumen-filler test.** A model painting collagen
  over vessels keeps the whitespace area and loses the loops; area alone cannot see
  it. This is the one descriptor that is *designed* to catch a specific failure.
- **The lumen reference carries no floor.** It is the same physical section the
  model generated from, so any discrepancy is model error rather than inter-level
  biology. That is why the lumen half is the stronger evidence even though the
  collagen half is the more familiar measurement.
- `tissue_fraction` is reported but not calibrated: it is H&E-derived on both
  sides, so it has zero variance and zero error.

**Regions.** Non-overlapping grid, 2048 px at 0.221 µm/px = 0.452 mm, sized in
pixels so the heatmap tiles without a seam. Partial edge regions are dropped so
every region has the same area. Regions below 25% tissue coverage are excluded.

**The tissue footprint** comes from the H&E tissue masks the CPA pipeline already
applies (`apply_he_mask.py`), with holes filled so internal lumens count as inside
tissue. Worth one sentence in methods, because it means the study uses **one**
definition of tissue throughout, and because it keeps the whitespace threshold out
of every denominator — the threshold then affects only which pixels are called
lumen, not what they are divided by.

**Thresholds.** The whitespace cut is a per-cohort measurement, not a constant, and
must be reported: H&E stable window 0.500–0.675, SR 0.600–0.700, **neither with a
plateau** (12% and 9% change per 0.025 step). The committed value is 0.65, inside
both windows. Report it as a convention held fixed across arms rather than as a
measured property — and note that `lumen_fraction`'s absolute value is therefore
convention-dependent, while its *ranking* across regions is what the calibration
uses.

---

## 4. The calibration (methods + results)

Per descriptor, over regions: σ from the ensemble, error against the real
reference, then

- **Spearman ρ(σ, |error|)** — the headline. Noise in the reference (a floor, a
  registration offset) attenuates ρ toward zero, so a positive value is
  conservative and a null is ambiguous.
- **E|z| where z = |error|/σ**, reported as E|z| / 0.80. Above 1 means
  over-confident: errors exceed the ensemble's own spread.
- A reliability curve in raw units, and a normalised ECE for continuity.

### Conventions to state explicitly, or a reviewer will

1. **σ is a predictive SD** — the spread of members, not the standard error of the
   mean. σ/√50 would be tiny and the test would collapse into a test of bias.
   Ensembles share systematic error, which is what makes the comparison meaningful
   and why "over-confident" is the usual finding.
2. **E|e| = σ·√(2/π) ≈ 0.80σ for Gaussian error.** The reliability line is 0.80σ,
   not the diagonal; a diagonal would call a perfectly calibrated ensemble 20%
   over-confident.
3. **Claim ranking, not absolute calibration**, for the collagen arm — the floor is
   in the target. The lumen arm has no floor and can support the absolute claim.
4. **Cluster on the case** (n = 20) for every interval. Regions within a slide are
   spatially correlated, and in `--prediction fold` the five subset predictions for
   a region share one target.

### The secondary claim the crossed grid uniquely supports

`--prediction grand` (mean of 50, total spread) versus `--prediction fold` (each
subset's mean, procedural spread alone). If the first calibrates better, the
data-exposure component earns its place. **A flat seed-only ensemble cannot ask
this** — worth saying, since it justifies the 5×10 design over a 50-seed run.

---

## 5. Figures available

| figure | from | shows |
|---|---|---|
| `floor_sweep.png` | `plot_floor_sweep.py` | floor ÷ signal against region size, with the evidence behind each point |
| `floor.png` | `estimate_floor.py` | per-descriptor verdict, and the variogram it rests on |
| `calibration_phi.png` | `calibrate_phi.py` | reliability per descriptor + a ρ summary panel |
| `<wsi>_uncertainty.png` | `plot_uncertainty_heatmap.py` | σ and σ/μ per descriptor over the slide |
| `white_thresh.png` | `calibrate_white_thresh.py` | how the threshold was chosen (supplementary) |
| `<wsi>_*_lumen.tif` | `--qc_dir` | the lumen call itself, for a supplementary QC panel |

**On the heatmap figure:** show σ *and* σ/μ. For a count-based descriptor σ rises
with how much structure a region holds, so a raw σ map can be a collagen-density
map wearing an uncertainty label. A reviewer who spots that unaided will not be
kind about it.

---

## 6. Still open before submission

- [ ] **The calibration has not been run on real data.** Everything in §4 is
      validated against injected synthetic ground truth only (calibrated → 1.01,
      2.5× over-confident → 2.50, uninformative → ρ = 0.002).
- [ ] **Step 0 — is the real SR on the H&E frame?** Decides whether the collagen
      arm pairs at region level (~6000 points) or falls back to WSI level (n = 20).
      The lumen arm is unaffected.
- [ ] **Re-run the floor sweep on current code** before quoting the §2 table. Those
      three runs predate two fixes (the inverted split-half bracket, the NaN-column
      crash). Neither changes the verdicts, but the manuscript should not cite
      figures produced by code that has since been corrected.
- [ ] **Negative control**: shuffle σ across regions, confirm ρ collapses.
- [ ] **Confound test**: correlate σ_lumen against mean region brightness. If they
      track, the lumen descriptor is measuring how pale each member rendered the
      tissue rather than where it placed vessels — the specific risk of
      thresholding the SR.
- [ ] **Cluster-robust intervals**, case as the unit. Not implemented anywhere yet.
- [ ] **Resolution parity**, if the collagen arm is used: the virtual arm's content
      was synthesised at 0.442 µm/px and upsampled, the real SR is genuinely 0.221.
      Same pixel size, different effective resolution, and the segmenter sees both.
      Sharper edges give more components — indistinguishable from model error.
- [ ] **`--min_object_px` must match across the two lumen runs.** Both write it to
      `lumen_masks.json`; nothing compares them.
- [ ] Kidney arm: none of the above has been run on it.
