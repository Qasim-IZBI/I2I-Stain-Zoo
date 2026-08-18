# What the manuscript needs to say

> Written 2026-08-16, against the code at `8ce9ea4`. A handover from the codebase to
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

**Narrowed on 2026-08-16:** the calibration is over the **four collagen
descriptors only**. The three lumen terms cannot be measured on this cohort (§3a),
which removes the floor-free arm and makes the SR/H&E frame check (§6) the gate for
the whole calibration rather than for half of it.

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

| # | descriptor | read from | referenced against | status |
|---|---|---|---|---|
| 1 | `task_specific_value` (CPA) | member's collagen mask | real SR | measured |
| 2 | `beta0_per_mm2` | member's collagen mask | real SR | measured |
| 3 | `beta1_per_mm2` | member's collagen mask | real SR | measured |
| 4 | `regional_dispersion` | member's collagen mask | real SR | measured |
| 5 | `lumen_fraction` | member's **generated SR** | real **H&E** | *unmeasurable here, §3a* |
| 6 | `beta0_lumen_per_mm2` | member's generated SR | real H&E | *unmeasurable here* |
| 7 | `beta1_lumen_per_mm2` | member's generated SR | real H&E | *unmeasurable here* |

Points the methods section has to make:

- **Mask, never intensity.** Every collagen term comes from the thresholded binary
  mask, so a global colour offset in the virtual stain cannot masquerade as
  structural error.
- **Counts are densities**, per mm² of tissue, so regions of differing size stay
  comparable. The **lumen** densities are per mm² of the *H&E footprint* — the only
  denominator available on both sides, since the real H&E has no collagen labels.
- **β₁ of the lumen space was the lumen-filler test** — a model painting collagen
  over vessels keeps the whitespace area and loses the loops, which area alone
  cannot see. It is the one descriptor designed to catch a specific failure, and
  §3a explains why it could not be computed. β₁ of the *collagen* mask remains and
  catches part of the same thing.
- **The lumen reference would have carried no floor** — same physical section, so
  any discrepancy is model error rather than inter-level biology. That was the
  design's main strength, and it is the one lost by §3a: with the lumen terms
  unmeasurable, every remaining descriptor is referenced to a different section
  and carries the floor.
- `tissue_fraction` is reported but not calibrated: it is H&E-derived on both
  sides, so it has zero variance and zero error.

**Regions.** Non-overlapping grid, 2048 px at 0.221 µm/px = 0.452 mm, sized in
pixels so the heatmap tiles without a seam. Partial edge regions are dropped so
every region has the same area. Regions below 25% coverage **of the H&E tissue
footprint** are excluded — the footprint, not a member's collagen mask, because a
model output would make the region set depend on which subset produced it, and the
five subsets would then no longer be measuring the same regions. Worth a clause:
it is what makes the variance decomposition well defined.

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

## 3a. Two findings about representation, and why the lumen arm is closed

Both are results in their own right and belong in the paper, not only in the
limitations.

**The generated stain does not reproduce whitespace.** Its intensity histogram has
no bimodality: the brightness-derived tissue footprint sweeps from 7% of the canvas
to 100% across thresholds 0.50–0.725, and at 0.675 only 13.7% of the canvas is
bright where the slide background alone is ~35–40%. At the H&E's own threshold
(0.65) it labels **22% of the slide as whitespace against the H&E's 4% on the same
tissue**. The model occupies a narrow tonal band and renders neither lumens nor
slide background as white. This is a concrete characterisation of what unpaired
translation preserves and what it does not, and it is why intensity-derived
descriptors were never going to work here — vindicating the mask-never-intensity
rule the collagen descriptors already follow.

**The collagen segmenter counts lumen as tissue.** `Dataset314_SR_light` is trained
that way, so no mask — real or virtual — contains enclosed background from which a
lumen could be recovered. Two consequences for methods: the lumen descriptors have
no route on either arm, and **CPA's denominator is tissue-including-lumen**
throughout. The latter is consistent across arms so it does not bias the
comparison, but it is not the denominator a histologist would assume and should be
stated.

Together these close the H&E-referenced half of φ on this cohort. The seven-term
vector and its tooling remain in the codebase for a cohort whose generated stain
retains whitespace; the paper should describe the four collagen terms as what was
measured, and can cite the two findings above as why.

## 4. The calibration (methods + results)

Over the four collagen descriptors (see §3a), per region: σ from the ensemble,
error against the real SR, then

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
3. **Claim ranking, not absolute calibration.** The floor is in the target, and
   with the lumen arm closed there is no floor-free descriptor left to carry an
   absolute claim.
4. **Cluster on the case** (n = 20) for every interval. Regions within a slide are
   spatially correlated, and in `--prediction fold` the five subset predictions for
   a region share one target.

### First run on real data (2026-08-17, liver, 2850 regions / 20 cases)

`--prediction grand`, 2048 px regions, four collagen descriptors:

| descriptor | ρ(σ, \|error\|) | E\|z\|/0.80 | reads as |
|---|---|---|---|
| `task_specific_value` (CPA) | **+0.217** | 0.71 | ranks its error; slightly under-confident |
| `beta0_per_mm2` | −0.007 | 0.66 | no relationship |
| `beta1_per_mm2` | −0.032 | 0.96 | no relationship |
| `regional_dispersion` | −0.002 | 2.17 | no relationship, and over-confident |

**CPA calibrates; the topological terms do not.** Two things make this coherent
rather than disappointing:

- CPA had the *best* floor-to-signal ratio of the four in the §2 sweep (0.87 at
  1.5 mm against 0.97–1.19), and it is the one where ρ survives. The descriptor
  with the most headroom is the one that shows signal.
- Floor noise in the target attenuates ρ toward zero, so **+0.217 is a lower
  bound** and the three nulls are *ambiguous*, not negative. At 2048 px
  (0.45 mm) the floor is worse than anywhere in the §2 sweep. Do not write "β₀
  uncertainty is uninformative"; write that it cannot be demonstrated against a
  reference this noisy.

Do **not** quote the naive p (1.3e-31 for CPA). It treats 2850 regions as
independent when they come from 20 cases. `--n_boot` resamples whole slides;
quote that CI. The shuffled control must sit near 0.

**Component comparison (2026-08-18).** All three σ are scored against the same
error, so only σ moves:

| CPA | ρ | 95% CI (by case) | E\|z\|/0.80 |
|---|---|---|---|
| total | +0.217 | [+0.107, +0.329] | 0.71 |
| **procedural** | **+0.274** | **[+0.149, +0.376]** | **1.01** |
| data-exposure | +0.169 | [+0.037, +0.307] | 1.10 |

All three exclude zero; the shuffled control is 0.015. Two separable claims:
procedural σ **ranks** best, while data-exposure σ **scales** best — its bin
ratios are flat across the σ range (1.22, 1.02, 1.08, 1.07, 1.19, 1.01, 0.93,
0.94, 0.97, 0.93) where procedural drifts 0.66 → 1.22. Data exposure is ~50% of
region-level CPA variance (median 0.508, IQR 0.44–0.57).

**Subset-level (`--prediction fold`), scored per subset.** Do NOT pool: subsets
sit at different σ and different error levels, so pooling induces a between-subset
trend present in none of them. Pooled β₀ read ρ = +0.312 against per-subset
+0.015, −0.017, +0.109, +0.123, +0.091 — larger than any subset. Per subset, CPA
gives +0.018, −0.055, +0.275, +0.318, −0.029: **the sign flips**, and only two of
five subsets calibrate.

The scale numbers are the more interesting half: subsets 1, 2 and 5 are
over-confident by 3.3–4.1×, subsets 3 and 4 are near-calibrated (0.65, 1.48). So
*which seven slides a member saw* determines not only its prediction but whether
its uncertainty means anything — and the 50-member ensemble is better calibrated
than any 10-member subset within it. That is an argument for the full crossed
grid, and it is the claim a flat seed-only ensemble cannot make.

### Lead with risk-coverage, not with rho (2026-08-18)

rho = 0.22 is real but modest, and a reviewer can fairly ask what it buys. The
same data answers that directly — rank regions by sigma, discard the least
certain, measure CPA error on what remains:

| coverage | CPA MAE | vs keeping all | random baseline |
|---|---|---|---|
| 100% | 0.0397 | — | 0.0397 |
| 90% | 0.0381 | −4.2% | 0.0398 |
| **80%** | **0.0366** | **−7.8%** | 0.0398 |
| 70% | 0.0346 | −13.0% | 0.0397 |
| 50% | 0.0298 | **−25.0%** | 0.0398 |

At 80% coverage the reduction is **−8.1%, 95% CI [−15.1%, −1.4%]** clustered on
case. Random discarding is flat at 0.0398 — the control that makes the claim.
The oracle (rank by true error) reaches −41.2%, so the ensemble captures ~20% of
what a perfect uncertainty would; **report that gap**, it is the honest measure of
how far this is from solved.

The deployable sentence: *the ensemble identifies which regions to trust for
fibrosis quantification.* Use it as the headline; keep rho and the reliability
diagram as the evidence behind it.

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
| `risk_coverage.png` | `calibrate_phi.py` | **the headline figure** — MAE vs coverage per descriptor, with the oracle ceiling and bootstrap bands |
| `risk_coverage.csv` | `calibrate_phi.py` | its numbers, incl. `capture_of_oracle` |
| `reliability_phi.png` | `calibrate_phi.py` | **the reliability diagram** — one panel per descriptor, total / procedural / data-exposure σ overlaid against the same error |
| `reliability_bins.csv` | `calibrate_phi.py` | the numbers behind that figure, one row per descriptor × bin |
| `calibration_phi.png` | `calibrate_phi.py` | working panel: reliability per descriptor + a ρ summary |
| `<wsi>_uncertainty.png` | `plot_uncertainty_heatmap.py` | σ and σ/μ per descriptor over the slide |
| `white_thresh.png` | `calibrate_white_thresh.py` | how the threshold was chosen (supplementary) |
| `<wsi>_*_lumen.tif` | `--qc_dir` | the lumen call itself, for a supplementary QC panel |

**On the heatmap figure:** show σ *and* σ/μ. For a count-based descriptor σ rises
with how much structure a region holds, so a raw σ map can be a collagen-density
map wearing an uncertainty label. A reviewer who spots that unaided will not be
kind about it.

---

## 6. Still open before submission

- [x] ~~**The calibration has not been run on real data.**~~ Run 2026-08-17 on
      liver; see §4. Still to do: re-run with `--n_boot` for the CI, and
      `--prediction fold` for the data-exposure claim.
- [ ] **Step 0 — is the real SR on the H&E frame? THE gate.** With the lumen arm
      closed, this decides whether there is a region-level calibration (~6000
      points) or only a WSI-level one (n = 20) — for the entire study, not half of
      it. `sbatch scripts/check_frame_alignment.sh` — seconds, header reads only.
- [ ] **Re-run the floor sweep on current code** before quoting the §2 table. Those
      three runs predate two fixes (the inverted split-half bracket, the NaN-column
      crash). Neither changes the verdicts, but the manuscript should not cite
      figures produced by code that has since been corrected.
- [x] ~~**Negative control**: shuffle σ across regions, confirm ρ collapses.~~
      Implemented as `rho_shuffled` in `calibrate_phi.py`, on by default with
      `--n_boot`. Report it beside every ρ.
- [x] ~~**Confound test**: correlate σ_lumen against mean region brightness.~~
      Superseded — the threshold sweep on the generated SR settled it directly
      (§3a). The lumen arm is closed, so there is nothing left to confound.
- [x] ~~**Cluster-robust intervals**, case as the unit.~~ `--n_boot 2000`
      resamples whole slides. A slide drawn twice contributes its regions twice,
      so the interval reflects how much the answer depends on which 20 slides
      were collected. Quote this, never the naive p.
- [ ] **Resolution parity**, if the collagen arm is used: the virtual arm's content
      was synthesised at 0.442 µm/px and upsampled, the real SR is genuinely 0.221.
      Same pixel size, different effective resolution, and the segmenter sees both.
      Sharper edges give more components — indistinguishable from model error.
- [ ] **`--min_object_px` must match across the two lumen runs.** Both write it to
      `lumen_masks.json`; nothing compares them.
- [ ] Kidney arm: none of the above has been run on it.
