# Non-Adjacent Section Data — OOD Probe and Bias-Term Feasibility (Liver + Kidney)

> Scope: a decision record for the available **non-adjacent** (same block, different level)
> H&E + PSR WSI sets — **liver 20 case pairs** and **kidney 40 case pairs** — covering (a) their use as the
> out-of-distribution probe for **E3**, and (b) whether the **E4 bias term** is computable
> without serial sections, including the choice of the feature map φ. The liver corpus
> remains the training and primary-evaluation data; these sets are held-out benchmarks.
> See `uncertainty_strategy.md` for E3/E4/E5 and `evaluation_strategy.md` for the structural
> metrics reused as descriptors here.
>
> Created 2026-07-31 · updated 2026-08-01.

---

## 1. The Questions

Available: kidney and liver WSIs in **H&E** and **PSR**, **same case / different levels of the
same block** — *not* serial sections.

- **Liver — 20 case pairs** (40 slides). Disjoint from the training specimens, so the §8
  hold-out requirement is satisfied.
- **Kidney — 40 case pairs** (80 slides).
- **Same magnification and pixel width for both organs**, so organ shift is a single clean
  knob and is *not* confounded with resolution — which is exactly the one-knob property
  argued for at the end of this section. Residual to check: identical pixel size does not
  guarantee the same scanner or staining run; the §2.2 encoder sanity check is what would
  detect appearance shift riding along.

1. Does the OOD experiment require serial-section pairing? → **No** (§2).
2. Can the bias term still be estimated from non-adjacent levels? → **Yes, at region level,
   conditional on the floor** (§4).
3. Can φ(·) be CPA rather than a UNI embedding? → **Yes, and it suits this data better —
   but not as the only φ** (§5).

E3's stated requirements are *"UNI/Virchow encoder; a shifted held-out split"*
(`uncertainty_strategy.md:124`). Nothing about pairing. Serial sections are demanded only by
**E4** (`:149`), **E5** (`:163`) and **V3** (`evaluation_strategy.md:275`), and §4 below shows
even those are partially recoverable.

Kidney is the **best available shift**: `uncertainty_strategy.md:173` names it directly —
*"Feed OOD tiles (unseen scanner/organ) → distributional epistemic spikes."* Organ shift moves
one knob only; scanner/batch shift is muddier because it perturbs stain appearance as well as
content.

---

## 2. Kidney as the OOD Probe

### 2.1 From kidney H&E alone — E3 in full
- **OOD detection.** Mahalanobis in UNI/Virchow2 feature space; held-out liver H&E = ID,
  kidney H&E = OOD; report AUROC.
- **Double-dissociation (V).** Run all four uncertainty channels on kidney tiles; expect the
  distributional term to spike while aleatoric stays flat and procedural/data move modestly.

**Lost without co-location:** E3's second evaluation criterion, *"correlation with forward
error on shifted tiles"* — unless §4 recovers region-level pairing, in which case it returns.

### 2.2 From kidney PSR, registration-free
- **E7 step 3** (`evaluation_strategy.md:230`) is explicitly *"unpaired, distribution-level"*:
  compare persistence images / landscapes / Betti curves / persistence entropy of virtual
  kidney PSR collagen against **real** kidney PSR. No registration required, by design.
- **Distribution-level realism failure.** FID/KID of virtual-PSR-from-kidney-H&E against real
  kidney PSR *and* against real liver PSR. Expected OOD signature: the model projects kidney
  onto the liver-collagen manifold and sits closer to liver PSR than to the kidney PSR it
  should be matching.
- **Encoder sanity check.** Score real kidney PSR against the liver-PSR training distribution
  to confirm the Mahalanobis detector responds to tissue architecture, not stain batch/colour.

### 2.3 From same-case pairing
This is the `evaluation_strategy.md:210` regime: *"paired-per-case-but-not-per-pixel."*
Per-case CPA agreement (virtual vs that case's real PSR), Bland–Altman, and a **signed** bias
in clinical units. Superseded and generalised by §4–§5, which put this on the E4 footing.

---

## 3. "Not Serial" Is Partly Recoverable

E5's premise is *"Serial sections aren't pixel-aligned; per-pixel calibration is impossible"*
(`uncertainty_strategy.md:157`) — the design **already assumes** misalignment, and §6.1 plans
for *"tissue-level registration (thumbnail affine/elastic) → co-located tile grids, explicitly
accepting sub-tile non-alignment."* Different levels of the same block is a *degree* of that
problem, not a different kind of problem.

Kidney is unusually forgiving: glomeruli are **~150–250 µm** across and persist through many
levels; large vessels and the cortex/medulla boundary persist further still. If levels are
tens of microns apart, the same glomeruli are physically present in both sections. Liver
sinusoidal architecture decorrelates far faster — kidney is the better tissue for this.

**Action:** attempt thumbnail affine registration before assuming it is unavailable. Success
upgrades kidney from E3-only to a weakened E4/E5 at region level.

---

## 4. The Bias Term From Non-Adjacent Levels

### 4.1 Why non-adjacency is survivable
The identity (`uncertainty_strategy.md:62`) is
`E_m‖φ(G_m(x)) − φ(y)‖² = Var + B + floor`. With non-adjacent levels you observe not `y` but
`y'` from a different level. Expanding:

```
‖μ − φ(y')‖² = ‖μ − φ(y)‖² + 2⟨μ − φ(y), φ(y) − φ(y')⟩ + ‖φ(y) − φ(y')‖²
```

If the level offset is **zero-mean** in φ-space — cutting deeper changes *which* structures
are seen but not their statistical composition — the cross term vanishes in expectation,
leaving `bias² + floor²`. This is exactly the "subtract the biological floor" design of §6.1.

**Non-adjacency does not break the estimator; it inflates the floor.**

The zero-mean condition is the assumption to defend. It holds for aggregate statistics over a
homogeneous block and fails where there is a systematic gradient along the cutting axis — in
kidney, **cortex/medulla proportion drift** is the concrete threat. Mask to cortex.

### 4.2 Scale — regions, not tiles
The floor shrinks with region size (structure displacements average out); bias does not.
Compute bias on regions large relative to the displacement scale: **~1–2 mm**, i.e.
~2048–4096 px at 0.5 µm/px — **not** 256² tiles. Liver lobular architecture is ~1–2 mm with
portal tracts ~1 mm apart; kidney glomeruli are ~200 µm at ~300–500 µm spacing. Both land in
that range.

Budget: a WSI with 50–500 mm² of tissue yields 25–250 such regions — so ~500–5,000 regions
for liver and ~1,000–10,000 for kidney. Ample; **cases, not regions, are the binding n**.
Regions within a slide are correlated — use cluster-robust SEs or treat WSI as the unit, or
confidence intervals will be badly optimistic.

### 4.3 What 20 + 40 buys that serial liver alone would not
Bias measured **identically on in-distribution and OOD tissue with the same pairing
structure**. This converts E4's expected result — *"bias is a large, tissue-rarity-dependent
fraction of forward error; uncorrelated with epistemic-variance"* — from a within-liver
stratification into a clean two-condition contrast, with organ shift as the extreme end of
the rarity axis.

**Target figure:** per region, **bias² vs the E3 Mahalanobis score**, pooling both organs. If
bias rises with distributional distance while ensemble variance stays flat, that demonstrates
the completion argument (E4 point 4) rather than asserting it. Serial liver alone cannot
produce this curve.

Both arms clear the doc's own ask of ~15–30 case pairs (`uncertainty_strategy.md:192`):
liver 20, kidney 40. Correlation coefficients are reportable for each organ separately, and
the pooled figure has 60 cases.

**The asymmetry runs the wrong way, though**, and is worth planning around: the OOD arm is
twice the ID arm, whereas the ID arm carries more inferential load — it anchors the floor
estimate that is subtracted from *every* bias number, and it sets the baseline against which
the OOD contrast is read. A noisy OOD arm blurs one end of the curve; a noisy ID arm shifts
the whole thing. Two mitigations: lean on the floor-free geometric terms (§6.0), whose
precision does not depend on pairing at all; and consider reporting liver-only and kidney-only
fits alongside the pooled one.

### 4.4 Still off the table
Per-pixel bias maps and any per-structure comparison. E4 point 5 (bias hotspots overlapping
structural-metric failures) survives only at region granularity — coarser than written, still
meaningful.

---

## 5. Choice of φ — CPA, UNI, or a Structural Vector

### 5.1 CPA is a formally valid φ
The identity is an exact Euclidean decomposition and holds for **any** map φ into an
inner-product space. ℝ¹ is one. With φ = CPA it reduces to the ordinary scalar bias–variance
decomposition:

- `μ(x) = E_m[CPA(G_m(x))]`
- `Var(x) = E_m (CPA(G_m(x)) − μ(x))²`
- `B(x) = (μ(x) − CPA(y))²`
- `E_m (CPA(G_m(x)) − CPA(y))² = Var(x) + B(x)` — exact.

Nothing in the derivation required φ to be high-dimensional or learned. The rule about
averaging in feature space rather than pixel space still holds and is satisfied: average CPA
*values* across ensemble members, never the images.

### 5.2 Why CPA suits non-adjacent data — signal-to-floor ratio
The argument is not dimensionality, it is **which directions you measure in**.

A single tile's UNI embedding is dominated by configuration detail — where each structure
sits — which decorrelates completely between non-adjacent levels; per-tile bias in UNI space
is unrecoverable from this data. Region-averaged UNI does concentrate, but retains many
nuisance directions (stain colour, texture, scanner) where real-vs-real cross-level variance
is large and carries no biological signal. The floor eats you there.

CPA is a hand-picked projection onto a direction where cross-level variance is **small** — a
marginal statistic, invariant to where the collagen sits, sensitive only to how much — and
biological signal is **large**. Its floor is the level-to-level sampling variability of a
scalar mean, shrinking roughly as `σ²/N` in the number of independent structures per region.

Two further wins: the floor becomes trivially estimable (split-half within a slide gives CPA's
spatial sampling variance directly, in 1-D rather than 512-D), and **bias lands in clinical
units** — "over-calls collagen by 2.3 percentage points" is actionable for a pathology
audience; UNI-space bias magnitude is not.

### 5.3 The cost — and it bites the headline
CPA is a scalar projection, so **CPA-bias is a lower bound on total bias, blind to spatial
hallucination.** A model that fills lumens with collagen while stealing it from septa — total
collagen fraction exactly right — shows **zero** CPA bias. That is precisely the failure mode
E7 and V1 exist to catch, and the one the headline claim ("hallucination *is* the bias term")
most needs to demonstrate.

The same compression hits the variance side: ensemble members differing spatially but agreeing
on total CPA look near-identical. This can be framed positively — it is the uncertainty of the
*clinical readout*, the decision-relevant quantity — but that is a **different claim** from
uncertainty of the image, and the two must not blur together in the write-up.

### 5.4 Recommendation — a whitened structural vector, plus UNI in parallel

Use a small vector of **marginal structural descriptors** rather than CPA alone:

```
φ_struct = [ task_specific_value,      # CPA for PSR; pluggable per stain
             β₀, β₁,                   # shape of the positive mask
             regional_dispersion,      # arrangement of the positive mask
             lumen_fraction,           # geometric, H&E-referenced
             tissue_fraction ]         # geometric, H&E-referenced
```

Every component is an aggregate statistic, so all inherit CPA's level-offset robustness and
all stay interpretable — but the space now has enough directions to register architectural
bias that scalar CPA misses. The **β₁ component is what catches the lumen-filler** of §5.3.

The components are on incommensurable scales (fractions vs counts), so the vector must be
normalised before any `‖·‖²` is taken — see **§5.5**, which is a prerequisite for every bias
number in §4.

#### 5.4.1 Constraint — measurable in the target stain
Descriptors must be computable **in the generated stain, on both real and virtual images**.
This removes two candidates from earlier drafts:

- **Nuclear density — dropped.** PSR without a nuclear counterstain does not resolve nuclei, so
  the descriptor is not measurable in the output modality and no choice of reference recovers
  it. *(Worth one check before it is settled: classical picrosirius protocols often include a
  Weigert's haematoxylin counterstain, which would make nuclei visible and restore the term.)*
- **H&E colour deconvolution is not the route to a collagen channel here.** Macenko estimates
  stain vectors from the image itself and does transfer to PSR with PSR stain vectors — but the
  repo's `cross_stain_consistency.py` implementation deconvolved *H&E* for the cross-stain
  comparison, which is a different question from real-PSR vs virtual-PSR.

#### 5.4.2 What is actually stain-specific
Stain-specificity is the wrong axis: every mask-derived descriptor above is computationally
generic — they are shape statistics of a binary mask. CPA is the most transferable of all
(`positive / tissue`, an area fraction). What varies is *which* mask you feed them and what the
number means biologically: on collagen β₁ counts fibrous loops, on CK19 it counts duct lumens.

The axis that costs effort is:

| | needs a per-stain segmenter |
|---|---|
| task_specific_value, β₀, β₁, regional_dispersion | **yes** — all need the positive-class mask |
| lumen_fraction, tissue_fraction | **no** — geometric, from the H&E |

**All four mask-derived terms ride on one segmenter per stain.** Keeping four costs no more
segmentation work than keeping one, so the only saving from a shorter vector is analysis and
write-up complexity. Hence: abstract the clinically-named quantity into a single pluggable
`task_specific_value` slot, and keep the shape terms alongside it rather than reducing further.

Caveat on the abstraction: if `task_specific_value` is the only interpretable term, the
"bias in clinical units" headline (§5.2) rests entirely on it and the shape terms become
supporting evidence. That is a defensible framing — *over-calls collagen by 2.3 percentage
points, and here is evidence it also puts it in the wrong places* — but it should be a
deliberate choice in the write-up, not an accident.

#### 5.4.3 Failure-mode coverage
| Failure mode | task_value | β₀ | β₁ | dispersion |
|---|---|---|---|---|
| Wrong amount | **✓** | | | |
| Right amount, fills lumens | | | **✓** | |
| Right amount, fragmented | | **✓** | | |
| Right amount, scrambled directions | | | | **✓** |
| Right amount, fibres too thick/thin | | | | *(uncovered — see below)* |

**Fibre thickness and local coherence were considered and dropped.** Both are computable
(thickness = 2 × mean of the Euclidean distance transform sampled on the skeleton; local
coherence = mean structure-tensor coherence `((λ₁−λ₂)/(λ₁+λ₂))²`). Local coherence overlaps
β₀/β₁ — the amorphous-vs-fibrous case is already caught topologically. Thickness is the one
genuinely uncovered failure mode, but at 0.442 µm/px a 2 µm fibre spans ~4.5 px and a 1 µm
fibre ~2.3 px, so the descriptor saturates near the sampling limit. **Verify the fibre-width
distribution on a few real SR slides before deciding**; if most fibres land at 2–3 px it
carries little information and the omission is free.

Note the distinction that motivated the split: "orientation coherence" conflates two different
measurements. *Local* coherence asks whether structures are locally elongated (high for any
fibrous texture); *regional dispersion* asks whether they point the same way, via circular
statistics on the doubled angle, `1 − |mean(exp(2iθ))|`. Only the second distinguishes aligned
from scrambled fibres of identical thickness — so it is the one retained.

#### 5.4.4 Compute the descriptors on the binary mask, not the intensity
β₀, β₁ and regional dispersion should be evaluated on the **thresholded positive mask**, never
on the greyscale. §6.2 warns that a global colour offset in the virtual stain would masquerade
as genuine bias; structure-tensor and intensity-based measures are sensitive to exactly that,
whereas mask-derived shape statistics are immune to any colour or contrast discrepancy between
real and virtual PSR. This removes a confound from the headline measurement at no cost.

β₀ and β₁ require **no classifier and no new segmentation** — they are read off the same mask
that yields the task-specific value, in four lines of `scipy` (already a dependency):

```python
from scipy import ndimage
b0 = ndimage.label(mask)[1]                       # connected components
holes = ndimage.binary_fill_holes(mask) & ~mask   # enclosed background
b1 = ndimage.label(holes)[1]                      # loops
```

Caveat: topology is far more sensitive to mask noise than area is — β₀ counts every speck as a
component. Apply `remove_small_objects` and a morphological closing first, fix those parameters
once, apply them identically to real and virtual, and report them.

#### 5.4.5 Keep UNI as a parallel track, not a replacement
The two answer different questions: `φ_struct` is interpretable but partial; UNI is sensitive
but uninterpretable and floor-limited on this data. Report both. Agreement on the ranking of
high-bias regions strengthens the result; **disagreement is itself informative** about what
kind of bias is present.

### 5.5 Normalising the Vector

The components live on wildly different scales — CPA and the fractions are in [0,1], β₀ and β₁
are counts in the hundreds — so an unnormalised `‖·‖²` is meaningless. Simulating plausible
per-region values with a genuine bias injected into β₁, the share each dimension takes of the
squared norm:

| | raw | z-scored | **floor-whitened** |
|---|---|---|---|
| CPA | 0.0% | 10.6% | 7.4% |
| **β₀** | **67.7%** | 9.2% | 6.4% |
| **β₁** | 32.3% | 48.9% | **64.0%** |
| regional_dispersion | 0.0% | 13.2% | 9.3% |
| lumen_fraction | 0.0% | 9.0% | 6.4% |
| tissue_fraction | 0.0% | 9.1% | 6.4% |

**Raw:** β₀ takes 68% purely because it is a count. The four bounded descriptors contribute
*nothing* — any bias in them is numerically invisible.

**Z-scoring is not sufficient.** It fixes the scale but not the correlation. β₀ and β₁ are
strongly correlated (more collagen → more components *and* more loops), so diagonal scaling
double-counts that shared direction and still credits β₀ with 9% despite it carrying almost no
independent bias.

**Full whitening** concentrates 64% on β₁, where the bias actually is.

#### 5.5.1 The recipe
Estimate the cross-level covariance `Σ` of `φ_struct` from the real-vs-real bracket (§6.1),
then use the Mahalanobis norm `‖·‖²_{Σ⁻¹}`. Two properties follow:

- **Signal-to-noise weighting.** Directions where levels naturally disagree are downweighted
  automatically — exactly right when the floor is the adversary.
- **The subtraction becomes exact.** With `Σ = Cov(δ)` for the level-offset noise δ,
  `E‖δ‖²_{Σ⁻¹} = d`, so `bias² = observed²_{Σ⁻¹} − d`. In simulation this recovers
  `15.56 − 6 = 9.56` against a true `9.51`.

#### 5.5.2 Four prerequisites
1. **Counts → densities.** β₀ scales with region area, so raw counts are ill-defined when
   regions differ in size. Use per mm² of tissue. This precedes scale-matching; it is about
   the descriptor being well-defined at all.
2. **Variance-stabilise the counts.** β₀/β₁ are Poisson-like (variance grows with the mean)
   while whitening assumes roughly elliptical structure. Apply `sqrt` or `log1p` first.
3. **Estimate Σ from the floor, never from the observed discrepancies.** The trap: whitening
   by the covariance of virtual-vs-real differences normalises away the bias being measured.
   Σ must come from real-vs-real (§6.1). Same failure mode as §6.2's warning against
   regressing out a global colour offset.
4. **Shrink it.** `d = 6` means 21 free parameters. Regions supply thousands of samples but
   are correlated within slide, so the effective n for the between-case component is nearer
   20 (liver). Use Ledoit–Wolf shrinkage rather than the raw empirical covariance.

#### 5.5.3 Common Σ or per-organ Σ
Whitening by each organ's *own* floor puts bias in that organ's floor-SD units — so "2
floor-SDs" denotes different biological magnitudes in liver and kidney. That is a coherent
signal-to-noise reading, but it is **not** a magnitude comparison, and §4.3's pooled
bias²-vs-Mahalanobis figure needs the latter. Use a **common Σ** for the pooled plot; report
per-organ whitening alongside, and say which is which.

#### 5.5.4 You may not need the scalar at all
Nothing forces a collapse to one number. Per-descriptor bias in **native units** — *"over-calls
CPA by 2.3 percentage points; 12 fewer collagen loops per mm²"* — requires no normalisation and
is far more interpretable for a pathology audience. The whitened scalar is needed only for the
single pooled figure.

**Report both:** native units in the tables, whitened scalar for §4.3.

---

## 6. The Biological Floor

### 6.0 Not every descriptor pays it — two reference classes

The floor is a consequence of *which reference* a descriptor is compared against, and the
vector splits in two. Recall the geometry: H&E is at **level A**, so the virtual stain depicts
**level A** tissue; the real PSR is at **level B**.

- **Stain-dependent terms** (`task_specific_value`, β₀, β₁, `regional_dispersion`) are not
  visible in H&E, so their only reference is the real PSR at level B. They span levels and
  **pay the floor**. These are the terms §6.1–§6.2 below are about.
- **Geometric terms** (`lumen_fraction`, `tissue_fraction`) *are* visible in the H&E input.
  Compare them against the **H&E at level A**, not the real PSR at level B: same physical
  section, pixel-aligned by construction because inference is tile-for-tile at identical
  coordinates. **No level offset, therefore no floor.**

The second class is a strictly better measurement, and it changes what is being asked: *does
the model preserve the structure that was present in its own input?* Any deviation is pure
model error — the lumen was visible in the H&E, so the model has no excuse. It also consumes
no real PSR, which means it survives even if the §7 pilot finds no headroom for the
stain-dependent terms.

One caveat on the floor-free class: whitespace also arises from tears, folds and processing
artefacts, which do differ between sections. Threshold conservatively on near-white and do not
over-interpret small differences.

| Term | Reference | Floor |
|---|---|---|
| task_specific_value, β₀, β₁, regional_dispersion | real PSR, level B | **yes** |
| lumen_fraction, tissue_fraction | H&E input, level A | **none** |

### 6.1 Bracketing the floor for the stain-dependent terms

Any agreement number needs the floor subtracted (`uncertainty_strategy.md:198`), but
real-vs-real cross-level discrepancy cannot be measured directly — there is only one stain per
level. Bracket it from both sides:

- **Upper bound — stain-invariant structures across the two levels.** Lumen/vessel fraction,
  tissue fraction, and E7's sublevel-filtration-on-tissue-density
  (`evaluation_strategy.md:225`) are computable from **both** H&E and PSR. Compute one on real
  H&E at level A and real PSR at level B; the discrepancy is pure level-offset **plus** stain
  and protocol differences. It genuinely spans the two levels and needs no extra slides.
  Absorbing the stain differences makes it an *over*-estimate → under-states bias →
  conservative, the right direction.
  *(Nuclear density was listed here in an earlier draft; it is not computable from PSR without
  a nuclear counterstain — see §5.4.1 — so it cannot serve as a cross-stain bracket.)*
- **Lower bound — split-half within a slide.** Partition the tissue of one real PSR slide into
  two disjoint region sets, compute the descriptor on each; the spread is a spatial sampling
  floor at that region size. Does not span levels, so it under-estimates.

**Report bias with a sensitivity band across the two bounds** rather than a single number.
More honest than either alone, and pre-empts the obvious reviewer objection.

### 6.2 Positive-mask confound — colour offset and segmenter drift
The task-specific value depends on how the positive mask is obtained. Applied to real PSR
versus virtual PSR, an identical rule can behave differently if the virtual stain's colour
statistics are slightly off — and that measurement artefact is **indistinguishable from genuine
model bias**. A global colour offset would masquerade as exactly the systematic error being
claimed.

Two routes, with different exposure:

- **Deconvolution + threshold.** Directly vulnerable as above; the threshold is the confound.
- **Learned segmenter (what the repo does today).** CPA currently comes from
  `compare_psr.py:compute_psr_fraction` counting **nnU-Net Dataset314_SR_light** mask labels —
  there is no deconvolution and no threshold to tune. The exposure moves rather than
  disappears: that network was trained on **liver** SR, so applying it to **kidney** PSR is
  itself an out-of-distribution use, and segmenter failure on kidney would be
  indistinguishable from the model bias under measurement. **Validate the segmenter on real
  kidney PSR against manual annotation before any kidney number is trusted** — this belongs
  in the §7 go/no-go alongside the floor estimate.

Note that the shape terms (β₀, β₁, `regional_dispersion`) are computed on the binary mask and
are therefore immune to colour and contrast discrepancy (§5.4.4) — but *not* to segmenter
drift, since they inherit whatever mask they are given, and topology is more sensitive to mask
error than area is.

Characterise before trusting any number: run the CPA pipeline on real PSR across staining
batches, quantify the threshold-induced spread, fold it into the floor. **Do not** fix it by
regressing out a global offset — that removes the global bias term being measured.

---

## 7. Go / No-Go Pilot

**Estimate the floor before building the pipeline.** If observed virtual-vs-real discrepancy
lands close to the floor there is no headroom, and `bias² = observed² − floor²` comes out near
zero or negative. Cheap to check on a handful of slides; a genuine go/no-go.

When running it, **report negative point estimates rather than clipping at zero** — clipping
biases the whole error budget upward. Bootstrap CIs over WSIs.

---

## 8. Caveats

- **Cortex/medulla confound.** The compartments differ sharply in collagen content, so levels
  sampling different proportions shift CPA for purely anatomical reasons. Restrict to a cortex
  mask or report compartment proportions alongside.
- **Aggregate statistics only** (absent successful registration, §3). Different levels means
  different anatomy at fine scale — aggregate over regions ≥1 mm, never per-tile.
- **Priors do not transfer.** Kidney fibrosis is interstitial; liver fibrosis is
  portal/bridging. The E5 collagen-plausibility prior and the β₁ ↔ fibrosis-stage
  interpretation are calibrated on liver. Frame kidney as **failure detection**, not as an
  accuracy benchmark — which is what an OOD set is for.
- **Hold-out discipline.** §6.1 of `uncertainty_strategy.md` is explicit that the benchmark is
  not the training corpus. In-sample bias reads artificially small.

---

## 9. Open Questions

1. **Level spacing?** Consecutive ribbon sections vs a re-cut hundreds of microns deeper are
   very different. Under ~100 µm → structurally serial for glomeruli and vessels, and §3
   becomes likely to work.
2. ~~Pairs or slides?~~ **Resolved (2026-08-01):** 20 liver *pairs* and 40 kidney *pairs*.
   Both clear the ~15-case threshold for reporting correlation coefficients.
3. ~~Are the liver WSIs held out?~~ **Resolved (2026-08-01):** yes, the 20 liver pairs are
   disjoint from the training specimens. §8 hold-out discipline satisfied.
4. ~~Resolution / scanner confound?~~ **Resolved (2026-08-01):** same magnification and pixel
   width across organs. Scanner and staining batch not yet confirmed — see §1.
5. **Is a second real PSR level available for any case?** Would replace the §6 bracket with a
   direct floor measurement. Worth re-asking now the kidney set is 40 pairs: even a handful of
   two-level cases would materially de-risk the §7 go/no-go.

---

## 10. Summary

| Experiment | Non-adjacent data sufficient? | Notes |
|---|---|---|
| E3 — distributional OOD | **Yes, fully** | kidney H&E alone; AUROC |
| V — double dissociation | **Yes** | kidney = unseen-organ knob |
| E7 step 3 — topological realism | **Yes** | registration-free by design |
| Distribution-level FID/KID | **Yes** | vs real kidney *and* liver PSR |
| **E4 bias, region level, φ_struct** | **Yes** | §4–§5; needs floor bracket (§6) and floor-whitening (§5.5) |
| E4 money figure (var vs bias) | **Yes, at region level** | scalar/vector φ compresses variance — see §5.3 |
| bias² vs Mahalanobis, both organs | **Yes** | the §4.3 target figure |
| E4 / E5 with φ = UNI | **Weak** | floor-limited; parallel track only (§5.4.5) |
| E4 / E5 at tile level | **Maybe** | only if thumbnail registration succeeds (§3) |
| E4 / E5 per-pixel | **No** | never; not the design anyway |
