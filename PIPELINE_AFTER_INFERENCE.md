# What to do after inference — liver + kidney

> A runbook for the stage after the ensemble has translated H&E → virtual PSR on the
> held-out cohorts. Companion to `uncertainty_strategy.md` (the experiments),
> `kidney_ood_data_plan.md` (the data and the floor) and `CLAUDE.md` (full flag
> reference). Written 2026-08-02, revised 2026-08-10.

> **Revision note.** This document previously described the **UGAC** chain. UGAC was
> retired as the generator on 2026-08-09 — the GGD-NLL cycle loss did not produce
> usable virtual stain — and the decomposition identity carries no aleatoric term, so
> nothing downstream needed the heads. The chain below is **vanilla CycleGAN** on a
> crossed subset × seed grid (`scripts/*_grid.sh`). The `*_ugac.sh` scripts remain in
> the repository for provenance; do not mix their outputs with these.
>
> Also corrected: the kidney cohort is **20 case pairs**, not 40.
> `kidney_ood_data_plan.md` says 40 and is wrong.

---

## 0. The one thing to understand first

The work splits into **two branches with very different risk**:

```
                      masks exist
                           │
            ┌──────────────┴──────────────┐
            │                             │
     UNCERTAINTY branch            BIAS branch
     no gates, no registration     two gates, needs registration
     runnable immediately          may not survive the gates
            │                             │
     procedural + data              bias² per region
     exposure variance              vs the real PSR
```

The **uncertainty branch needs nothing you don't already have.** Every ensemble
member generates from the same H&E, so region *r* is the same tissue across
members; disagreement between them is pure model uncertainty. No real target, no
registration, no floor.

The **bias branch compares against the real PSR**, which sits at a different
section level. That needs region correspondence and a measured floor — and either
gate can come back saying "no headroom here".

> **It did.** The §7 floor pilot was run on the liver cohort at 0.75, 1.5 and
> 2.5 mm regions, and no descriptor reaches `usable` at any of them (§4a). The
> bias branch is **closed on this cohort**; branch A is the result. What follows
> still describes how to run branch B, because the kidney arm and any cohort with
> a second PSR level need it.

> **And branch A gained a second half.** Quantifying the spread does not show it
> is *meaningful* — a reviewer will ask whether low variance means correct. §3a
> calibrates the spread against φ_struct of the real tissue, per descriptor.
> That half was designed with a floor-free lumen arm referenced to the real H&E;
> on this cohort the lumen descriptors turned out to be unmeasurable (§3a), so
> the calibration is over the four collagen descriptors against the real SR — and
> therefore needs the same region correspondence branch B does.

**So: run the uncertainty branch while you resolve the bias gates.** Do not
sequence them one after the other; the first does not depend on the second.

---

## 1. What you should have before starting

| Artefact | Where from | Used by |
|---|---|---|
| Virtual PSR **tiles**, per member | `infer_ensemble_cyclegan_grid.sh` | stage 2 |
| Real **PSR** WSIs (level B) | the cohort | bias, floor |
| Real **H&E** WSIs (level A) | reconstruct from testA tiles | geometric descriptors |
| `tiles_metadata.csv` per WSI | tiling | region grids everywhere |
| **Cortex masks**, kidney only | manual annotation | `--roi_dir`, §3 |

Cohorts: **liver 20 case pairs** (40 WSIs, in-distribution, held out from training)
and **kidney 20 case pairs** (40 WSIs, out-of-distribution, same animals).

The ensemble grid is **K = 5 disjoint training subsets × S = 10 seeds = 50 members**,
spanning training folders 001–035 in subsets of seven. Folders 031–035 were tiled on
2026-08-10, so all five subsets are live and every script runs its full array.

> The committed scripts point at the 5-WSI BMVC test set. For the 20-case cohorts,
> change `TEST_A` at the top of the inference script, `N_WSIS` in the recon and
> segmentation scripts, and `WSI_COUNT` in the two post-processing scripts, then scale
> `--array` accordingly. The decomposition constants (`RANGE_STARTS`, `RANGE_ENDS`,
> `N_MEMBERS`) must stay identical across all six or the array indices stop lining up.

---

## 2. Tiles → masks (both cohorts)

```bash
sbatch scripts/recon_ensemble_grid.sh       # tiles  -> reconstructed WSIs
sbatch scripts/segment_psr_grid.sh          # WSIs   -> wsi_masks/
sbatch scripts/apply_he_mask_grid.sh        #        -> wsi_masks_cleaned/
sbatch scripts/fill_tissue_holes_grid.sh    #        -> wsi_masks_final/
```

Run them in order; each has a skip guard, so a re-submit after an interruption is
safe.

**Why reconstruct at all, rather than working on tiles?** β₀ and β₁ count connected
components and loops, and those cross tile boundaries — the topology of a region is
not a function of its tiles' topologies. Stitching is a hard prerequisite, not an
optimisation.

**Watch out:** reconstructions come back at the **source** resolution (0.221 µm/px),
not the 0.442 µm/px the model saw, because `reconstruct_wsi` upsamples each tile back
to `tile_size`. Everything downstream sizes regions in millimetres for exactly this
reason. Leave `--mpp` at its default unless your reconstructions genuinely differ.

You also need the **real** PSR put through segmentation and the same two
post-processing steps. Two routes, differing only in what gets segmented:

```bash
# (a) stitched testB tiles — the BMVC route. Still no committed segmentation
#     script: run nnUNetv2_predict with Dataset314_SR_light over the output by hand.
sbatch scripts/recon_real_psr.sh
sbatch scripts/apply_he_mask_real.sh
sbatch scripts/fill_tissue_holes_real.sh

# (b) the original thumbnail-registered SR WSIs — no reconstruction at all
sbatch scripts/segment_psr_real.sh            # --array=0-19, one slide per task
sbatch scripts/apply_he_mask_real_sr.sh
sbatch scripts/fill_tissue_holes_real_sr.sh
```

Route (b) buys you the sharper input — no 512→256→512 round trip — at the cost of
two things you have to check yourself, because neither is visible in the output.
**Resolution parity:** `Dataset314_SR_light` is a 2d model with a fixed patch size,
so if the originals are not at the same mpp as the virtual arm's reconstructions,
every CPA difference is confounded with scale; `segment_psr_real.sh` logs each
slide's geometry for comparison. **Footprint accuracy:** on the virtual arm the H&E
mask is exact by construction, here the SR is a serial section registered only at
thumbnail level, and the nearest-neighbour resize fixes scale but not offset.

Both routes end at `psr_masks/real/psr_masks_wsi_final/`, which is what
`compare_psr.py` and `estimate_floor.py --real_psr` consume.

---

## 3. Branch A — uncertainty (start this immediately)

```bash
python compute_phi_uncertainty.py \
    --fold /work2/.../ensemble_grid/cyclegan/data_001_007/model_small/wsi_masks_final \
    --fold /work2/.../data_008_014/model_small/wsi_masks_final \
    --fold /work2/.../data_015_021/model_small/wsi_masks_final \
    --fold /work2/.../data_022_028/model_small/wsi_masks_final \
    --fold /work2/.../data_029_035/model_small/wsi_masks_final \
    --tiles_metadata /path/tiles/testA \
    --he_dir /path/reconstructed_he \
    --outdir ./phi_uncertainty_liver/
```

On SLURM that is `sbatch scripts/compute_phi_uncertainty_grid.sh`, with every path
taken from the environment so the kidney arm needs no edit. On a 20-case cohort
prefer the array form, one WSI per task:

```bash
sbatch scripts/compute_phi_uncertainty_grid_array.sh          # --array=0-19
python aggregate_phi_uncertainty.py \
    --indir  /work2/.../phi_uncertainty/per_wsi \
    --outdir /work2/.../phi_uncertainty --expect 20
```

Splitting over WSIs is exact rather than an approximation — `decompose()` works
region by region and regions never cross slide boundaries, so the per-WSI files hold
final per-region numbers and only the three means need pooling. Every task still
reads **all five folds**: the split is over WSIs, never over folds, because one fold
alone yields procedural variance and no data-exposure term at all.

Pass `QC_DIR=` to write one region per WSI as a TIF pair — the label mask
(0 outside, 1 tissue, 2 lumen) beside its H&E crop, openable in Fiji — and look at
a couple before pooling. `lumen_fraction` is thresholded, so the number alone
cannot distinguish "found the lumens" from "found pale tissue". `QC_MAX_PX=2000`
keeps the crops small.

**Check `mu_lumen_fraction` in `per_region.csv` on the first slide that finishes.** A
value around 1e-5 means `--white_thresh` (`WHITE_THRESH=` on the SLURM scripts, default
0.85) sits above this cohort's lumens and they are being counted as tissue. `he_bright`
needs **every channel** over the cut, so set it from the per-pixel channel *minimum* —
an 8-bit conversion shows a channel average, which is always higher. On the UC liver
cohort the committed value is **0.65**, from `scripts/calibrate_white_thresh.sh`:
the H&E footprint is stable over 0.500-0.675 and the SR over 0.600-0.700, and
neither stain has a plateau inside its window — so the number is a convention held
fixed across both arms, not a measurement. Only the two H&E-referenced descriptors move when you change
it; both are identical across members and contribute zero variance, so the
procedural/data split does not need recomputing — but re-run before quoting the
level-A columns, and use the same value for `estimate_floor.py`.

`--he_dir` also accepts the **original H&E WSIs** rather than reconstructions. Tiling
starts at `(0,0)` with stride = tile size and `reconstruct_wsi` upsamples tiles back
to `tile_size`, so both sit in the same pixel frame and region boxes index identical
pixels; the original just skips the resampling round trip.

One `--fold` per training subset, all five. With several folds you get the full split:

```
Var_total  =  Var_k( E_s[·] )  +  E_k[ Var_s(·) ]
                data-exposure       procedural
```

Members inside a subset share a training set and differ only by seed, so their spread
is **procedural**. Subset means differ because the subsets saw different slides, so
their spread is **data-exposure**. Pass a single `--ensemble` instead and you get
procedural only, with the data component reported as *undefined* — not zero, because
one subset cannot support the claim that data exposure contributes nothing.

**Outputs:** `per_region.csv` (μ per descriptor, Var, procedural, data_exposure) and
`summary.json`. Note the CSV's two totals: `var_total_descriptor_space` is the pooled
plug-in variance, which ignores the fold structure, while `var_total_anova` is the
ANOVA total `summary.json` reports and the one that equals `procedural +
data_exposure`.

`summary.json` reports `"bias": {"computed": false}` by design. Bias is branch B and
has not passed its gates.

### 3a. Does the spread predict the error?

Ensemble spread measures disagreement between members, not error, and the BMVC
2026 result is that cycle-reconstruction error does not calibrate it. This scores
the spread against an external target — φ_struct of the real tissue.

> **Liver, 2026-08-16: only the collagen arm is available, so Step 0 gates the
> whole study.** Both routes to a lumen mask are closed. Thresholding the
> generated SR does not work — its histogram has no bimodality, the footprint
> sweeps from 7% of the canvas to 100% across the sweep, and at the H&E's own 0.65
> it calls 22% of the slide lumen against the H&E's 4% on the same tissue; the
> model does not reproduce whitespace. And `Dataset314_SR_light` labels lumen as
> tissue, so no mask holds enclosed background to find — a property of the
> segmenter, applying equally to the real SR. Run without `--lumen_root` and
> `--real_lumen`; the three terms report as having no reference. This removes the
> floor-free arm, so the earlier advice to "build the lumen half first, it is
> unblocked either way" no longer holds.
>
> ```bash
> sbatch scripts/check_frame_alignment.sh     # Step 0 — run this first
> ```
>
> Header reads only, seconds across twenty multi-GB slides. It pairs by the
> `SR_`/`HE_` rule, reports unpaired stems first (a naming mismatch otherwise
> reads as a dimension mismatch), gives the scale ratio on any differing case —
> a clean 2.000 is a magnification level apart, a different fix from a
> registration problem — flags mismatched `XResolution`, and checks each slide
> against the *region extent* from `tiles_metadata`, which is what
> `calibrate_phi` actually compares against. Identical dimensions are necessary
> but not sufficient: two slides can match in size and still be offset, so
> overlay one case before trusting it.

```bash
# 1. lumen masks: the virtual side per member, then the reference from the H&E
sbatch scripts/make_lumen_masks_grid.sh
python make_lumen_masks.py --rgb_dir ${HE_RGB} --he_masks ${HE_TISSUE} \
    --white_thresh 0.65 --min_object_px 64 --outdir /path/lumen_masks_real

# 2. phi with the per-member lumen, on a pixel grid the heatmap can tile
sbatch --export=ALL,WHITE_THRESH=0.65,LUMEN_ROOT=...,REGION_PX=2048 \
    scripts/compute_phi_uncertainty_grid_array.sh
python aggregate_phi_uncertainty.py --indir .../per_wsi --outdir ... --expect 20

# 3. calibrate, and map
python calibrate_phi.py --phi_csv .../per_region.csv \
    --real_lumen /path/lumen_masks_real --he_masks ${HE_TISSUE} \
    --real_psr /path/psr_masks/real/psr_masks_wsi_final --outdir ./calibration_phi/
python plot_uncertainty_heatmap.py --phi_csv .../per_region.csv --downsample 32
```

**One footprint everywhere.** `--he_masks` is the same tissue mask
`apply_he_mask.py` applies to the collagen, filled so internal lumens count as
inside tissue. Pass it to `make_lumen_masks`, to the φ run and to `calibrate_phi`:
the enclosure test and every density denominator have to be built the same way on
both sides, and taking them from a mask rather than a threshold keeps
`white_thresh` out of the denominator — which is where the threshold sweep showed
it does most damage.

**The collagen arm needs the frame.** `--real_psr` scores the four collagen
descriptors against the real SR, which is only paired correctly if that SR was
resampled onto the H&E grid — `calibrate_phi` checks the geometry and exits rather
than scoring different tissue under the same region id. With the lumen arm closed
on this cohort, that check decides whether there is a region-level calibration at
all, or only a WSI-level one at n = 20.

**Read ρ, not the ECE.** ρ(σ, |error|) is the claim that survives noise in the
reference — a floor or a registration offset attenuates it toward zero, so a
positive value is conservative. E|z|/0.80 is the scale: above 1 the ensemble is
over-confident, errors exceeding its own spread. The normalised ECE is reported for
continuity with the BMVC pipeline but cannot distinguish a calibrated ensemble from
an uninformative one on synthetic data.

**Run it twice.** `--prediction grand` pairs the mean of all 50 members with the
total spread; `--prediction fold` pairs each subset's mean with its procedural
spread alone. Whether the first calibrates better than the second is the
data-exposure claim, and a flat seed-only ensemble cannot pose it.

### The kidney run is cortex-only

```bash
python compute_phi_uncertainty.py \
    --fold ... (all five, as above) \
    --tiles_metadata /path/tiles/testA_kidney \
    --he_dir /path/reconstructed_he_kidney \
    --roi_dir /path/cortex_masks/ \
    --outdir ./phi_uncertainty_kidney/
```

`--roi_dir` takes per-WSI binary masks named `<wsi_stem>.tif`, resized
nearest-neighbour if they were annotated at thumbnail magnification. Two reasons the
kidney arm needs it, and they are independent:

- **Clinical.** Cortex and medulla differ systematically in fibrosis distribution and
  are conventionally analysed apart. A grid sampling both mixes two populations, and
  CPA moves for anatomical reasons that have nothing to do with the model.
- **Methodological.** The variogram floor assumes rough isotropy at region scale, and
  cortex/medulla layering is precisely a directional structure.

A region must be **≥ 50 % covered** to be kept (`--min_roi_fraction`), not merely
centred inside: a region half in medulla is not a cortex measurement. A WSI with no
mask in `--roi_dir` is **excluded and warned about**, never passed through whole —
a missing case is recoverable, a silently contaminated one is not.

The liver run takes no `--roi_dir`.

Expect the data-exposure component to rise on the kidney arm; that is the OOD result.
Note that the kidney arm carries **20 cases, not 40**, so its intervals are wider than
earlier drafts of this document assumed.

---

## 4. Branch B — the two gates

Both are cheap and either can stop the bias work. Do them before building anything on
top.

### Gate 1 — is the segmenter trustworthy on kidney?

`Dataset314_SR_light` was trained on **liver** SR. Kidney is an out-of-distribution
use, and a segmenter failure there is indistinguishable from the model bias you are
trying to measure.

**1a. Look at the masks.** Free, and it catches catastrophic failure — near-zero or
near-total collagen. If it survives this, the uncertainty branch is fine regardless of
what follows.

**1b. Bound the differential.**

```bash
python stain_sensitivity.py make-series \
    --real_psr /path/real_psr_kidney/ \
    --virtual_psr /path/.../reconstructed/model_01/ \
    --outdir /work2/.../perturbation_kidney/

sbatch scripts/segment_psr_perturbation.sh

python stain_sensitivity.py analyse \
    --masks /work2/.../perturbation_kidney/masks/ \
    --tiles_metadata /path/tiles/testB \
    --outdir /work2/.../perturbation_kidney/
```

**Why eyeballing is not enough.** Applying the same segmenter to both arms cancels
*anatomy-driven* error, which is common to real and virtual. It does **not** cancel
*appearance-driven* error, because appearance is exactly where the two arms differ —
so the measurement error is correlated with the quantity being measured. Both masks
can look entirely reasonable while differing by a percentage point of CPA purely from
colour.

The test holds anatomy fixed and moves only appearance: a real slide is transformed
toward the virtual's colour statistics, t = 0 → 1, and re-segmented at each step. The
tissue never changes, so any descriptor drift is measurement artefact.

Read `shift_over_region_sd`. Above ~0.25 the segmenter reacts to colour at a scale
comparable to real biological variation — fold that shift into the floor and treat a
bias of similar size as unproven.

Also check `series.json` for the out-of-gamut fraction. Clipping is non-invertible and
would break the fixed-anatomy premise; if it climbs past a percent, narrow
`--fractions`.

### Gate 2 — is there any headroom above the floor?

```bash
sbatch scripts/estimate_floor.sh          # 12 h / 96 G, single job, not an array

# region-size sweep — the one knob that moves the verdict
sbatch --export=ALL,REGION_MM=0.75,OUTDIR=./floor_075 scripts/estimate_floor.sh
python plot_floor_sweep.py --runs ./floor_075 ./floor_150 ./floor_250 \
    --outdir ./floor_sweep/
```

`--tiles_metadata` is optional and the committed script omits it: the real SR has
no tiling, so the grid is sized from each mask. Outputs now include **`floor.png`**
— panel A the verdict per descriptor over the usable/marginal/floor-limited bands,
panel B the variogram curves. Read B first: a floor from a sill that never
flattened is an under-estimate, and an under-estimated floor makes bias read high.

`--real_psr` gives masks; the cross-stain bound also needs the PSR **RGB**
(`--real_psr_rgb`), since the two stain-invariant descriptors must be measured on both
images. `--real_he` alone produces no cross-stain bound and says so. Set
`--white_thresh` to the same value used for `compute_phi_uncertainty.py`, or the floor
and the quantity it bounds are measured differently; `--white_thresh_psr` exists because
the two stains sit at different whitespace levels and defaults to `--white_thresh`.

### 4a. What it returned on the liver cohort

| region | CPA | β₀ | β₁ | dispersion | regions | pairs / lag span |
|---|---|---|---|---|---|---|
| 0.75 mm | 1.06 | 1.11 | 1.00 | 1.23 | 1058 | 13,690 / 4.9× |
| 1.5 mm | 0.87 | 0.97 | 0.80 | 1.19 | 279 | 1,197 / 2.2× |
| 2.5 mm | 0.71 | 0.76 | 0.67 | 0.93 | 99 | 120 / 1.4× |

Ratios improve with region size, as §4.2 predicts — the floor averages out faster
than the biology. But **nothing reaches `usable` (<0.5)**, and the best numbers
rest on the weakest evidence: at 2.5 mm the sill spans 1.4× of lag over 120 pairs,
where a flat variogram is the absence of evidence rather than evidence of a
plateau. At 0.75 mm, where the estimate is well conditioned, everything is
floor-limited. CPA would need ~6 mm regions to clear 0.5, leaving ~15 regions
across 20 slides.

The cross-stain arm contributed nothing here and is off in the committed script:
it bounds only the two level-A descriptors, and the SR can measure neither — its
footprint is unstable across the whole threshold sweep. Those rows read *unknown*.

**Conclusion: bias is not claimable on the liver cohort.** Report branch A and
this sweep as the documented reason. `--psr_level_b` — a second real PSR level per
case — is the only estimator that supersedes the variogram and would reopen it.

**What the floor is.** The real PSR sits at a different section level from the H&E the
model saw. Two levels of the same block differ for purely biological reasons, and that
difference is subtracted from every bias number: `bias² = observed² − floor²`. If the
observed discrepancy lands near the floor, bias comes out at or below zero and there is
nothing to report.

Four estimators, precedence applied **per descriptor**:

| Source | Covers | Direction |
|---|---|---|
| `direct` (`--psr_level_b`) | all six | measured — needs a second real PSR level |
| `variogram` (default) | all six | conservative |
| `cross_stain` (`--real_he` + `--real_psr_rgb`) | lumen, tissue only | conservative |
| `split_half` (always) | all six | **anti-conservative** |

You have no second PSR level, so the four collagen descriptors rest on the
**variogram**: in-plane spatial variation standing in for through-plane. It reads the
sill — the plateau of the semivariance curve — which over-estimates the floor and
therefore under-states bias, the safe direction. Note that `cross_stain` cannot reach
the collagen terms at all, since collagen is not measurable in H&E, so for those four
the variogram *is* the upper bound rather than a fallback behind one.

Read three columns:

- `floor_to_signal` — floor SD over between-region SD. `usable` < 0.5, `marginal`,
  `floor-limited` ≥ 0.9.
- `floor_source` — which estimator each descriptor fell back to.
- `bound_direction` — anything marked `anti-conservative` can support an *upper-bound*
  claim about bias, never a point estimate.

**If the topological terms come back floor-limited**, CPA stands alone and the
lumen-filler blind spot reopens: a model that gets total collagen right while putting
it in the wrong places would pass unnoticed. That is a real result worth reporting,
not a failure.

Run the kidney floor on the **cortex mask** too, for the isotropy reason above.

---

## 5. Branch B — bias itself (not yet built)

Blocked on two things:

1. **Region correspondence.** `region_grid` indexes the virtual side only. Bias needs
   a mapping from the H&E grid onto the real PSR WSI — a thumbnail affine, *not* pixel
   registration. Sub-region misalignment is expected and fine; the descriptors are
   densities, so only the expected count per mm² needs to match, never the individual
   structures.
2. **A floor with headroom**, from gate 2.

Once both land, bias per region is `bias² = ‖μ − φ(y')‖²_{Σ⁻¹} − d` with Σ from
`uncertainty_phi/floor.py` — never from the observed discrepancies, which would
normalise away the very thing being measured.

**A side benefit of the registration work:** once thumbnails are aligned, look at
whether the same large vessels and glomeruli overlap. If they do, the levels are closer
than assumed and a finer analysis scale opens up. If they do not, you are confirmed at
region scale — which is what everything is already built for.

---

## 6. Not built, and honest about it

| Piece | Needed for | Status |
|---|---|---|
| **Bias on the liver cohort** | — | **closed: no headroom at 0.75/1.5/2.5 mm (§4a)** |
| Region correspondence to real PSR | bias | in progress; moot on liver until a second PSR level exists |
| Real-PSR segmentation script | bias, floor | `scripts/segment_psr_real.sh` for the original registered SR WSIs; still manual for the stitched-testB route |
| **Cluster-robust intervals, case as unit** | every reported interval | **not started** |
| Cortex masks themselves | kidney arm | manual annotation; `--roi_dir` consumes them |
| UNI/Virchow encoder + Mahalanobis | E3 OOD detection | not started |
| AUSE / sparsification | E5 calibration | recoverable from `6b90c20` |
| Second real PSR level | direct floor | no data — variogram substitutes |

The cluster-robust intervals matter more than their one line suggests. Regions within
a slide are spatially correlated, so treating regions as independent understates every
interval, and the manuscript already commits to clustering on the case. Nothing in
`uncertainty_phi/` does this yet — `per_region.csv` is the input it will need.

---

## 7. Order of operations, condensed

```
1.  reconstruct + segment + post-process        both cohorts        SLURM
2.  segment the real PSR                        both cohorts        SLURM from the
                                                                    original SR WSIs,
                                                                    manual from tiles
2b. annotate cortex masks                       kidney only         manual
    │
    ├─ 3.  compute_phi_uncertainty.py           ── no gates, start now
    │        liver: no --roi_dir
    │        kidney: --roi_dir cortex_masks/
    │        20 cases: _grid_array.sh + aggregate_phi_uncertainty.py
    │
    └─ 4a. eyeball kidney masks                 ── free
       4b. stain_sensitivity.py                 ── hours, GATE
       4c. estimate_floor.py                    ── hours, GATE
           sweep region size, pool with plot_floor_sweep.py
           LIVER: returned no headroom at any size — branch closed
           │
           └─ 5. bias, once registration lands
```

If you only do one thing after the masks exist: **run step 3**. It is the largest,
safest and most complete part of the work, and it does not depend on anything still
unresolved.

If you only chase one loose end: **ask the archive for a second PSR level**, even on
five cases. It replaces the variogram substitute with a direct measurement and removes
the biggest assumption left in the bias argument.
