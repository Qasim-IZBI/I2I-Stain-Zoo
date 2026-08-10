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
post-processing steps. `scripts/recon_real_psr.sh` stitches real tiles, but there is
**no committed script that segments the real PSR** — run `nnUNetv2_predict` with
`Dataset314_SR_light` over its output by hand, then `apply_he_mask_real.sh` →
`fill_tissue_holes_real.sh`.

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
`summary.json`.

`summary.json` reports `"bias": {"computed": false}` by design. Bias is branch B and
has not passed its gates.

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
python estimate_floor.py \
    --real_psr /path/real_psr_kidney/psr_masks_wsi_final/ \
    --tiles_metadata /path/tiles/testB \
    --real_he /path/reconstructed_he/ \
    --outdir ./floor_kidney/
```

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
| `cross_stain` (`--real_he`) | lumen, tissue only | conservative |
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
| Region correspondence to real PSR | bias | in progress |
| Real-PSR segmentation script | bias, floor | run `nnUNetv2_predict` by hand |
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
2.  segment the real PSR                        both cohorts        manual
2b. annotate cortex masks                       kidney only         manual
    │
    ├─ 3.  compute_phi_uncertainty.py           ── no gates, start now
    │        liver: no --roi_dir
    │        kidney: --roi_dir cortex_masks/
    │
    └─ 4a. eyeball kidney masks                 ── free
       4b. stain_sensitivity.py                 ── hours, GATE
       4c. estimate_floor.py                    ── hours, GATE
           │
           └─ 5. bias, once registration lands
```

If you only do one thing after the masks exist: **run step 3**. It is the largest,
safest and most complete part of the work, and it does not depend on anything still
unresolved.

If you only chase one loose end: **ask the archive for a second PSR level**, even on
five cases. It replaces the variogram substitute with a direct measurement and removes
the biggest assumption left in the bias argument.
