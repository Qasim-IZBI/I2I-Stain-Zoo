# Uncertainty Estimation and Calibration

Notes for paper writing. Covers the rationale, methodology, and how to interpret
and report epistemic uncertainty results for the unpaired image-to-image
translation models in this repo.

---

## 1. Why estimate uncertainty for virtual staining?

The translated images produced by these models (CycleGAN, UNIT, MUNIT, DCLGAN,
UVCGAN, MIUDiff) have no ground-truth target — the H&E and IHC sections come
from different physical slices, so per-pixel correctness cannot be measured
directly. A model can produce a plausible-looking IHC image while being wrong in
clinically meaningful ways (e.g. hallucinating PSR-positive regions where the
underlying tissue does not actually contain collagen).

For downstream clinical use, we need to know **where the model is unsure**.
Per-pixel uncertainty maps support:
- flagging tiles that need pathologist review,
- gating downstream automated analyses (e.g. PSR quantification),
- comparing models on a dimension orthogonal to FID/SSIM/LPIPS.

The two open questions answered by this pipeline:

1. *Where in each image is the model unsure?* → produced by `uncertainty.py`
2. *Can we trust those uncertainty values?* (calibration) → produced by `uncertainty_calibration.py`

---

## 2. Epistemic uncertainty via deep ensembles

We use the **deep ensemble** approach (Lakshminarayanan et al., 2017): train the
same architecture *N* independent times with different random seeds and
different SGD trajectories. Each member converges to a different local minimum.
Where the members disagree on their predictions, the model is uncertain
(epistemic uncertainty — uncertainty due to model ignorance, not data noise).

This works for image translation because:
- it requires no architectural changes (drop-in for any of the 6 models),
- it is fully post-hoc — uncertainty is computed at inference time from outputs
  alone,
- it captures the kind of uncertainty that matters in unpaired translation:
  *which features of the output were determined by the data vs. by the random
  initialisation of the generator*.

### Per-pixel uncertainty score

For ensemble members `m = 1 … N` with translated outputs `B_m(x, y, c)` (RGB):

1. **Per-channel sample variance** across the ensemble:
   $$\sigma_c^2(x, y) = \mathrm{Var}_{m=1..N}\big[B_m(x, y, c)\big],\quad \text{ddof}=1$$
2. **Scalar uncertainty map** by summing channels:
   $$U(x, y) = \sigma_R^2 + \sigma_G^2 + \sigma_B^2$$
3. Optional `log1p` compression to suppress heavy tails.
4. **Global percentile normalisation** across all tiles in the run
   (`p1`–`p99` by default) → `U_norm ∈ [0, 1]` for visualisation.

This gives one heatmap per tile with comparable scale across tiles and WSIs.

### Implementation: `uncertainty.py`

Inputs:
- `--data` directory containing `model_01/, model_02/, …` subfolders, each with
  the same set of translated TIF tiles.

Outputs (under `<out>/<arch>/`):
- `raw_npy/<stem>.npy`  — unnormalised `U(x,y)`, shape `(H, W)`. **Use this for calibration analysis.**
- `norm_npy/<stem>.npy` — globally normalised to `[0, 1]` (clipped at p1/p99).
- `heatmaps/<stem>.png` — magma-coloured visualisation with colorbar.
- `overlays/<stem>.png` — heatmap blended on the first ensemble member's image.
- `summary.json` — global bounds + per-image stats (mean, median, p95, p99).

**Important**: `norm_npy` is *clipped*, which destroys information at the
distribution tails. For any quantitative analysis (calibration, correlation),
read from `raw_npy/`.

---

## 3. Calibration: turning a heatmap into a number

A single heatmap is not enough to claim that uncertainty is meaningful.
**Calibration** asks: when the model says "I'm uncertain here", is it actually
wrong here? We need to compare uncertainty against an error signal.

In paired regression (e.g. depth estimation) the natural error signal is
$|y_\mathrm{pred} - y_\mathrm{true}|$. For *unpaired* translation, we have no
$y_\mathrm{true}$. The closest available proxy is **cycle-reconstruction
error**:

$$E(x, y) = \big|A(x, y) - F(G(A))(x, y)\big|$$

where $G: A \to B$ is the forward generator and $F: B \to A$ the inverse. A
pixel that the round trip cannot reconstruct is one where the forward
translation discarded information — a signature of translation difficulty.

This proxy is implemented in `evaluation.py:compute_regen_error` (cycle
reconstruction MAE in `[0, 255]`). The patched flag `--save_error_npy` writes
the per-pixel error map as `error_npy/<stem>.npy`.

### Two variants of the proxy

The repo supports two related proxies, exposed as separate metrics in
`evaluation.py`:

1. **Self-cycle** (`--metric regen_error`):
   $E_\mathrm{self}(x, y) = |A(x, y) - F_\mathrm{model}(G_\mathrm{model}(A))(x, y)|$
   Each model under evaluation provides both $G$ and $F$. Available only for
   architectures with both directions: cyclegan, unit, munit, dclgan, uvcgan.
   **Not available for MIUDiff** — diffusion models are one-way ($A \to B$
   only) and have no learned inverse.

2. **Judge-based** (`--metric judge_regen_error`):
   $E_\mathrm{judge}(x, y) = |A(x, y) - F_\mathrm{judge}(B')(x, y)|$
   where $B' = G_\mathrm{model}(A)$ is read from disk and $F_\mathrm{judge}$
   is a single fixed external GAN inverter applied to *every* model under
   evaluation. Works for any forward translator including MIUDiff.

**Recommendation for the paper**: use **judge-based** uniformly across all six
architectures. Two reasons:

- It is the only option that yields a calibration number for MIUDiff at all.
- Even for the GAN models, judging every model with the *same* fixed inverter
  removes a confound: with self-cycle, a model whose forward and inverse are
  jointly biased in the same way (e.g. both ignore a tissue feature) reports
  low cycle error despite poor translation. The external judge cannot be
  jointly biased with the model under test.

The judge can be any of cyclegan / unit / munit / dclgan / uvcgan; freeze
one trained checkpoint and reuse it everywhere. Document in the methods
section which architecture and checkpoint serves as judge.

### General caveats on the proxy

- Cycle error and translation error are correlated but not equivalent. A
  perfect generator pair can have non-zero cycle error if the inverse is
  imperfect; a poor generator can have low cycle error if both directions are
  jointly biased the same way (mitigated by judge-based variant).
- Cycle error is computed in the **source domain (A)**, while uncertainty is
  defined on the **target domain (B′)**. They are pixel-aligned because the
  spatial geometry is preserved by the translation, but they describe
  complementary aspects of the same translation.
- Calibration results should always be **reported alongside the assumption**:
  *"using cycle-reconstruction error as the per-pixel error proxy"* (or
  *"using judge-based reconstruction error with judge = X"*). We do not claim
  absolute calibration, only calibration with respect to this proxy.

---

## 4. Four calibration metrics

Once each tile has a paired `(U(x,y), E(x,y))`, we ask four complementary
questions.

### 4.1 Within-tile Spearman ρ — *spatial calibration*

> Within a single tile, do uncertain pixels tend to be the wrong pixels?

Per tile, flatten `U` and `E` over tissue pixels, compute Spearman rank
correlation. One scalar per tile.

- ρ → +1: high uncertainty co-locates with high error → well-calibrated spatially.
- ρ → 0: uncertainty spatial structure unrelated to error → uninformative.
- ρ < 0: anti-calibrated.

Aggregate across tiles: report **mean ± std of ρ**. This is the headline
"spatial calibration" number.

Spearman is preferred over Pearson here because `U` and `E` are on different
scales (variance in `[0, 255]^2` units vs MAE in `[0, 255]`), and only the rank
relationship matters.

### 4.2 Across-tile Pearson / Spearman — *tile-level calibration*

> Across tiles, do uncertain *tiles* tend to be the wrong tiles overall?

Reduce each tile to two scalars: `mean(U)`, `mean(E)`. Correlate across all
tiles. One number for the whole dataset.

This catches the case where a model is well-calibrated within each tile (ρ ≈ +1
locally) but assigns the same average uncertainty everywhere — useless for
tile-level triage.

Both Pearson and Spearman are reported because tile averages tend to be more
linearly related than per-pixel values.

### 4.3 Sparsification curve and AUSE

The standard depth-uncertainty calibration metric (Ilg et al., 2018; Poggi et al., 2020).

**Construction**: sort pixels by predicted uncertainty *descending*. Sweep
fraction $k \in [0, 1]$ and remove the top-$k$ most-uncertain pixels. Compute
the mean error on the remaining $(1-k)$ fraction → **predicted sparsification
curve $S_\mathrm{pred}(k)$**.

If uncertainty is informative, removing high-uncertainty pixels removes
high-error pixels first, and $S_\mathrm{pred}(k)$ drops quickly.

**Oracle**: same construction, but sort by actual error $E$ → $S_\mathrm{oracle}(k)$.
This is the steepest possible drop — the lower envelope.

**AUSE (Area Under Sparsification Error)**:

$$\mathrm{AUSE} = \int_0^1 \big[S_\mathrm{pred}(k) - S_\mathrm{oracle}(k)\big]\, dk$$

- AUSE = 0: predicted ordering = oracle ordering → perfectly calibrated.
- AUSE > 0: predicted ordering misses high-error pixels.
- AUSE ≈ AUSE-of-random: uncertainty has no information about error.

Computed per tile, averaged across tiles. The sparsification curve in the figure
is also tile-averaged.

**Why AUSE complements Spearman**: Spearman penalises any rank inversion
equally. AUSE weights inversions by the error magnitude they cost — a model
that ranks the few highest-error pixels correctly but misranks low-error
pixels can still have low AUSE (it gets the *important* pixels right).

### 4.4 Reliability diagram and ECE — *monotonicity / shape*

Bin pixels by `U`-quantile (10 bins, equal-mass). For each bin, plot
`mean(U_n)` (x) vs `mean(E_n)` (y), where both are min-max normalised to
`[0, 1]` using global p1–p99 bounds.

- Perfect calibration: all bins fall on the y = x diagonal.
- Monotonically increasing curve below diagonal: under-confident (uncertainty
  is informative but underestimates error magnitude).
- Flat curve: uncertainty does not discriminate.
- Decreasing curve: anti-calibrated.

**ECE-like score**:

$$\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}\,\big|\overline{U_n}_b - \overline{E_n}_b\big|$$

Weighted L1 deviation from the diagonal. Lower = better calibrated *in shape*,
not just rank.

**Caveat**: this is not the classical classification ECE (Guo et al., 2017),
because U and E are not on the same probability scale. We define ECE on
jointly-normalised `[0, 1]` versions to give the diagonal a meaning. Always
cite the construction explicitly when reporting it.

### What each metric catches that the others miss

| Failure mode | Spearman | AUSE | ECE | Across-tile ρ |
|---|---|---|---|---|
| Random uncertainty | low | high | mid | low |
| Uncertainty = constant scaling of error | +1 | 0 | non-zero (offset) | +1 |
| Per-tile calibrated but tile-level flat | +1 | 0 | low | low |
| Anti-calibrated | -1 | high | high | low |

Reporting all four in the paper is the safest way to characterise calibration.

---

## 5. Pipeline implementation

### `uncertainty_calibration.py`

End-to-end script. Consumes precomputed maps; does not run any models.

**CLI**:
```
python uncertainty_calibration.py \
    --uncertainty_dir uncertainty_out/<arch>/raw_npy/ \
    --error_dirs     <error_dir_1> [<error_dir_2> …] \
    --mask_dir       <tissue_mask_dir> \
    --tiles_metadata <dataset_root>           # optional, for per-WSI rollup
    --outdir         calibration_<arch>/
```

**`--error_dirs` semantics**:
- One directory → use that single member's cycle error (fast, default).
- Multiple directories → average per-pixel across all of them ("ensemble-mean
  error") before calibration. Pairs the variance signal with the average error
  signal. More principled, ~Nx slower because each member needs its own
  `regen_error` run.

**Tissue masking**: required by default. Background pixels are trivially
low-uncertainty and low-error; including them inflates Spearman and ECE
spuriously. Pass `--no_mask` only for diagnostic comparisons.

**Subsampling**: `--reliability_sample 4096` (default) caps the per-tile pixel
count contributed to the dataset-level reliability diagram so that ECE
computation does not OOM on large datasets. Per-tile metrics (Spearman, AUSE)
always use all tissue pixels.

### Outputs

`calibration_<arch>/`:
- `per_tile.csv` — tile_stem, source_wsi, n_tissue_pixels, spearman_rho,
  pearson_rho_within, mean_u, mean_e, ause
- `per_wsi.csv` — per-WSI rollup (mean ρ, mean AUSE, mean U, mean E, n_tiles).
  Only written when `--tiles_metadata` is provided.
- `summary.json` — all dataset-level aggregates, the parameters used, and the
  reliability/sparsification arrays needed to redraw the plot.
- `calibration.png` — 2×2 figure: reliability diagram, sparsification curve,
  histogram of within-tile ρ, scatter of mean(U) vs mean(E) across tiles.

---

## 6. Tile overlap and pixel double-counting

`tile.py` supports `--overlap` so adjacent tiles share pixels at their
borders (e.g. `--overlap 0.25` means an interior pixel can appear in up to
four tiles). Overlap is useful when reconstructing full WSIs because it
smooths seams between tiles, but it has a non-trivial effect on calibration
metrics because **the same physical pixel is then counted multiple times**.

### Where overlap matters and where it doesn't

| Step | Affected? | Why |
|---|---|---|
| `uncertainty.py` ensemble variance per tile | No | Computed locally per tile; overlap with neighbours is irrelevant. |
| `regen_error` per-pixel error map per tile | No | Same — local. |
| **Within-tile Spearman ρ** | **No** | Computed inside one tile; the ranking inside this tile does not see the neighbours. |
| Mean of within-tile ρ across tiles | Mildly | Overlap-region tiles are near-duplicates → their ρ values cluster, inflating apparent sample size for the std but barely changing the mean. |
| **Across-tile Pearson / Spearman** | **Yes** | Each pixel contributes to multiple tile-averages. Effective N is smaller than `n_tiles`. |
| **Reliability diagram / ECE** | **Yes** | Pixels in overlap regions are sampled into the global pool from multiple tiles. ECE is biased toward whatever those regions look like. |
| **Average sparsification curve / AUSE** | **Yes** | Tile-mean curves include the same physical pixels multiple times. |
| Per-WSI rollup (`per_wsi.csv`) | Modestly | Tile counts per WSI scale with overlap; within-WSI std estimates are not iid. |

### Magnitude of the bias

For overlap fraction $o$, an interior pixel appears in approximately
$1 / (1 - o)^2$ tiles. So `--overlap 0.25` ≈ 1.78× redundancy on interior
pixels (less at the WSI border). Effective independent sample size for
across-tile statistics is roughly $n_\mathrm{tiles} \times (1 - o)^2$.

### How to handle it

In increasing order of effort:

1. **Recommended: re-tile the test set with `--overlap 0` for calibration
   analysis.** Keep the overlapping tile set in parallel if you still need it
   for visual reconstruction (`testA/` with overlap, `testA_no_overlap/`
   without). The calibration pipeline runs on the latter; reconstruction
   figures use the former. This sidesteps every overlap-related issue at the
   cost of one extra `tile.py` call.

2. **Reconstruct full WSIs first, compute calibration per WSI.** Run
   `reconstruct.py` on both the uncertainty maps (mean blending) and the
   error maps, producing one `(U_\mathrm{wsi}, E_\mathrm{wsi})` pair per
   WSI. Compute Spearman / AUSE / ECE on each WSI, get one row per WSI. No
   double counting, but coarser-grained statistics (n = number of WSIs).
   Useful as a secondary check that confirms the tile-level numbers.

3. **Keep overlap and document the bias.** Report `--overlap o` in the
   methods and note that effective independent sample size for across-tile
   metrics is $n_\mathrm{tiles}\times(1-o)^2$. Acceptable if overlap is small
   ($o \le 0.1$); not recommended for the values typical in this repo
   ($o = 0.25$).

### What this script does *not* try to fix

`uncertainty_calibration.py` does not detect or correct for overlap on its
own — it consumes whatever `.npy` files it is pointed at. The user is
responsible for choosing a non-overlapping test set (or accepting the bias).
The metadata CSVs do contain `x`, `y`, and `overlap` columns that would in
principle allow per-tile masking of overlap regions, but this is not
implemented and is unlikely to be needed if the test set is re-tiled
correctly.

---

## 7. End-to-end recipe

```bash
# (a) Train N independent ensemble members with different --seed.
#     This step is outside the calibration pipeline.

# (b) Run inference for each member into model_01/, model_02/, ... layout.
#     Each subfolder contains the same set of translated tiles.

# (c) Compute uncertainty maps.
python uncertainty.py \
    --model cyclegan \
    --data ./ensemble_outputs_cyclegan/ \
    --output ./uncertainty_out/

# (d) Compute per-pixel error maps. Choose ONE of the two variants:
#
# (d-self) Self-cycle, GAN models only (cyclegan, unit, munit, dclgan, uvcgan).
#          Each member judges its own translation. Not available for MIUDiff.
python evaluation.py \
    --metric regen_error \
    --path_A /path/to/testA \
    --model cyclegan --ckpt ./ensemble_outputs_cyclegan/model_01.pt \
    --direction A2B \
    --overlay_dir ./regen_cyclegan_m01/ \
    --save_error_npy
#
# (d-judge) External judge — works for ANY model including MIUDiff, and is the
#           recommended variant for the paper because the same judge applies
#           to all 6 architectures. B' tiles are read from the inference output
#           directory; the judge runs B'→A_judge.
python evaluation.py \
    --metric judge_regen_error \
    --path_A /path/to/testA \
    --path_B_generated ./ensemble_outputs_cyclegan/model_01/ \
    --judge_model cyclegan --judge_ckpt ./judge_cyclegan.pt --judge_direction B2A \
    --overlay_dir ./judge_err_cyclegan_m01/ \
    --save_error_npy
# Reuse the SAME --judge_ckpt across every architecture's calibration run.

# (e) Run calibration analysis.
python uncertainty_calibration.py \
    --uncertainty_dir ./uncertainty_out/cyclegan/raw_npy/ \
    --error_dirs     ./regen_cyclegan_m01/error_npy/ \
    --mask_dir       ./tissue_masks_flat/ \
    --tiles_metadata /path/to/testA \
    --outdir         ./calibration_cyclegan/
```

For ensemble-mean error, repeat step (d) per member with distinct
`--overlay_dir`s and pass all of them to `--error_dirs` in step (e).

---

## 8. What to report in the paper

### Headline numbers (one row per architecture)

| Quantity | What it captures |
|---|---|
| Within-tile Spearman ρ (mean ± std over tiles) | Spatial calibration |
| Across-tile Spearman ρ | Tile-level calibration |
| AUSE (mean ± std) | Error-weighted ranking quality |
| ECE (10 bins, p1–p99 normalised) | Calibration *shape* |

### Figures

- **Per-architecture calibration figure** (`calibration.png`): four panels per
  model. Already produced by the pipeline.
- **Cross-architecture comparison**: bar chart of mean within-tile Spearman ρ
  and AUSE across the 6 models, with error bars from per-tile std. Easy to
  build from the `per_tile.csv` outputs.
- **Per-WSI table or boxplot**: from `per_wsi.csv`, show per-WSI ρ to detect
  WSIs where calibration breaks down (e.g. an outlier WSI with anomalous tissue).

### Phrasing templates

- "Epistemic uncertainty was estimated from a deep ensemble of *N* independently
  trained generators (Lakshminarayanan et al., 2017). Per-pixel uncertainty was
  defined as the sum of per-channel sample variances across the ensemble."
- "Calibration was measured against per-pixel reconstruction MAE as an
  unsupervised error proxy, since paired ground truth is unavailable in this
  unpaired translation setting. The same external GAN inverter (judge model
  X, frozen) was applied to every architecture's translated outputs to
  produce a comparable error signal — including MIUDiff, which has no
  inverse generator of its own."
- "We report (i) within-tile Spearman ρ between uncertainty and cycle error
  (spatial calibration), (ii) AUSE (Ilg et al., 2018) for the ranking quality
  of pixels by uncertainty, and (iii) ECE on jointly normalised values for the
  shape of the reliability curve."

---

## 9. Limitations and honest caveats

1. **Ensemble size**. Five members is the practical minimum for sample variance
   to be meaningful (`ddof=1`). More members give tighter estimates but
   diminishing returns past ~10. Report *N* in the methods section.

2. **Reconstruction error is a proxy, not a ground truth**. Numbers should be
   reported as "calibration with respect to reconstruction error", not
   "calibration of translation error". The self-cycle variant can be gamed
   by a model whose forward and inverse are jointly biased the same way; the
   judge-based variant removes that confound but introduces a dependency on
   the judge's quality. Either way, report calibration alongside FID / SSIM /
   downstream task results so reviewers can triangulate.

3. **Tissue masking matters**. Background pixels would dominate Spearman/ECE
   because both U and E are near-zero there and trivially co-vary. Always
   compute on tissue masks; report this in the methods.

4. **Global vs per-WSI normalisation for ECE**. We use *dataset-global* p1–p99
   bounds. This makes ECE comparable across architectures within one dataset
   but not across datasets. Per-WSI normalisation would give different (less
   useful) numbers; do not switch silently.

5. **Cross-section variability**. Because the test sections are not registered
   to the H&E sections, even a perfect translation can have non-trivial cycle
   error from biological differences (different cells visible in different
   sections). This adds an irreducible noise floor to AUSE and ECE that no
   model can cross. Acknowledge this in the discussion.

6. **The four metrics are not independent**. Don't stack them as if they each
   add independent evidence — they characterise the same underlying property
   from different angles. Treat them as a converging picture, not a
   significance battery.

---

## 10. References to cite

- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and scalable
  predictive uncertainty estimation using deep ensembles.* NeurIPS.
- Ilg, E., Çiçek, Ö., Galesso, S., Klein, A., Makansi, O., Hutter, F., & Brox, T.
  (2018). *Uncertainty estimates and multi-hypotheses networks for optical flow.*
  ECCV. — origin of AUSE.
- Poggi, M., Aleotti, F., Tosi, F., & Mattoccia, S. (2020). *On the uncertainty
  of self-supervised monocular depth estimation.* CVPR. — extends AUSE to
  monocular depth, sparsification methodology.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of
  modern neural networks.* ICML. — origin of ECE (classification setting).
- Kendall, A., & Gal, Y. (2017). *What uncertainties do we need in Bayesian
  deep learning for computer vision?* NeurIPS. — epistemic vs aleatoric
  decomposition; useful framing in the introduction.
