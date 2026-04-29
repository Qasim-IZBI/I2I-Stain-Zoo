# Task-Based Evaluation: PSR Segmentation

Notes for paper writing. Covers the rationale, design choices, and how to interpret and report results.

---

## 1. What is task-based evaluation and why use it?

Standard generative model metrics (FID, SSIM, LPIPS) measure visual quality and distribution similarity at the pixel level. They say nothing about whether the generated image contains biologically meaningful content.

Task-based evaluation asks a different question: **does a downstream analysis model behave the same way on a generated image as it does on a real image?** In this case, the downstream task is PSR-positive area segmentation — a clinically relevant quantification of fibrosis.

If a generative model produces SR images with realistic PSR staining, a segmentation model should find the same amount (and distribution) of PSR-positive tissue as it finds in real SR images. If the model fails to reproduce the PSR signal, the segmentation distribution will shift (too much, too little, or wrong spread).

---

## 2. Why absolute segmentation accuracy does not matter

The nnUNet segmentation model is a **fixed function** applied identically to both real SR and generated SR images. Any systematic errors it makes — over-segmentation, under-segmentation, label confusion — affect both conditions equally and cancel out in the comparison.

What remains after cancellation is a clean signal: the difference in segmentation output is driven entirely by differences in the input images, i.e. how faithfully the generative model reproduced the PSR staining pattern.

This argument justifies using even an imperfect segmentation model for this evaluation, which is important because:
- The nnUNet model was trained on the same data used to train the generative models (no leakage concern since nnUNet is a downstream tool, not a generative model)
- Any domain-shift bias the segmentation model has is shared between conditions

---

## 3. Data setup and what it allows

**Test set structure**: testA (H&E) and testB (SR) come from the same mouse tissue samples, but the H&E and SR sections are several sections apart — there is no pixel-level or close-to-pixel registration.

This has two implications:

**What it rules out:**
- Pixel-level comparison (Dice, IoU between generated mask and real mask) — meaningless because the sections show slightly different tissue states
- Per-WSI paired correlation — unreliable because even a perfect model would produce a slightly different PSR fraction than testB, simply due to inter-section biological variability

**What it supports:**
- **Distribution-level comparison**: pool PSR fractions across all WSIs per condition (real SR, generated SR from model X) and ask whether the two distributions are compatible
- The same tissue biology (same mouse, same organ) means the overall PSR fraction distribution should be similar, even if individual WSI values shift slightly between sections

---

## 4. Evaluation design: two-level analysis

### WSI-level (primary for statistics)
- One PSR fraction per WSI: `PSR_pixels / (Tissue_pixels + PSR_pixels)` — background pixels excluded
- This is the statistically honest unit of analysis given the data (n = 5 WSIs per condition)
- Report: mean ± std for each condition, plus pairwise metrics below

### Tile-level (for visualisation, supplementary)
- WSI masks can be split back into tiles using the metadata CSVs; each tile gives one PSR fraction
- Many more data points (thousands per WSI), but tiles from the same WSI are correlated (pseudoreplication)
- Use tile-level distributions only for figures; do not apply statistical tests to tile-level data without accounting for the WSI-level clustering
- If tile-level analysis is reported: state explicitly that n=5 WSIs, tile count is for display only

---

## 5. Metrics

All pairwise comparisons are generated condition vs. real SR.

| Metric | Interpretation |
|---|---|
| **Wasserstein-1 distance** | Overall distributional gap; lower = more similar to real SR. Robust to shape differences between distributions. |
| **Bootstrap 95% CI on Wasserstein-1** | Resampling at the WSI level (not tile level) gives honest uncertainty given n=5. Wide CIs indicate the estimate is unreliable at this sample size. |
| **KS test (statistic + p-value)** | Tests whether real and generated fractions could be from the same distribution. At n=5, power is very low — use for directional evidence only, not as a decision criterion. |
| **Mean difference (generated − real)** | Signed bias: positive = model over-segments PSR relative to real; negative = under-segments. |
| **Std ratio (generated / real)** | Spread ratio: >1 means generated PSR fractions are more variable than real; <1 means they collapse to a narrower range (failure mode: model always outputs the same PSR level regardless of input). |

---

## 6. Sample size caveat (n = 5 WSIs)

**What you can say with n = 5:**
- Visual: "The distribution of PSR-positive area fractions in images generated by model X closely matches that of real SR images" — supported by overlapping box plots / dot plots
- Directional: effect size and sign (model X tends to over/under-segment PSR relative to real)
- Qualitative ranking: which models preserve PSR biology better
- Bootstrap CIs on Wasserstein-1 quantify uncertainty honestly

**What you cannot say with n = 5:**
- Statistically significant differences between models (KS test has near-zero power at n=5)
- Tight confidence intervals on any distributional metric

**How to frame in the paper:**
- Acknowledge n=5 explicitly in the methods section
- Report bootstrap CIs to show the range of plausible Wasserstein values
- Describe this as a proof-of-concept / pilot task-based evaluation; expanding the test set is future work
- Emphasise that the visual agreement in the distribution plots is the primary evidence

---

## 7. Figure recommendations

**Main figure (for results section):**
- Box plot + individual data points (dots) per condition — standard in biology for small n
- Real SR as leftmost bar (gray/reference), generated conditions in color
- Dashed horizontal line at real SR mean
- Label the y-axis clearly: "PSR-positive area fraction (PSR / (Tissue + PSR))"
- Caption should note: n=5 WSIs per condition; same nnUNet v2 model applied to all conditions

**Optional supplementary:**
- Tile-level PSR fraction histograms (one per WSI, overlaid real vs. generated) — shows spatial variability within slides
- summary.json values formatted as a table

---

## 8. Scripts

| Script | Role |
|---|---|
| `reconstruct.py` | Reconstruct WSI TIFs from inference tiles (run before segment_psr.py) |
| `segment_psr.py` | Run nnUNet v2 on reconstructed WSIs → mask TIFs (labels: 0=bg, 1=tissue, 2=PSR) |
| `compare_psr.py` | Compute PSR fractions, run Wasserstein/KS/mean-diff, bootstrap CI, save CSV/JSON/plot |

Typical workflow:
```bash
# 1. Reconstruct real SR test WSIs
python reconstruct.py --metadata /path/to/testB --output ./wsis_real/

# 2. Reconstruct generated SR tiles into WSIs (repeat per model config)
python reconstruct.py --metadata /path/to/testB \
    --tile_dir /path/to/cyclegan/inference/ --output ./wsis_cyclegan/

# 3. Segment PSR in all WSI sets
python segment_psr.py --data ./wsis_real/     --outdir ./masks_real/     --nnunet_dataset 1 ...
python segment_psr.py --data ./wsis_cyclegan/ --outdir ./masks_cyclegan/ --nnunet_dataset 1 ...

# 4. Compare distributions
python compare_psr.py \
    --masks_real ./masks_real/ \
    --masks_generated ./masks_cyclegan/ ./masks_unit/ ./masks_munit/ \
    --labels cyclegan unit munit \
    --outdir ./psr_comparison/
```

Outputs in `psr_comparison/`:
- `per_wsi.csv` — one row per WSI with condition and PSR fraction
- `summary.json` — per-condition stats and pairwise Wasserstein/KS/mean-diff vs real
- `comparison.png` — box + dot plot

---

## 9. Key sentences for the paper (draft)

**Methods:**
> "To evaluate whether generated SR images preserve biologically relevant staining patterns, we performed task-based evaluation using PSR-positive area segmentation. A nnUNet v2 model trained independently on the same tissue dataset was applied to both real and generated SR images reconstructed from test WSIs. Because the same segmentation model is applied to all conditions, any systematic errors in segmentation cancel out in the comparison; residual differences in the predicted PSR fraction reflect differences in staining fidelity between generative models. PSR-positive area fraction was defined as the ratio of PSR-labelled pixels to all tissue-labelled pixels (background excluded). Wasserstein-1 distance with bootstrap 95% confidence intervals (1000 iterations, WSI-level resampling) was used to quantify distributional agreement between real and generated SR."

**Results:**
> "Model X produced a PSR-positive area fraction distribution closely matching that of real SR (Wasserstein-1 = X.XX, 95% CI [X.XX, X.XX]; mean difference = ±X.XX), whereas model Y systematically over/under-estimated PSR-positive area (Wasserstein-1 = X.XX; mean difference = +/−X.XX), indicating failure to reproduce the fibrous staining pattern."

**Limitations:**
> "The task-based evaluation was conducted on five test WSIs per condition, limiting the statistical power of distributional comparisons. Bootstrap confidence intervals are reported to reflect this uncertainty. Expanding the test set is a priority for future work."
