# Uncertainty Estimation Strategy for Unpaired Virtual Staining (H&E → PSR, H&E → CK19)

> Scope: this document covers **only the uncertainty-estimation** goal of the project.
> It defines the gap, the diagnosis of why our current epistemic estimate is uncalibrated,
> the experiments to run, how to run them, expected results, how each is evaluated, and the
> requirements. The **bias-term experiment (E4)** is the headline contribution; the rest form
> the supporting decomposition around it. See `evaluation_strategy.md` for the structural
> metrics reused here as calibration targets.

---

## 1. The Gap

### 1.1 Observed problem
We compute an epistemic-uncertainty estimate for the unpaired I2I model, but it is **not
calibrated** when checked against the **regeneration (cycle-reconstruction) error**.

### 1.2 Diagnosis — two independent reasons
**(a) Regeneration error validates aleatoric, not epistemic.** In UGAC (our aleatoric
backbone), the generalized-Gaussian likelihood is placed on the *cycle-consistency residual*;
the per-pixel scale/shape heads are fit to explain reconstruction error. So **regeneration
error is the aleatoric channel by construction** and has no reason to track epistemic
uncertainty. Epistemic must instead be validated against **forward translation error**
(H&E→PSR/CK19 vs the real stain) and against **out-of-distribution (OOD) behaviour**.
[UGAC, arXiv:2110.12467](https://arxiv.org/abs/2110.12467)

**(b) Standard epistemic omits the bias term.** Variance-based / second-order disentanglement
(deep ensembles, MC-dropout, evidential) decomposes *predictive variance* into aleatoric +
epistemic but **drops the bias² term** of the classical bias–variance decomposition. When a
model is systematically biased (underfits under-represented tissue), the bias is invisible to
epistemic-as-variance (all models agree on the wrong answer) and is misattributed to aleatoric.
Epistemic then looks **falsely low and even decreases with more data**.
[Position paper, arXiv:2505.23506](https://arxiv.org/abs/2505.23506)

**Consequence for us:** our uncertainty *is* the variance term; forward error is
`variance + bias + floor`; therefore uncertainty undershoots error by exactly `bias + floor`.
**The calibration gap we observed is the bias term.**

### 1.3 The open gap
- No one has **operationalized the bias term** for image-to-image translation, let alone
  unpaired virtual staining.

  **Precise status of 2505.23506** *(corrected — see note)*. It is a position paper, and it
  contains **no method and no I2I application** — but it is *not* toy-only:
  - §4.4 — synthetic illustration (Beta + sin, heteroscedastic noise, MLP 4×100), sample
    sizes N ∈ {50, 100, 500}.
  - §4.5 — **real-data experiment**: NYC taxi trip duration, 1.2M train / 230K test, MLPs.
    There it decomposes epistemic variance into **procedural vs data** components (Fig. 4)
    and finds Deep Ensembles capture mostly the *procedural* part.

  **The bias term itself is measured on synthetic data only** (Figs. 1a, 2, 3, 5, 6) — and for
  a principled reason, not an oversight: bias² = ‖μ − truth‖² needs the data-generating process
  p(y|x) to be known, which holds only in simulation. On the taxi data the authors must assume
  a normal target and cannot compute bias at all:
  > "in statistical simulations, where the data-generating process p(y|x) is known, the bias of
  > the reference distribution can also be measured."

  > *Note — superseded wording.* An earlier version of this document described 2505.23506 as
  > "a position paper on toy regression — no method, no application." The *position paper* and
  > *no method / no application* parts stand; "toy regression" was inaccurate, because §4.5 is
  > a real-data experiment. Only the **bias** analysis is synthetic-only.

  **This makes our gap sharper, not weaker.** The bias term has never been measured on *any*
  real dataset, because no real dataset supplies the reference distribution it requires. Our
  contribution is precisely to sidestep that requirement: serial sections provide a co-located
  real target φ(y), so bias² = ‖μ(x) − φ(y)‖² becomes computable from ensemble features and
  target features alone, without knowing p(y|x) — with the real-vs-real serial discrepancy
  (§6.1) as the floor that keeps the substitution honest.

  Note also that the **procedural-vs-data split (our E2) is already demonstrated on real data**
  by §4.5, so E2 is a transfer to a new domain rather than a new decomposition; the novelty
  concentrates in E4.
- In adversarial unpaired I2I, mode-seeking training makes ensembles **agree on the dominant
  mode**, so high bias coincides with low variance → **confident hallucination**. This means
  **virtual-staining hallucination is the bias term of the uncertainty decomposition**, which
  is precisely why variance-based epistemic cannot flag it. This mechanistic link between
  uncertainty disentanglement and the field's central hallucination problem is unclaimed.

---

## 2. Conceptual Framework

### 2.1 Feature-space error decomposition (the backbone identity)
Let φ be a frozen pathology encoder (UNI/Virchow) or the structural-metric outputs from
`evaluation_strategy.md`, `y` a co-located real target tile, `x` an input tile, and `{G_m}`
an ensemble / stochastic set of generators. In feature space the decomposition is an **exact
Euclidean identity**:

- mean prediction `μ(x) = E_m[ φ(G_m(x)) ]`
- **epistemic-variance** `Var(x) = E_m ‖ φ(G_m(x)) − μ(x) ‖²`  *(what we currently estimate)*
- **bias²** `B(x) = ‖ μ(x) − φ(y) ‖²`  *(the missing term)*
- identity: `E_m‖ φ(G_m(x)) − φ(y) ‖² = Var(x) + B(x)` (+ biological floor)

Hence **bias = forward error − epistemic-variance − floor**, measurable from ensemble features
and target features alone. (Average in **feature space**, never in pixel space — averaging GAN
outputs blurs into a non-image.)

### 2.2 Which error signal validates which channel

| Error signal | Validates | Uncertainty channel |
|---|---|---|
| Regeneration / cycle residual | Aleatoric (UGAC GGD head) | irreducible + noise |
| Forward error vs real target (tile-representation level) | Epistemic (variance) | model-disagreement |
| Ensemble-mean forward error (bias) | **Missing epistemic piece** | model-form / data-scarcity bias |

### 2.3 Uncertainty taxonomy (target components)
Following 2505.23506 + 2605.18329:

- **Aleatoric** — irreducible; per-pixel GGD (UGAC). σ² = α²Γ(3/β)/Γ(1/β).
- **Epistemic**
  - **Procedural** — algorithmic randomness (seeds, init). → Deep Ensemble / MC-dropout.
  - **Data (finite-sample)** — training-set variability. → Bootstrap / CV ensemble.
  - **Distributional** — train↔test shift. → feature-density / Mahalanobis OOD.
  - **Model-form bias** — systematic consensus error. → E4 (bias identity above).

---

## 3. Models
- **Primary:** CycleGAN unpaired I2I (UGAC-style GGD head attaches directly to the cycle loss).
- **Stretch (if time permits):** DCLGAN and CycleDiffusion. These are more mode-committed, so
  the bias / confident-hallucination effect (E4) should be **stronger** — a built-in ablation
  that strengthens rather than complicates the story.

---

## 4. Experiments

Each lists: **Gap addressed · Method · Requirements · Expected result · How evaluated**.

### E1 — Aleatoric via UGAC *(adopt / extend)*
- **Gap:** Need a principled aleatoric estimate for unpaired I2I in staining.
- **Method:** Add generalized-Gaussian per-pixel heads (scale α, shape β) to the cycle loss;
  closed-form aleatoric σ² = α²Γ(3/β)/Γ(1/β). Extend UGAC from natural/MRI images to
  H&E→PSR/CK19.
- **Requirements:** CycleGAN + GGD heads; UGAC reference implementation.
- **Expected result:** Aleatoric peaks at genuinely ambiguous regions (ductular-reaction
  margins, faint collagen); correlates with the **regeneration residual**.
- **How evaluated:** Correlation of aleatoric map with cycle-reconstruction residual (UGAC's
  own validation); qualitative overlap with eval-doc "ignore-regions".

### E2 — Epistemic disentanglement: procedural vs data *(adaptation)*
- **Gap:** MC-dropout captures only procedural uncertainty; need to separate procedural from
  data-scarcity.
- **Method:** Train a **deep ensemble** (N seeds, full data → procedural) and a **bootstrap/CV
  ensemble** (resampled training WSIs / folds → data-exposure). Decompose total ensemble
  variance over the **(fold × seed) grid** via the law of total variance:
  `Var_total = Var_folds(E_seed[·]) + E_folds[Var_seed(·)]` → data vs procedural components.
- **Requirements:** N-member ensembles; enough training WSIs to resample. [2605.18329]
- **Expected result:** Data component dominates in under-represented tissue; procedural
  component is comparatively flat.
- **How evaluated:** Component magnitudes vs tissue frequency; contribution of each component
  to forward-error prediction (E5).

### E3 — Distributional / OOD epistemic *(standard, for completeness)*
- **Gap:** Epistemic should spike under distribution shift; variance estimates often don't.
- **Method:** Feature-space density / **Mahalanobis distance** in the encoder space; evaluated
  on held-out **scanner / site / stain-batch** shift and unseen tissue.
- **Requirements:** UNI/Virchow encoder; a shifted held-out split.
- **Expected result:** Distributional term rises sharply on OOD tiles where in-distribution
  epistemic-variance stays low.
- **How evaluated:** OOD-detection AUROC; correlation with forward error on shifted tiles.

### E4 — Bias-term experiment *(HEADLINE — novel)*
- **Gap:** The dropped bias² term = confident hallucination; never operationalized for I2I.
- **Method:** On the serial-section triples, for each co-located tile compute μ, Var, bias²
  (§2.1), aleatoric (E1), and **subtract the biological floor** (real-vs-real serial
  discrepancy). Then:
  1. **Error budget** — bias² as a fraction of forward error, stratified by tissue rarity.
  2. **Money figure** — scatter epistemic-variance vs bias; isolate the **high-bias /
     low-variance quadrant = confident hallucinations**.
  3. **False-confidence-with-data** — subsample training data at several sizes; show
     variance-epistemic *decreases* while error stays high (reproduce 2505.23506 **Fig. 3** in
     the staining setting — Deep Ensembles at N ∈ {50, 100, 500}, synthetic there, real here).
     *(Corrected: earlier drafts cited "Fig-4"; Fig. 4 is the real-data taxi procedural/data
     decomposition, not the false-confidence result.)*
  4. **Calibration recovery** — variance-alone fails AUSE vs forward error; the completed
     estimate (E2 data + E3 distributional + a bias proxy) restores it.
  5. **Clinical grounding** — show high-bias-low-variance tiles are exactly where the
     `evaluation_strategy.md` structural metrics fail (CK19 epithelial-specificity precision
     drops, early fibrosis missed) → bias → hallucination → eval metrics.
- **Requirements:** Serial-section triples (§6), ensembles (E2), encoder φ, structural metrics.
- **Expected result:** Bias is a large, tissue-rarity-dependent fraction of forward error;
  uncorrelated with epistemic-variance; grows more pronounced for DCLGAN/CycleDiffusion.
- **How evaluated:** Fraction-of-error plots; variance–bias correlation (expect ≈0);
  calibration curves before/after adding bias; overlap of bias hotspots with structural-metric
  failures.

### E5 — Calibration at tile-representation level *(the correct validation)*
- **Gap:** Serial sections aren't pixel-aligned; per-pixel calibration is impossible.
- **Method:** Define per-tile forward error in **feature space** (φ distance to real target) or
  via **registration-free structural metrics** (lumen-Dice, nuclei consistency, CK19 epithelial
  specificity). Assess calibration **above the biological floor** using **AUSE /
  sparsification-error curves**, **Spearman**(uncertainty, tile-error), and **error-retention
  curves**. Not per-pixel ECE.
- **Requirements:** Serial-section triples; encoder φ; structural metrics from eval doc.
- **Expected result:** Completed/disentangled uncertainty attains lower AUSE and higher
  Spearman than the original variance-only estimate.
- **How evaluated:** AUSE, Spearman, retention curves vs the original baseline and vs FID/SSIM
  proxies.

### V — Controlled double-dissociation (proves disentanglement is real)
Move one knob at a time and show only the intended component responds:
- Shrink training data / subsample a tissue type → **data** epistemic + **bias** rise; aleatoric flat.
- Inject label ambiguity / faint staining → **aleatoric** rises (ties to eval-doc ignore-regions).
- Feed OOD tiles (unseen scanner/organ) → **distributional** epistemic spikes.
- Force underfitting (reduce capacity / early stop) → reproduce **bias-masquerading-as-aleatoric**.

---

## 5. Uncertainty ↔ Evaluation Linkage
- Structural-disagreement maps (eval E1/E2) are a **reference-free spatial forward-error proxy**
  and serve as calibration targets here (E5).
- Ductular-reaction **ignore-regions** (eval E3/E4) are designed **aleatoric** targets — the
  model should be uncertain there; validating that is positive evidence of calibration.
- **Headline unification:** the same measurement that evaluates the stain (structural
  faithfulness) also localizes where to distrust it (bias/uncertainty). Report the negative
  correlation between structural faithfulness and total uncertainty.

---

## 6. Dataset & Requirements

### 6.1 Serial-section calibration benchmark (contribution + released with paper)
- **Serial triples** H&E / PSR(SR) / CK19, **~15–30 WSI per stain**, open-source.
- Role: the **held-out calibration & bias benchmark** (not the training corpus — training uses
  the larger unpaired pool). Tiled, these give ample tiles for tile-level statistics; tissue
  diversity is the main limit (enough for aggregate bias, thin for fine per-type stratification).
- **Tissue-level registration** (thumbnail affine/elastic) → co-located tile grids, explicitly
  accepting sub-tile non-alignment (why calibration is representation-level).
- **Biological floor**: measure real-vs-real serial tile discrepancy; subtract everywhere.

### 6.2 Methods / tooling (no bespoke detectors trained)
- CycleGAN (+ DCLGAN, CycleDiffusion) with **UGAC** GGD heads.
- **Deep ensembles** (Lakshminarayanan 2017) and **MC-dropout** (Gal & Ghahramani 2016);
  bootstrap/CV ensembles.
- **UNI / Virchow2** frozen encoders for feature space φ; Mahalanobis OOD.
- Calibration: **AUSE / sparsification** (Ilg 2018), Spearman, retention curves.
- Structural metrics from `evaluation_strategy.md` (StarDist, HoVer-Net, QuPath, scikit-image).

---

## 7. References
1. Position paper on epistemic disentanglement / bias contamination — [arXiv:2505.23506](https://arxiv.org/abs/2505.23506)
2. CV vs deep ensembles in medical segmentation (procedural vs data-exposure) — [arXiv:2605.18329](https://arxiv.org/abs/2605.18329)
3. UGAC — Uncertainty-aware Generalized Adaptive Cycle Consistency (aleatoric, unpaired I2I) — [arXiv:2110.12467](https://arxiv.org/abs/2110.12467)
4. AQuA — reference-free hallucination detection via cycle-consistency (uncertainty accumulation) — [arXiv:2404.18458](https://arxiv.org/abs/2404.18458)
5. Kendall & Gal — aleatoric vs epistemic in deep learning, NeurIPS 2017.
6. Lakshminarayanan et al. — Deep Ensembles, NeurIPS 2017.
7. Gal & Ghahramani — MC-dropout, ICML 2016.
8. Ilg et al. — uncertainty estimation & sparsification/AUSE, ECCV 2018.
9. Zhu et al. — CycleGAN, ICCV 2017. Han et al. — DCLGAN, 2021. CycleDiffusion (Wu & De la Torre), 2023.
10. UNI — Chen et al., Nat Med 2024; Virchow — Vorontsov et al. 2024.
11. Liver fibrosis measurement with uncertainty (context) — [medRxiv 2025.05.12.25326981](https://www.medrxiv.org/content/10.1101/2025.05.12.25326981)

> Cross-reference: `evaluation_strategy.md` (structural metrics reused as calibration targets).
