# Diffusion Models — Architecture Diagrams

All four diffusion models share the same `DDPMUNet` backbone and linear noise schedule
(`beta_start=1e-4`, `beta_end=2e-2`, `T=1000`) but differ in conditioning strategy,
loss composition, and training stages.

**Colour key (all diagrams)**
- Blue border — Domain A input
- Orange border — Domain B input
- Green border — sampled noise / random
- Purple border — neural network
- Red border — loss term

---

## Shared: DDIM Denoising Step

One iteration of the reverse loop used at inference by all four models:

```mermaid
flowchart LR
    yt["y_t"]
    cond["cond\n(optional)"]
    cat["cat([y_t, cond])"]
    net["ε_net( · , t)"]
    epspred["ε_pred"]
    x0pred["x0_pred = (y_t − sqrt(1−a_bar_t) · ε_pred) / sqrt(a_bar_t)"]
    ynext["y_{t−1} = sqrt(a_bar_{t−1}) · x0_pred + sqrt(1−a_bar_{t−1}) · ε_pred"]

    yt --> cat
    cond --> cat
    cat --> net
    net --> epspred
    epspred --> x0pred
    x0pred --> ynext
```

---

## MIUDiff

Three-stage training. Stage 1 builds an unconditional domain-B prior. Stage 2 adds
conditional A→B translation. Stage 3 adds patch contrastive loss (PCL).
Stage 2 weights are warm-started from stage 1 (`eps_uncond → eps_cond`, extra
conditioning channel initialised from channel mean).

---

### MIUDiff — Stage 1 Training (Unconditional DDPM on B)

```mermaid
flowchart TD
    xB(["xB ∈ Domain B"])
    eps(["ε ~ N(0,I)"])
    t(["t ~ Uniform[0,T)"])

    xB & eps & t --> yt["y_t = sqrt(a_bar_t)·xB + sqrt(1−a_bar_t)·ε"]
    yt --> uncond["eps_uncond\n3-ch UNet"]
    t --> uncond
    uncond --> loss[/"loss = MSE(ε_pred, ε)"/]

    style xB fill:#fff3e0,stroke:#e65100
    style eps fill:#e8f5e9,stroke:#388e3c
    style t fill:#e8f5e9,stroke:#388e3c
    style uncond fill:#f3e5f5,stroke:#7b1fa2
    style loss fill:#ffebee,stroke:#c62828
```

---

### MIUDiff — Stage 2 Training (Conditional + optional struct loss)

```mermaid
flowchart TD
    xA(["xA ∈ Domain A"])
    xB(["xB ∈ Domain B"])
    eps(["ε ~ N(0,I)"])
    t(["t ~ Uniform[0,T)"])

    xA --> struct["extract_struct\ngray or sobel → 1ch"]
    xB & eps & t --> yt["y_t = sqrt(a_bar_t)·xB + sqrt(1−a_bar_t)·ε"]

    yt & struct --> cat["cat([y_t, cond]) → 4ch"]
    cat --> condnet["eps_cond\n4-ch UNet"]
    t --> condnet
    condnet --> epspred["ε_pred"]

    epspred --> losseps[/"loss_eps = MSE(ε_pred, ε)"/]

    epspred --> x0pred["x0_pred (reconstruct)"]
    x0pred --> graypred["gray(x0_pred)"]
    xA --> grayA["gray(xA)"]
    graypred & grayA --> lossstruct[/"loss_struct = L1 × lambda_struct\n(only if lambda_struct > 0)"/]

    losseps & lossstruct --> total[/"loss_G = loss_eps + loss_struct"/]

    style xA fill:#e8f4f8,stroke:#1565c0
    style xB fill:#fff3e0,stroke:#e65100
    style eps fill:#e8f5e9,stroke:#388e3c
    style t fill:#e8f5e9,stroke:#388e3c
    style condnet fill:#f3e5f5,stroke:#7b1fa2
    style losseps fill:#ffebee,stroke:#c62828
    style lossstruct fill:#ffebee,stroke:#c62828
    style total fill:#ffebee,stroke:#c62828
```

---

### MIUDiff — Stage 3 Training (Conditional + struct + PCL)

Stage 3 loads the stage-2 checkpoint directly (no weight copying). PCL is applied at
**all** timesteps during training; `t0_prime` thresholding is inference-only.

```mermaid
flowchart TD
    xA(["xA ∈ Domain A"])
    xB(["xB ∈ Domain B"])
    eps(["ε ~ N(0,I)"])
    t(["t ~ Uniform[0,T)"])

    xA --> struct["extract_struct → cond"]
    xB & eps & t --> yt["y_t = sqrt(a_bar_t)·xB + sqrt(1−a_bar_t)·ε"]

    yt & struct --> cat["cat([y_t, cond]) → 4ch"]
    cat --> condnet["eps_cond\n4-ch UNet"]
    t --> condnet
    condnet --> epspred["ε_pred"]

    epspred --> losseps[/"loss_eps = MSE(ε_pred, ε)"/]

    epspred --> x0pred["x0_pred"]
    x0pred --> graypred["gray(x0_pred)"]
    xA --> grayA["gray(xA)"]
    graypred & grayA --> lossstruct[/"loss_struct = L1 × lambda_struct"/]

    x0pred --> featfake["patch features\n(x0_pred)"]
    xA --> featanc["patch features\n(xA — anchor)"]
    xB --> featneg["patch features\n(xB — negatives)"]
    featfake & featanc & featneg --> losspcl[/"loss_PCL = InfoNCE × lambda_pcl"/]

    losseps & lossstruct & losspcl --> total[/"loss_G"/]

    style xA fill:#e8f4f8,stroke:#1565c0
    style xB fill:#fff3e0,stroke:#e65100
    style eps fill:#e8f5e9,stroke:#388e3c
    style t fill:#e8f5e9,stroke:#388e3c
    style condnet fill:#f3e5f5,stroke:#7b1fa2
    style losseps fill:#ffebee,stroke:#c62828
    style lossstruct fill:#ffebee,stroke:#c62828
    style losspcl fill:#ffebee,stroke:#c62828
    style total fill:#ffebee,stroke:#c62828
```

---

### MIUDiff — Inference

`--miu_stage pretrain` uses only `eps_uncond` (unconditional sampling from B prior).
`--miu_stage finetune` uses `eps_cond` with optional MI guidance and PCL refinement.

```mermaid
flowchart TD
    subgraph pretrain ["--miu_stage pretrain"]
        pnoise(["z ~ N(0,I)"])
        pddim["DDIM loop t: T−1→0\neps_uncond (3-ch, no cond)"]
        pout(["Sampled B image"])
        pnoise --> pddim --> pout
    end

    subgraph finetune ["--miu_stage finetune (default)"]
        xA(["xA (test image)"])
        fnoise(["z ~ N(0,I)"])
        struct["extract_struct → cond"]
        ddim["DDIM loop t: T−1→0\neps_cond + optional MI guidance"]
        pcl["PCL latent refinement\n(--miu_pcl --pcl_refine_steps N)"]
        fout(["Translated B image"])

        xA --> struct
        struct --> ddim
        xA -->|"--miu_guidance > 0"| ddim
        fnoise --> ddim
        ddim --> pcl
        pcl --> fout
    end

    style xA fill:#e8f4f8,stroke:#1565c0
    style fnoise fill:#e8f5e9,stroke:#388e3c
    style pnoise fill:#e8f5e9,stroke:#388e3c
    style ddim fill:#f3e5f5,stroke:#7b1fa2
    style pddim fill:#f3e5f5,stroke:#7b1fa2
    style fout fill:#fff3e0,stroke:#e65100
    style pout fill:#fff3e0,stroke:#e65100
```

---

## UNIT-DDPM

Two-stage training. Architecture is identical to MIUDiff but without MI guidance or PCL.
Default conditioning is full RGB (`cond_type=rgb`); `gray` and `sobel` are available.
Stage 2 warm-starts `eps_cond` from the stage-1 `eps_uncond` weights.

---

### UNIT-DDPM — Stage 1 Training (Unconditional DDPM on B)

Identical to MIUDiff Stage 1.

```mermaid
flowchart TD
    xB(["xB ∈ Domain B"])
    eps(["ε ~ N(0,I)"])
    t(["t ~ Uniform[0,T)"])

    xB & eps & t --> yt["y_t = sqrt(a_bar_t)·xB + sqrt(1−a_bar_t)·ε"]
    yt --> uncond["eps_uncond\n3-ch UNet"]
    t --> uncond
    uncond --> loss[/"loss = MSE(ε_pred, ε)"/]

    style xB fill:#fff3e0,stroke:#e65100
    style eps fill:#e8f5e9,stroke:#388e3c
    style t fill:#e8f5e9,stroke:#388e3c
    style uncond fill:#f3e5f5,stroke:#7b1fa2
    style loss fill:#ffebee,stroke:#c62828
```

---

### UNIT-DDPM — Stage 2 Training (Conditional A→B)

`cond_type=rgb` → 6-ch input. `cond_type=gray` or `sobel` → 4-ch input.

```mermaid
flowchart TD
    xA(["xA ∈ Domain A"])
    xB(["xB ∈ Domain B"])
    eps(["ε ~ N(0,I)"])
    t(["t ~ Uniform[0,T)"])

    xA --> cond["extract_cond\nrgb 3ch / gray 1ch / sobel 1ch"]
    xB & eps & t --> yt["y_t = sqrt(a_bar_t)·xB + sqrt(1−a_bar_t)·ε"]

    yt & cond --> cat["cat([y_t, cond]) → 4ch or 6ch"]
    cat --> condnet["eps_cond\n4-ch or 6-ch UNet"]
    t --> condnet
    condnet --> epspred["ε_pred"]

    epspred --> losseps[/"loss_eps = MSE(ε_pred, ε)"/]

    epspred --> x0pred["x0_pred (if lambda_struct > 0)"]
    x0pred --> graypred["gray(x0_pred)"]
    xA --> grayA["gray(xA)"]
    graypred & grayA --> lossstruct[/"loss_struct = L1 × lambda_struct\n(only if lambda_struct > 0)"/]

    losseps & lossstruct --> total[/"loss_G"/]

    style xA fill:#e8f4f8,stroke:#1565c0
    style xB fill:#fff3e0,stroke:#e65100
    style eps fill:#e8f5e9,stroke:#388e3c
    style t fill:#e8f5e9,stroke:#388e3c
    style condnet fill:#f3e5f5,stroke:#7b1fa2
    style losseps fill:#ffebee,stroke:#c62828
    style lossstruct fill:#ffebee,stroke:#c62828
    style total fill:#ffebee,stroke:#c62828
```

---

### UNIT-DDPM — Inference

Always uses `eps_cond`. Starts from pure Gaussian noise (no SB-style initialisation).

```mermaid
flowchart LR
    xA(["xA (test image)"])
    noise(["z ~ N(0,I)"])

    xA --> cond["extract_cond\nrgb / gray / sobel"]
    noise --> ddim["DDIM loop t: T−1→0\neps_cond"]
    cond --> ddim
    ddim --> out(["Translated B image"])

    style xA fill:#e8f4f8,stroke:#1565c0
    style noise fill:#e8f5e9,stroke:#388e3c
    style ddim fill:#f3e5f5,stroke:#7b1fa2
    style out fill:#fff3e0,stroke:#e65100
```

---

## CycleDiffusion

Single-stage joint training of two fully unconditional DDPMs, one per domain.
No spatial conditioning — structure is transferred implicitly via the shared
DDIM noise latent. No discriminator, no cycle loss.

---

### CycleDiffusion — Training (Joint, Single Stage)

```mermaid
flowchart TD
    subgraph branchA ["Domain A branch"]
        xA(["xA ∈ Domain A"])
        epsA(["εA ~ N(0,I)"])
        tA(["t ~ Uniform[0,T)"])
        ytA["xA_t = sqrt(a_bar_t)·xA + sqrt(1−a_bar_t)·εA"]
        epsnetA["eps_A\n3-ch UNet"]
        lossA[/"loss_A = MSE(ε_predA, εA)"/]

        xA & epsA & tA --> ytA
        ytA --> epsnetA
        tA --> epsnetA
        epsnetA --> lossA
    end

    subgraph branchB ["Domain B branch"]
        xB(["xB ∈ Domain B"])
        epsB(["εB ~ N(0,I)"])
        tB(["t ~ Uniform[0,T)"])
        ytB["xB_t = sqrt(a_bar_t)·xB + sqrt(1−a_bar_t)·εB"]
        epsnetB["eps_B\n3-ch UNet"]
        lossB[/"loss_B = MSE(ε_predB, εB)"/]

        xB & epsB & tB --> ytB
        ytB --> epsnetB
        tB --> epsnetB
        epsnetB --> lossB
    end

    lossA & lossB --> total[/"loss_G = loss_A + loss_B"/]

    style xA fill:#e8f4f8,stroke:#1565c0
    style epsA fill:#e8f5e9,stroke:#388e3c
    style tA fill:#e8f5e9,stroke:#388e3c
    style epsnetA fill:#f3e5f5,stroke:#7b1fa2
    style lossA fill:#ffebee,stroke:#c62828
    style xB fill:#fff3e0,stroke:#e65100
    style epsB fill:#e8f5e9,stroke:#388e3c
    style tB fill:#e8f5e9,stroke:#388e3c
    style epsnetB fill:#f3e5f5,stroke:#7b1fa2
    style lossB fill:#ffebee,stroke:#c62828
    style total fill:#ffebee,stroke:#c62828
```

---

### CycleDiffusion — Inference A→B

DDIM inversion runs the forward process (t: 0→T−1) deterministically using `eps_A`
to encode xA into a noise code `z`. DDIM decode then runs the reverse process
(t: T−1→0) using `eps_B` to reconstruct an image in domain B from the same `z`.
B→A is symmetric (swap eps_A ↔ eps_B).

```mermaid
flowchart LR
    xA(["xA (test image)"])

    xA --> invert["DDIM inversion\nt: 0 → T−1\neps_A encodes xA → z"]
    invert --> z["noise code z\n(shared latent)"]
    z --> decode["DDIM decode\nt: T−1 → 0\neps_B decodes z → image"]
    decode --> out(["Translated B image"])

    style xA fill:#e8f4f8,stroke:#1565c0
    style z fill:#e8f5e9,stroke:#388e3c
    style invert fill:#f3e5f5,stroke:#7b1fa2
    style decode fill:#f3e5f5,stroke:#7b1fa2
    style out fill:#fff3e0,stroke:#e65100
```

---

## UNSB

Single-stage training. Score network `z_theta` takes noisy domain-B images
conditioned on xA and predicts the added noise. An adversarial discriminator
pushes the predicted clean image `x0_pred` toward the domain-B distribution.
Both losses pull in the same direction: score matching anchors `x0_pred` to xB,
adversarial reinforces the B distribution boundary.

---

### UNSB — Training

```mermaid
flowchart TD
    xA(["xA ∈ Domain A"])
    xB(["xB ∈ Domain B"])
    realB(["xB ∈ Domain B\n(discriminator real samples)"])
    eps(["ε ~ N(0,I)"])
    t(["t ~ Uniform[0,T)"])

    xB & eps & t --> xt["x_t = sqrt(a_bar_t)·xB + sqrt(1−a_bar_t)·ε\n(noisy domain B)"]
    xt & xA --> cat["cat([x_t, xA]) → 6ch\n(xA is structural conditioning)"]
    cat --> ztheta["z_theta\n6-ch UNet"]
    t --> ztheta
    ztheta --> epspred["ε_pred"]

    epspred --> lossscore[/"loss_score = MSE(ε_pred, ε) × lambda_score"/]

    epspred --> x0pred["x0_pred = (x_t − sqrt(1−a_bar_t)·ε_pred) / sqrt(a_bar_t)"]
    x0pred --> Gadv["D_adv (fake — real label)"]
    Gadv --> lossadvG[/"loss_adv_G × lambda_adv"/]

    lossscore & lossadvG --> lossG[/"loss_G"/]

    realB --> Dreal["D_adv (real)"]
    x0pred --> pool["Image pool\n(50-sample replay buffer)"]
    pool --> Dfake["D_adv (fake)"]
    Dreal --> lossD[/"loss_D = 0.5 · (loss_real + loss_fake)"/]
    Dfake --> lossD

    style xA fill:#e8f4f8,stroke:#1565c0
    style xB fill:#fff3e0,stroke:#e65100
    style realB fill:#fff3e0,stroke:#e65100
    style eps fill:#e8f5e9,stroke:#388e3c
    style t fill:#e8f5e9,stroke:#388e3c
    style ztheta fill:#f3e5f5,stroke:#7b1fa2
    style Gadv fill:#e3f2fd,stroke:#1565c0
    style Dreal fill:#e3f2fd,stroke:#1565c0
    style Dfake fill:#e3f2fd,stroke:#1565c0
    style lossscore fill:#ffebee,stroke:#c62828
    style lossadvG fill:#ffebee,stroke:#c62828
    style lossG fill:#ffebee,stroke:#c62828
    style lossD fill:#ffebee,stroke:#c62828
```

---

### UNSB — Inference A→B

The SB initial distribution starts from heavily-noised xA. Because `a_bar_{T-1} ≈ 0`
at t = T−1, the xA contribution is negligible (≈1%) and the starting point is
effectively pure Gaussian noise. `z_theta` is conditioned on xA at every step.

```mermaid
flowchart LR
    xA(["xA (test image)"])
    eps(["ε ~ N(0,I)"])

    xA & eps --> init["y_T = sqrt(a_bar_{T-1})·xA + sqrt(1−a_bar_{T-1})·ε\n≈ pure noise  (a_bar_{T-1} ≈ 0)"]
    init --> ddim["DDIM loop t: T−1→0\nz_theta(cat([y_t, xA]), t)\nxA concatenated at every step"]
    xA -->|"conditioning at each step"| ddim
    ddim --> out(["Translated B image"])

    style xA fill:#e8f4f8,stroke:#1565c0
    style eps fill:#e8f5e9,stroke:#388e3c
    style ddim fill:#f3e5f5,stroke:#7b1fa2
    style out fill:#fff3e0,stroke:#e65100
```

---

## Model Comparison

| | MIUDiff | UNIT-DDPM | CycleDiffusion | UNSB |
|---|---|---|---|---|
| **Stages** | 3 | 2 | 1 | 1 |
| **Score matching target** | xB | xB | xA and xB (separate) | xB |
| **Conditioning input** | gray/sobel xA (1ch) | rgb/gray/sobel xA | None | rgb xA (3ch) |
| **Adversarial loss** | No | No | No | Yes (on x0_pred) |
| **Structural alignment** | MI guidance + optional L1 + PCL | optional L1 | Shared noise latent | Adversarial only |
| **Inference start** | Pure noise | Pure noise | Noisy xA (DDIM invert) | ≈ Pure noise |
| **Unpaired-safe** | Partially (MI guidance helps) | Limited | Yes | Limited |
