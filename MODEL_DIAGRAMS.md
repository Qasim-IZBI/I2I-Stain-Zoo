# Model Architecture Diagrams

Mermaid.js diagrams for all six models in I2I-Stain-Zoo.
Each diagram shows the **A→B translation path** at inference, plus the discriminators and loss signals used during training.

---

## 1. CycleGAN

Split-encoder design. Two fully-independent generator paths share no weights. A replay buffer (ImagePool) stabilises discriminator training.

```mermaid
flowchart TD
    subgraph Generator ["Generator (A→B translation path)"]
        xA([real_A]) --> EncA["Enc_A\n(Conv stride-2 ×2)"]
        EncA --> BnA["Bn_A\n(ResBlocks ×9)"]
        BnA --> DecB["Dec_B\n(ConvTranspose ×2 + Tanh)"]
        DecB --> fakeB([fake_B])
    end

    subgraph Cycle ["Cycle path (training only)"]
        fakeB --> EncB2["Enc_B"]
        EncB2 --> BnB2["Bn_B"]
        BnB2 --> DecA2["Dec_A"]
        DecA2 --> recA([rec_A])
    end

    subgraph Discriminator ["Discriminators (training only)"]
        fakeB --> DB["D_B\n(PatchGAN 70×70)"]
        recA -.->|cycle loss L1| xA
    end

    subgraph Losses ["Losses"]
        DB -->|GAN loss| LG["loss_G\n= λ_GAN + λ_cycle·L1_cycle\n+ λ_idt·L1_identity"]
    end

    style Generator fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style Cycle fill:#fff8e1,stroke:#e6a817,stroke-width:1px,stroke-dasharray:5 3
    style Discriminator fill:#fce4ec,stroke:#c62828,stroke-width:1px
    style Losses fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
```

---

## 2. UNIT

Shared bottleneck enforces domain-invariant representation. The variational reparameterisation adds KL regularisation. Both domains pass through the **same** shared ResBlocks.

```mermaid
flowchart TD
    subgraph EncoderA ["Domain A encoder"]
        xA([real_A]) --> EA["E_A\n(Conv stride-2 ×2)"]
        EA --> LatA["LatentHeads_A\n(1×1 Conv → μ_A, logvar_A)"]
        LatA -->|reparameterise z~N(μ,σ²)| zA([z_A])
    end

    subgraph EncoderB ["Domain B encoder"]
        xB([real_B]) --> EB["E_B\n(Conv stride-2 ×2)"]
        EB --> LatB["LatentHeads_B\n(1×1 Conv → μ_B, logvar_B)"]
        LatB -->|reparameterise| zB([z_B])
    end

    subgraph Bottleneck ["Bottleneck (A→B path shown)"]
        zA --> PreA["bn_pre_A\n(private ResBlocks)"]
        PreA --> Shared["bn_shared\n(shared weights ←→ both domains)"]
        Shared --> PostB["bn_post_B\n(private ResBlocks)"]
    end

    subgraph Decoder ["Decoder"]
        PostB --> DecB["Dec_B\n(ConvTranspose ×2 + Tanh)"]
        DecB --> fakeB([fake_B])
    end

    subgraph Training ["Training signals"]
        fakeB --> DB["D_B\n(PatchGAN)"]
        DB -->|GAN| LG["loss_G = λ_GAN + λ_recon·L1 + λ_KL·KL"]
        LatA -.->|KL loss| LG
    end

    style EncoderA fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style EncoderB fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style Bottleneck fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Decoder fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style Training fill:#fce4ec,stroke:#c62828,stroke-width:1px
```

---

## 3. MUNIT

Disentangles content (spatial structure) and style (global appearance). At inference, style can be sampled randomly for multimodal output, or extracted from a reference image.

```mermaid
flowchart TD
    subgraph EncodeA ["Encode domain A"]
        xA([real_A]) --> EcA["Ec_A\n(Conv stride-2 ×2)"]
        EcA --> BnA["Bn_A\n(content ResBlocks)"]
        BnA --> cA([content_A])
        xA --> EsA["Es_A\n(Conv ×3 + GlobalPool + FC)"]
        EsA --> sA([style_A])
    end

    subgraph Translate ["Translate A→B"]
        cA --> AdaINB["AdaIN_B\n(ResBlocks with style injection)"]
        sB_rand([s_B ~ N&#40;0,1&#41;\nor style ref]) -->|StyleMLP → γ,β per block| AdaINB
        AdaINB --> DecB["Dec_B\n(ConvTranspose ×2 + Tanh)"]
        DecB --> fakeB([fake_B])
    end

    subgraph ReEncode ["Re-encode for consistency (training)"]
        fakeB --> EcB2["Ec_B"]
        EcB2 --> BnB2["Bn_B"]
        BnB2 --> cA_hat([content_A_hat])
        fakeB --> EsB2["Es_B"]
        EsB2 --> sB_hat([style_B_hat])
    end

    subgraph Losses ["Losses (training)"]
        cA_hat -.->|content recon L1| cA
        sB_hat -.->|style recon L1| sB_rand
        fakeB --> DB["D_B\n(PatchGAN)"]
        DB -->|GAN| LG["loss_G = λ_GAN + λ_img·L1\n+ λ_content·L1 + λ_style·L1"]
    end

    style EncodeA fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style Translate fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style ReEncode fill:#fff8e1,stroke:#e6a817,stroke-width:1px,stroke-dasharray:5 3
    style Losses fill:#fce4ec,stroke:#c62828,stroke-width:1px
```

---

## 4. DCLGAN

CycleGAN backbone augmented with **Dual Contrastive Learning (DCL)**. Feature maps at multiple depths are sampled as patches and pulled together via InfoNCE, encouraging structural consistency without an explicit cycle.

```mermaid
flowchart TD
    subgraph GenA2B ["G_A2B = Enc_A → Bn_A → Dec_B"]
        xA([real_A]) --> EncA["Enc_A\n(Conv stride-2 ×2)\n↳ features @ layers 3,6,9"]
        EncA --> BnA["Bn_A\n(ResBlocks ×9)\n↳ features @ blocks 0, mid, last"]
        BnA --> DecB["Dec_B\n(ConvTranspose ×2 + Tanh)"]
        DecB --> fakeB([fake_B])
    end

    subgraph Features ["Feature collection (training)"]
        EncA -->|enc_out + 3 enc feats| FeatQ["feats_A\n[enc_out, bn_0, bn_mid, bn_last]"]
        BnA --> FeatQ
    end

    subgraph DCL ["Dual Contrastive Learning (training)"]
        FeatQ --> PSF["PatchSampleF\n(PatchSampler + per-layer MLP + L2-norm)"]
        FakeFeats["feats_fakeB\n(same layers from fakeB pass)"] --> PSF
        PSF -->|InfoNCE per layer| LDCL["loss_DCL"]
    end

    subgraph Losses ["Total generator loss"]
        fakeB --> DB["D_B\n(PatchGAN)"]
        DB -->|GAN| LG["loss_G = λ_GAN + λ_cycle·L1\n+ λ_idt·L1 + λ_DCL·InfoNCE"]
        LDCL --> LG
    end

    style GenA2B fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style Features fill:#fff8e1,stroke:#e6a817,stroke-width:1px,stroke-dasharray:5 3
    style DCL fill:#ede7f6,stroke:#4527a0,stroke-width:2px
    style Losses fill:#fce4ec,stroke:#c62828,stroke-width:1px
```

---

## 5. UVCGAN

UNet generator with a **Vision Transformer (ViT) bottleneck**. Skip connections preserve spatial detail; the ViT models long-range relationships at the bottleneck. Supports an optional masked-image pretraining stage.

```mermaid
flowchart TD
    subgraph PretrainNote ["Stage 1 — Masked Image Modelling (optional)"]
        masked([masked_A]) -->|random patch mask| Ginf["G_B2A reconstructs\nmasked regions of A"]
        Ginf --> recA_pt([rec_A])
        recA_pt -.->|L1 on masked region| PT_loss["loss_pretrain"]
    end

    subgraph Generator ["Stage 2 — Cycle-consistent translation (G_A2B shown)"]
        xA([real_A]) --> Stem["UNetEncoder\nStem: ReflPad + Conv7 + IN + ReLU"]
        Stem --> D1["Down ×1\nConv stride-2 + IN + ReLU\n↳ skip_0"]
        D1 --> D2["Down ×2\n↳ skip_1"]
        D2 --> D3["Down ×3\n↳ skip_2"]
        D3 --> D4["Down ×4 (last)\nno skip stored"]
        D4 --> ViT["ViTBottleneck\nLinear proj_in\n+ pos_embed\n+ TransformerBlocks ×N\n+ LayerNorm\n+ Linear proj_out"]
        ViT --> U1["UNetDecoder\nUpsample + concat skip_2 + merge"]
        U1 --> U2["Upsample + concat skip_1 + merge"]
        U2 --> U3["Upsample + concat skip_0 + merge"]
        U3 --> Head["ReflPad + Conv7 + Tanh"]
        Head --> fakeB([fake_B])
    end

    subgraph Losses ["Training losses (Stage 2)"]
        fakeB --> DB["D_B\n(PatchGAN)"]
        fakeB -->|cycle: G_B2A| recA2([rec_A])
        DB -->|GAN| LG["loss_G = λ_GAN + λ_cycle·L1\n+ λ_idt·L1"]
        recA2 -.->|cycle L1| xA
    end

    style PretrainNote fill:#fff8e1,stroke:#e6a817,stroke-width:1px,stroke-dasharray:5 3
    style Generator fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style Losses fill:#fce4ec,stroke:#c62828,stroke-width:1px
```

---

## 6. MIUDiff

Three-stage conditional diffusion model. Stage 1 trains an unconditional DDPM on domain B. Stage 2 adds a conditional UNet guided by the source image and mutual information. Stage 3 adds patch-contrastive refinement.

### Stage 1 — Unconditional Pretraining (`eps_uncond`)

```mermaid
flowchart LR
    subgraph Forward ["Forward process q(y_t | y_0)"]
        y0([y_0 ~ domain B]) -->|add noise α̅_t| yT([y_t])
    end

    subgraph UNet ["Unconditional UNet: eps_uncond"]
        yT --> UNC["DDPMUNet\nin: y_t &#40;3ch&#41; + t_frac\n\nEncoder: ResBlocks + Downsample\nMid: ResBlock + Attention + ResBlock\nDecoder: ResBlock + Upsample + skip-concat"]
        UNC --> eps_pred(["ε̂_uncond"])
    end

    subgraph MI ["MI Estimator (separate optimiser)"]
        y0 -->|Sobel grad| gy(["∇y_0"])
        gy --> MINet["MIEstimator\n(patch MLP)"]
        y0 --> MINet
        MINet -->|MINE lower bound| MI_loss["loss_MI\n(separate backward)"]
    end

    subgraph Loss ["Diffusion loss"]
        eps_pred -.->|MSE vs ε| loss_eps["loss_eps = MSE(ε̂, ε)"]
    end

    style Forward fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style UNet fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style MI fill:#ede7f6,stroke:#4527a0,stroke-width:1px,stroke-dasharray:5 3
    style Loss fill:#fce4ec,stroke:#c62828,stroke-width:1px
```

### Stage 2 — Conditional Finetuning (`eps_cond`)

```mermaid
flowchart LR
    subgraph Input ["Inputs"]
        xA([real_A]) -->|to_gray| xStruct([x_struct &#40;1ch&#41;])
        y0([y_0 ~ domain B]) -->|add noise| yT([y_t &#40;3ch&#41;])
        yT --> concat([concat: y_t ⊕ x_struct &#40;4ch&#41;])
        xStruct --> concat
    end

    subgraph UNet ["Conditional UNet: eps_cond"]
        concat --> COND["DDPMUNet\nin: y_t ⊕ x_struct &#40;4ch&#41; + t_frac\n\nSame architecture as eps_uncond\nbut extra input channel for conditioning"]
        COND --> eps_pred(["ε̂_cond"])
    end

    subgraph Loss ["Loss"]
        eps_pred -.->|MSE vs ε| loss_eps["loss_eps + optional λ_PCL·InfoNCE"]
    end

    subgraph Inference ["DDIM sampling (inference)"]
        noise([z ~ N&#40;0,I&#41;]) -->|T denoising steps| DDIM["DDIM loop\nε̂ = guidance_scale·ε̂_cond\n       + &#40;1−guidance_scale&#41;·ε̂_uncond"]
        xStruct2([x_struct]) --> DDIM
        DDIM --> fakeB([fake_B])
    end

    style Input fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style UNet fill:#e8f4e8,stroke:#4a9,stroke-width:2px
    style Loss fill:#fce4ec,stroke:#c62828,stroke-width:1px
    style Inference fill:#fff8e1,stroke:#e6a817,stroke-width:2px
```

### Stage 3 — PCL Refinement (optional, added on top of Stage 2)

```mermaid
flowchart TD
    subgraph PCL ["Patch Contrastive Learning (late timesteps only: t ≤ t0_prime)"]
        xStruct([x_struct]) -->|Sobel| gx(["∇x"])
        x0_pred([x̂_0 from ε̂_cond]) -->|to_gray → Sobel| gy(["∇ŷ"])
        gx --> FeatX["SmallFeatNet_x\n(Conv × 3 + GroupNorm + SiLU)"]
        gy --> FeatY["SmallFeatNet_y\n(Conv × 3 + GroupNorm + SiLU)"]
        FeatX -->|PatchSampler| q(["q patches"])
        FeatY -->|same patch ids| k(["k patches"])
        q --> Proj["PatchProjector\n(Linear + ReLU + Linear)"]
        k --> Proj
        Proj -->|InfoNCE| PCL_loss["loss_PCL"]
    end

    PCL_loss -->|λ_PCL| LG["total loss_G = loss_eps + λ_PCL·loss_PCL"]

    style PCL fill:#ede7f6,stroke:#4527a0,stroke-width:2px
```

---

## Shared building blocks (`base_models.py`)

```mermaid
flowchart LR
    subgraph Encoder ["Encoder (shared by CycleGAN, UNIT, MUNIT, DCLGAN)"]
        IN["input 3ch"] --> RC["ReflPad + Conv7"] --> DN["Conv stride-2 ×n_down\n+ InstanceNorm + ReLU"]
    end

    subgraph ResnetBottleneck ["ResnetBottleneck"]
        RB1["ResBlock_1"] --> RB2["..."] --> RBn["ResBlock_n"]
    end

    subgraph Decoder ["Decoder (shared)"]
        UP["ConvTranspose ×n_up\n+ InstanceNorm + ReLU"] --> HEAD["ReflPad + Conv7 + Tanh"]
    end

    subgraph Discriminator ["NLayerDiscriminator (PatchGAN 70×70)"]
        DIN["input 3ch"] --> DL1["Conv + LeakyReLU"] --> DLn["Conv ×n_layers"] --> DOUT["Conv → patch scores"]
    end

    Encoder --> ResnetBottleneck --> Decoder

    style Encoder fill:#e8f4e8,stroke:#4a9,stroke-width:1px
    style ResnetBottleneck fill:#fff3e0,stroke:#e65100,stroke-width:1px
    style Decoder fill:#e8f4e8,stroke:#4a9,stroke-width:1px
    style Discriminator fill:#fce4ec,stroke:#c62828,stroke-width:1px
```
