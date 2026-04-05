# CLI Pipeline Flowchart

End-to-end flow from raw WSI files to evaluation results and reconstructed whole-slide images.

---

## Full Pipeline Overview

```mermaid
flowchart TD
    WSI_A(["🔬 Raw WSIs\nDomain A"])
    WSI_B(["🔬 Raw WSIs\nDomain B"])

    TILE_A["tile.py\n--image_type trainA / testA"]
    TILE_B["tile.py\n--image_type trainB / testB"]

    TILES_A[("tiles/trainA/\n001/images/*.tif\n...\ntiles_metadata.csv")]
    TILES_B[("tiles/trainB/\n001/images/*.tif\n...\ntiles_metadata.csv")]
    TILES_TEST[("tiles/testA/\n001/images/*.tif\n...\ntiles_metadata.csv")]

    TRAIN["train.py\n--model --steps --amp\n--dataA --dataB --output"]
    CKPT[("run/\n  checkpoints/step_N.pt\n  loss_log.csv\n  training_meta.json\n  samples/")]

    PLOT["plot_training.py\n--run ./run/"]
    PLOTS[("run/\n  loss_plots/*.png\n  training_summary.json")]

    INFER["inference.py\n--model --ckpt --data\n--direction A2B --outdir"]
    FAKE[("translated_tiles/\n  *.tif")]

    EVAL["evaluation.py\n--metric fid|ssim|lpips\n         |patch_ssim|regen_error"]
    METRICS[("results.csv\n  FID / SSIM / LPIPS\n  regen_error heatmaps")]

    RECON["reconstruct.py\n--metadata --tile_dir --output"]
    WSI_OUT(["🔬 Reconstructed WSI\n  slide_001.tif"])

    ENSEMBLE["Run inference N times\n(N model seeds / checkpoints)"]
    ENS_DIRS[("ensemble_out/\n  model_01/*.tif\n  model_02/*.tif\n  ...")]
    UNCERT["uncertainty.py\n--model --data --output"]
    UNCERT_OUT[("uncertainty_out/\n  heatmaps/*.png\n  raw_npy/\n  norm_npy/\n  summary.json")]

    WSI_A --> TILE_A --> TILES_A
    WSI_B --> TILE_B --> TILES_B
    WSI_A --> TILE_A

    TILES_A --> TRAIN
    TILES_B --> TRAIN
    TRAIN --> CKPT

    CKPT --> PLOT --> PLOTS
    CKPT --> INFER

    TILES_TEST --> INFER --> FAKE

    TILES_B --> EVAL
    FAKE --> EVAL --> METRICS

    TILES_TEST --> RECON
    FAKE --> RECON --> WSI_OUT

    CKPT --> ENSEMBLE --> ENS_DIRS --> UNCERT --> UNCERT_OUT

    style WSI_A fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style WSI_B fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style WSI_OUT fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style TILES_A fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style TILES_B fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style TILES_TEST fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style FAKE fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style CKPT fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style PLOTS fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style METRICS fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style ENS_DIRS fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style UNCERT_OUT fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style TILE_A fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style TILE_B fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style TRAIN fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style PLOT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style INFER fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style EVAL fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style RECON fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style ENSEMBLE fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style UNCERT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
```

---

## Multi-stage Training Detail

For models with a pretrain stage (**MIUDiff**, **UVCGAN**), the `train.py` step expands into multiple runs, each with its own output directory.

```mermaid
flowchart TD
    TILES_A[("tiles/trainA/")]
    TILES_B[("tiles/trainB/")]

    subgraph GAN ["Single-stage GAN models\n(CycleGAN · UNIT · MUNIT · DCLGAN)"]
        T1["train.py --model cyclegan\n--dataA --dataB\n--steps 5000000 --output ./run/"]
        C1[("run/checkpoints/step_N.pt")]
        T1 --> C1
    end

    subgraph UVCGAN_FLOW ["UVCGAN (2 stages)"]
        direction TB
        UP["train.py --model uvcgan\n--uvcgan_stage pretrain\n--output ./uvcgan_pt/"]
        UPC[("uvcgan_pt/checkpoints/step_N.pt")]
        UF["train.py --model uvcgan\n--uvcgan_stage finetune\n--uvcgan_init_ckpt uvcgan_pt/.../step_N.pt\n--output ./uvcgan/"]
        UFC[("uvcgan/checkpoints/step_N.pt")]
        UP --> UPC --> UF --> UFC
    end

    subgraph MIU_FLOW ["MIUDiff (3 stages)"]
        direction TB
        S1["train.py --model miudiff\n--miu_stage pretrain\n--output ./stage1/"]
        S1C[("stage1/checkpoints/step_N.pt")]
        S2["train.py --model miudiff\n--miu_stage finetune\n--miu_init_ckpt stage1/.../step_N.pt\n--output ./stage2/"]
        S2C[("stage2/checkpoints/step_N.pt")]
        S3["train.py --model miudiff\n--miu_stage finetune --miu_pcl\n--miu_init_ckpt stage2/.../step_N.pt\n--output ./stage3/"]
        S3C[("stage3/checkpoints/step_N.pt")]
        S1 --> S1C --> S2 --> S2C --> S3 --> S3C
    end

    TILES_A --> GAN
    TILES_B --> GAN
    TILES_A --> UVCGAN_FLOW
    TILES_B --> UVCGAN_FLOW
    TILES_A --> MIU_FLOW
    TILES_B --> MIU_FLOW

    style GAN fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px
    style UVCGAN_FLOW fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style MIU_FLOW fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style TILES_A fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style TILES_B fill:#fff8e1,stroke:#f9a825,stroke-width:1px
```

---

## Inference Modes

```mermaid
flowchart TD
    CKPT[("checkpoint.pt")]
    TILES_TEST[("tiles/testA/ or testB/")]

    subgraph GAN_INF ["GAN models (CycleGAN · UNIT · DCLGAN · UVCGAN)"]
        GI["inference.py --model cyclegan\n--direction A2B\n--data tiles/testA/\n--ckpt model.pt --outdir ./out/"]
        GO[("out/stem.tif\none tile per input")]
        GI --> GO
    end

    subgraph MUNIT_INF ["MUNIT (random or reference style)"]
        MI1["inference.py --model munit --direction A2B\n--num_samples 3\n→ out/stem_0.tif, stem_1.tif, stem_2.tif"]
        MI2["inference.py --model munit --direction A2B\n--style_image ref.png\n→ out/stem.tif  (fixed style)"]
    end

    subgraph MIU_INF ["MIUDiff (3 modes)"]
        MD1["--miu_stage pretrain\n--data tiles/testB/ (filename anchor)\n→ out/stem_uncond.tif"]
        MD2["--miu_stage pretrain\n--num_uncond_samples 50\n→ out/uncond_0000.tif ... uncond_0049.tif"]
        MD3["--miu_stage finetune (default)\n--data tiles/testA/\n→ out/stem.tif"]
        MD4["--miu_stage finetune --miu_pcl\n--pcl_refine_steps 3\n→ out/stem.tif  (PCL latent refined)"]
    end

    CKPT --> GAN_INF
    CKPT --> MUNIT_INF
    CKPT --> MIU_INF
    TILES_TEST --> GAN_INF
    TILES_TEST --> MUNIT_INF
    TILES_TEST --> MIU_INF

    style GAN_INF fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style MUNIT_INF fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px
    style MIU_INF fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style CKPT fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style TILES_TEST fill:#fff8e1,stroke:#f9a825,stroke-width:1px
```

---

## Evaluation Metrics

```mermaid
flowchart LR
    REAL[("real tiles\n(domain B)")]
    FAKE[("translated tiles\n(inference output)")]
    CKPT[("checkpoint.pt")]
    ORIG_A[("original tiles\n(domain A)")]

    subgraph UNPAIRED ["Unpaired / distribution-level"]
        FID["evaluation.py --metric fid\n--backend inception|dino\n→ FID score"]
    end

    subgraph PAIRED ["Paired (matched by filename)"]
        SSIM["evaluation.py --metric ssim\n→ mean SSIM"]
        PSSIM["evaluation.py --metric patch_ssim\n--patch_size 64 --patches_per_image 16\n→ patch-level SSIM"]
        LPIPS["evaluation.py --metric lpips\n→ mean LPIPS (lower = better)"]
    end

    subgraph REGEN ["Cycle reconstruction (no paired ground truth needed)"]
        REGEN_E["evaluation.py --metric regen_error\n--path_A --model --ckpt --direction\n→ MAE in [0,255] + optional heatmaps"]
    end

    REAL --> UNPAIRED
    FAKE --> UNPAIRED
    REAL --> PAIRED
    FAKE --> PAIRED
    ORIG_A --> REGEN
    CKPT --> REGEN

    UNPAIRED --> CSV[("results.csv")]
    PAIRED --> CSV
    REGEN --> CSV
    REGEN --> HEAT[("overlay heatmaps/")]

    style UNPAIRED fill:#fce4ec,stroke:#c62828,stroke-width:1px
    style PAIRED fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    style REGEN fill:#fff3e0,stroke:#e65100,stroke-width:1px
    style REAL fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style FAKE fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style CKPT fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style ORIG_A fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style CSV fill:#fff8e1,stroke:#f9a825,stroke-width:1px
    style HEAT fill:#fff8e1,stroke:#f9a825,stroke-width:1px
```

---

## Tile directory structure (output of `tile.py`)

```mermaid
flowchart TD
    ROOT["tiles/\n└── trainA/  or  testA/  or  trainB/ ..."]
    ROOT --> META["tiles_metadata.csv\n(x, y, stride, overlap, image_path per tile)"]
    ROOT --> W001["001/\n└── images/\n    ├── 0000001.tif\n    ├── 0000002.tif\n    └── ...\n└── masks/  (if --mask provided)\n    └── 0000001.tif ..."]
    ROOT --> W002["002/ ..."]
    ROOT --> WDOT["..."]

    META -.->|used by reconstruct.py| RECON["reconstruct.py"]

    style ROOT fill:#fff8e1,stroke:#f9a825,stroke-width:2px
    style META fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    style W001 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px
    style W002 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:1px
```
