# Model Size Reference

Parameter counts are for the **A→B single-direction forward pass** (inference).
All counts were verified with `python train.py --model <model> --count_params <args>`.

> **MIUDiff note**: both `eps_uncond` and `eps_cond` are counted because classifier-free guidance
> requires both UNets at inference time. Stage-1 pretrain uses `eps_uncond` only (~half the A→B count).
>
> **Option A** (default): original channel multipliers, `--miu_num_res_blocks 2` (2 ResBlocks per level).
> **Option B** (simpler): 3-level UNet, `--miu_num_res_blocks 1` — fewer residual accumulations, faster per step.
> Both variants include the output-norm stability fix in `ResBlock`.

> **UNIT-DDPM note**: A→B params = `eps_cond` only. `eps_uncond` is a training artefact (no
> classifier-free guidance); it is not used at finetune inference. Total model ≈ 2× the A→B count.
> Stage-1 pretrain uses `eps_uncond` only.

> **CycleDiffusion note**: A→B params = `eps_A + eps_B` = full model. Both UNets are active at
> inference — `eps_A` for DDIM inversion of the source, `eps_B` for DDIM decoding to the target.
> Because CycleDiffusion needs two UNets for one direction, a given config gives 2× the A→B params
> of UNIT-DDPM/UNSB at the same channel width.

> **UNSB note**: A→B params = `z_theta` (score network) only. The discriminator is training-only
> and adds ~3M to the total but is not used at inference. Total ≈ A→B + 3M.

---

## Small (~10M parameters)

| Model | `--ngf` / channels | blocks | other args | A→B params |
|-------|-------------------|--------|------------|------------|
| CycleGAN | `--cyclegan_ngf 64` | `--cyclegan_n_blocks 8` | — | **10.20M** |
| UNIT | `--unit_ngf 64` | `--unit_n_blocks 8 --unit_n_blocks_shared 2` | — | **10.33M** |
| MUNIT | `--munit_ngf 64` | `--munit_n_content_blocks 3 --munit_n_adain_blocks 4` | — | **10.14M** |
| DCLGAN | `--dclgan_ngf 64` | `--dclgan_n_blocks 8` | — | **10.20M** |
| UVCGAN | `--uvcgan_ngf 48` | `--uvcgan_vit_blocks 6` | `--uvcgan_vit_features 96` | **10.01M** |
| MIUDiff (A) | `--miu_base_channels 32` | `--miu_num_res_blocks 2` | `--miu_channel_mult 1,2,4,4` | **11.4M** |
| MIUDiff (B) | `--miu_base_channels 48` | `--miu_num_res_blocks 1` | `--miu_channel_mult 1,2,4` | **10.4M** |
| UNIT-DDPM | `--unitddpm_base_channels 64` | `--unitddpm_num_res_blocks 1` | `--unitddpm_channel_mult 1,2,2,4` | **9.49M** |
| CycleDiffusion | `--cd_base_channels 48` | `--cd_num_res_blocks 1` | `--cd_channel_mult 1,2,2,4` | **10.68M** |
| UNSB | `--unsb_base_channels 64` | `--unsb_num_res_blocks 1` | `--unsb_channel_mult 1,2,2,4` | **9.49M** |

---

## Medium (~50M parameters)

| Model | `--ngf` / channels | blocks | other args | A→B params |
|-------|-------------------|--------|------------|------------|
| CycleGAN | `--cyclegan_ngf 128` | `--cyclegan_n_blocks 10` | — | **50.18M** |
| UNIT | `--unit_ngf 128` | `--unit_n_blocks 10 --unit_n_blocks_shared 2` | — | **50.71M** |
| MUNIT | `--munit_ngf 128` | `--munit_n_content_blocks 5 --munit_n_adain_blocks 4` | — | **47.64M** |
| DCLGAN | `--dclgan_ngf 128` | `--dclgan_n_blocks 10` | — | **50.18M** |
| UVCGAN | `--uvcgan_ngf 96` | `--uvcgan_vit_blocks 6` | `--uvcgan_vit_features 384` | **48.28M** |
| MIUDiff (A) | `--miu_base_channels 64` | `--miu_num_res_blocks 2` | `--miu_channel_mult 1,2,2,4,4` | **49.1M** |
| MIUDiff (B) | `--miu_base_channels 112` | `--miu_num_res_blocks 1` | `--miu_channel_mult 1,2,4` | **51.5M** |
| UNIT-DDPM | `--unitddpm_base_channels 128` | `--unitddpm_num_res_blocks 2` | `--unitddpm_channel_mult 1,2,4` | **50.02M** |
| CycleDiffusion | `--cd_base_channels 84` | `--cd_num_res_blocks 2` | `--cd_channel_mult 1,2,2,4` | **50.29M** |
| UNSB | `--unsb_base_channels 128` | `--unsb_num_res_blocks 2` | `--unsb_channel_mult 1,2,4` | **50.02M** |

---

## Large (~100M parameters)

| Model | `--ngf` / channels | blocks | other args | A→B params |
|-------|-------------------|--------|------------|------------|
| CycleGAN | `--cyclegan_ngf 192` | `--cyclegan_n_blocks 9` | — | **102.26M** |
| UNIT | `--unit_ngf 192` | `--unit_n_blocks 9 --unit_n_blocks_shared 3` | — | **103.44M** |
| MUNIT | `--munit_ngf 192` | `--munit_n_content_blocks 4 --munit_n_adain_blocks 4` | — | **94.87M** |
| DCLGAN | `--dclgan_ngf 192` | `--dclgan_n_blocks 9` | — | **102.26M** |
| UVCGAN | `--uvcgan_ngf 128` | `--uvcgan_vit_blocks 17` | `--uvcgan_vit_features 384` | **96.72M** |
| MIUDiff (A) | `--miu_base_channels 64` | `--miu_num_res_blocks 2` | `--miu_channel_mult 1,2,4,8` | **99.7M** |
| MIUDiff (B) | `--miu_base_channels 160` | `--miu_num_res_blocks 1` | `--miu_channel_mult 1,2,4` | **103.9M** |
| UNIT-DDPM | `--unitddpm_base_channels 168` | `--unitddpm_num_res_blocks 2` | `--unitddpm_channel_mult 1,2,2,4` | **100.49M** |
| CycleDiffusion | `--cd_base_channels 128` | `--cd_num_res_blocks 2` | `--cd_channel_mult 1,2,4` | **100.02M** |
| UNSB | `--unsb_base_channels 168` | `--unsb_num_res_blocks 2` | `--unsb_channel_mult 1,2,2,4` | **100.49M** |

---

## Architecture CLI args reference

| Model | Arg | Default | Controls |
|-------|-----|---------|---------|
| CycleGAN | `--cyclegan_ngf` | 64 | Base channel width; bottleneck = `ngf × 4` |
| CycleGAN | `--cyclegan_n_blocks` | 9 | ResNet blocks in bottleneck |
| UNIT | `--unit_ngf` | 64 | Base channel width; `z_dim` auto-set to `ngf × 4` |
| UNIT | `--unit_n_blocks` | 9 | Total bottleneck blocks (pre + shared + post) |
| UNIT | `--unit_n_blocks_shared` | 3 | Shared blocks; pre = post = `(total − shared) / 2` |
| MUNIT | `--munit_ngf` | 64 | Base channel width |
| MUNIT | `--munit_n_content_blocks` | 4 | Content ResNet blocks (domain-specific) |
| MUNIT | `--munit_n_adain_blocks` | 4 | AdaIN decoder blocks (style injection) |
| DCLGAN | `--dclgan_ngf` | 64 | Base channel width; bottleneck = `ngf × 4` |
| DCLGAN | `--dclgan_n_blocks` | 9 | ResNet blocks in bottleneck |
| UVCGAN | `--uvcgan_ngf` | 64 | UNet encoder/decoder channel width |
| UVCGAN | `--uvcgan_vit_blocks` | 6 | Transformer blocks in ViT bottleneck |
| UVCGAN | `--uvcgan_vit_features` | 192 | ViT hidden dimension (must be divisible by `vit_n_heads=6`) |
| MIUDiff | `--miu_base_channels` | 64 | DDPM UNet base width (must be divisible by 32) |
| MIUDiff | `--miu_channel_mult` | `1,2,2,4` | Per-level channel multipliers; each `base × mult` must be divisible by 32 |
| MIUDiff | `--miu_num_res_blocks` | 2 | ResBlocks per level (use 1 for Option B simpler variant) |
| UNIT-DDPM | `--unitddpm_base_channels` | 64 | DDPM UNet base width |
| UNIT-DDPM | `--unitddpm_channel_mult` | `1,2,2,4` | Per-level channel multipliers |
| UNIT-DDPM | `--unitddpm_num_res_blocks` | 2 | ResBlocks per UNet level |
| UNIT-DDPM | `--unitddpm_cond_type` | `rgb` | Source conditioning: `rgb` (3ch), `gray` (1ch), `sobel` (1ch) |
| CycleDiffusion | `--cd_base_channels` | 64 | DDPM UNet base width (applies to both eps_A and eps_B) |
| CycleDiffusion | `--cd_channel_mult` | `1,2,2,4` | Per-level channel multipliers |
| CycleDiffusion | `--cd_num_res_blocks` | 2 | ResBlocks per UNet level |
| UNSB | `--unsb_base_channels` | 64 | Score network UNet base width |
| UNSB | `--unsb_channel_mult` | `1,2,2,4` | Per-level channel multipliers |
| UNSB | `--unsb_num_res_blocks` | 2 | ResBlocks per UNet level |
| UNSB | `--unsb_lambda_adv` | 0.1 | Adversarial loss weight (generator); increase for stronger domain transfer |
| UNSB | `--unsb_lambda_score` | 1.0 | Score-matching (MSE) loss weight; keep ≥ 0.5 for diffusion regularisation |

---

## Full example commands

```bash
# Small CycleGAN
python train.py --model cyclegan --cyclegan_ngf 64 --cyclegan_n_blocks 8 \
    --dataA /tiles/trainA --dataB /tiles/trainB --steps 5000000 --amp --output ./results/

# Medium UVCGAN
python train.py --model uvcgan --uvcgan_ngf 96 --uvcgan_vit_features 384 --uvcgan_vit_blocks 6 \
    --dataA /tiles/trainA --dataB /tiles/trainB --steps 5000000 --amp --output ./results/

# Large MIUDiff Option A (stage 2)
python train.py --model miudiff --miu_base_channels 64 --miu_channel_mult 1,2,4,8 --miu_num_res_blocks 2 \
    --miu_stage finetune --dataA /tiles/trainA --dataB /tiles/trainB --steps 500000 --output ./results/

# MIUDiff Option B — simpler 3-level, 1-ResBlock-per-level (stage 2, large)
python train.py --model miudiff --miu_base_channels 160 --miu_channel_mult 1,2,4 --miu_num_res_blocks 1 \
    --miu_stage finetune --dataA /tiles/trainA --dataB /tiles/trainB --steps 500000 --output ./results/

# Small UNIT-DDPM (stage 1 pretrain)
python train.py --model unitddpm --unitddpm_stage pretrain \
    --unitddpm_base_channels 64 --unitddpm_channel_mult 1,2,2,4 --unitddpm_num_res_blocks 1 \
    --dataA /tiles/trainA --dataB /tiles/trainB --steps 500000 --output ./unitddpm_stage1/

# Small UNIT-DDPM (stage 2 finetune, warm-start from stage 1)
python train.py --model unitddpm --unitddpm_stage finetune \
    --unitddpm_base_channels 64 --unitddpm_channel_mult 1,2,2,4 --unitddpm_num_res_blocks 1 \
    --unitddpm_init_ckpt ./unitddpm_stage1/checkpoints/step_500000.pt \
    --dataA /tiles/trainA --dataB /tiles/trainB --steps 500000 --output ./unitddpm_stage2/

# Medium CycleDiffusion (single training run)
python train.py --model cyclediffusion \
    --cd_base_channels 84 --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 2 \
    --dataA /tiles/trainA --dataB /tiles/trainB --steps 1000000 --output ./cyclediffusion/

# Large UNSB
python train.py --model unsb \
    --unsb_base_channels 168 --unsb_channel_mult 1,2,2,4 --unsb_num_res_blocks 2 \
    --dataA /tiles/trainA --dataB /tiles/trainB --steps 1000000 --output ./unsb/

# Dry-run parameter count (no data needed)
python train.py --model cyclegan --count_params --cyclegan_ngf 128 --cyclegan_n_blocks 10
python train.py --model unitddpm --unitddpm_stage finetune --count_params \
    --unitddpm_base_channels 128 --unitddpm_channel_mult 1,2,4 --unitddpm_num_res_blocks 2
python train.py --model cyclediffusion --count_params \
    --cd_base_channels 84 --cd_channel_mult 1,2,2,4 --cd_num_res_blocks 2
python train.py --model unsb --count_params \
    --unsb_base_channels 128 --unsb_channel_mult 1,2,4 --unsb_num_res_blocks 2
```
