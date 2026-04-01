# Model Size Reference

Parameter counts are for the **A→B single-direction forward pass** (inference).
All counts were verified with `python train.py --model <model> --count_params <args>`.

> **MIUDiff note**: both `eps_uncond` and `eps_cond` are counted because classifier-free guidance
> requires both UNets at inference time. Stage-1 pretrain uses `eps_uncond` only (~half the A→B count).

---

## Small (~10M parameters)

| Model | `--ngf` / channels | blocks | other args | A→B params |
|-------|-------------------|--------|------------|------------|
| CycleGAN | `--cyclegan_ngf 64` | `--cyclegan_n_blocks 8` | — | **10.20M** |
| UNIT | `--unit_ngf 64` | `--unit_n_blocks 8 --unit_n_blocks_shared 2` | — | **10.33M** |
| MUNIT | `--munit_ngf 64` | `--munit_n_content_blocks 3 --munit_n_adain_blocks 4` | — | **10.14M** |
| DCLGAN | `--dclgan_ngf 64` | `--dclgan_n_blocks 8` | — | **10.20M** |
| UVCGAN | `--uvcgan_ngf 48` | `--uvcgan_vit_blocks 6` | `--uvcgan_vit_features 96` | **10.01M** |
| MIUDiff | `--miu_base_channels 32` | — | `--miu_channel_mult 1,2,4,4` | **10.31M** |

---

## Medium (~50M parameters)

| Model | `--ngf` / channels | blocks | other args | A→B params |
|-------|-------------------|--------|------------|------------|
| CycleGAN | `--cyclegan_ngf 128` | `--cyclegan_n_blocks 10` | — | **50.18M** |
| UNIT | `--unit_ngf 128` | `--unit_n_blocks 10 --unit_n_blocks_shared 2` | — | **50.71M** |
| MUNIT | `--munit_ngf 128` | `--munit_n_content_blocks 5 --munit_n_adain_blocks 4` | — | **47.64M** |
| DCLGAN | `--dclgan_ngf 128` | `--dclgan_n_blocks 10` | — | **50.18M** |
| UVCGAN | `--uvcgan_ngf 96` | `--uvcgan_vit_blocks 6` | `--uvcgan_vit_features 384` | **48.28M** |
| MIUDiff | `--miu_base_channels 64` | — | `--miu_channel_mult 1,2,2,4,4` | **47.97M** |

---

## Large (~100M parameters)

| Model | `--ngf` / channels | blocks | other args | A→B params |
|-------|-------------------|--------|------------|------------|
| CycleGAN | `--cyclegan_ngf 192` | `--cyclegan_n_blocks 9` | — | **102.26M** |
| UNIT | `--unit_ngf 192` | `--unit_n_blocks 9 --unit_n_blocks_shared 3` | — | **103.44M** |
| MUNIT | `--munit_ngf 192` | `--munit_n_content_blocks 4 --munit_n_adain_blocks 4` | — | **94.87M** |
| DCLGAN | `--dclgan_ngf 192` | `--dclgan_n_blocks 9` | — | **102.26M** |
| UVCGAN | `--uvcgan_ngf 128` | `--uvcgan_vit_blocks 17` | `--uvcgan_vit_features 384` | **96.72M** |
| MIUDiff | `--miu_base_channels 64` | — | `--miu_channel_mult 1,2,4,8` | **98.60M** |

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

---

## Full example commands

```bash
# Small CycleGAN
python train.py --model cyclegan --cyclegan_ngf 64 --cyclegan_n_blocks 8 \
    --dataA /tiles/trainA --dataB /tiles/trainB --epochs 100 --amp --output ./results/

# Medium UVCGAN
python train.py --model uvcgan --uvcgan_ngf 96 --uvcgan_vit_features 384 --uvcgan_vit_blocks 6 \
    --dataA /tiles/trainA --dataB /tiles/trainB --epochs 100 --amp --output ./results/

# Large MIUDiff (stage 2)
python train.py --model miudiff --miu_base_channels 64 --miu_channel_mult 1,2,4,8 \
    --miu_stage finetune --dataA /tiles/trainA --dataB /tiles/trainB --epochs 10 --amp --output ./results/

# Dry-run parameter count (no data needed)
python train.py --model cyclegan --count_params --cyclegan_ngf 128 --cyclegan_n_blocks 10
```
