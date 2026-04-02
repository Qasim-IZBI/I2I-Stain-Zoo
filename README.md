# I2I-Stain-Zoo

Tested bash code
## Train model

### Train CycleGAN/UNIT/MUNIT/DCLGAN
```bash
!python train.py --model cyclegan \
    --dataA /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainA/images/ \
    --dataB /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainB/images/ \
    --steps 5000000 \
    --amp \
    --output /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/models/cyclegan/
```

### Train MIUDIFF
#### Stage 1
```bash
!python train.py --model miudiff \
    --dataA /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainA/images/ \
    --dataB /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainB/images/ \
    --steps 500000 \
    --amp \
    --miu_stage pretrain \
    --output /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Models/miudiff/stage1/
```
#### Stage 2
```bash
!python train.py --model miudiff \
    --dataA /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainA/images/ \
    --dataB /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainB/images/ \
    --steps 500000 \
    --amp \
    --miu_stage finetune \
    --output /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Models/miudiff/stage1/
```
#### Stage 3
```bash
!python train.py --model miudiff \
    --miu_stage finetune \
    --miu_pcl \
    --lambda_pcl 0.1 \
    --pcl_n_patches 256 \
    --pcl_proj_dim 128 \
    --dataA /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainA/images/ \
    --dataB /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainB/images/ \
    --steps 500000 \
    --output /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Models/miudiff/stage3/ \
        --amp
```

## Inference
### CycleGAN/UNIT/DCLGAN
```bash
!python inference.py \
  --model munit \
  --direction A2B \
  --data /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testA/images \
  --ckpt /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Models/munit/checkpoints/step_5000000.pt \
  --outdir /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/fake_munit \
```

### MUNIT
```bash
!python inference.py \
  --model munit \
  --direction A2B \
  --data /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testA/images \
  --ckpt /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Models/munit/checkpoints/step_5000000.pt \
  --outdir /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/fake_munit \
    --num_samples 1
```

### MIUDIFF
```bash
!python inference.py \
    --model miudiff \
    --direction A2B \
    --data /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testA/images \
    --ckpt /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Models/miudiff/stage3/checkpoints/step_500000.pt \
    --outdir /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/fake_miudiff \
    --miu_pcl \
    --pcl_refine_steps 3 \
    --pcl_refine_lr 0.05 \
    --miu_steps 200 \
    --miu_guidance 1.0 \

```

## Evaluation
```bash
!python evaluation.py \
    --path_real /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/images \
    --path_fake /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/fake_unit \
    --backend inception \
    --device cuda
``
