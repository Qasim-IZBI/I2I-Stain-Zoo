# I2I-Stain-Zoo

Tested bash code
## Train model
```bash
!python train.py --model cyclegan \
    --dataA /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainA/images/ \
        --dataB /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/trainB/images/ \
            --epochs 10 \
                --amp \
                    --output /home/qasim/Desktop/Computer/Hoehme_Git/Qasim/demo_virtualstaining/models/cyclegan/
```

## Inference
```bash
!python inference.py \
  --model munit \
  --direction A2B \
  --data /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testA/images \
  --ckpt /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Models/munit/checkpoints/epoch_50.pt \
  --outdir /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/fake_munit \
    --num_samples 1
```

## Evaluation
```bash
!python evaluation.py \
    --path_real /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/images \
        --path_fake /home/qasim/Desktop/Computer/Projects/Qasim/Ahmed/Virtual_Staining/20032025/Data/tiles/testB/fake_unit \
            --backend inception \
                --device cuda
``
