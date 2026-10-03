# Troubleshooting

## Issue downloading Dino

Their training script apparently has a data race where multiple processes try to download Dino at once and trip each other up. Solution is to download it manually, e.g. using this script:
```python
import torch

model = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vitg14',
)
```

## Restarting from a checkpoint

You can check saved checkpoints under `<output-dir>/<exp-name>/checkpoints`. You can resume from a checkpoint by running the training command above with `--resume-step=<checkpoint-step>`.
  
## Training script

Training script `--data-dir` argument has to point to a folder which contains raw images under `images/` and VAE latents under `vae-sd`
