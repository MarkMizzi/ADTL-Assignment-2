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

## Figuring out right value for `--projector-embed-dims` when generating samples

If when generating samples you get an error like
```
[rank0]: RuntimeError: Error(s) in loading state_dict for SiT:                                                                                                                               
[rank0]:        size mismatch for projectors.0.4.weight: copying a param with shape torch.Size([1024, 2048]) from checkpoint, the shape in current model is torch.Size([1536, 2048]).        
[rank0]:        size mismatch for projectors.0.4.bias: copying a param with shape torch.Size([1024]) from checkpoint, the shape in current model is torch.Size([1536]).                      
E1004 12:42:31.269000 109094 torch/distributed/elastic/multiprocessing/api.py:1002] failed (exitcode: 1) local_rank: 0 (pid: 109175) of binary: /work/data/REPA/.venv/bin/python3.12     
```
you have to set `--projector-embed-dims` to the size shown in the first entry of first `torch.Size()` shown (in this case `1024`).
