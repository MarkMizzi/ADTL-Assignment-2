# ATDL Assignment 2

This is the main code repo for the reproduction study carried out by Group 19 as Assignment 2 of Advanced Topics in Deep Learning (ATDL). The paper reproduced is ["Representation Alignment for Generation: Training Diffusion Transformers Is Easier Than You Think"](https://arxiv.org/abs/2410.06940).

## Scope

When choosing the scope of reproduction carried out, the team opted for training models from scratch using the code provided by the authors of the paper in [this repo](https://github.com/sihyun-yu/REPA). Some minor modifications to the code were required to get it to run, and these were tracked in [this fork of the repo](https://github.com/MarkMizzi/REPA), which is also provided as a submodule of the current repo.

The models evaluated in the paper were tabulated and assigned to team members for training according to the table below (Note that due to time constraints, not all models and experiments from the paper could be recreated):

|    | Training Iterations | Training Dataset | Vision Model | w. REPA? | Target Repr. (for representation alignment) | Depth | To do                                  | By who?                                                       | Training time       | Energy Estimate* |
|----|---------------------|------------------|--------------|----------|---------------------------------------------|-------|----------------------------------------|---------------------------------------------------------------|---------------------|------------------|
| 1  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | MAE-L                                       | 8     | [X]                                    | Ondrej                                                        |                     |                  |
| 2  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINO-B                                      | 8     | [X]                                    | Ondrej                                                        |                     |                  |
| 3  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | MoCov3-L                                    | 8     | [X]                                    | Ondrej                                                        |                     |                  |
| 4  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | I-JEPA-H                                    | 8     | [X]                                    | Ondrej                                                        |                     |                  |
| 5  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | CLIP-L                                      | 8     | [X]                                    | Mark                                                          | 50:01:05            | ~50 kWh          |
| 6  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | SigLIP-L                                    | 8     | [ ]                                    |                                                               |                     |                  |
| 7  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-B                                    | 8     | [X]                                    | Mark                                                          | 42:08:51            | ~42 kWh          |
| 8  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L                                    | 8     | [X]                                    | Mark                                                          | 48:00:53            | ~48 kWh          |
| 9  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-g                                    | 8     | [X]                                    | Mark                                                          | ~64:00:00           | ~64 kWh          |
| 10 | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L                                    | 6     | [X]                                    | FREDERIK                                                      | done around oct 4-5 |                  |
| 11 | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L                                    | 10    | [X]                                    | FREDERIK                                                      | done around oct 4-5 |                  |
| 12 | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L                                    | 12    | [X]                                    | FREDERIK                                                      | done around oct 4-5 |                  |
| 13 | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L                                    | 14    | [X]                                    | FREDERIK                                                      | done around oct 4-5 |                  |
| 14 | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L                                    | 16    | [X]                                    | FREDERIK                                                      | done around oct 4-5 |                  |
| 15 | 400K                | ImageNet 256x256 | DiT-L/2      | No       | N/A                                         | N/A   | [ ]                                    |                                                               |                     |                  |
| 16 | 400K                | ImageNet 256x256 | DiT-XL/2     | No       | N/A                                         | N/A   | [ ]                                    |                                                               |                     |                  |
| 17 | 7M                  | ImageNet 256x256 | DiT-XL/2     | No       | N/A                                         | N/A   | [ ]                                    |                                                               |                     |                  |
| 18 | 400K                | ImageNet 256x256 | SiT-B/2      | No       | N/A                                         | N/A   | [X]                                    | [Baseline (Ma et al. 2024)](https://arxiv.org/abs/2401.08740) | N/A                 | N/A              |
| 19 | 400K                | ImageNet 256x256 | SiT-L/2      | No       | N/A                                         | N/A   | [X]                                    | [Baseline (Ma et al. 2024)](https://arxiv.org/abs/2401.08740) | N/A                 | N/A              |
| 20 | 400K                | ImageNet 256x256 | SiT-XL/2     | No       | N/A                                         | N/A   | [X]                                    | [Baseline (Ma et al. 2024)](https://arxiv.org/abs/2401.08740) | N/A                 | N/A              |
| 21 | 7M                  | ImageNet 256x256 | SiT-XL/2     | No       | N/A                                         | N/A   | [X]                                    | [Baseline (Ma et al. 2024)](https://arxiv.org/abs/2401.08740) | N/A                 | N/A              |
| 22 | 400K                | ImageNet 256x256 | DiT-L/2      | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 23 | 400K                | ImageNet 256x256 | DiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 24 | 7M                  | ImageNet 256x256 | DiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 25 | 400K                | ImageNet 256x256 | SiT-B/2      | Yes      | DINOv2-L?                                   | 8     | [X]                                    | Natasha                                                            |                     |                  |
| 26 | 700K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L?                                   | 8     | [X]                                    | Natasha                                                            |                     |                  |
| 27 | 150K                | ImageNet 256x256 | SiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [X]                                    | Natasha                                                            |                     |                  |
| 28 | 400K                | ImageNet 256x256 | SiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [X]                                    | Natasha                                                            |                     |                  |
| 29 | 1M                  | ImageNet 256x256 | SiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |

*Energy estimate only takes into account energy consumed by GPU during training. The B200 used ran near its maximum TDP of 1000W for the entire duration of model training.

## Dataset

For all their experiments, the REPA authors used a pre-processed version of ImageNet 256x256 downloaded from [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/).

The dataset was downloaded using the following Python script (in a venv with `kagglehub` installed):
```python
import kagglehub 
kagglehub.login()
path = kagglehub.competition_download('imagenet-object-localization-challenge', output_dir="data")
```

## Sharing Data and Models

Since the project required sharing data and models which are larger than what is feasible (or good practice) to store on GitHub, we used [this link here](https://sharepoint.markmizzi.com/atdl-assignment-2) as a sharepoint, which hosts the files on Mark's personal storage server. The access password for this share is `P$rx4NxzEBBReLeP`.

The dataset was pre-processed and Stability VAE latents were generated by Mark using the commands given in [this README](./REPA/preprocessing/README.md). Batching was added to the `dataset_tools.py` script to bring the time down from around 60 hours to `1:25:22` hours. The [preprocessed dataset](https://sharepoint.markmizzi.com/atdl-assignment-2/images.zip) and [latents](https://sharepoint.markmizzi.com/atdl-assignment-2/vae-sd.zip) can be downloaded from the given links.

## Training

Our training process is documented extensively in guide form [in this page](./TRAINING.md)

## Evaluation

To calculate evaluation metrics for the trained models, commands similar to the following were first run on the UCloud servers (from the root of the `REPA` repo) to generate a sample dataset:
```bash
torchrun             \
  --nnodes=1         \
  --nproc_per_node=1 \
  generate.py        \
  --model SiT-L/2    \
  --num-fid-samples 50000   \
  --ckpt exps/linear-sit-l-2-dinov2-g-enc8/checkpoints/0400000.pt \
  --mode sde         \
  --cfg-scale=1      \
  --projector-embed-dims=1536  \
  --num-steps=250
```

Latents generated from the command above are compared with a standard ImageNet 256x256 latent obtained [from this link](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz), using the following command:
```bash
python guided-diffusion/evaluations/evaluator.py \
    VIRTUAL_imagenet256_labeled.npz              \
    <samples-dir>/SiT-L-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde.npz
```
which generates all metrics we reproduce from the paper.

NOTE: In order to run the `evaluator.py` script, you should set up a Python venv with all dependencies in [./guided-diffusion-requirements.txt](./guided-diffusion-requirements.txt) installed.

## Artifacts

Artifacts such as trained models and images generated for evaluation are available at [this link](https://sharepoint.markmizzi.com/atdl-assignment-2/), and are organized in the following format:

```text
├── models
|   └── linear-sit-l-2-<enc-type>-enc<enc-depth>
|       ├── checkpoints                              
|       |   ├── 0400000.pt                           # Final training checkpoint, containing trained model + optimizer state
|       |   └── <other-checkpoint>.pt
|       └── samples                                  # Samples generated for evaluation
|           ├── SiT-L-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde.npz        # Compressed archive of evaluation samples
|           └── SiT-L-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde-<other-checkpoint-number>.npz    # Same as above but for checkpoint ≠ 400K
└── VIRTUAL_imagenet256_labeled.npz                  # Reference latent for ImageNet 256x256
```

The images generated for evaluation can be extracted from the `.npz` archives using the [./extract_images_from_npz.py](./extract_images_from_npz.py), e.g.
```bash
python extract_images_from_npz.py SiT-L-2-0400000-size-256-vae-ema-cfg-1.0-seed-0-sde.npz outdir
```

## Troubleshooting

See [this page](./TROUBLESHOOTING.md)
