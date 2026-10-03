# ATDL Assignment 2

I suggest splitting tasks related to training and tasks related to experiments, as once trained models can be shared between the team for experiments. I have identified the following models we would need to train to do component-wise analysis and system-wide (unconditional) comparisions:

|    | Training Iterations | Training Dataset | Vision Model | w. REPA? | Target Repr. (for representation alignment) | Depth | To do                                  | By who?                                                       | Training time       | Energy Estimate* |
|----|---------------------|------------------|--------------|----------|---------------------------------------------|-------|----------------------------------------|---------------------------------------------------------------|---------------------|------------------|
| 1  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | MAE-L                                       | 8     | [ ]                                    | Ondrej                                                        |                     |                  |
| 2  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINO-B                                      | 8     | [ ]                                    | Ondrej                                                        |                     |                  |
| 3  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | MoCov3-L                                    | 8     | [ ]                                    | Ondrej                                                        |                     |                  |
| 4  | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | I-JEPA-H                                    | 8     | [ ]                                    | Ondrej                                                        |                     |                  |
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
| 18 | 400K                | ImageNet 256x256 | SiT-B/2      | No       | N/A                                         | N/A   | [ ]                                    |                                                               |                     |                  |
| 19 | 400K                | ImageNet 256x256 | SiT-L/2      | No       | N/A                                         | N/A   | [X]                                    | [Baseline (Ma et al. 2024)](https://arxiv.org/abs/2401.08740) | N/A                 | N/A              |
| 20 | 400K                | ImageNet 256x256 | SiT-XL/2     | No       | N/A                                         | N/A   | [ ]                                    |                                                               |                     |                  |
| 21 | 7M                  | ImageNet 256x256 | SiT-XL/2     | No       | N/A                                         | N/A   | [ ]                                    |                                                               |                     |                  |
| 22 | 400K                | ImageNet 256x256 | DiT-L/2      | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 23 | 400K                | ImageNet 256x256 | DiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 24 | 7M                  | ImageNet 256x256 | DiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 25 | 400K                | ImageNet 256x256 | SiT-B/2      | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 26 | 400K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L?                                   | 8     | [X] Mark has done or is doing this one |                                                               |                     |                  |
| 27 | 700K                | ImageNet 256x256 | SiT-L/2      | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 28 | 150K                | ImageNet 256x256 | SiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 29 | 400K                | ImageNet 256x256 | SiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |
| 30 | 1M                  | ImageNet 256x256 | SiT-XL/2     | Yes      | DINOv2-L?                                   | 8     | [ ]                                    |                                                               |                     |                  |

*Energy estimate only takes into account energy consumed by GPU during training. The B200 used ran near its maximum TDP of 1000W for the entire duration of model training.

This is probably too much to do in time so we need to decide which ones we will train. In their training script I cannot see an easy option to train DiT models so these are a good candidate to remove from our list.

## Dataset

The REPA authors use a pre-processed version of ImageNet 256x256 downloaded from [here](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/).

## Sharing Data and Models

We need to share data and models which are larger than what is feasible (or good practice) to store on GitHub, so we can use [this link here](https://sharepoint.markmizzi.com/atdl-assignment-2) as a sharepoint, it hosts the files on my personal storage server. I have set the password to be `P$rx4NxzEBBReLeP`.

## Journal

### Dataset preparation

- 26/09 Mark: I made a Kaggle account, created an API key, then ran the following code in Python (from venv with `kagglehub` installed):
  ```python
  import kagglehub 
  kagglehub.login()
  path = kagglehub.competition_download('imagenet-object-localization-challenge', output_dir="data")
  ```
  to download the raw dataset used in the REPA paper. Next I ran the following command:
  ```bash
  python dataset_tools.py convert --source=[DIR]/ILSVRC/Data/CLS-LOC/train \
    --dest=[DIR]/images --resolution=256x256 --transform=center-crop-dhariwal
  ```
  which they give in the `preprocessing/` folder of their repo to apply some simple image transformation to the images. You can download these images (which are also needed for training) [here](https://sharepoint.markmizzi.com/atdl-assignment-2/images.zip).

- 27/09 Mark: Pre-processed latents were generated on UCloud server and are available [here](https://sharepoint.markmizzi.com/atdl-assignment-2/vae-sd.zip). Pre-processing was performed on a B200 (in `1:25:22` hours), I had to make some changes to the `dataset_tools.py` script to bring the time down to this. In particular I added batching support (see [theirs](https://github.com/sihyun-yu/REPA/blob/main/preprocessing/dataset_tools.py#L395-L409) vs. [ours](https://github.com/MarkMizzi/REPA/blob/main/preprocessing/dataset_tools.py#L398-L431)).

- 28/09 Mark: Training failed with an error like subclass of `torch.nn.Module` does not have `module` property, I tried fixing this in commit [535b419](https://github.com/MarkMizzi/REPA/commit/535b41915efa99a6189756dd0358d8b98497ebcf). The repository does have a big problem for reproducibility in that the project requirements were not even frozen to a major version, meaning deprecated or removed features can silently break. I froze deps we are using in [57a2105](https://github.com/MarkMizzi/REPA/commit/57a2105ec9ca5d321598e543d389aa52edb437d8).

## Guide to Training on UCloud

See [this page](./GUIDE.md).

## Troubleshooting

See [this page](./TROUBLESHOOTING.md)
