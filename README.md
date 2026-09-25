
I suggest splitting tasks related to training and tasks related to experiments, as once trained models can be shared between the team for experiments. I have identified the following models we would need to train to do component-wise analysis and system-wide (unconditional) comparisions:

| | Training Iterations | Training Dataset | Vision Model | w. REPA? | Target Repr. (for representation alignment) | Depth | To do |
|--|--|--|--|--|--|--|--|
| 1 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | MAE-L | 8 | [ ] |
| 2 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINO-B | 8 | [ ] |
| 3 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | MoCov3-L | 8 | [ ] |
| 4 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | I-JEPA-H | 8 | [ ] |
| 5 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | CLIP-L | 8 | [ ] |
| 6 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | SigLIP-L | 8 | [ ] |
| 7 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [ ] |
| 8 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-B | 8 | [ ] |
| 9 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [ ] |
| 10 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-g | 8 | [ ] |
| 11 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 6 | [ ] |
| 12 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [ ] |
| 13 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 10 | [ ] |
| 14 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 12 | [ ] |
| 15 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 14 | [ ] |
| 16 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 16 | [ ] |
| 17 | 400K | ImageNet 256x256 | DiT-L/2 | No | N/A | N/A | [ ] |
| 18 | 400K | ImageNet 256x256 | DiT-XL/2 | No | N/A | N/A | [ ] |
| 19 | 7M | ImageNet 256x256 | DiT-XL/2 | No | N/A | N/A | [ ] |
| 20 | 400K | ImageNet 256x256 | SiT-B/2 | No | N/A | N/A | [ ] |
| 21 | 400K | ImageNet 256x256 | SiT-L/2 | No | N/A | N/A | [ ] |
| 22 | 400K | ImageNet 256x256 | SiT-XL/2 | No | N/A | N/A | [ ] |
| 23 | 7M | ImageNet 256x256 | SiT-XL/2 | No | N/A | N/A | [ ] |
| 24 | 400K | ImageNet 256x256 | DiT-L/2 | Yes | DINOv2-L? | 8 | [ ] |
| 25 | 400K | ImageNet 256x256 | DiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 26 | 7M | ImageNet 256x256 | DiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 27 | 400K | ImageNet 256x256 | SiT-B/2 | Yes | DINOv2-L? | 8 | [ ] |
| 28 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L? | 8 | [ ] |
| 29 | 700K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L? | 8 | [ ] |
| 30 | 150K | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 31 | 400K | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |
| 32 | 1M | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] |

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
  which they give in the `preprocessing/` folder of their repo to apply some simple image transformation to the images.
