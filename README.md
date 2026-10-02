# ATDL Assignment 2

I suggest splitting tasks related to training and tasks related to experiments, as once trained models can be shared between the team for experiments. I have identified the following models we would need to train to do component-wise analysis and system-wide (unconditional) comparisions:

| | Training Iterations | Training Dataset | Vision Model | w. REPA? | Target Repr. (for representation alignment) | Depth | To do | By who? | Training time | Energy Estimate* |
|--|--|--|--|--|--|--|--|--|--|--|
| 1 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | MAE-L | 8 | [ ] | | | |
| 2 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINO-B | 8 | [ ] | | | |
| 3 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | MoCov3-L | 8 | [ ] | | | |
| 4 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | I-JEPA-H | 8 | [ ] | | | |
| 5 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | CLIP-L | 8 | [X] | Mark | 50:01:05 | ~50 kWh |
| 6 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | SigLIP-L | 8 | [ ] | | | |
| 7 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [X] | Mark | | |
| 8 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-B | 8 | [X] | Mark | | |
| 9 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [X] | Mark | | |
| 10 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-g | 8 | [X] | Mark | ~64:00:00 | ~64 kWh |
| 11 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 6 | [X] |FREDERIK | done around oct 4-5| |
| 12 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 8 | [X] | Mark | | |
| 13 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 10 | [X] |FREDERIK | done around oct 4-5 | |
| 14 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 12 | [X] |FREDERIK | done around oct 4-5 | |
| 15 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 14 | [X] |FREDERIK| done around oct 4-5 | |
| 16 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L | 16 | [X] |FREDERIK | done around oct 4-5 | |
| 17 | 400K | ImageNet 256x256 | DiT-L/2 | No | N/A | N/A | [ ] | | | |
| 18 | 400K | ImageNet 256x256 | DiT-XL/2 | No | N/A | N/A | [ ] | | | |
| 19 | 7M | ImageNet 256x256 | DiT-XL/2 | No | N/A | N/A | [ ] | | | |
| 20 | 400K | ImageNet 256x256 | SiT-B/2 | No | N/A | N/A | [ ] | | | |
| 21 | 400K | ImageNet 256x256 | SiT-L/2 | No | N/A | N/A | [X] | [Baseline (Ma et al. 2024)](https://arxiv.org/abs/2401.08740) | N/A | N/A |
| 22 | 400K | ImageNet 256x256 | SiT-XL/2 | No | N/A | N/A | [ ] | | | |
| 23 | 7M | ImageNet 256x256 | SiT-XL/2 | No | N/A | N/A | [ ] | | | |
| 24 | 400K | ImageNet 256x256 | DiT-L/2 | Yes | DINOv2-L? | 8 | [ ] | | | |
| 25 | 400K | ImageNet 256x256 | DiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] | | | |
| 26 | 7M | ImageNet 256x256 | DiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] | | | |
| 27 | 400K | ImageNet 256x256 | SiT-B/2 | Yes | DINOv2-L? | 8 | [ ] | | | |
| 28 | 400K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L? | 8 | [X] Mark has done or is doing this one | | | |
| 29 | 700K | ImageNet 256x256 | SiT-L/2 | Yes | DINOv2-L? | 8 | [ ] | | | |
| 30 | 150K | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] | | | |
| 31 | 400K | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] | | | |
| 32 | 1M | ImageNet 256x256 | SiT-XL/2 | Yes | DINOv2-L? | 8 | [ ] | | | |

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

1. Download [`images.zip`](https://sharepoint.markmizzi.com/atdl-assignment-2/images.zip) and [`vae-sd.zip`](https://sharepoint.markmizzi.com/atdl-assignment-2/vae-sd.zip).
2. Upload them to UCloud under your `Jobs` directory
3. Right click > "Uncompress" on both uploaded archives. A job will start, wait for it to finish.
4. Move both uncompressed dirs into a `data` directory under `Jobs`. Make sure that the dirs are the ones which contain data folders directly underneath them.
5. From the left side bar, click "Applications" > "Development" and then pick "Terminal":
   <img width="1277" height="695" alt="screenshot_2026-09-27_18:23:36" src="https://github.com/user-attachments/assets/32b6b3c1-90ea-45b8-8eee-3edde7c0408c" />
6. Use the following parameters for the new job:
   <img width="1279" height="695" alt="screenshot_2026-09-27_18:26:41" src="https://github.com/user-attachments/assets/2d446fae-e03a-4538-96fb-dff9ef14ead5" />
   100 hours is enough, you can increase this later. Make sure everything matches the screenshot (except for name)
7. Under "Storage" set "Folder #1" to the `data` directory created earlier:
   <img width="833" height="155" alt="screenshot_2026-09-27_18:28:33" src="https://github.com/user-attachments/assets/d3fc9a41-0acf-484a-8952-bf0b27c86cf4" />

   It is **VERY IMPORTANT** that every output is saved in this directory, as anything else will not be saved.
9. Click "Submit"
10. From the left sidebar click "Jobs" > `<Name of the job created in the last step>`. You will be shown a status page:
    <img width="1280" height="696" alt="screenshot_2026-09-27_18:31:41" src="https://github.com/user-attachments/assets/02395242-6f53-4156-b17b-ffeafa7fe6be" />
11. Click "Open terminal" and a bash session will start in a new tab
12. Run `tmux`, this will ensure that your work is not lost if you close the tab
13. `cd` to the `data` directory (which should be under `/work`), and then copy-paste the contents of [this script](./setup_training_env) and run them.
14. You are now ready to run a training command, when running from the `REPA` folder, it should look something like this:
    ```bash
    accelerate launch train.py   \
      --report-to="wandb"        \
      --allow-tf32               \
      --mixed-precision="fp16"   \
      --seed=0                   \
      --path-type="linear"       \
      --prediction="v"           \
      --weighting="uniform"      \
      --model="SiT-L/2"          \
      --enc-type="dinov2-vit-g"  \
      --proj-coeff=0.5           \
      --encoder-depth=8          \
      --output-dir="exps"        \
      --exp-name="linear-sit-l-2-dinov2-g-enc8" \
      --data-dir=../
    ```
    with appropriate changes to `--model`, `--enc-type`, `--encoder-depth`. See [REPA README.md](./REPA/README.md) for more details on the options. Model weights will be saved under `exps` in the `REPA` repo.
15. Monitor the job over time and check if it needs more time. You can add more hours from the status page.


## Troubleshooting

### Issue downloading Dino

Their training script apparently has a data race where multiple processes try to download Dino at once and trip each other up. Solution is to download it manually, e.g. using this script:
```python
import torch

model = torch.hub.load(
    'facebookresearch/dinov2',
    'dinov2_vitg14',
)
```

### Restarting from a checkpoint

You can check saved checkpoints under `<output-dir>/<exp-name>/checkpoints`. You can resume from a checkpoint by running the training command above with `--resume-step=<checkpoint-step>`.
  
### Training script

Training script `--data-dir` argument has to point to a folder which contains raw images under `images/` and VAE latents under `vae-sd`
