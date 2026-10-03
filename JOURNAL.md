# Journal

## Dataset preparation

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
