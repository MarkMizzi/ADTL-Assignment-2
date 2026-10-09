import os, sys, zipfile
import numpy as np
from PIL import Image
from tqdm import tqdm

npz_path, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

with zipfile.ZipFile(npz_path) as zf, zf.open("arr_0.npy") as f:
    version = np.lib.format.read_magic(f)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
    else:
        raise ValueError(f"Unsupported .npy version: {version}")

    assert not fortran and dtype == np.uint8, (fortran, dtype)
    n, h, w, c = shape
    per_img = h * w * c
    for i in tqdm(range(n), desc="Extracting", unit="img"):
        img = np.frombuffer(f.read(per_img), dtype=np.uint8).reshape(h, w, c)
        Image.fromarray(img).save(os.path.join(out_dir, f"{i:06d}.png"))
