# Guide to Training on UCloud

1. Download [`images.zip`](https://sharepoint.markmizzi.com/atdl-assignment-2/images.zip) and [`vae-sd.zip`](https://sharepoint.markmizzi.com/atdl-assignment-2/vae-sd.zip).
2. Upload them to UCloud under your `Jobs` directory
3. Right click > "Uncompress" on both uploaded archives. A job will start, wait for it to finish.
4. Move both uncompressed dirs into a `data` directory under `Jobs`. Make sure that the dirs are the ones which contain data folders directly underneath them.
5. From the left side bar, click "Applications" > "Development" and then pick "Terminal":
   <img width="1277" height="695" alt="screenshot_2026-09-27_18:23:36" src="https://github.com/user-attachments/assets/32b6b3c1-90ea-45b8-8eee-3edde7c0408c" />
6. Use the following parameters for the new job:
   <img width="1278" height="695" alt="screenshot_2026-10-03_13:51:22" src="https://github.com/user-attachments/assets/55e8e6e4-f694-464e-ab7e-acdb370ef379" />

   100 hours is enough, you can increase this later. Make sure everything matches the screenshot (except for name)
8. Under "Storage" set "Folder #1" to the `data` directory created earlier:
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


