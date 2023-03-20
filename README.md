# [NTIRE 2023 Challenge on Image Denoising](https://cvlai.net/ntire/2023/) @ [CVPR 2023](https://cvpr2023.thecvf.com/)

## How to test the model?

1. `git clone https://github.com/m-hmy/NTIRE2023_Dn50_MegNR.git`
2. Select the model you would like to test from [`run.sh`](./run.sh)
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test_demo.py --data_dir [path to your data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - Be sure the change the directories `--data_dir` and `--save_dir`.

## Pre-train weights
because the pre-train weight is a bit large, so you should download from follow link after 

`mkdir ./model_zoo`

pre-train weight link: https://drive.google.com/file/d/18R5k6g_bpsRu8kXB2eHcX_4mhfZzlpKn/view?usp=sharing

or you can download with shell script `./download_model.sh`