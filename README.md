# TriStab

English | [中文](README.zh-CN.md)

## Pretrained Weights

The model weights (3.1 GB) are hosted on Hugging Face: https://huggingface.co/ruoykw/tristab

Download and place the file at `checkpoint/checkpoints/best.ckpt`:

```bash
# huggingface_hub CLI
pip install -U huggingface_hub
hf download ruoykw/tristab checkpoints/best.ckpt --local-dir checkpoint
```

The training configuration for this checkpoint (Hydra configs) is included under [`data/.hydra/`](data/.hydra).

## Dataset

`data/dataset/` is included in the repository, except for the oversized file
`data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (666 MB, over GitHub's 100 MB per-file limit).
That file comes from the Tsuboyama et al. 2023 mega-scale dataset; download it separately and place it back at the original path.
