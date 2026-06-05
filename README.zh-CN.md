# TriStab

[English](README.md) | 中文

## 预训练权重

模型权重（3.1GB）托管在 Hugging Face：https://huggingface.co/ruoykw/tristab

下载后放到 `checkpoint/checkpoints/best.ckpt`：

```bash
# huggingface_hub CLI
pip install -U huggingface_hub
hf download ruoykw/tristab checkpoints/best.ckpt --local-dir checkpoint
```

该权重对应的训练配置（Hydra 配置）已包含在 [`data/.hydra/`](data/.hydra) 下。

## 数据集

`data/dataset/` 已随仓库提供，但其中体积过大的
`data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv`（666MB，超过 GitHub 单文件 100MB 上限）未包含。
该文件来自 Tsuboyama et al. 2023 的 mega-scale 数据集，请自行下载后放回原路径。
