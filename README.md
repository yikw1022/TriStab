# TriStab

## Pretrained Weights

The model weights (3.1 GB) are hosted on Hugging Face: https://huggingface.co/ruoykw/tristab

Download and place the file at `checkpoint/checkpoints/best.ckpt`:

```bash
# huggingface_hub CLI
pip install -U huggingface_hub
hf download ruoykw/tristab checkpoints/best.ckpt --local-dir checkpoint
```

The training configuration for this checkpoint (Hydra configs) is included under [`data/.hydra/`](data/.hydra).


