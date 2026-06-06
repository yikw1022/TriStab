# TriStab


Official code for "Predicting mutation-induced protein stability changes from single sequences via triadic message passing".

## Table of Contents

- [Overview](#overview)
- [Hardware Requirements](#hardware-requirements)
- [Getting Started](#getting-started)
- [Data](#data)
- [Training](#training)
- [Inference](#inference)
- [Pretrained Weights](#pretrained-weights)
- [License](#license)

## Overview

Predicting mutation-induced protein stability changes (ΔΔG) is an effective way to improve the efficiency of protein engineering. TriStab is a framework that relies only on single-sequence input: built on ESM2 sequence representations and ProteinMPNN structural features, it uses a site-level substitution branch to model the global dependencies of the mutation site, and a local perturbation branch based on triadic message passing to incorporate structural micro-environment information over a residue neighbor graph. The outputs of the two branches are fused to predict the stability change, covering single-point, multi-point, and insertion/deletion mutation types.

## Hardware Requirements

Experiments were tested on a single NVIDIA RTX 3090 (24GB).

## Getting Started

Set up the environment.

```bash
conda env create -f environment.yml
conda activate TriStab
```

## Data

The datasets are provided in the `data` folder.

## Training

To train the model from scratch, run

```bash
python train.py
```

## Inference

To test the model on different test data, run

```bash
python test.py
```

## Pretrained Weights

The model weights (3.1 GB) are hosted on Hugging Face: https://huggingface.co/ruoykw/tristab

Download and place the file at `checkpoint/checkpoints/best.ckpt`:

```bash
# huggingface_hub CLI
pip install -U huggingface_hub
hf download ruoykw/tristab checkpoints/best.ckpt --local-dir checkpoint
```

The training configuration for this checkpoint (Hydra configs) is included under [`data/.hydra/`](data/.hydra).

## License

This project is licensed under the Apache-2.0 License — see [LICENSE](LICENSE).
