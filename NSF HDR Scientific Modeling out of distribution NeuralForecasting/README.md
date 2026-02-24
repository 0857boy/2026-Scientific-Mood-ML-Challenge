# NSF HDR Scientific Mood Challenge — NeuralForecasting

This repository contains training code and submissions for the [NSF HDR Scientific Mood (Modeling out of Distribution) Challenge: NeuralForecasting Track](https://www.codabench.org/competitions/9854/).

**Model:** AMAG (Adaptive Multi-scale Attention Graph)  
**Author:** Joe Liao, National Central University (NCU)

## Repository Structure

For your submission, you will want the following:

```
submission/
  model_affi.pth
  model_beignet.pth
  model.py
  requirements.txt
```

We also recommend that you include a [CITATION.cff](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) for your work.

### Structure of this Repository

```
NSF-HDR-NeuralForecasting/
│
├── baselines/
│   ├── submissions/
│   │   └── AMAG/
│   │       ├── model_affi.pth
│   │       ├── model_affi_seed50.pth
│   │       ├── model_affi_seed51.pth
│   │       ├── model_affi_seed52.pth
│   │       ├── model_beignet.pth
│   │       ├── model_beignet_seed50.pth
│   │       ├── model_beignet_seed51.pth
│   │       ├── model_beignet_seed52.pth
│   │       ├── model.py
│   │       └── requirements.txt
│   └── training/
│       └── AMAG/
│           ├── model.py
│           ├── train.py
│           ├── evaluation.py
│           └── utils.py
├── .gitignore
├── LICENSE
├── CITATION.cff
└── README.md
```

> [!IMPORTANT]
> Do not zip the whole submission folder when submitting your model to Codabench. Only select the `model.py` and relevant weight and `requirements.txt` files to make the tarball.

## Installation & Running (for Training)

### Installation

If you have `uv`, simply run `uv sync`. Otherwise you can use the `requirements.txt` file with either `conda` or `pip`:

```bash
conda create -n neural-forecasting -c conda-forge pip -y
conda activate neural-forecasting
pip install -r baselines/submissions/AMAG/requirements.txt
```

### Training

An example training run can be executed by running the following:

```bash
python baselines/training/AMAG/train.py --monkey affi --epochs 50
python baselines/training/AMAG/train.py --monkey beignet --epochs 50
```

With `uv`:

```bash
uv run python baselines/training/AMAG/train.py --monkey affi --epochs 50
```

### Evaluation

After training, you can locally evaluate your model by running the following:

```bash
python baselines/training/AMAG/evaluation.py --monkey affi
python baselines/training/AMAG/evaluation.py --monkey beignet
```

With `uv`:

```bash
uv run python baselines/training/AMAG/evaluation.py --monkey affi
```

## Model Overview

**AMAG** (Adaptive Multi-scale Attention Graph) is a neural time-series forecasting model that combines:

- **RevIN** — Reversible Instance Normalization for distribution-shift robustness
- **Multi-Scale Attention** — Fuses local (depthwise convolution) and global (multi-head attention) temporal patterns
- **Adaptive Graph Layer** — Learns a sparse adjacency matrix from node embeddings with sample-dependent modulation

The model takes input of shape `(B, 20, N, F)` where:
- `B` = batch size
- `20` = 10 observation + 10 prediction time steps
- `N` = number of nodes (239 for Affi, 87 for Beignet)
- `F` = 9 input features

## References

- Kim, T., et al. "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." *ICLR 2022*.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
