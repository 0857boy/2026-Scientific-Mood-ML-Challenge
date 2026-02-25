# 🧪 Scientific-Mood ML Challenge

> **Scientific Modeling out of Distribution (Scientific-Mood)**  
> NSF HDR ML Challenge — Modeling Out-of-Domain Extrapolation on Critical Scientific Processes

[![Challenge](https://img.shields.io/badge/Challenge-Scientific--Mood-blueviolet)](https://www.codabench.org/)

Machine learning models excel at interpolating across training datasets. This challenge asks models to **extend beyond their training** by performing out-of-domain extrapolation on practical, critical scientific processes that have not yet been well studied.

The HDR ML Challenge program presents **three scientific benchmarks** for modeling out of distribution in critical areas, plus one combined challenge.

📅 **Challenge Period:** September 18, 2025 — February 22, 2026 (11:59 pm AOE)

---

## 📂 Repository Structure

```
Scientific-Mood ML Challenge/
│
├── NSF HDR Scientific Modeling out of distribution NeuralForecasting/
│   ├── baselines/
│   │   ├── submissions/AMAG/        # Submission-ready model weights & code
│   │   └── training/AMAG/           # Training, evaluation & utility scripts
│   ├── CITATION.cff
│   ├── LICENSE
│   └── README.md
│
├── Beetles Scientific-Mood ML Challenge/
│   ├── submissions/                  # Inference & submission code
│   ├── training/                     # Training scripts & utilities
│   └── README.md
│
├── Predicting Coastal Flooding Events - Scientific Out-of-Distribution Challenge/
│   ├── submissions/xgboost_regression/   # Submission model & config
│   ├── training/xgboost_regression/      # Data conversion & training scripts
│   ├── LICENSE.txt
│   └── README.md
│
└── README.md                         # ← You are here
```

---

## 🧠 Sub-Challenge 1: Neural Forecasting

> Forecast activations of a cluster of neurons from prior signals — vital for brain-chip interfaces and artificial limb control.

| Item | Detail |
|---|---|
| **Model** | **AMAG** (Adaptive Multi-scale Attention Graph) |
| **Key Techniques** | RevIN, Multi-Scale Attention, Adaptive Graph Layer |
| **Input Shape** | `(B, 20, N, F)` — 10 observation + 10 prediction steps |
| **Subjects** | Affi (239 neurons) · Beignet (87 neurons) |
| **Platform** | [Codabench — NeuralForecasting](https://www.codabench.org/competitions/9806/) |

### Quick Start

```bash
# Install
pip install -r baselines/submissions/AMAG/requirements.txt

# Train
python baselines/training/AMAG/train.py --monkey affi --epochs 50
python baselines/training/AMAG/train.py --monkey beignet --epochs 50

# Evaluate
python baselines/training/AMAG/evaluation.py --monkey affi
python baselines/training/AMAG/evaluation.py --monkey beignet
```

📖 [Full Details →](NSF%20HDR%20Scientific%20Modeling%20out%20of%20distribution%20NeuralForecasting/README.md)

---

## 🪲 Sub-Challenge 2: Climate Prediction Using Ecological Data (Beetles)

> Predict drought conditions (SPEI) over multiple timescales using images of ecological indicator organisms (ground beetles).

| Item | Detail |
|---|---|
| **Model** | Hybrid Regressor (BioClip-2 ViT-B/16 + ConvNeXt Base) |
| **Regression Head** | Hidden Size: 512 → 256 → 6 with categorical metadata embeddings |
| **Target** | SPEI at 30-day, 1-year, and 2-year scales |
| **Input Features** | Beetle satellite images, scientific names, Domain IDs |
| **Training** | 5-Fold CV · 50 epochs · AdamW + Cosine Annealing LR |
| **Data & Weights** | [🤗 Hugging Face: jason79461385/beetles](https://huggingface.co/jason79461385/beetles) |

**Key Training Innovations:**
- **Extreme-Value Weighted Loss** — `(MSE × (1 + |target|)).mean()` to emphasize extreme drought/wet conditions
- **Layer Unfreezing** (BioClip) — Only last 2 ResBlocks unfrozen to prevent catastrophic forgetting
- **Dual Learning Rate** (BioClip) — 10⁻⁵ for backbone / 10⁻⁴ for regression head

### Quick Start

```bash
# Install
cd training
pip install -r requirements.txt

# Train (BioClip)
python train_bioclip.py

# Train (ConvNeXt)
python train_convnext.py
```

📖 [Full Details →](Beetles%20Scientific-Mood%20ML%20Challenge/README.md)

---

## 🌊 Sub-Challenge 3: Coastal Flooding Prediction Over Time

> Model sea levels at multiple sites across decades to predict coastal floods driven by climate change.

| Item | Detail |
|---|---|
| **Model** | XGBoost Regression |
| **Input Window** | 168 hours (7 days) |
| **Metric** | MCC (Matthews Correlation Coefficient) |
| **Data Sources** | Historical `.mat` station data (1950–2020, 12 stations) |

### Quick Start

```bash
# Install
pip install -r training/xgboost_regression/requirements.txt

# Generate training data
cd training/xgboost_regression/
python convert_to_parquet.py

# Train
python train_xgb_regression.py

# Evaluate
cd ../../submissions/xgboost_regression/
python model.py --test_hourly <path_to_test_hourly.csv> --test_index <path_to_test_index.csv> --predictions_out submission.csv
```

📖 [Full Details →](Predicting%20Coastal%20Flooding%20Events%20-%20Scientific%20Out-of-Distribution%20Challenge/README.md)

---

## 🏆 Challenge Organizers

<details>
<summary><b>Imageomics</b></summary>

- Elizabeth G. Campolongo
- Wei-Lun Chao
- Chandra Earl (NEON)
- Hilmar Lapp
- Kayla Perry
- Sydne Record
- Eric Sokol (NEON)

**Student Organizers:** David E. Carlyn · Alyson East · Connor Kilrain · Fangxun Liu · Zheda Mai · S M Rayeed · Jiaman Wu
</details>

<details>
<summary><b>A3D3</b></summary>

- Yuan-Tang Chou
- Ekaterina Govorkova
- Philip Harris
- Shih-Chieh Hsu
- Mark S. Neubauer
- Amy Orsborn
- Leo Scholl
- Eli Shlizerman

**Student Organizer:** Jingyuan Li
</details>

<details>
<summary><b>iHARP</b></summary>

- Ratnaksha Lele
- Aneesh Subramanian
- Josephine Namayanja
- Bayu Tama
- Vandana Janeja

**Student Organizers:** Sai Vikas Amaraneni · Emam Hossain · Maloy Kumar Devnath · Subhankar Ghosh
</details>

