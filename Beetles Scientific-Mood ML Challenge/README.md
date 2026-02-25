# Beetles Scientific-Mood ML Challenge 🪲

This project aims to predict the Standardized Precipitation Evapotranspiration Index (SPEI) for 30-day, 1-year, and 2-year periods to assess environmental drought conditions. The predictions are made using beetle satellite images, scientific names, and geographic Domain IDs. 

The project utilizes a Hybrid Regressor architecture, combining advanced vision models (such as BioClip and ConvNeXt) with categorical feature embeddings.

---

## ⚠️ Important: Download Data and Weights First!

Before running any training or inference scripts, you **must** download the required datasets and pre-trained model weights from our Hugging Face repository:

🔗 **[Hugging Face Repository: jason79461385/beetles](https://huggingface.co/jason79461385/beetles)**

* **Training Data**: Download the training datasets (`.csv` and Hugging Face `.arrow` database files) and place them in your working directory before starting the training process.
* **Model Weights**: Download the pre-trained model weights (`.pth` files) if you wish to run inference or make a submission directly.

---

## ⚙️ Training Configurations & Innovations

To achieve robust performance, we trained two separate architectures and incorporated several key training strategies:

### 1. Model Architectures & Hyperparameters
We utilized two distinct visual backbones, both combined with a custom regression head (Hidden Size: $512 \rightarrow 256 \rightarrow 6$) that embeds categorical metadata (Species and Domain IDs).

| Parameter | BioClip-2 Model | ConvNeXt Base Model |
| :--- | :--- | :--- |
| **Backbone** | `hf-hub:imageomics/bioclip-2` | `convnext_base.fb_in22k_ft_in1k` |
| **Input Image Size** | 224x224 | 224x224 |
| **Batch Size** | 32 | 64 |
| **Epochs** | 50 | 50 |
| **Optimizer** | AdamW (Weight Decay: $10^{-3}$) | AdamW (Weight Decay: $10^{-3}$) |
| **Learning Rate** | **Dual LR**: $10^{-5}$ (Backbone) / $10^{-4}$ (Head) | $10^{-5}$ (Global) |
| **Scheduler** | Cosine Annealing LR | Cosine Annealing LR |
| **Cross-Validation**| 5-Fold | 5-Fold |

### 2. Key Training Innovations
* **Extreme-Value Weighted Loss:** Instead of a standard MSE, we implemented a custom weighted loss function: `Loss = (MSE * (1.0 + |target|)).mean()`. This forces the model to pay more attention to extreme drought or wet conditions (high absolute SPEI values), which significantly improved our $R^2$ score.
* **Layer Unfreezing Strategy (BioClip):** To prevent catastrophic forgetting of the pre-trained biological features in BioClip, we froze the entire visual transformer and only unfreezed the last 2 ResBlocks (`n_last_trainable_resblocks=2`) alongside the newly initialized regression head.
* **Dual Learning Rate (BioClip):** We applied a smaller learning rate ($10^{-5}$) to the unfreezed BioClip backbone to preserve its representations, while using a larger learning rate ($10^{-4}$) for the randomly initialized regression head to accelerate convergence.

---

## 📂 Project Structure

The project is mainly divided into two core directories: `training/` and `submissions/`.

### 1. `training/` (Training Module)
This directory contains all the scripts required to train the models from scratch.
* `train_bioclip.py`: Training script for fine
