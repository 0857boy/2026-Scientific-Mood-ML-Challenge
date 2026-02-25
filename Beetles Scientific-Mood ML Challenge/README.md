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

## 📂 Project Structure

The project is mainly divided into two core directories: `training/` and `submissions/`.

### 1. `training/` (Training Module)
This directory contains all the scripts required to train the models from scratch.
* `train_bioclip.py`: Training script for fine-tuning the [Imageomics BioClip-2](https://huggingface.co/imageomics/bioclip-2) (ViT-B/16) model. It includes 5-Fold Cross-Validation and an extreme-weather weighted Loss function.
* `train_convnext.py`: Training script utilizing the ConvNeXt Base model.
* `utils.py`: Shared utility functions (e.g., data preprocessing, metrics calculation, debugging tools).
* `mappings.json`: Dictionary mapping Species (scientific names) and Domain IDs to numerical integer IDs.
* `requirements.txt`: Python dependencies required for training.

### 2. `submissions/` (Inference & Submission Module)
This directory contains the code to be zipped and uploaded to the official evaluation system.
* `model.py`: The core inference script. It contains the `Model` class responsible for loading weights, preprocessing, and making predictions. This script is heavily optimized to meet the evaluation system's strict memory limits and output formatting requirements (Scalar Aggregation).
* `mappings.json`: Required ID mapping dictionary for inference.
* `requirements.txt`: Python dependencies required by the evaluation environment (e.g., `timm`, `open_clip_torch`, `pandas`).

---

## 🚀 Quick Start

### Model Training
1. Ensure you have downloaded the dataset from the Hugging Face repository to your local machine.
2. Navigate to the training directory and install dependencies:
   ```bash
   cd training
   pip install -r requirements.txt
