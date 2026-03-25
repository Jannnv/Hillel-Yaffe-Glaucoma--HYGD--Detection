<div align="center">
  <h1>👁️ Glaucoma Detection Pipeline</h1>
  <p><strong>End-to-End Quality-Aware Multi-Task Deep Learning Framework</strong></p>
  
  [![Dataset](https://img.shields.io/badge/Dataset-HYGD-blue.svg)](https://physionet.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-🔥-red.svg)](https://pytorch.org/)
  [![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](#)
</div>

<br/>

## 📖 Overview

This repository contains a comprehensive deep learning pipeline for detecting Glaucomatous Optic Neuropathy (GON) from Deep Fundus Images (DFIs). Built on top of the **Hillel Yaffe Glaucoma Dataset (HYGD)**, it utilizes a **Quality-Aware Multi-Task EfficientNet** framework that jointly predicts the presence of glaucoma and the image quality score.

The codebase is highly modularized into 6 distinct phases ranging from image preprocessing to clinical explainability (XAI).

---

## 🏆 Performance

Based on 5-Fold Cross-Validation, the model achieves state-of-the-art unbiased metrics:

- **AUC-ROC**: `0.9947`
- **AUC-PRC (Average Precision)**: `0.9979`
- **Accuracy**: `96.76%`
- **Sensitivity (Recall)**: `98.15%`
- **Specificity**: `92.95%`
- **F1-Score**: `0.9780`

*(Using the Youden's Index optimal threshold of `0.5365`, Accuracy reaches **98.11%** and Specificity **98.00%**).*

---

## 📂 Project Structure

| File | Description |
| :--- | :--- |
| `config.py` | Central configuration file for hyperparameters, learning rates, paths, and constants. |
| `preprocessing.py` | **[Phase 1]** Handles data preprocessing: fundus cropping, circular masking, CLAHE contrast enhancement, and low quality filtering. |
| `dataset.py` | **[Phase 2]** PyTorch Dataset implementation, image augmentation (Albumentations), and strict patient-level K-Fold splitting to prevent data leakage. |
| `model.py` | **[Phase 3]** The Quality-Aware Multi-Task EfficientNet architecture definition and Custom Loss functions. |
| `train.py` | **[Phase 4]** Training loop, employing MixUp, Automatic Mixed Precision (AMP), early stopping, and model checkpointing. |
| `evaluate.py` | **[Phase 5]** Thorough performance evaluation: metrics, ROC/PR Curves, baseline Grad-CAM, and confidence calibration. |
| `clinical_interpretability.py` | **[Phase 6]** Advanced Explainability (XAI): SHAP values, Grad-CAM++ visualizations, and individual patient Clinical Reports generation. |
| `main.py` | **⚡ Entry Point** to execute any phase or the entire pipeline using CLI arguments. |

---

## 🚀 Usage Guide (`main.py`)

The `main.py` script acts as the main controller. You can run the entire pipeline end-to-end or execute specific phases using the `--phase` argument.

### Full Pipeline
Run Phase 1 through 6 sequentially (Great for a fresh start):
```bash
python main.py
# OR
python main.py --phase 0
```

### Modular Execution
If you want to run or test specific parts of the pipeline:

- **Phase 1: Preprocessing & Filtering** (Test CLAHE, mask, cropping, etc.)
  ```bash
  python main.py --phase 1
  ```
- **Phase 2: Dataset & Augmentation** (Verify dataloaders and patient-level CV splits)
  ```bash
  python main.py --phase 2
  ```
- **Phase 3: Model Architecture** (Run a dummy forward pass and test Loss calculations)
  ```bash
  python main.py --phase 3
  ```
- **Phase 4: Training** (Start K-Fold Cross Validation Training)
  ```bash
  python main.py --phase 4 --epochs 50 --batch_size 16
  ```
- **Phase 5: Evaluation & Metrics** (Evaluate testing sets using saved checkpoints)
  ```bash
  python main.py --phase 5 --fold 1 --checkpoint output/checkpoints/best_model_fold1.pth
  ```
- **Phase 6: Clinical Explainability (XAI)** (Generate SHAP, Grad-CAM++, and Clinical Reports)
  ```bash
  python main.py --phase 6 --fold 1 --gradcam_samples 20 --shap_samples 10
  ```

### Custom Configurations
Easily override parameters via command line variables:
```bash
python main.py --phase 4 --img_size 224 --epochs 30 --batch_size 32
```

