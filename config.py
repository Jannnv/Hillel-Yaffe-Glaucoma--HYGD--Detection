"""
=============================================================================
config.py — Konfigurasi & Hyperparameters untuk GON Detection Pipeline
=============================================================================
Hillel Yaffe Glaucoma Dataset (HYGD) - Quality-Aware GON Detection
"""

import os
from pathlib import Path

# ===========================================================================
# Path Configuration
# ===========================================================================
BASE_DIR = Path(__file__).parent
IMAGES_DIR = BASE_DIR / "Images"
LABELS_CSV = BASE_DIR / "Labels.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_DIR = OUTPUT_DIR / "results"
GRADCAM_DIR = OUTPUT_DIR / "gradcam"
PHASE6_DIR = OUTPUT_DIR / "phase6_interpretability"

# Buat direktori output
for d in [OUTPUT_DIR, CHECKPOINT_DIR, RESULTS_DIR, GRADCAM_DIR, PHASE6_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ===========================================================================
# Dataset Configuration
# ===========================================================================
RANDOM_SEED = 42
NUM_FOLDS = 5              # Patient-level K-Fold CV
QUALITY_THRESHOLD = 3.0    # Minimum quality score (filter gambar < threshold)

# ===========================================================================
# Image Preprocessing
# ===========================================================================
IMG_SIZE = 512              # Resize target (512×512)
IMG_SIZE_SMALL = 224        # Untuk baseline/faster training
CLAHE_CLIP_LIMIT = 2.0     # CLAHE contrast enhancement
CLAHE_GRID_SIZE = (8, 8)

# Normalisasi (ImageNet stats)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ===========================================================================
# Data Augmentation
# ===========================================================================
AUG_PARAMS = {
    "horizontal_flip_p": 0.5,
    "vertical_flip_p": 0.5,
    "rotation_limit": 30,
    "brightness_limit": 0.2,
    "contrast_limit": 0.2,
    "saturation_limit": 0.2,
    "hue_limit": 0.05,
    "shift_limit": 0.1,
    "scale_limit": 0.1,
    "gaussian_blur_limit": (3, 5),
    "elastic_alpha": 50,
    "elastic_sigma": 10,
    "cutout_num_holes": 4,
    "cutout_max_h_size": 40,
    "cutout_max_w_size": 40,
    "mixup_alpha": 0.4,
}

# ===========================================================================
# Model Configuration
# ===========================================================================
MODEL_NAME = "efficientnet_b0"   # Backbone: efficientnet_b0, resnet50, convnext_small
NUM_CLASSES = 1                   # Binary classification (sigmoid output)
PRETRAINED = True
DROP_RATE = 0.3
DROP_PATH_RATE = 0.2

# Quality-Aware Attention Module (QAAM)
QAAM_REDUCTION = 16               # Channel attention reduction ratio
QAAM_QS_EMBED_DIM = 64            # Quality score embedding dimension

# ===========================================================================
# Training Hyperparameters
# ===========================================================================
BATCH_SIZE = 16
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15

# Optimizer
OPTIMIZER = "adamw"
LR_BACKBONE = 1e-4               # Learning rate untuk backbone (pretrained)
LR_HEAD = 1e-3                   # Learning rate untuk classification heads
WEIGHT_DECAY = 1e-4

# Scheduler
SCHEDULER = "cosine_warm_restarts"
COSINE_T0 = 10
COSINE_T_MULT = 2
WARMUP_EPOCHS = 3

# Loss
FOCAL_GAMMA = 2.0                 # Focal Loss gamma
FOCAL_ALPHA = 0.73                # Weight untuk kelas positif (GON+)
LABEL_SMOOTHING = 0.1
QUALITY_LOSS_WEIGHT = 0.3         # Beta: bobot untuk quality regression loss
USE_QUALITY_WEIGHTING = True      # Gunakan quality score sebagai sample weight

# Mixed Precision
USE_AMP = True

# ===========================================================================
# Evaluation
# ===========================================================================
CONFIDENCE_THRESHOLD = 0.5        # Default threshold
SENSITIVITY_TARGET = 0.95         # Target minimum sensitivity
TTA_TRANSFORMS = 5                # Number of TTA augmentations

# Grad-CAM
GRADCAM_TARGET_LAYER = "backbone.bn2"         # Layer untuk Grad-CAM (BatchNorm setelah conv_head, 1280ch, post-activation)
GRADCAM_NUM_SAMPLES = 20          # Jumlah sampel untuk visualisasi

# Phase 6: Clinical Interpretability
RISK_THRESHOLD_HIGH = 0.70        # P(GON+) >= 0.70 → High Risk
RISK_THRESHOLD_MEDIUM = 0.40      # 0.40 <= P(GON+) < 0.70 → Medium Risk
CONFIDENCE_REVIEW_THRESHOLD = 0.85  # confidence < 0.85 → manual review
QUALITY_GATE_THRESHOLD = 4.0      # QS < 4.0 → insufficient quality
SHAP_BACKGROUND_SAMPLES = 50      # Background samples untuk SHAP
SHAP_EXPLAIN_SAMPLES = 10         # Samples to explain via SHAP

# ===========================================================================
# Device
# ===========================================================================
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = min(4, os.cpu_count() or 1)
PIN_MEMORY = True if torch.cuda.is_available() else False

print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Dataset: {IMAGES_DIR}")
print(f"[CONFIG] Output: {OUTPUT_DIR}")
