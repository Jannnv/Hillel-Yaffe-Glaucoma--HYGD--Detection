"""
=============================================================================
evaluate.py — Phase 5: Evaluation, Metrics & Explainability (Grad-CAM++)
=============================================================================
Features:
  - Comprehensive metrics: AUC-ROC, AUC-PRC, Sensitivity, Specificity, F1
  - Optimal threshold via Youden's Index
  - Confidence calibration (Temperature Scaling)
  - Stratified metrics by quality score
  - Grad-CAM++ for explainability
  - Test-Time Augmentation (TTA)
  - Clinical report generation
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, cohen_kappa_score,
    classification_report, accuracy_score
)

import config
from model import GONDetectionModel, create_model

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ===========================================================================
# Core Metrics
# ===========================================================================

def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float = 0.5) -> dict:
    """
    Hitung comprehensive metrics untuk binary classification.

    Args:
        y_true: Ground truth labels (0 or 1)
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        dict dengan semua metrics
    """
    # Pastikan input valid
    y_true = np.array(y_true).flatten()
    y_prob = np.array(y_prob).flatten()

    # Binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # AUC-ROC
    try:
        metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc_roc"] = 0.0

    # AUC-PRC (Average Precision)
    try:
        metrics["auc_prc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        metrics["auc_prc"] = 0.0

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Sensitivity (Recall / True Positive Rate)
    metrics["sensitivity"] = tp / max(tp + fn, 1)

    # Specificity (True Negative Rate)
    metrics["specificity"] = tn / max(tn + fp, 1)

    # Precision (Positive Predictive Value)
    metrics["precision"] = tp / max(tp + fp, 1)

    # F1 Score
    metrics["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Cohen's Kappa
    metrics["cohens_kappa"] = cohen_kappa_score(y_true, y_pred)

    # Youden's Index (J = Sensitivity + Specificity - 1)
    metrics["youdens_index"] = metrics["sensitivity"] + metrics["specificity"] - 1

    # Negative Predictive Value
    metrics["npv"] = tn / max(tn + fn, 1)

    # Counts
    metrics["tp"] = int(tp)
    metrics["fp"] = int(fp)
    metrics["tn"] = int(tn)
    metrics["fn"] = int(fn)
    metrics["threshold"] = threshold

    return metrics


def find_optimal_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           method: str = "youden") -> float:
    """
    Temukan threshold optimal menggunakan Youden's Index atau
    sensitivity target.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        method: 'youden' atau 'sensitivity_target'

    Returns:
        Optimal threshold
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    if method == "youden":
        # Youden's Index: maximize (sensitivity + specificity - 1)
        youdens = tpr - fpr
        optimal_idx = np.argmax(youdens)
        optimal_threshold = thresholds[optimal_idx]

    elif method == "sensitivity_target":
        # Temukan threshold yang memberikan sensitivity >= target
        target = config.SENSITIVITY_TARGET
        valid_idx = np.where(tpr >= target)[0]
        if len(valid_idx) > 0:
            # Pilih threshold yang memaksimalkan specificity
            # sambil mempertahankan sensitivity >= target
            best_idx = valid_idx[np.argmax(1 - fpr[valid_idx])]
            optimal_threshold = thresholds[best_idx]
        else:
            optimal_threshold = 0.5

    return float(optimal_threshold)


# ===========================================================================
# Stratified Evaluation by Quality Score
# ===========================================================================

def evaluate_by_quality_strata(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                quality_scores: np.ndarray) -> dict:
    """
    Evaluasi model performance per strata kualitas gambar.

    Strata:
      - Low Quality:  QS < 4.0
      - Medium Quality: 4.0 <= QS < 6.0
      - High Quality: QS >= 6.0
    """
    strata = {
        "low_quality (QS < 4.0)": quality_scores < 4.0,
        "medium_quality (4.0 <= QS < 6.0)": (quality_scores >= 4.0) & (quality_scores < 6.0),
        "high_quality (QS >= 6.0)": quality_scores >= 6.0,
    }

    results = {}
    for stratum_name, mask in strata.items():
        if mask.sum() > 0:
            metrics = compute_metrics(y_true[mask], y_prob[mask])
            metrics["n_samples"] = int(mask.sum())
            results[stratum_name] = metrics
            print(f"  {stratum_name}: n={mask.sum()}, "
                  f"AUC={metrics['auc_roc']:.4f}, "
                  f"Sens={metrics['sensitivity']:.4f}, "
                  f"Spec={metrics['specificity']:.4f}")
        else:
            results[stratum_name] = {"n_samples": 0}

    return results


# ===========================================================================
# Temperature Scaling (Confidence Calibration)
# ===========================================================================

class TemperatureScaling(nn.Module):
    """
    Temperature Scaling untuk kalibrasi confidence.

    P_calibrated(y|x) = sigmoid(z / T)

    T dioptimasi pada validation set menggunakan NLL loss.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


def calibrate_model(model: GONDetectionModel,
                    val_loader,
                    max_iter: int = 100) -> TemperatureScaling:
    """
    Kalibrasi model menggunakan Temperature Scaling.

    Args:
        model: Trained model
        val_loader: Validation DataLoader
        max_iter: Maximum optimization iterations

    Returns:
        TemperatureScaling module
    """
    model.eval()
    temp_scaling = TemperatureScaling().to(config.DEVICE)

    # Collect all logits dan labels
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            qs = batch["quality_score"].to(config.DEVICE)

            outputs = model(images, qs)
            all_logits.append(outputs["gon_logits"])
            all_labels.append(labels)

    logits = torch.cat(all_logits).detach()
    labels = torch.cat(all_labels).detach()

    # Optimize temperature
    optimizer = torch.optim.LBFGS([temp_scaling.temperature], lr=0.01, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled_logits = temp_scaling(logits)
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    print(f"[CALIBRATION] Optimal temperature: {temp_scaling.temperature.item():.4f}")

    return temp_scaling


# ===========================================================================
# Test-Time Augmentation (TTA)
# ===========================================================================

@torch.no_grad()
def predict_with_tta(model: GONDetectionModel,
                     image: np.ndarray,
                     quality_score: float,
                     tta_transforms: list) -> float:
    """
    Prediksi dengan Test-Time Augmentation.

    Multiple augmented views diprediksi dan di-aggregate.

    Args:
        model: Trained model
        image: Preprocessed image (numpy, uint8, RGB)
        quality_score: Normalized quality score
        tta_transforms: List of albumentations transforms

    Returns:
        Aggregated probability
    """
    model.eval()
    predictions = []

    qs_tensor = torch.tensor([[quality_score]], dtype=torch.float32).to(config.DEVICE)

    for transform in tta_transforms:
        augmented = transform(image=image)
        img_tensor = augmented["image"].unsqueeze(0).to(config.DEVICE)

        output = model(img_tensor, qs_tensor)
        prob = output["gon_probs"].cpu().item()
        predictions.append(prob)

    # Quality-weighted average (higher quality -> higher weight)
    # Untuk TTA sederhana, gunakan mean
    avg_prob = np.mean(predictions)

    return avg_prob


# ===========================================================================
# Grad-CAM++ Explainability
# ===========================================================================

class GradCAMPlusPlus:
    """
    Grad-CAM++ untuk visualisasi attention model.

    Menunjukkan area pada fundus image yang paling berpengaruh
    terhadap prediksi model (idealnya: optic disc region).
    """

    def __init__(self, model: GONDetectionModel, target_layer: str = None):
        self.model = model
        self.model.eval()

        self.gradients = None
        self.activations = None

        # Register hooks pada target layer
        if target_layer:
            target = self._find_layer(model, target_layer)
        else:
            # Default: last convolutional layer
            target = self._find_last_conv(model)

        if target is not None:
            target.register_forward_hook(self._save_activation)
            target.register_full_backward_hook(self._save_gradient)

    def _find_layer(self, model, layer_name):
        """Find layer by name."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        print(f"[WARNING] Layer '{layer_name}' not found.")
        return None

    def _find_last_conv(self, model):
        """Find best target layer for Grad-CAM in the backbone."""
        backbone = getattr(model, 'backbone', model)
        
        # Prioritas 1: bn2 (BatchNorm setelah conv_head, post-activation, 1280ch)
        if hasattr(backbone, 'bn2'):
            return backbone.bn2
        
        # Prioritas 2: conv_head (1x1 conv, 320->1280 channels)
        if hasattr(backbone, 'conv_head'):
            return backbone.conv_head
        
        # Prioritas 3: blok MBConv terakhir
        if hasattr(backbone, 'blocks') and len(backbone.blocks) > 0:
            return backbone.blocks[-1]
        
        # Final fallback: last Conv2d
        last_conv = None
        for module in backbone.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, image: torch.Tensor,
                 quality_score: torch.Tensor = None,
                 original_image: np.ndarray = None) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap.

        Args:
            image: Input tensor [1, 3, H, W]
            quality_score: Quality score tensor [1, 1]
            original_image: Optional preprocessed image (numpy, uint8, RGB)
                            Used to create a data-driven fundus mask.

        Returns:
            Heatmap (numpy array, float32, [H, W], range [0, 1])
        """
        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        # Forward
        output = self.model(image, quality_score)
        logits = output["gon_logits"]

        # Backward (dari GON prediction)
        logits.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            print("[WARNING] No gradients captured. Returning empty heatmap.")
            return np.zeros((image.shape[2], image.shape[3]))

        # Grad-CAM++ weights
        grads = self.gradients  # [1, C, H, W]
        acts = self.activations  # [1, C, H, W]

        # ---- Grad-CAM++ weight computation (paper-correct) ----
        grads_power_2 = grads ** 2
        grads_power_3 = grads ** 3
        sum_acts = torch.sum(acts, dim=(2, 3), keepdim=True) + 1e-7

        alpha_numer = grads_power_2
        alpha_denom = 2.0 * grads_power_2 + sum_acts * grads_power_3 + 1e-7
        alpha = alpha_numer / alpha_denom  # [1, C, H, W]

        # Weights: sum over spatial of (alpha * relu(grads))
        weights = torch.sum(alpha * F.relu(grads), dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activations
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # [1, 1, H, W]

        # ReLU & normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()

        # Handle edge case: scalar output
        if cam.ndim == 0:
            cam = cam.reshape(1, 1)

        # Resize to input image dimensions (before masking/normalization)
        target_h, target_w = image.shape[2], image.shape[3]
        cam = cv2.resize(cam, (target_w, target_h),
                         interpolation=cv2.INTER_LINEAR)

        # ---- Apply fundus mask BEFORE normalization ----
        if original_image is not None:
            fundus_mask = self._create_fundus_mask(original_image, target_h, target_w)
            cam = cam * fundus_mask

        # ---- Normalize using only non-zero (masked) region ----
        valid_region = cam[cam > 0] if (cam > 0).any() else cam.ravel()
        cam_min = valid_region.min()
        cam_p99 = np.percentile(valid_region, 99)
        cam_max = valid_region.max()

        norm_max = cam_p99 if (cam_p99 - cam_min) > 1e-8 else cam_max
        if norm_max - cam_min > 1e-8:
            cam = np.where(cam > 0, (cam - cam_min) / (norm_max - cam_min), 0)
            cam = np.clip(cam, 0, 1)
        else:
            cam = np.zeros_like(cam)

        # Mild Gaussian smoothing
        cam = cv2.GaussianBlur(cam, (0, 0), sigmaX=5)

        # Re-apply mask after blur to ensure edges stay at zero
        if original_image is not None:
            cam = cam * fundus_mask

        # Final normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam

    @staticmethod
    def _create_fundus_mask(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """
        Create a soft mask from the preprocessed fundus image.
        Detects fundus region, erodes inward, and blurs for smooth falloff.
        """
        img_resized = cv2.resize(image, (target_w, target_h))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        erode_size = max(int(min(target_h, target_w) * 0.15), 10)
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size * 2 + 1, erode_size * 2 + 1)
        )
        eroded = cv2.erode(binary, kernel_erode)

        mask = eroded.astype(np.float32) / 255.0
        blur_sigma = max(erode_size * 1.5, 10)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)

        if mask.max() > 0:
            mask = mask / mask.max()

        return mask


def generate_gradcam_visualization(model: GONDetectionModel,
                                    image: np.ndarray,
                                    quality_score: float,
                                    image_name: str,
                                    save_dir: Path = config.GRADCAM_DIR) -> Path:
    """
    Generate dan simpan Grad-CAM++ visualization.

    Args:
        model: Trained model
        image: Preprocessed image (numpy, uint8, RGB, HxWx3)
        quality_score: Normalized quality score
        image_name: Nama file gambar
        save_dir: Direktori output

    Returns:
        Path ke saved visualization
    """
    grad_cam = GradCAMPlusPlus(model, target_layer=config.GRADCAM_TARGET_LAYER)

    # Prepare input
    from dataset import get_val_transforms
    transform = get_val_transforms(config.IMG_SIZE)
    augmented = transform(image=image)
    img_tensor = augmented["image"].unsqueeze(0).to(config.DEVICE)
    qs_tensor = torch.tensor([[quality_score]], dtype=torch.float32).to(config.DEVICE)

    # Forward + generate heatmap
    img_tensor.requires_grad_(True)
    heatmap = grad_cam.generate(img_tensor, qs_tensor, original_image=image)

    # Visualize
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original
        display_img = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
        axes[0].imshow(display_img)
        axes[0].set_title("Original Fundus Image")
        axes[0].axis("off")

        # Heatmap
        axes[1].imshow(heatmap, cmap='jet', alpha=0.8)
        axes[1].set_title("Grad-CAM++ Heatmap")
        axes[1].axis("off")

        # Overlay
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(display_img, 0.6, heatmap_colored, 0.4, 0)
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay (Attention Areas)")
        axes[2].axis("off")

        safe_name = image_name.replace('.jpg', '').replace('.png', '')
        save_path = save_dir / f"gradcam_{safe_name}.png"
        plt.suptitle(f"Grad-CAM++ Analysis: {image_name}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        return save_path

    return None


# ===========================================================================
# Plotting Functions
# ===========================================================================

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   save_path: Path = None, fold: int = None) -> None:
    """Plot ROC curve dengan AUC."""
    if not HAS_MATPLOTLIB:
        return

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2,
             label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')

    # Mark optimal threshold (Youden)
    optimal_idx = np.argmax(tpr - fpr)
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], c='red', s=100, zorder=5,
                label=f'Optimal (Thresh={thresholds[optimal_idx]:.3f})')

    title = "ROC Curve"
    if fold is not None:
        title += f" - Fold {fold}"

    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray,
                                 save_path: Path = None) -> None:
    """Plot Precision-Recall curve dengan AUPRC."""
    if not HAS_MATPLOTLIB:
        return

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, 'b-', linewidth=2,
             label=f'PR Curve (AUPRC = {auprc:.4f})')

    # Baseline (prevalence)
    prevalence = y_true.mean()
    plt.axhline(y=prevalence, color='r', linestyle='--',
                label=f'Baseline (prevalence={prevalence:.3f})')

    plt.xlabel("Recall (Sensitivity)", fontsize=12)
    plt.ylabel("Precision (PPV)", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_history(history_path: Path,
                          save_path: Path = None) -> None:
    """Plot training history (loss, AUC, sensitivity)."""
    if not HAS_MATPLOTLIB:
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = history["epoch"]

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history["val_loss"], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training & Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # AUC
    axes[0, 1].plot(epochs, history["val_auc"], 'g-', linewidth=2)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("AUC-ROC")
    axes[0, 1].set_title("Validation AUC-ROC")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='Target (0.90)')
    axes[0, 1].legend()

    # Sensitivity & Specificity
    axes[1, 0].plot(epochs, history["val_sensitivity"], 'b-', label='Sensitivity')
    axes[1, 0].plot(epochs, history["val_specificity"], 'r-', label='Specificity')
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Sensitivity & Specificity")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0.95, color='b', linestyle='--', alpha=0.3, label='Sens Target')

    # Learning Rate
    axes[1, 1].plot(epochs, history["learning_rate"], 'purple', linewidth=2)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_visual(y_true: np.ndarray, y_pred: np.ndarray,
                                  save_path: Path = None) -> None:
    """Plot confusion matrix heatmap."""
    if not HAS_MATPLOTLIB:
        return

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    classes = ['GON- (Normal)', 'GON+ (Glaucoma)']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')

    # Text annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, f'{cm[i, j]}',
                    ha='center', va='center', color=color, fontsize=16)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ===========================================================================
# Clinical Report Generation
# ===========================================================================

def generate_clinical_report(y_true: np.ndarray,
                              y_prob: np.ndarray,
                              quality_scores: np.ndarray = None,
                              fold: int = None,
                              save_path: Path = None) -> str:
    """
    Generate clinical evaluation report.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        quality_scores: Quality scores (opsional)
        fold: Fold number (opsional)
        save_path: Path untuk menyimpan report

    Returns:
        Report text
    """
    # Find optimal threshold
    optimal_thresh = find_optimal_threshold(y_true, y_prob, method="youden")
    sensitivity_thresh = find_optimal_threshold(y_true, y_prob, method="sensitivity_target")

    # Metrics dengan berbagai thresholds
    metrics_default = compute_metrics(y_true, y_prob, threshold=0.5)
    metrics_optimal = compute_metrics(y_true, y_prob, threshold=optimal_thresh)
    metrics_sens = compute_metrics(y_true, y_prob, threshold=sensitivity_thresh)

    report_lines = [
        "=" * 60,
        "  CLINICAL EVALUATION REPORT",
        f"  GON Detection - Quality-Aware Multi-Task Model",
    ]

    if fold:
        report_lines.append(f"  Fold: {fold}")

    report_lines.extend([
        "=" * 60,
        "",
        f"  Total Samples: {len(y_true)}",
        f"  GON+ (Positive): {int(y_true.sum())} ({y_true.mean()*100:.1f}%)",
        f"  GON- (Negative): {int((1-y_true).sum())} ({(1-y_true).mean()*100:.1f}%)",
        "",
        "-" * 60,
        "  PERFORMANCE METRICS",
        "-" * 60,
        "",
        "  >> Threshold = 0.5 (Default)",
        f"     AUC-ROC:      {metrics_default['auc_roc']:.4f}",
        f"     AUC-PRC:      {metrics_default['auc_prc']:.4f}",
        f"     Sensitivity:  {metrics_default['sensitivity']:.4f}",
        f"     Specificity:  {metrics_default['specificity']:.4f}",
        f"     F1-Score:     {metrics_default['f1_score']:.4f}",
        f"     Cohen's k:    {metrics_default['cohens_kappa']:.4f}",
        f"     Accuracy:     {metrics_default['accuracy']:.4f}",
        "",
        f"  >> Threshold = {optimal_thresh:.4f} (Youden's Optimal)",
        f"     Sensitivity:  {metrics_optimal['sensitivity']:.4f}",
        f"     Specificity:  {metrics_optimal['specificity']:.4f}",
        f"     F1-Score:     {metrics_optimal['f1_score']:.4f}",
        f"     Youden's J:   {metrics_optimal['youdens_index']:.4f}",
        "",
        f"  >> Threshold = {sensitivity_thresh:.4f} (Sensitivity >= {config.SENSITIVITY_TARGET})",
        f"     Sensitivity:  {metrics_sens['sensitivity']:.4f}",
        f"     Specificity:  {metrics_sens['specificity']:.4f}",
        f"     F1-Score:     {metrics_sens['f1_score']:.4f}",
        "",
        "-" * 60,
        "  CONFUSION MATRIX (Optimal Threshold)",
        "-" * 60,
        f"                 Predicted GON-  Predicted GON+",
        f"  Actual GON-:   {metrics_optimal['tn']:>10}     {metrics_optimal['fp']:>10}",
        f"  Actual GON+:   {metrics_optimal['fn']:>10}     {metrics_optimal['tp']:>10}",
        "",
    ])

    # Quality-stratified metrics
    if quality_scores is not None:
        report_lines.extend([
            "-" * 60,
            "  QUALITY-STRATIFIED PERFORMANCE",
            "-" * 60,
            ""
        ])
        strata_results = evaluate_by_quality_strata(y_true, y_prob, quality_scores)
        for stratum_name, metrics in strata_results.items():
            if metrics.get("n_samples", 0) > 0:
                report_lines.extend([
                    f"  {stratum_name} (n={metrics['n_samples']}):",
                    f"    AUC={metrics['auc_roc']:.4f}, "
                    f"Sens={metrics['sensitivity']:.4f}, "
                    f"Spec={metrics['specificity']:.4f}",
                    ""
                ])

    report_lines.extend([
        "-" * 60,
        "  CLINICAL SAFETY NOTES",
        "-" * 60,
        "",
        "  [!] Model ini adalah alat bantu screening, BUKAN pengganti",
        "    pemeriksaan oftalmologi komprehensif.",
        "  [!] Prediksi dengan confidence < 0.85 harus di-review manual.",
        "  [!] Gambar dengan Quality Score < 4.0 harus difoto ulang.",
        "",
        "=" * 60,
    ])

    report_text = "\n".join(report_lines)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"[REPORT] Saved to: {save_path}")

    return report_text


# ===========================================================================
# Full Evaluation Pipeline
# ===========================================================================

def evaluate_fold(model: GONDetectionModel,
                  val_loader,
                  fold: int,
                  save_plots: bool = True) -> dict:
    """
    Evaluasi lengkap untuk satu fold.

    Args:
        model: Trained model
        val_loader: Validation DataLoader
        fold: Fold number
        save_plots: Apakah simpan plot

    Returns:
        dict dengan semua results
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_quality_scores = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(config.DEVICE)
            labels = batch["label"].to(config.DEVICE)
            qs = batch["quality_score"].to(config.DEVICE)

            outputs = model(images, qs)
            all_preds.extend(outputs["gon_probs"].cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_quality_scores.extend(qs.cpu().numpy().flatten() * 10)  # Denormalize

    y_true = np.array(all_labels)
    y_prob = np.array(all_preds)
    quality_scores = np.array(all_quality_scores)

    # Generate report
    report_path = config.RESULTS_DIR / f"clinical_report_fold{fold}.txt"
    report = generate_clinical_report(
        y_true, y_prob, quality_scores, fold, report_path
    )
    print(report)

    # Generate plots
    if save_plots:
        plot_roc_curve(y_true, y_prob,
                       save_path=config.RESULTS_DIR / f"roc_curve_fold{fold}.png",
                       fold=fold)

        plot_precision_recall_curve(y_true, y_prob,
                                    save_path=config.RESULTS_DIR / f"pr_curve_fold{fold}.png")

        optimal_thresh = find_optimal_threshold(y_true, y_prob)
        y_pred = (y_prob >= optimal_thresh).astype(int)
        plot_confusion_matrix_visual(y_true, y_pred,
                                      save_path=config.RESULTS_DIR / f"confusion_matrix_fold{fold}.png")

        # Training history plot
        history_path = config.RESULTS_DIR / f"history_fold{fold}.json"
        if history_path.exists():
            plot_training_history(history_path,
                                  save_path=config.RESULTS_DIR / f"training_history_fold{fold}.png")

    return {
        "y_true": y_true,
        "y_prob": y_prob,
        "quality_scores": quality_scores,
        "metrics": compute_metrics(y_true, y_prob, find_optimal_threshold(y_true, y_prob)),
    }


if __name__ == "__main__":
    """Test metrics computation."""
    np.random.seed(42)

    # Simulate predictions
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_prob = np.clip(y_true + np.random.normal(0, 0.3, n), 0, 1)
    qs = np.random.uniform(3, 8, n)

    print("=== Test Metrics ===")
    metrics = compute_metrics(y_true, y_prob)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n=== Optimal Threshold ===")
    thresh = find_optimal_threshold(y_true, y_prob)
    print(f"  Youden: {thresh:.4f}")

    print(f"\n=== Quality-Stratified ===")
    evaluate_by_quality_strata(y_true, y_prob, qs)

    print(f"\n=== Clinical Report ===")
    report = generate_clinical_report(y_true, y_prob, qs, fold=1,
                                       save_path=config.RESULTS_DIR / "test_report.txt")
