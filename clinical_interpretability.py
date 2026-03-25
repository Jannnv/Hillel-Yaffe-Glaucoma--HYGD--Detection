"""
=============================================================================
clinical_interpretability.py — Phase 6: Clinical Interpretability & Explainability
=============================================================================
Features:
  1. Grad-CAM++ Batch Visualization
     - Heatmap pada optic disc region untuk verifikasi klinis
     - Overlay attention map pada gambar asli
  2. SHAP Analysis (Deep Explainer)
     - Feature importance analysis per region fundus
     - Global & local explanations
  3. Clinical Report Generation
     - Confidence score & risk categorization (Low/Medium/High)
     - Quality gate: gambar QS < 4.0 → flag insufficient quality
     - Attention map overlay untuk referensi klinisi
  4. Summary Dashboard
     - Distribusi risk categories
     - Confidence calibration overview
     - Per-sample clinical decision support
"""

import json
import os
import warnings
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

import config
from model import GONDetectionModel

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import LinearSegmentedColormap
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


# ===========================================================================
# Configuration Phase 6
# ===========================================================================

PHASE6_DIR = config.OUTPUT_DIR / "phase6_interpretability"
PHASE6_GRADCAM_DIR = PHASE6_DIR / "gradcam_batch"
PHASE6_SHAP_DIR = PHASE6_DIR / "shap_analysis"
PHASE6_REPORTS_DIR = PHASE6_DIR / "clinical_reports"
PHASE6_SUMMARY_DIR = PHASE6_DIR / "summary"

for d in [PHASE6_DIR, PHASE6_GRADCAM_DIR, PHASE6_SHAP_DIR,
          PHASE6_REPORTS_DIR, PHASE6_SUMMARY_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Risk thresholds
RISK_THRESHOLDS = {
    "high": 0.70,     # P(GON+) >= 0.70 → High Risk
    "medium": 0.40,   # 0.40 <= P(GON+) < 0.70 → Medium Risk
    "low": 0.0,       # P(GON+) < 0.40 → Low Risk
}

# Confidence threshold for manual review flag
CONFIDENCE_REVIEW_THRESHOLD = 0.85

# Quality gate threshold
QUALITY_GATE_THRESHOLD = 4.0


# ===========================================================================
# 1. Grad-CAM++ Batch Visualization
# ===========================================================================

class GradCAMPlusPlusInterp:
    """
    Grad-CAM++ untuk Phase 6: Clinical Interpretability.

    Mengenerate heatmap yang menunjukkan area fundus yang paling
    berpengaruh terhadap prediksi GON (idealnya: optic disc region,
    optic cup, neuroretinal rim, peripapillary RNFL).
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
            target = self._find_last_conv(model)

        if target is not None:
            target.register_forward_hook(self._save_activation)
            target.register_full_backward_hook(self._save_gradient)
        else:
            print("[WARNING] Tidak dapat menemukan target layer untuk Grad-CAM++.")

    def _find_layer(self, model, layer_name):
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        print(f"[WARNING] Layer '{layer_name}' tidak ditemukan.")
        return None

    def _find_last_conv(self, model):
        """Find best target layer for Grad-CAM in the backbone."""
        backbone = getattr(model, 'backbone', model)
        
        # Prioritas 1: bn2 (BatchNorm setelah conv_head, post-activation, 1280ch)
        # Ini adalah representasi paling kaya sebelum pooling
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
            if isinstance(module, torch.nn.Conv2d):
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
                            Used to create a data-driven fundus mask to suppress
                            edge artifacts from circular mask preprocessing.

        Returns:
            Heatmap (numpy array, float32, [H, W], range [0, 1])
        """
        self.model.zero_grad()
        self.gradients = None
        self.activations = None

        # Forward
        output = self.model(image, quality_score)
        logits = output["gon_logits"]

        # Backward
        logits.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            print("[WARNING] No gradients captured.")
            return np.zeros((image.shape[2], image.shape[3]))

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

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().detach().numpy()

        # Handle edge case: scalar output (single spatial cell)
        if cam.ndim == 0:
            cam = cam.reshape(1, 1)

        # Resize to input image dimensions (before masking/normalization)
        target_h, target_w = image.shape[2], image.shape[3]
        cam = cv2.resize(cam, (target_w, target_h),
                         interpolation=cv2.INTER_LINEAR)

        # ---- Apply fundus mask BEFORE normalization ----
        # This ensures edge activations are zeroed BEFORE they can affect
        # the normalization range. Essential for suppressing boundary artifacts.
        if original_image is not None:
            fundus_mask = self._create_fundus_mask(original_image, target_h, target_w)
            cam = cam * fundus_mask

        # ---- Normalize using only non-zero (masked) region ----
        # This prevents the zeroed-out edge region from skewing normalization
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

        # Mild Gaussian smoothing (sigma=5 for natural look without losing focus)
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

        Detects the actual fundus region (non-black pixels), erodes it inward
        to move away from the boundary, and applies Gaussian blur for smooth falloff.

        Args:
            image: Preprocessed fundus image (numpy, uint8, RGB)
            target_h, target_w: Target dimensions for the mask

        Returns:
            Soft mask (float32, [target_h, target_w], range [0, 1])
        """
        # Convert to grayscale and threshold to find fundus area
        img_resized = cv2.resize(image, (target_w, target_h))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)

        # Morphological close to fill small holes
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

        # Erode aggressively to move away from the fundus edge
        # Use 15% of image size as erosion radius
        erode_size = max(int(min(target_h, target_w) * 0.15), 10)
        kernel_erode = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_size * 2 + 1, erode_size * 2 + 1)
        )
        eroded = cv2.erode(binary, kernel_erode)

        # Convert to float and apply heavy Gaussian blur for smooth falloff
        mask = eroded.astype(np.float32) / 255.0
        blur_sigma = max(erode_size * 1.5, 10)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=blur_sigma)

        # Normalize to [0, 1]
        if mask.max() > 0:
            mask = mask / mask.max()

        return mask


def generate_gradcam_batch(model: GONDetectionModel,
                           val_loader,
                           val_df,
                           num_samples: int = 20,
                           save_dir: Path = PHASE6_GRADCAM_DIR) -> list:
    """
    Generate Grad-CAM++ visualization untuk batch sampel dari validation set.

    Memverifikasi bahwa model fokus pada area klinis yang relevan:
    optic cup, neuroretinal rim, peripapillary RNFL.

    Args:
        model: Trained model
        val_loader: Validation DataLoader
        val_df: Validation DataFrame
        num_samples: Jumlah sampel yang divisualisasi
        save_dir: Direktori output

    Returns:
        List of dicts dengan info setiap sampel
    """
    from dataset import get_val_transforms

    grad_cam = GradCAMPlusPlusInterp(model, target_layer=config.GRADCAM_TARGET_LAYER)
    transform = get_val_transforms(config.IMG_SIZE)
    results = []

    sample_df = val_df.head(min(num_samples, len(val_df)))
    print(f"\n[Phase 6] Generating Grad-CAM++ for {len(sample_df)} samples...")

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_name = row["Image Name"]
        img_path = config.IMAGES_DIR / img_name
        qs_raw = row["Quality Score"]
        qs_norm = qs_raw / 10.0
        label = row["Label"]

        if not img_path.exists():
            print(f"  [SKIP] {img_name}: file not found")
            continue

        try:
            from preprocessing import preprocess_single_image
            img = preprocess_single_image(str(img_path), config.IMG_SIZE)

            # Prepare tensors
            augmented = transform(image=img)
            img_tensor = augmented["image"].unsqueeze(0).to(config.DEVICE)
            qs_tensor = torch.tensor([[qs_norm]], dtype=torch.float32).to(config.DEVICE)

            img_tensor.requires_grad_(True)

            # Generate heatmap (pass original_image for edge masking)
            heatmap = grad_cam.generate(img_tensor, qs_tensor, original_image=img)

            # Forward pass untuk prediksi
            model.eval()
            with torch.no_grad():
                output = model(
                    augmented["image"].unsqueeze(0).to(config.DEVICE),
                    qs_tensor
                )
                prob = output["gon_probs"].cpu().item()
                qs_pred = output["quality_pred"].cpu().item() * 10

            # Risk categorization
            risk = categorize_risk(prob)

            # Visualize
            if HAS_MATPLOTLIB:
                save_path = _visualize_gradcam_clinical(
                    img, heatmap, img_name, prob, qs_raw, qs_pred,
                    label, risk, save_dir
                )
            else:
                save_path = None

            results.append({
                "image_name": img_name,
                "label": int(label) if hasattr(label, 'item') else label,
                "quality_score": float(qs_raw),
                "prediction": float(prob),
                "risk_category": risk,
                "gradcam_path": str(save_path) if save_path else None,
            })

            print(f"  [{idx+1}/{len(sample_df)}] {img_name}: "
                  f"P(GON+)={prob:.3f} | Risk={risk} | QS={qs_raw:.2f}")

        except Exception as e:
            print(f"  [ERROR] {img_name}: {e}")

    # Save results JSON
    results_path = save_dir / "gradcam_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 6] Grad-CAM++ batch visualization selesai: {len(results)} sampel")
    return results


def _visualize_gradcam_clinical(image: np.ndarray,
                                 heatmap: np.ndarray,
                                 image_name: str,
                                 prob: float,
                                 qs_raw: float,
                                 qs_pred: float,
                                 label,
                                 risk: str,
                                 save_dir: Path) -> Path:
    """
    Visualisasi Grad-CAM++ dengan informasi klinis lengkap.

    Layout:
    [Original] [Heatmap] [Overlay] + Info panel
    """
    fig = plt.figure(figsize=(20, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.6])

    display_img = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))

    # 1. Original Image
    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(display_img)
    ground_truth = "GON+" if (label == 1 or label == "GON+") else "GON-"
    ax1.set_title(f"Original ({ground_truth})", fontsize=12, fontweight='bold')
    ax1.axis("off")

    # 2. Heatmap
    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(heatmap, cmap='jet', alpha=0.9)
    ax2.set_title("Grad-CAM++ Heatmap", fontsize=12, fontweight='bold')
    ax2.axis("off")

    # 3. Overlay (Attention Map pada gambar asli)
    ax3 = fig.add_subplot(gs[2])
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_img, 0.55, heatmap_colored, 0.45, 0)
    ax3.imshow(overlay)
    ax3.set_title("Attention Overlay", fontsize=12, fontweight='bold')
    ax3.axis("off")

    # 4. Clinical Info Panel
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")

    # Risk color
    risk_colors = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
    risk_color = risk_colors.get(risk, "#95a5a6")

    info_text = (
        f"━━━ Clinical Info ━━━\n\n"
        f"Image: {image_name}\n"
        f"Ground Truth: {ground_truth}\n\n"
        f"━━━ Prediction ━━━\n\n"
        f"P(GON+): {prob:.4f}\n"
        f"Risk: {risk}\n\n"
        f"━━━ Quality ━━━\n\n"
        f"QS (actual): {qs_raw:.2f}\n"
        f"QS (predicted): {qs_pred:.2f}\n"
    )

    # Quality gate flag
    if qs_raw < QUALITY_GATE_THRESHOLD:
        info_text += "\n⚠️ LOW QUALITY\nRetake recommended"

    # Confidence flag
    if 0.3 < prob < 0.7:
        info_text += "\n\n⚠️ UNCERTAIN\nManual review needed"

    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       edgecolor='gray', alpha=0.9))

    # Risk badge
    ax4.text(0.5, 0.02, f"RISK: {risk.upper()}",
             transform=ax4.transAxes, fontsize=14, fontweight='bold',
             ha='center', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=risk_color,
                       edgecolor='darkgray', alpha=0.9),
             color='white')

    plt.suptitle(f"Phase 6: Grad-CAM++ Clinical Interpretability — {image_name}",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    safe_name = image_name.replace('.jpg', '').replace('.png', '')
    save_path = save_dir / f"gradcam_clinical_{safe_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


# ===========================================================================
# 2. SHAP Analysis
# ===========================================================================

def run_shap_analysis(model: GONDetectionModel,
                      val_loader,
                      val_df,
                      num_background: int = 50,
                      num_explain: int = 10,
                      save_dir: Path = PHASE6_SHAP_DIR) -> dict:
    """
    SHAP (SHapley Additive exPlanations) Analysis.

    Menghitung feature importance untuk setiap region fundus
    terhadap prediksi GON.

    Args:
        model: Trained model
        val_loader: Validation DataLoader
        val_df: Validation DataFrame
        num_background: Jumlah background samples untuk SHAP
        num_explain: Jumlah gambar yang di-explain
        save_dir: Direktori output

    Returns:
        dict dengan SHAP results
    """
    if not HAS_SHAP:
        print("[Phase 6] SHAP library not installed. Menggunakan Gradient-based "
              "feature attribution sebagai alternatif.")
        return _gradient_based_attribution(model, val_loader, val_df,
                                            num_explain, save_dir)

    print(f"\n[Phase 6] Running SHAP Deep Explainer Analysis...")
    model.eval()

    # Collect background data
    background_images = []
    background_qs = []
    count = 0

    for batch in val_loader:
        images = batch["image"].to(config.DEVICE)
        qs = batch["quality_score"].to(config.DEVICE)

        for i in range(images.shape[0]):
            if count >= num_background:
                break
            background_images.append(images[i:i+1])
            background_qs.append(qs[i:i+1])
            count += 1
        if count >= num_background:
            break

    background_tensor = torch.cat(background_images, dim=0)
    background_qs_tensor = torch.cat(background_qs, dim=0)

    # Wrapper function untuk SHAP
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, qs):
            super().__init__()
            self.model = model
            self.qs = qs

        def forward(self, x):
            # Expand quality score jika batch size berbeda
            bs = x.shape[0]
            qs = self.qs[:bs] if bs <= self.qs.shape[0] else \
                 self.qs[0:1].expand(bs, -1)
            output = self.model(x, qs)
            return output["gon_logits"]

    wrapped_model = ModelWrapper(model, background_qs_tensor)

    try:
        # Deep Explainer
        explainer = shap.DeepExplainer(wrapped_model, background_tensor)

        # Collect explain data
        explain_images = []
        explain_names = []
        explain_labels = []
        explain_qs = []
        count = 0

        sample_df = val_df.head(min(num_explain, len(val_df)))
        from dataset import get_val_transforms
        from preprocessing import preprocess_single_image
        transform = get_val_transforms(config.IMG_SIZE)

        for _, row in sample_df.iterrows():
            img_path = config.IMAGES_DIR / row["Image Name"]
            if img_path.exists():
                img = preprocess_single_image(str(img_path), config.IMG_SIZE)
                augmented = transform(image=img)
                img_tensor = augmented["image"].unsqueeze(0).to(config.DEVICE)
                explain_images.append(img_tensor)
                explain_names.append(row["Image Name"])
                explain_labels.append(row["Label"])
                explain_qs.append(row["Quality Score"])
                count += 1
            if count >= num_explain:
                break

        explain_tensor = torch.cat(explain_images, dim=0)

        # Compute SHAP values
        shap_values = explainer.shap_values(explain_tensor)

        # Visualize SHAP
        if HAS_MATPLOTLIB and shap_values is not None:
            _visualize_shap_results(
                shap_values, explain_tensor, explain_names,
                explain_labels, explain_qs, save_dir
            )

        results = {
            "method": "SHAP_DeepExplainer",
            "num_background": num_background,
            "num_explained": len(explain_names),
            "samples": explain_names,
        }

        # Save results
        results_path = save_dir / "shap_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        print(f"[Phase 6] SHAP analysis selesai: {len(explain_names)} sampel")
        return results

    except Exception as e:
        print(f"[Phase 6] SHAP Deep Explainer error: {e}")
        print("[Phase 6] Fallback ke gradient-based attribution...")
        return _gradient_based_attribution(model, val_loader, val_df,
                                            num_explain, save_dir)


def _gradient_based_attribution(model: GONDetectionModel,
                                 val_loader,
                                 val_df,
                                 num_samples: int = 10,
                                 save_dir: Path = PHASE6_SHAP_DIR) -> dict:
    """
    Gradient-based feature attribution sebagai fallback untuk SHAP.

    Menghitung |∂y/∂x| × x (Gradient × Input) untuk setiap pixel.
    """
    print(f"\n[Phase 6] Running Gradient-based Feature Attribution...")
    model.eval()

    from dataset import get_val_transforms
    from preprocessing import preprocess_single_image
    transform = get_val_transforms(config.IMG_SIZE)

    results = []
    sample_df = val_df.head(min(num_samples, len(val_df)))

    for idx, (_, row) in enumerate(sample_df.iterrows()):
        img_name = row["Image Name"]
        img_path = config.IMAGES_DIR / img_name
        qs_raw = row["Quality Score"]
        qs_norm = qs_raw / 10.0
        label = row["Label"]

        if not img_path.exists():
            continue

        try:
            img = preprocess_single_image(str(img_path), config.IMG_SIZE)
            augmented = transform(image=img)
            img_tensor = augmented["image"].unsqueeze(0).to(config.DEVICE)
            qs_tensor = torch.tensor([[qs_norm]], dtype=torch.float32).to(config.DEVICE)

            img_tensor.requires_grad_(True)

            # Forward
            output = model(img_tensor, qs_tensor)
            logits = output["gon_logits"]
            prob = output["gon_probs"].cpu().item()

            # Backward
            model.zero_grad()
            logits.backward()

            # Gradient × Input attribution
            gradients = img_tensor.grad.data.abs()
            attribution = (gradients * img_tensor.data.abs()).squeeze()

            # Aggregate across channels
            attribution_map = attribution.mean(dim=0).cpu().numpy()

            # Normalize
            if attribution_map.max() > 0:
                attribution_map = attribution_map / attribution_map.max()

            # Visualize
            if HAS_MATPLOTLIB:
                save_path = _visualize_attribution(
                    img, attribution_map, img_name, prob, qs_raw, label, save_dir
                )
            else:
                save_path = None

            results.append({
                "image_name": img_name,
                "label": int(label) if hasattr(label, 'item') else label,
                "prediction": float(prob),
                "method": "Gradient_x_Input",
                "save_path": str(save_path) if save_path else None,
            })

            print(f"  [{idx+1}/{len(sample_df)}] {img_name}: Attribution map generated")

        except Exception as e:
            print(f"  [ERROR] {img_name}: {e}")

    # Save results
    results_path = save_dir / "attribution_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"[Phase 6] Feature attribution selesai: {len(results)} sampel")
    return {"method": "Gradient_x_Input", "num_explained": len(results), "results": results}


def _visualize_shap_results(shap_values, explain_tensor, names,
                             labels, quality_scores, save_dir):
    """Visualize SHAP values sebagai image plots."""
    for i in range(len(names)):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Original image (denormalize)
        img = explain_tensor[i].cpu().numpy().transpose(1, 2, 0)
        img = img * np.array(config.IMAGENET_STD) + np.array(config.IMAGENET_MEAN)
        img = np.clip(img, 0, 1)
        axes[0].imshow(img)
        gt = "GON+" if labels[i] == 1 else "GON-"
        axes[0].set_title(f"Original ({gt})", fontsize=12)
        axes[0].axis("off")

        # SHAP values heatmap
        sv = shap_values[i] if isinstance(shap_values, list) else shap_values[i]
        if isinstance(sv, np.ndarray):
            sv_agg = np.abs(sv).mean(axis=0)
        else:
            sv_agg = np.abs(sv.cpu().numpy()).mean(axis=0)

        if sv_agg.max() > 0:
            sv_agg = sv_agg / sv_agg.max()

        axes[1].imshow(sv_agg, cmap='hot')
        axes[1].set_title("SHAP Feature Importance", fontsize=12)
        axes[1].axis("off")

        # Overlay
        sv_colored = cv2.applyColorMap(
            (sv_agg * 255).astype(np.uint8), cv2.COLORMAP_HOT
        )
        sv_colored = cv2.cvtColor(sv_colored, cv2.COLOR_BGR2RGB) / 255.0
        overlay = 0.6 * img + 0.4 * sv_colored
        overlay = np.clip(overlay, 0, 1)
        axes[2].imshow(overlay)
        axes[2].set_title("SHAP Overlay", fontsize=12)
        axes[2].axis("off")

        plt.suptitle(f"SHAP Analysis: {names[i]} (QS={quality_scores[i]:.2f})",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        safe_name = names[i].replace('.jpg', '').replace('.png', '')
        save_path = save_dir / f"shap_{safe_name}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()


def _visualize_attribution(image, attribution_map, image_name,
                            prob, qs_raw, label, save_dir):
    """Visualize Gradient×Input attribution map."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    display_img = cv2.resize(image, (config.IMG_SIZE, config.IMG_SIZE))
    gt = "GON+" if label == 1 else "GON-"

    # Original
    axes[0].imshow(display_img)
    axes[0].set_title(f"Original ({gt})", fontsize=12, fontweight='bold')
    axes[0].axis("off")

    # Attribution heatmap
    attr_resized = cv2.resize(attribution_map,
                               (config.IMG_SIZE, config.IMG_SIZE))
    axes[1].imshow(attr_resized, cmap='hot')
    axes[1].set_title("Feature Attribution (Gradient×Input)", fontsize=12,
                      fontweight='bold')
    axes[1].axis("off")

    # Overlay
    heatmap_colored = cv2.applyColorMap(
        (attr_resized * 255).astype(np.uint8), cv2.COLORMAP_HOT
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display_img, 0.55, heatmap_colored, 0.45, 0)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay — P(GON+)={prob:.3f}", fontsize=12,
                      fontweight='bold')
    axes[2].axis("off")

    plt.suptitle(f"Feature Attribution: {image_name} (QS={qs_raw:.2f})",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    safe_name = image_name.replace('.jpg', '').replace('.png', '')
    save_path = save_dir / f"attribution_{safe_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ===========================================================================
# 3. Clinical Report Generation
# ===========================================================================

def categorize_risk(probability: float) -> str:
    """
    Kategorisasi risiko berdasarkan probabilitas prediksi.

    - High Risk:   P(GON+) >= 0.70
    - Medium Risk: 0.40 <= P(GON+) < 0.70
    - Low Risk:    P(GON+) < 0.40
    """
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    elif probability >= RISK_THRESHOLDS["medium"]:
        return "Medium"
    else:
        return "Low"


def generate_per_sample_clinical_report(model: GONDetectionModel,
                                         val_loader,
                                         val_df,
                                         save_dir: Path = PHASE6_REPORTS_DIR) -> list:
    """
    Generate clinical report per-sample dengan:
    - Confidence score
    - Risk categorization (Low/Medium/High)
    - Quality gate (QS < 4.0 → insufficient quality flag)
    - Manual review flag (confidence < 0.85)

    Args:
        model: Trained model
        val_loader: Validation DataLoader
        val_df: Validation DataFrame
        save_dir: Direktori output

    Returns:
        List of per-sample clinical reports
    """
    print(f"\n[Phase 6] Generating per-sample clinical reports...")
    model.eval()

    all_reports = []
    batch_idx = 0

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(config.DEVICE)
            labels = batch["label"].cpu().numpy().flatten()
            qs = batch["quality_score"].to(config.DEVICE)
            qs_raw = qs.cpu().numpy().flatten() * 10  # Denormalize

            outputs = model(images, qs)
            probs = outputs["gon_probs"].cpu().numpy().flatten()
            qs_preds = outputs["quality_pred"].cpu().numpy().flatten() * 10

            # Get image names if available
            if "image_name" in batch:
                img_names = batch["image_name"]
            else:
                img_names = [f"sample_{batch_idx * len(labels) + i}"
                             for i in range(len(labels))]

            for i in range(len(labels)):
                prob = float(probs[i])
                risk = categorize_risk(prob)
                confidence = max(prob, 1 - prob)  # Confidence = certainty
                quality = float(qs_raw[i])

                report = {
                    "image_name": img_names[i] if isinstance(img_names, list) else str(img_names[i]),
                    "ground_truth": "GON+" if labels[i] == 1 else "GON-",
                    "prediction": {
                        "probability_gon_positive": round(prob, 4),
                        "confidence": round(confidence, 4),
                        "risk_category": risk,
                        "predicted_class": "GON+" if prob >= 0.5 else "GON-",
                    },
                    "quality_assessment": {
                        "quality_score_actual": round(quality, 2),
                        "quality_score_predicted": round(float(qs_preds[i]), 2),
                        "quality_sufficient": quality >= QUALITY_GATE_THRESHOLD,
                        "quality_flag": "INSUFFICIENT — Retake recommended"
                                       if quality < QUALITY_GATE_THRESHOLD
                                       else "OK",
                    },
                    "clinical_flags": {
                        "needs_manual_review": confidence < CONFIDENCE_REVIEW_THRESHOLD,
                        "low_quality_warning": quality < QUALITY_GATE_THRESHOLD,
                        "high_risk_alert": risk == "High",
                    },
                    "clinical_recommendation": _get_recommendation(prob, quality, risk),
                }

                all_reports.append(report)

            batch_idx += 1

    # Save full report JSON
    full_report_path = save_dir / "per_sample_clinical_reports.json"
    with open(full_report_path, 'w', encoding='utf-8') as f:
        json.dump(all_reports, f, indent=2, ensure_ascii=False)

    # Generate readable text report
    _generate_text_report(all_reports, save_dir)

    print(f"[Phase 6] Clinical reports generated: {len(all_reports)} sampel")
    print(f"  → JSON: {full_report_path}")
    return all_reports


def _get_recommendation(probability: float, quality: float, risk: str) -> str:
    """Generate clinical recommendation berdasarkan prediksi dan kualitas."""
    recommendations = []

    if quality < QUALITY_GATE_THRESHOLD:
        recommendations.append(
            "Kualitas gambar tidak memadai (QS < 4.0). "
            "Disarankan untuk mengambil ulang foto fundus dengan "
            "pencahayaan dan fokus yang lebih baik."
        )

    if risk == "High":
        recommendations.append(
            "RISIKO TINGGI terdeteksi. Rujuk pasien ke spesialis "
            "glaukoma untuk pemeriksaan komprehensif meliputi: "
            "tonometri, perimetri, OCT RNFL, dan gonioskopi."
        )
    elif risk == "Medium":
        recommendations.append(
            "RISIKO SEDANG terdeteksi. Disarankan follow-up dalam 3-6 bulan "
            "dengan pemeriksaan tonometri dan funduskopi ulang. "
            "Pertimbangkan rujukan bila ada faktor risiko tambahan."
        )
    else:
        recommendations.append(
            "Risiko rendah. Lanjutkan pemeriksaan rutin sesuai jadwal "
            "(setiap 1-2 tahun untuk populasi umum, atau sesuai indikasi klinisi)."
        )

    confidence = max(probability, 1 - probability)
    if confidence < CONFIDENCE_REVIEW_THRESHOLD:
        recommendations.append(
            f"⚠️ Confidence model rendah ({confidence:.2%}). "
            "Hasil ini HARUS di-review oleh oftalmologis secara manual."
        )

    recommendations.append(
        "DISCLAIMER: Hasil prediksi ini adalah alat bantu screening, "
        "BUKAN pengganti pemeriksaan oftalmologi komprehensif oleh dokter."
    )

    return " | ".join(recommendations)


def _generate_text_report(reports: list, save_dir: Path):
    """Generate human-readable text report dari per-sample reports."""
    lines = [
        "=" * 70,
        "  PHASE 6: CLINICAL INTERPRETABILITY REPORT",
        f"  GON Detection — Quality-Aware Multi-Task Model",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"  Total Samples Analyzed: {len(reports)}",
        "",
    ]

    # Risk distribution summary
    risk_counts = {"High": 0, "Medium": 0, "Low": 0}
    quality_insufficient = 0
    manual_review_needed = 0
    correct_predictions = 0

    for r in reports:
        risk_counts[r["prediction"]["risk_category"]] += 1
        if r["clinical_flags"]["low_quality_warning"]:
            quality_insufficient += 1
        if r["clinical_flags"]["needs_manual_review"]:
            manual_review_needed += 1
        pred_class = r["prediction"]["predicted_class"]
        gt = r["ground_truth"]
        if pred_class == gt:
            correct_predictions += 1

    lines.extend([
        "-" * 70,
        "  RISK DISTRIBUTION",
        "-" * 70,
        f"  🔴 High Risk:    {risk_counts['High']:>4} "
        f"({risk_counts['High']/len(reports)*100:.1f}%)",
        f"  🟡 Medium Risk:  {risk_counts['Medium']:>4} "
        f"({risk_counts['Medium']/len(reports)*100:.1f}%)",
        f"  🟢 Low Risk:     {risk_counts['Low']:>4} "
        f"({risk_counts['Low']/len(reports)*100:.1f}%)",
        "",
        "-" * 70,
        "  CLINICAL FLAGS",
        "-" * 70,
        f"  ⚠️  Low Quality (QS < {QUALITY_GATE_THRESHOLD}): "
        f"{quality_insufficient} samples",
        f"  🔍 Manual Review Needed: {manual_review_needed} samples",
        f"  📊 Prediction Accuracy: "
        f"{correct_predictions}/{len(reports)} "
        f"({correct_predictions/len(reports)*100:.1f}%)",
        "",
    ])

    # Per-sample details
    lines.extend([
        "-" * 70,
        "  PER-SAMPLE RESULTS",
        "-" * 70,
        "",
        f"  {'Image':<15} {'GT':<6} {'Pred':<6} {'P(GON+)':<10} "
        f"{'Risk':<8} {'QS':<6} {'Flags':<20}",
        f"  {'-'*13:<15} {'-'*4:<6} {'-'*4:<6} {'-'*8:<10} "
        f"{'-'*6:<8} {'-'*4:<6} {'-'*18:<20}",
    ])

    for r in reports:
        flags = []
        if r["clinical_flags"]["high_risk_alert"]:
            flags.append("🔴HIGH")
        if r["clinical_flags"]["needs_manual_review"]:
            flags.append("🔍REVIEW")
        if r["clinical_flags"]["low_quality_warning"]:
            flags.append("⚠️LQ")
        flags_str = " ".join(flags) if flags else "—"

        name = r["image_name"][:14]
        lines.append(
            f"  {name:<15} {r['ground_truth']:<6} "
            f"{r['prediction']['predicted_class']:<6} "
            f"{r['prediction']['probability_gon_positive']:<10.4f} "
            f"{r['prediction']['risk_category']:<8} "
            f"{r['quality_assessment']['quality_score_actual']:<6.2f} "
            f"{flags_str}"
        )

    lines.extend([
        "",
        "-" * 70,
        "  CLINICAL SAFETY REMINDERS",
        "-" * 70,
        "",
        "  [!] Model ini adalah alat bantu screening, BUKAN pengganti",
        "      pemeriksaan oftalmologi komprehensif oleh dokter spesialis.",
        f"  [!] Prediksi dengan confidence < {CONFIDENCE_REVIEW_THRESHOLD:.0%} "
        "harus di-review manual.",
        f"  [!] Gambar dengan Quality Score < {QUALITY_GATE_THRESHOLD:.1f} "
        "harus difoto ulang.",
        "  [!] Pasien dengan Risk HIGH harus dirujuk ke spesialis glaukoma.",
        "",
        "=" * 70,
    ])

    report_text = "\n".join(lines)

    # Save
    text_path = save_dir / "clinical_report_phase6.txt"
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"  → Text report: {text_path}")
    return report_text


# ===========================================================================
# 4. Summary Dashboard
# ===========================================================================

def generate_summary_dashboard(reports: list,
                                gradcam_results: list = None,
                                shap_results: dict = None,
                                save_dir: Path = PHASE6_SUMMARY_DIR):
    """
    Generate summary dashboard untuk Phase 6.

    Includes:
    - Risk distribution pie chart
    - Confidence distribution histogram
    - Quality vs Prediction scatter
    - Per-risk-category performance
    """
    if not HAS_MATPLOTLIB:
        print("[Phase 6] Matplotlib not available, skipping dashboard.")
        return

    print(f"\n[Phase 6] Generating summary dashboard...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Extract data
    probs = [r["prediction"]["probability_gon_positive"] for r in reports]
    risks = [r["prediction"]["risk_category"] for r in reports]
    qs_actual = [r["quality_assessment"]["quality_score_actual"] for r in reports]
    gts = [1 if r["ground_truth"] == "GON+" else 0 for r in reports]
    preds = [1 if r["prediction"]["predicted_class"] == "GON+" else 0 for r in reports]
    confidences = [r["prediction"]["confidence"] for r in reports]

    # 1. Risk Distribution Pie Chart
    risk_counts = {"High": risks.count("High"),
                   "Medium": risks.count("Medium"),
                   "Low": risks.count("Low")}
    colors_pie = ["#e74c3c", "#f39c12", "#27ae60"]
    labels_pie = [f"{k}\n({v})" for k, v in risk_counts.items()]
    wedges, texts, autotexts = axes[0, 0].pie(
        risk_counts.values(), labels=labels_pie, colors=colors_pie,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_fontweight('bold')
    axes[0, 0].set_title("Risk Category Distribution", fontsize=13,
                          fontweight='bold')

    # 2. Confidence Distribution Histogram
    axes[0, 1].hist(confidences, bins=20, color='steelblue',
                     edgecolor='white', alpha=0.85)
    axes[0, 1].axvline(x=CONFIDENCE_REVIEW_THRESHOLD, color='red',
                        linestyle='--', linewidth=2,
                        label=f'Review threshold ({CONFIDENCE_REVIEW_THRESHOLD})')
    axes[0, 1].set_xlabel("Confidence", fontsize=11)
    axes[0, 1].set_ylabel("Count", fontsize=11)
    axes[0, 1].set_title("Prediction Confidence Distribution", fontsize=13,
                          fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Quality Score vs P(GON+)
    # Color by ground truth
    colors_scatter = ['#e74c3c' if gt == 1 else '#3498db' for gt in gts]
    axes[1, 0].scatter(qs_actual, probs, c=colors_scatter, alpha=0.6, s=50,
                        edgecolors='gray', linewidth=0.5)
    axes[1, 0].axhline(y=RISK_THRESHOLDS["high"], color='red',
                        linestyle='--', alpha=0.5, label=f'High risk ({RISK_THRESHOLDS["high"]})')
    axes[1, 0].axhline(y=RISK_THRESHOLDS["medium"], color='orange',
                        linestyle='--', alpha=0.5, label=f'Medium risk ({RISK_THRESHOLDS["medium"]})')
    axes[1, 0].axvline(x=QUALITY_GATE_THRESHOLD, color='purple',
                        linestyle='--', alpha=0.5, label=f'Quality gate ({QUALITY_GATE_THRESHOLD})')
    axes[1, 0].set_xlabel("Quality Score", fontsize=11)
    axes[1, 0].set_ylabel("P(GON+)", fontsize=11)
    axes[1, 0].set_title("Quality Score vs Prediction", fontsize=13,
                          fontweight='bold')
    axes[1, 0].legend(fontsize=9, loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)

    # Custom legend for ground truth
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=8, label='GON+'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db',
               markersize=8, label='GON-'),
    ]
    axes[1, 0].legend(handles=legend_elements + axes[1, 0].get_legend_handles_labels()[0],
                       fontsize=9, loc='upper left')

    # 4. Performance per Risk Category
    risk_performance = {}
    for risk_cat in ["Low", "Medium", "High"]:
        indices = [i for i, r in enumerate(risks) if r == risk_cat]
        if indices:
            gt_sub = [gts[i] for i in indices]
            pred_sub = [preds[i] for i in indices]
            acc = sum(1 for g, p in zip(gt_sub, pred_sub) if g == p) / len(indices)
            risk_performance[risk_cat] = {
                "count": len(indices),
                "accuracy": acc,
            }

    categories = list(risk_performance.keys())
    accuracies = [risk_performance[c]["accuracy"] for c in categories]
    counts = [risk_performance[c]["count"] for c in categories]

    bar_colors = {"Low": "#27ae60", "Medium": "#f39c12", "High": "#e74c3c"}
    bars = axes[1, 1].bar(categories,
                           accuracies,
                           color=[bar_colors[c] for c in categories],
                           edgecolor='gray', alpha=0.85)

    for bar, count, acc in zip(bars, counts, accuracies):
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                        f'{acc:.1%}\n(n={count})', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

    axes[1, 1].set_ylim(0, 1.15)
    axes[1, 1].set_xlabel("Risk Category", fontsize=11)
    axes[1, 1].set_ylabel("Accuracy", fontsize=11)
    axes[1, 1].set_title("Accuracy per Risk Category", fontsize=13,
                          fontweight='bold')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    plt.suptitle("Phase 6: Clinical Interpretability Dashboard",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    dashboard_path = save_dir / "phase6_dashboard.png"
    plt.savefig(dashboard_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[Phase 6] Dashboard saved: {dashboard_path}")

    # Save summary JSON
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": len(reports),
        "risk_distribution": risk_counts,
        "quality_insufficient": sum(1 for r in reports
                                     if r["clinical_flags"]["low_quality_warning"]),
        "manual_review_needed": sum(1 for r in reports
                                     if r["clinical_flags"]["needs_manual_review"]),
        "risk_performance": risk_performance,
        "mean_confidence": float(np.mean(confidences)),
        "overall_accuracy": sum(1 for g, p in zip(gts, preds) if g == p) / len(gts),
    }

    summary_path = save_dir / "phase6_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"[Phase 6] Summary saved: {summary_path}")
    return summary


# ===========================================================================
# 5. Full Phase 6 Pipeline
# ===========================================================================

def run_phase6(checkpoint_path: str = None,
               fold: int = 1,
               num_gradcam_samples: int = 20,
               num_shap_samples: int = 10) -> dict:
    """
    Jalankan seluruh Phase 6: Clinical Interpretability & Explainability.

    Pipeline:
    1. Load model dari checkpoint
    2. Grad-CAM++ batch visualization
    3. SHAP / Gradient-based Attribution analysis
    4. Per-sample Clinical Report Generation (risk categorization)
    5. Summary Dashboard

    Args:
        checkpoint_path: Path ke checkpoint model
        fold: Fold number
        num_gradcam_samples: Jumlah sampel Grad-CAM++
        num_shap_samples: Jumlah sampel SHAP analysis

    Returns:
        dict dengan semua results Phase 6
    """
    from model import create_model
    from dataset import create_patient_level_splits, create_dataloaders

    print("\n" + "=" * 70)
    print("  PHASE 6: CLINICAL INTERPRETABILITY & EXPLAINABILITY")
    print("=" * 70)

    # --- Step 1: Load Model ---
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / f"best_model_fold{fold}.pth"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[INFO] Jalankan Phase 4 (training) terlebih dahulu.")
        return None

    print(f"\n[Phase 6] Loading model from: {checkpoint_path}")

    model = create_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()

    epoch = checkpoint.get('epoch', 'N/A')
    val_auc = checkpoint.get('val_auc', 'N/A')
    print(f"[Phase 6] Model loaded (epoch={epoch}, val_auc={val_auc})")

    # --- Step 2: Prepare validation data ---
    splits = create_patient_level_splits()
    train_df, val_df = splits[fold - 1]

    # DataLoader biasa (tanpa meta)
    _, val_loader = create_dataloaders(train_df, val_df,
                                        batch_size=config.BATCH_SIZE)

    # DataLoader dengan meta info
    from dataset import HYGDDataset, get_val_transforms
    val_dataset_meta = HYGDDataset(
        val_df,
        transform=get_val_transforms(config.IMG_SIZE),
        preprocess=True,
        img_size=config.IMG_SIZE,
        return_meta=True,
    )
    val_loader_meta = torch.utils.data.DataLoader(
        val_dataset_meta,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Avoid pickling issues
        pin_memory=config.PIN_MEMORY,
    )

    print(f"[Phase 6] Validation set: {len(val_df)} samples (Fold {fold})")

    # --- Step 3: Grad-CAM++ Batch Visualization ---
    gradcam_results = generate_gradcam_batch(
        model, val_loader, val_df,
        num_samples=num_gradcam_samples,
        save_dir=PHASE6_GRADCAM_DIR
    )

    # --- Step 4: SHAP / Attribution Analysis ---
    shap_results = run_shap_analysis(
        model, val_loader, val_df,
        num_explain=num_shap_samples,
        save_dir=PHASE6_SHAP_DIR
    )

    # --- Step 5: Clinical Report Generation ---
    clinical_reports = generate_per_sample_clinical_report(
        model, val_loader_meta, val_df,
        save_dir=PHASE6_REPORTS_DIR
    )

    # --- Step 6: Summary Dashboard ---
    summary = generate_summary_dashboard(
        clinical_reports, gradcam_results, shap_results,
        save_dir=PHASE6_SUMMARY_DIR
    )

    # --- Final Summary ---
    print("\n" + "=" * 70)
    print("  PHASE 6 RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Grad-CAM++ visualizations: {len(gradcam_results)} samples")
    print(f"  SHAP/Attribution analysis: {shap_results.get('num_explained', 0)} samples")
    print(f"  Clinical reports: {len(clinical_reports)} samples")
    print(f"\n  Output directory: {PHASE6_DIR}")
    print(f"    ├── gradcam_batch/     - Grad-CAM++ heatmaps")
    print(f"    ├── shap_analysis/     - SHAP feature importance")
    print(f"    ├── clinical_reports/  - Per-sample clinical reports")
    print(f"    └── summary/           - Dashboard & summary")
    print(f"\n[Phase 6] [DONE] Clinical Interpretability & Explainability selesai!")

    return {
        "gradcam_results": gradcam_results,
        "shap_results": shap_results,
        "clinical_reports": clinical_reports,
        "summary": summary,
    }


# ===========================================================================
# Standalone Execution
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase 6: Clinical Interpretability & Explainability"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path ke model checkpoint")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold number (default: 1)")
    parser.add_argument("--gradcam_samples", type=int, default=20,
                        help="Jumlah sampel Grad-CAM++ (default: 20)")
    parser.add_argument("--shap_samples", type=int, default=10,
                        help="Jumlah sampel SHAP (default: 10)")

    args = parser.parse_args()

    run_phase6(
        checkpoint_path=args.checkpoint,
        fold=args.fold,
        num_gradcam_samples=args.gradcam_samples,
        num_shap_samples=args.shap_samples,
    )
