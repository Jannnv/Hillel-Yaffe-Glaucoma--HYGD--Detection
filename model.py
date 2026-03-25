"""
=============================================================================
model.py — Phase 3: Quality-Aware Multi-Task GON Detection Model
=============================================================================
Arsitektur:
  - Backbone: EfficientNet-B0 (pretrained ImageNet)
  - Quality-Aware Attention Module (QAAM)
  - Multi-Task Heads:
      Head 1: GON Classification (Binary, Focal Loss)
      Head 2: Quality Score Regression (MSE Loss)
  - Quality-Conditioned Feature Refinement (QCFR)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[WARNING] timm not found. Install: pip install timm")

import config


# ===========================================================================
# Building Blocks
# ===========================================================================

class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) block untuk channel attention.

    Memberi bobot adaptif pada setiap channel berdasarkan
    global information dari seluruh spatial dimensions.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(in_channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial attention module.

    Mengidentifikasi DIMANA features penting berada
    (e.g., optic disc region pada fundus).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]

        # Concatenate & convolve
        attention = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        attention = self.conv(attention)  # [B, 1, H, W]

        return x * attention


class QualityConditionedFeatureRefinement(nn.Module):
    """
    Quality-Conditioned Feature Refinement (QCFR) — Modul Novel.

    Menggunakan quality score sebagai conditioning signal untuk
    memodulasi intermediate features secara adaptif.

    Idea: Model harus memproses gambar berkualitas rendah
    secara berbeda dari gambar berkualitas tinggi.

    QS → MLP → (γ, β) → Feature_refined = γ ⊙ Feature + β
    """

    def __init__(self, feature_dim: int, qs_embed_dim: int = 64):
        super().__init__()

        # MLP untuk menghasilkan scale (γ) dan shift (β) parameters
        self.conditioning_network = nn.Sequential(
            nn.Linear(1, qs_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(qs_embed_dim, qs_embed_dim),
            nn.ReLU(inplace=True),
        )

        # Separate heads untuk γ dan β
        self.gamma_head = nn.Linear(qs_embed_dim, feature_dim)
        self.beta_head = nn.Linear(qs_embed_dim, feature_dim)

        # Inisialisasi: γ ≈ 1, β ≈ 0 (identity mapping awalnya)
        nn.init.ones_(self.gamma_head.weight)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, features: torch.Tensor,
                quality_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, D] feature vector
            quality_score: [B, 1] normalized quality score

        Returns:
            Refined features [B, D]
        """
        # Condition on quality score
        qs_embed = self.conditioning_network(quality_score)  # [B, qs_embed_dim]

        # Generate scale and shift
        gamma = self.gamma_head(qs_embed)  # [B, D]
        beta = self.beta_head(qs_embed)    # [B, D]

        # Feature refinement: FiLM (Feature-wise Linear Modulation)
        refined = gamma * features + beta

        return refined


class QualityAwareAttentionModule(nn.Module):
    """
    Quality-Aware Attention Module (QAAM).

    Combine:
      1. Channel Attention (SE Block)
      2. Spatial Attention
      3. Quality-conditioned modulation
    """

    def __init__(self, in_channels: int,
                 reduction: int = config.QAAM_REDUCTION,
                 qs_embed_dim: int = config.QAAM_QS_EMBED_DIM):
        super().__init__()

        self.channel_attention = SqueezeExcitation(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # Quality score embedding → modulate spatial attention
        self.qs_spatial_modulator = nn.Sequential(
            nn.Linear(1, qs_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(qs_embed_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor,
                quality_score: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Feature map [B, C, H, W]
            quality_score: [B, 1] normalized quality score

        Returns:
            Attention-refined feature map [B, C, H, W]
        """
        # Channel attention
        x = self.channel_attention(x)

        # Spatial attention
        x = self.spatial_attention(x)

        # Quality-conditioned modulation (opsional)
        if quality_score is not None:
            qs_weight = self.qs_spatial_modulator(quality_score)  # [B, 1]
            qs_weight = qs_weight.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
            x = x * (0.5 + 0.5 * qs_weight)  # Soft modulation [0.5, 1.0]

        return x


# ===========================================================================
# Main Model: Quality-Aware Multi-Task GON Detection
# ===========================================================================

class GONDetectionModel(nn.Module):
    """
    Quality-Aware Multi-Task Learning Model untuk GON Detection.

    Architecture:
        Input → EfficientNet-B0 (Backbone)
              → QAAM (Quality-Aware Attention)
              → Global Average Pooling
              → QCFR (Quality-Conditioned Feature Refinement)
              → Multi-Task Heads:
                  ├── GON Classification Head → P(GON+)
                  └── Quality Regression Head → QS_predicted
    """

    def __init__(self,
                 backbone_name: str = config.MODEL_NAME,
                 pretrained: bool = config.PRETRAINED,
                 num_classes: int = config.NUM_CLASSES,
                 drop_rate: float = config.DROP_RATE,
                 drop_path_rate: float = config.DROP_PATH_RATE):
        super().__init__()

        self.backbone_name = backbone_name

        # ===== 1. Backbone =====
        if HAS_TIMM:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                features_only=False,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
            # Dapatkan dimensi output dari backbone
            self.feature_dim = self.backbone.num_features

            # Hapus classifier head bawaan
            if hasattr(self.backbone, 'classifier'):
                self.backbone.classifier = nn.Identity()
            elif hasattr(self.backbone, 'fc'):
                self.backbone.fc = nn.Identity()
            elif hasattr(self.backbone, 'head'):
                self.backbone.head = nn.Identity()
        else:
            # Fallback: simple CNN jika timm tidak tersedia
            self.backbone = self._build_simple_backbone()
            self.feature_dim = 512

        # ===== 2. Quality-Aware Attention Module =====
        # Note: QAAM diterapkan sebelum GAP.
        # Untuk EfficientNet, kita perlu hook ke feature map sebelum pooling
        self.qaam = QualityAwareAttentionModule(
            in_channels=self.feature_dim,
            reduction=config.QAAM_REDUCTION,
            qs_embed_dim=config.QAAM_QS_EMBED_DIM
        )

        # ===== 3. Pooling =====
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ===== 4. Quality-Conditioned Feature Refinement =====
        self.qcfr = QualityConditionedFeatureRefinement(
            feature_dim=self.feature_dim,
            qs_embed_dim=config.QAAM_QS_EMBED_DIM
        )

        # ===== 5. Multi-Task Heads =====

        # Head 1: GON Classification
        self.gon_head = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        # Head 2: Quality Score Regression
        self.quality_head = nn.Sequential(
            nn.Dropout(p=drop_rate * 0.5),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output range [0, 1] (normalized quality score)
        )

        # ===== 6. Untuk Grad-CAM: simpan feature map terakhir =====
        self._feature_map = None
        self._gradients = None

    def _build_simple_backbone(self):
        """Fallback backbone sederhana jika timm tidak tersedia."""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature map dari backbone.
        Handle berbagai arsitektur backbone.
        """
        if HAS_TIMM:
            # Untuk timm models, gunakan forward_features
            features = self.backbone.forward_features(x)
        else:
            features = self.backbone(x)

        return features

    def forward(self, x: torch.Tensor,
                quality_score: torch.Tensor = None) -> dict:
        """
        Forward pass.

        Args:
            x: Input images [B, 3, H, W]
            quality_score: Normalized quality scores [B, 1]

        Returns:
            dict with keys:
                - 'gon_logits': [B, 1] raw logits untuk GON classification
                - 'gon_probs': [B, 1] probability P(GON+)
                - 'quality_pred': [B, 1] predicted quality score (normalized)
                - 'features': [B, D] feature vectors (untuk visualization)
        """
        # 1. Extract feature map dari backbone
        feature_map = self._extract_features(x)  # [B, C, H, W]

        # Simpan untuk Grad-CAM
        self._feature_map = feature_map

        # 2. Quality-Aware Attention
        feature_map = self.qaam(feature_map, quality_score)

        # 3. Global Average Pooling
        features = self.global_pool(feature_map)  # [B, C, 1, 1]
        features = features.flatten(1)             # [B, C]

        # 4. Quality-Conditioned Feature Refinement
        if quality_score is not None:
            features = self.qcfr(features, quality_score)

        # 5. Multi-Task Predictions
        gon_logits = self.gon_head(features)           # [B, 1]
        gon_probs = torch.sigmoid(gon_logits)          # [B, 1]
        quality_pred = self.quality_head(features)      # [B, 1]

        return {
            "gon_logits": gon_logits,
            "gon_probs": gon_probs,
            "quality_pred": quality_pred,
            "features": features,
        }

    def get_feature_map(self) -> torch.Tensor:
        """Return feature map terakhir (untuk Grad-CAM)."""
        return self._feature_map


# ===========================================================================
# Loss Functions
# ===========================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss untuk class imbalance.

    L_focal(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    γ > 0 mengurangi loss untuk well-classified examples,
    memfokuskan training pada hard, misclassified examples.
    """

    def __init__(self, gamma: float = config.FOCAL_GAMMA,
                 alpha: float = config.FOCAL_ALPHA,
                 label_smoothing: float = config.LABEL_SMOOTHING):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1] raw logits
            targets: [B, 1] binary targets (0 or 1)

        Returns:
            Scalar focal loss
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Binary cross entropy (tanpa reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Probabilities
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combined
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class QualityWeightedMultiTaskLoss(nn.Module):
    """
    Quality-Weighted Multi-Task Loss.

    L_total = (1 / 2σ₁²) * L_gon * w_q + (1 / 2σ₂²) * L_qs + log(σ₁) + log(σ₂)

    Menggunakan uncertainty weighting (Kendall et al.) untuk
    otomatis menyeimbangkan kedua task.
    """

    def __init__(self,
                 focal_gamma: float = config.FOCAL_GAMMA,
                 focal_alpha: float = config.FOCAL_ALPHA,
                 quality_weight: float = config.QUALITY_LOSS_WEIGHT,
                 use_uncertainty_weighting: bool = True):
        super().__init__()

        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.quality_weight = quality_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting

        if use_uncertainty_weighting:
            # Learnable log-variance parameters (Kendall et al.)
            self.log_sigma_gon = nn.Parameter(torch.zeros(1))
            self.log_sigma_qs = nn.Parameter(torch.zeros(1))

    def forward(self,
                gon_logits: torch.Tensor,
                gon_targets: torch.Tensor,
                quality_pred: torch.Tensor,
                quality_targets: torch.Tensor,
                quality_weights: torch.Tensor = None) -> dict:
        """
        Args:
            gon_logits: [B, 1] GON classification logits
            gon_targets: [B, 1] GON binary labels
            quality_pred: [B, 1] predicted quality scores
            quality_targets: [B, 1] target quality scores (normalized)
            quality_weights: [B, 1] per-sample quality weights

        Returns:
            dict with 'total_loss', 'gon_loss', 'quality_loss'
        """
        # GON classification loss (Focal Loss)
        gon_loss = self.focal_loss(gon_logits, gon_targets)

        # Quality regression loss (MSE)
        quality_loss = F.mse_loss(quality_pred, quality_targets)

        # Quality-weighted GON loss (opsional)
        if quality_weights is not None and config.USE_QUALITY_WEIGHTING:
            # Weighted focal loss: gambar berkualitas tinggi mendapat bobot lebih
            weighted_bce = F.binary_cross_entropy_with_logits(
                gon_logits, gon_targets, reduction='none'
            )
            probs = torch.sigmoid(gon_logits)
            p_t = probs * gon_targets + (1 - probs) * (1 - gon_targets)
            focal_weight = (1 - p_t) ** self.focal_loss.gamma
            alpha_t = self.focal_loss.alpha * gon_targets + \
                      (1 - self.focal_loss.alpha) * (1 - gon_targets)

            weighted_loss = (alpha_t * focal_weight * weighted_bce * quality_weights)
            gon_loss = weighted_loss.mean()

        # Combine losses
        if self.use_uncertainty_weighting:
            # Kendall uncertainty weighting
            precision_gon = torch.exp(-2 * self.log_sigma_gon)
            precision_qs = torch.exp(-2 * self.log_sigma_qs)

            total_loss = (precision_gon * gon_loss + self.log_sigma_gon +
                          precision_qs * quality_loss + self.log_sigma_qs)
        else:
            total_loss = gon_loss + self.quality_weight * quality_loss

        return {
            "total_loss": total_loss,
            "gon_loss": gon_loss,
            "quality_loss": quality_loss,
        }


# ===========================================================================
# Model Factory
# ===========================================================================

def create_model(backbone_name: str = config.MODEL_NAME,
                 pretrained: bool = config.PRETRAINED) -> GONDetectionModel:
    """
    Factory function untuk membuat model.

    Args:
        backbone_name: Nama backbone architecture
        pretrained: Apakah gunakan pretrained weights

    Returns:
        GONDetectionModel instance
    """
    model = GONDetectionModel(
        backbone_name=backbone_name,
        pretrained=pretrained,
    )

    # Hitung parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[MODEL] {backbone_name}")
    print(f"  Feature dim: {model.feature_dim}")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    return model


if __name__ == "__main__":
    """Test model forward pass."""
    model = create_model()
    model.to(config.DEVICE)
    model.eval()

    # Dummy input
    x = torch.randn(4, 3, config.IMG_SIZE, config.IMG_SIZE).to(config.DEVICE)
    qs = torch.rand(4, 1).to(config.DEVICE)

    with torch.no_grad():
        output = model(x, qs)

    print(f"\n[TEST] Forward pass:")
    print(f"  Input:        {x.shape}")
    print(f"  GON logits:   {output['gon_logits'].shape}")
    print(f"  GON probs:    {output['gon_probs'].shape} -> {output['gon_probs'].squeeze().tolist()}")
    print(f"  Quality pred: {output['quality_pred'].shape} -> {output['quality_pred'].squeeze().tolist()}")
    print(f"  Features:     {output['features'].shape}")

    # Test loss
    criterion = QualityWeightedMultiTaskLoss()
    labels = torch.randint(0, 2, (4, 1)).float().to(config.DEVICE)
    qs_target = torch.rand(4, 1).to(config.DEVICE)
    qw = torch.rand(4, 1).to(config.DEVICE)

    losses = criterion(output['gon_logits'], labels, output['quality_pred'], qs_target, qw)
    print(f"\n[TEST] Loss:")
    print(f"  Total:   {losses['total_loss'].item():.4f}")
    print(f"  GON:     {losses['gon_loss'].item():.4f}")
    print(f"  Quality: {losses['quality_loss'].item():.4f}")
