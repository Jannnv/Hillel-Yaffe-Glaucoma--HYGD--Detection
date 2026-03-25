"""
=============================================================================
train.py — Phase 4: Training Pipeline
=============================================================================
Features:
  - Patient-level K-Fold Cross-Validation
  - Mixed precision training (AMP)
  - Learning rate warmup + cosine annealing
  - Early stopping
  - MixUp augmentation
  - Quality-weighted multi-task loss
  - Per-epoch metrics logging
  - Best model checkpointing
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import config
from model import GONDetectionModel, QualityWeightedMultiTaskLoss, create_model
from dataset import (
    create_patient_level_splits,
    create_dataloaders,
    mixup_data,
)
from evaluate import compute_metrics, find_optimal_threshold


# ===========================================================================
# Learning Rate Scheduler with Warmup
# ===========================================================================

class WarmupCosineScheduler:
    """
    Learning rate scheduler with linear warmup + cosine annealing.

    Warmup: LR linearly increases from 0 to base_lr over warmup_epochs
    Cosine: LR follows cosine decay after warmup
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(self.warmup_epochs, 1)
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i] * alpha
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(
                self.total_epochs - self.warmup_epochs, 1)
            alpha = 0.5 * (1 + np.cos(np.pi * progress))
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * alpha

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ===========================================================================
# Early Stopping
# ===========================================================================

class EarlyStopping:
    """
    Early stopping monitor.

    Menghentikan training ketika validation metric tidak membaik
    selama `patience` epochs berturut-turut.
    """

    def __init__(self, patience: int = config.EARLY_STOPPING_PATIENCE,
                 min_delta: float = 1e-4, mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ===========================================================================
# Training & Validation Loops
# ===========================================================================

def train_one_epoch(model: GONDetectionModel,
                    train_loader,
                    criterion: QualityWeightedMultiTaskLoss,
                    optimizer,
                    scaler,
                    epoch: int,
                    use_mixup: bool = True) -> dict:
    """
    Train model untuk satu epoch.

    Returns:
        dict with average losses dan metrics
    """
    model.train()
    running_losses = defaultdict(float)
    all_preds = []
    all_labels = []
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        images = batch["image"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)
        quality_scores = batch["quality_score"].to(config.DEVICE)
        quality_weights = batch["quality_weight"].to(config.DEVICE)

        # MixUp (opsional, 50% chance)
        use_mix = use_mixup and np.random.rand() < 0.5
        if use_mix:
            images, labels, quality_scores, lam = mixup_data(
                images, labels, quality_scores
            )

        # Forward pass with mixed precision
        optimizer.zero_grad()

        if config.USE_AMP and config.DEVICE.type == 'cuda':
            with autocast():
                outputs = model(images, quality_scores)
                losses = criterion(
                    outputs["gon_logits"], labels,
                    outputs["quality_pred"], quality_scores,
                    quality_weights
                )
            # Backward pass with scaler
            scaler.scale(losses["total_loss"]).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images, quality_scores)
            losses = criterion(
                outputs["gon_logits"], labels,
                outputs["quality_pred"], quality_scores,
                quality_weights
            )
            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Accumulate
        running_losses["total_loss"] += losses["total_loss"].item()
        running_losses["gon_loss"] += losses["gon_loss"].item()
        running_losses["quality_loss"] += losses["quality_loss"].item()

        # Collect predictions (tanpa mixup labels)
        if not use_mix:
            all_preds.extend(outputs["gon_probs"].detach().cpu().numpy().flatten())
            all_labels.extend(labels.detach().cpu().numpy().flatten())

        num_batches += 1

    # Average losses
    avg_losses = {k: v / num_batches for k, v in running_losses.items()}

    # Compute train metrics
    if all_preds:
        train_metrics = compute_metrics(
            np.array(all_labels),
            np.array(all_preds),
            threshold=config.CONFIDENCE_THRESHOLD
        )
        avg_losses.update({f"train_{k}": v for k, v in train_metrics.items()})

    return avg_losses


@torch.no_grad()
def validate(model: GONDetectionModel,
             val_loader,
             criterion: QualityWeightedMultiTaskLoss) -> dict:
    """
    Validasi model.

    Returns:
        dict with losses, metrics, predictions, dan labels
    """
    model.eval()
    running_losses = defaultdict(float)
    all_preds = []
    all_labels = []
    all_quality_preds = []
    all_quality_targets = []
    num_batches = 0

    for batch in val_loader:
        images = batch["image"].to(config.DEVICE)
        labels = batch["label"].to(config.DEVICE)
        quality_scores = batch["quality_score"].to(config.DEVICE)
        quality_weights = batch["quality_weight"].to(config.DEVICE)

        # Forward
        if config.USE_AMP and config.DEVICE.type == 'cuda':
            with autocast():
                outputs = model(images, quality_scores)
                losses = criterion(
                    outputs["gon_logits"], labels,
                    outputs["quality_pred"], quality_scores,
                    quality_weights
                )
        else:
            outputs = model(images, quality_scores)
            losses = criterion(
                outputs["gon_logits"], labels,
                outputs["quality_pred"], quality_scores,
                quality_weights
            )

        running_losses["total_loss"] += losses["total_loss"].item()
        running_losses["gon_loss"] += losses["gon_loss"].item()
        running_losses["quality_loss"] += losses["quality_loss"].item()

        all_preds.extend(outputs["gon_probs"].cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
        all_quality_preds.extend(outputs["quality_pred"].cpu().numpy().flatten())
        all_quality_targets.extend(quality_scores.cpu().numpy().flatten())

        num_batches += 1

    # Average losses
    avg_losses = {k: v / num_batches for k, v in running_losses.items()}

    # Compute metrics
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # Cari threshold optimal
    optimal_thresh = find_optimal_threshold(labels, preds)

    # Metrics dengan threshold default dan optimal
    metrics_default = compute_metrics(labels, preds, threshold=0.5)
    metrics_optimal = compute_metrics(labels, preds, threshold=optimal_thresh)

    result = {
        **avg_losses,
        "optimal_threshold": optimal_thresh,
        "predictions": preds,
        "labels_true": labels,
        "quality_preds": np.array(all_quality_preds),
        "quality_targets": np.array(all_quality_targets),
    }

    # Add metrics with prefix
    for k, v in metrics_default.items():
        result[f"val_{k}"] = v
    for k, v in metrics_optimal.items():
        result[f"val_opt_{k}"] = v

    return result


# ===========================================================================
# Full Training Pipeline
# ===========================================================================

def train_fold(fold_idx: int,
               train_df,
               val_df,
               img_size: int = config.IMG_SIZE,
               num_epochs: int = config.NUM_EPOCHS) -> dict:
    """
    Train model untuk satu fold.

    Args:
        fold_idx: Index fold (0-based)
        train_df: Training DataFrame
        val_df: Validation DataFrame
        img_size: Image size
        num_epochs: Maximum epochs

    Returns:
        dict dengan best metrics dan path ke checkpoint
    """
    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx + 1}/{config.NUM_FOLDS}")
    print(f"{'='*70}")

    # DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_df, val_df, img_size=img_size
    )

    # Model
    model = create_model()
    model = model.to(config.DEVICE)

    # Loss
    criterion = QualityWeightedMultiTaskLoss().to(config.DEVICE)

    # Optimizer: different LR untuk backbone vs heads
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config.LR_BACKBONE},
        {'params': head_params, 'lr': config.LR_HEAD},
        # Learnable loss weights (jika uncertainty weighting)
        {'params': criterion.parameters(), 'lr': config.LR_HEAD},
    ], weight_decay=config.WEIGHT_DECAY)

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        total_epochs=num_epochs
    )

    # AMP Scaler
    scaler = GradScaler() if config.USE_AMP else None

    # Early stopping
    early_stopping = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE, mode='max')

    # Training history
    history = defaultdict(list)
    best_auc = 0.0
    best_epoch = 0
    best_checkpoint_path = None

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Update learning rate
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()

        # ===== Train =====
        train_results = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            epoch, use_mixup=True
        )

        # ===== Validate =====
        val_results = validate(model, val_loader, criterion)

        epoch_time = time.time() - epoch_start

        # Log
        val_auc = val_results.get("val_auc_roc", 0)
        val_sens = val_results.get("val_opt_sensitivity", 0)
        val_spec = val_results.get("val_opt_specificity", 0)
        val_f1 = val_results.get("val_opt_f1_score", 0)

        print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {current_lr[0]:.2e} | "
              f"Train Loss: {train_results['total_loss']:.4f} | "
              f"Val Loss: {val_results['total_loss']:.4f} | "
              f"AUC: {val_auc:.4f} | "
              f"Sens: {val_sens:.4f} | "
              f"Spec: {val_spec:.4f} | "
              f"F1: {val_f1:.4f}")

        # Save history
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_results["total_loss"])
        history["val_loss"].append(val_results["total_loss"])
        history["val_auc"].append(val_auc)
        history["val_sensitivity"].append(val_sens)
        history["val_specificity"].append(val_spec)
        history["val_f1"].append(val_f1)
        history["learning_rate"].append(current_lr[0])

        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch + 1

            checkpoint_path = config.CHECKPOINT_DIR / f"best_model_fold{fold_idx+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_auc': val_auc,
                'val_results': {k: v for k, v in val_results.items()
                                if not isinstance(v, np.ndarray)},
                'config': {
                    'backbone': config.MODEL_NAME,
                    'img_size': img_size,
                    'fold': fold_idx + 1,
                },
            }, checkpoint_path)

            best_checkpoint_path = checkpoint_path
            print(f"  * New best AUC: {best_auc:.4f} (saved)")

        # Early stopping
        if early_stopping(val_auc):
            print(f"\n  [!] Early stopping triggered at epoch {epoch + 1}")
            print(f"    Best AUC: {best_auc:.4f} at epoch {best_epoch}")
            break

    # Save history
    history_path = config.RESULTS_DIR / f"history_fold{fold_idx+1}.json"
    history_json = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)

    return {
        "fold": fold_idx + 1,
        "best_auc": best_auc,
        "best_epoch": best_epoch,
        "checkpoint_path": str(best_checkpoint_path),
        "history_path": str(history_path),
    }


# ===========================================================================
# Main Training Function
# ===========================================================================

def train_all_folds(img_size: int = config.IMG_SIZE,
                    num_epochs: int = config.NUM_EPOCHS) -> dict:
    """
    Jalankan training untuk semua K folds.

    Returns:
        dict dengan summary results dari semua folds
    """
    print(f"\n{'#'*70}")
    print(f"  GON DETECTION TRAINING")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Image Size: {img_size}")
    print(f"  Device: {config.DEVICE}")
    print(f"  Folds: {config.NUM_FOLDS}")
    print(f"{'#'*70}")

    # Buat patient-level splits
    splits = create_patient_level_splits()

    # Train setiap fold
    fold_results = []

    for fold_idx, (train_df, val_df) in enumerate(splits):
        result = train_fold(fold_idx, train_df, val_df, img_size, num_epochs)
        fold_results.append(result)

    # Summary
    aucs = [r["best_auc"] for r in fold_results]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    for r in fold_results:
        print(f"  Fold {r['fold']}: AUC={r['best_auc']:.4f} (epoch {r['best_epoch']})")

    # Save summary
    summary = {
        "model": config.MODEL_NAME,
        "img_size": img_size,
        "num_folds": config.NUM_FOLDS,
        "mean_auc": float(mean_auc),
        "std_auc": float(std_auc),
        "fold_results": fold_results,
    }

    summary_path = config.RESULTS_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved to: {summary_path}")

    return summary


if __name__ == "__main__":
    train_all_folds()
