"""
=============================================================================
main.py -- Entry Point: GON Detection Pipeline
=============================================================================
Menjalankan seluruh pipeline Phase 1-6:
  Phase 1: Preprocessing (CLAHE, masking, quality filtering)
  Phase 2: Dataset + Augmentation + Patient-level splitting
  Phase 3: Model Architecture (Quality-Aware Multi-Task EfficientNet)
  Phase 4: Training (K-Fold CV, MixUp, AMP, early stopping)
  Phase 5: Evaluation (metrics, Grad-CAM++, clinical report)
  Phase 6: Clinical Interpretability & Explainability

Usage:
  # Full pipeline (semua folds):
  python main.py

  # Hanya preprocessing test:
  python main.py --phase 1

  # Hanya training:
  python main.py --phase 4

  # Hanya evaluasi dari checkpoint:
  python main.py --phase 5 --checkpoint output/checkpoints/best_model_fold1.pth

  # Phase 6: Clinical Interpretability:
  python main.py --phase 6 --checkpoint output/checkpoints/best_model_fold1.pth

  # Custom settings:
  python main.py --img_size 224 --epochs 50 --batch_size 8
"""

import argparse
import sys
import time
import warnings
import numpy as np
from pathlib import Path

import torch

warnings.filterwarnings("ignore", category=UserWarning)

import config


def run_phase1_preprocessing():
    """Phase 1: Test preprocessing pipeline."""
    print("\n" + "="*70)
    print("  PHASE 1: PREPROCESSING PIPELINE")
    print("="*70)

    import pandas as pd
    from preprocessing import (
        preprocess_single_image,
        filter_by_quality,
        compute_quality_weights,
        load_image,
    )

    # Load labels
    df = pd.read_csv(config.LABELS_CSV)
    df.columns = [c.strip() for c in df.columns]

    print(f"\n[Phase 1] Loaded {len(df)} entries from Labels.csv")

    # Quality filtering
    df_filtered = filter_by_quality(df, threshold=config.QUALITY_THRESHOLD)

    # Compute quality weights
    weights = compute_quality_weights(df_filtered["Quality Score"].values)
    print(f"[Phase 1] Quality weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Test preprocessing pada beberapa sampel
    sample_images = ["0_0.jpg", "187_0.jpg", "7_0.jpg", "250_0.jpg"]
    print(f"\n[Phase 1] Testing preprocessing on {len(sample_images)} samples:")

    for img_name in sample_images:
        img_path = config.IMAGES_DIR / img_name
        if img_path.exists():
            try:
                processed = preprocess_single_image(str(img_path), config.IMG_SIZE)
                row = df[df["Image Name"] == img_name]
                label = row["Label"].values[0] if len(row) > 0 else "N/A"
                qs = row["Quality Score"].values[0] if len(row) > 0 else 0
                print(f"  [OK] {img_name}: {processed.shape} | {label} | QS={qs}")
            except Exception as e:
                print(f"  [FAIL] {img_name}: ERROR - {e}")

    print("\n[Phase 1] [DONE] Preprocessing pipeline berhasil!")
    return df_filtered


def run_phase2_dataset():
    """Phase 2: Test dataset & splits."""
    print("\n" + "="*70)
    print("  PHASE 2: DATASET & AUGMENTATION")
    print("="*70)

    from dataset import (
        create_patient_level_splits,
        create_dataloaders,
        get_train_transforms,
        get_val_transforms,
    )

    # Patient-level splits
    splits = create_patient_level_splits()

    # Test first fold
    train_df, val_df = splits[0]

    print(f"\n[Phase 2] Testing DataLoader (Fold 1)...")
    train_loader, val_loader = create_dataloaders(
        train_df, val_df,
        img_size=config.IMG_SIZE_SMALL,  # Gunakan size kecil untuk test
        batch_size=4
    )

    # Load satu batch
    batch = next(iter(train_loader))
    print(f"\n[Phase 2] Sample batch:")
    print(f"  Image shape:    {batch['image'].shape}")
    print(f"  Label shape:    {batch['label'].shape}")
    print(f"  Labels:         {batch['label'].squeeze().tolist()}")
    print(f"  Quality scores: {batch['quality_score'].squeeze().tolist()}")
    print(f"  Quality weights:{batch['quality_weight'].squeeze().tolist()}")

    print("\n[Phase 2] [DONE] Dataset & augmentation pipeline berhasil!")
    return splits


def run_phase3_model():
    """Phase 3: Test model architecture."""
    print("\n" + "="*70)
    print("  PHASE 3: MODEL ARCHITECTURE")
    print("="*70)

    from model import create_model, QualityWeightedMultiTaskLoss

    # Create model
    model = create_model()
    model = model.to(config.DEVICE)
    model.eval()

    # Dummy forward pass
    x = torch.randn(2, 3, config.IMG_SIZE_SMALL, config.IMG_SIZE_SMALL).to(config.DEVICE)
    qs = torch.rand(2, 1).to(config.DEVICE)

    with torch.no_grad():
        output = model(x, qs)

    print(f"\n[Phase 3] Forward pass test:")
    print(f"  Input:         {x.shape}")
    print(f"  GON logits:    {output['gon_logits'].shape}")
    print(f"  GON probs:     {output['gon_probs'].squeeze().tolist()}")
    print(f"  Quality pred:  {output['quality_pred'].squeeze().tolist()}")
    print(f"  Features:      {output['features'].shape}")

    # Test loss
    criterion = QualityWeightedMultiTaskLoss().to(config.DEVICE)
    labels = torch.randint(0, 2, (2, 1)).float().to(config.DEVICE)
    losses = criterion(output['gon_logits'], labels,
                       output['quality_pred'], qs, qs)

    print(f"\n[Phase 3] Loss test:")
    print(f"  Total loss:    {losses['total_loss'].item():.4f}")
    print(f"  GON loss:      {losses['gon_loss'].item():.4f}")
    print(f"  Quality loss:  {losses['quality_loss'].item():.4f}")

    print("\n[Phase 3] [DONE] Model architecture berhasil!")
    return model


def run_phase4_training(img_size: int = config.IMG_SIZE,
                         num_epochs: int = config.NUM_EPOCHS):
    """Phase 4: Full training pipeline."""
    print("\n" + "="*70)
    print("  PHASE 4: TRAINING PIPELINE")
    print("="*70)

    from train import train_all_folds

    summary = train_all_folds(img_size=img_size, num_epochs=num_epochs)

    print("\n[Phase 4] [DONE] Training pipeline selesai!")
    return summary


def run_phase5_evaluation(checkpoint_path: str = None,
                           fold: int = 1):
    """Phase 5: Evaluation & explainability."""
    print("\n" + "="*70)
    print("  PHASE 5: EVALUATION & EXPLAINABILITY")
    print("="*70)

    from model import create_model
    from dataset import create_patient_level_splits, create_dataloaders
    from evaluate import (
        evaluate_fold,
        generate_gradcam_visualization,
        calibrate_model,
    )
    from preprocessing import preprocess_single_image

    # Load model dari checkpoint
    if checkpoint_path is None:
        checkpoint_path = config.CHECKPOINT_DIR / f"best_model_fold{fold}.pth"

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
        print("[INFO] Jalankan Phase 4 (training) terlebih dahulu.")
        return None

    print(f"[Phase 5] Loading checkpoint: {checkpoint_path}")

    # Create model & load weights
    model = create_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    model.eval()

    print(f"[Phase 5] Model loaded (epoch {checkpoint['epoch']}, "
          f"val_auc={checkpoint.get('val_auc', 'N/A')})")

    # Create validation set
    splits = create_patient_level_splits()
    _, val_df = splits[fold - 1]
    _, val_loader = create_dataloaders(
        splits[fold - 1][0], val_df, batch_size=config.BATCH_SIZE
    )

    # Full evaluation
    results = evaluate_fold(model, val_loader, fold, save_plots=True)

    # Confidence calibration
    print(f"\n[Phase 5] Calibrating model confidence...")
    temp_scaling = calibrate_model(model, val_loader)

    # Grad-CAM++ pada beberapa sampel
    print(f"\n[Phase 5] Generating Grad-CAM++ visualizations...")
    sample_images = val_df.head(min(config.GRADCAM_NUM_SAMPLES, len(val_df)))

    for _, row in sample_images.iterrows():
        img_name = row["Image Name"]
        img_path = config.IMAGES_DIR / img_name
        qs = row["Quality Score"] / 10.0  # Normalize

        if img_path.exists():
            try:
                img = preprocess_single_image(str(img_path), config.IMG_SIZE)
                save_path = generate_gradcam_visualization(
                    model, img, qs, img_name
                )
                if save_path:
                    print(f"  [OK] Grad-CAM++: {img_name} -> {save_path}")
            except Exception as e:
                print(f"  [FAIL] Grad-CAM++ error for {img_name}: {e}")

    print(f"\n[Phase 5] [DONE] Evaluation & explainability selesai!")
    print(f"  Results saved to: {config.RESULTS_DIR}")
    print(f"  Grad-CAM++ saved to: {config.GRADCAM_DIR}")

    return results


def run_phase6_interpretability(checkpoint_path: str = None,
                                fold: int = 1,
                                gradcam_samples: int = 20,
                                shap_samples: int = 10):
    """Phase 6: Clinical Interpretability & Explainability."""
    print("\n" + "="*70)
    print("  PHASE 6: CLINICAL INTERPRETABILITY & EXPLAINABILITY")
    print("="*70)

    from clinical_interpretability import run_phase6

    results = run_phase6(
        checkpoint_path=checkpoint_path,
        fold=fold,
        num_gradcam_samples=gradcam_samples,
        num_shap_samples=shap_samples,
    )

    print("\n[Phase 6] [DONE] Clinical Interpretability selesai!")
    return results


# ===========================================================================
# Argument Parser
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="GON Detection Pipeline - HYGD Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Run semua phase
  python main.py --phase 1                 # Hanya preprocessing test
  python main.py --phase 4 --epochs 50     # Training 50 epochs
  python main.py --phase 5 --fold 1        # Evaluasi fold 1
  python main.py --phase 6 --fold 1        # Clinical interpretability
  python main.py --img_size 224            # Gunakan resolusi kecil
        """
    )

    parser.add_argument("--phase", type=int, default=0,
                        choices=[0, 1, 2, 3, 4, 5, 6],
                        help="Phase yang akan dijalankan (0=semua)")
    parser.add_argument("--img_size", type=int, default=config.IMG_SIZE,
                        help=f"Image size (default: {config.IMG_SIZE})")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS,
                        help=f"Max epochs (default: {config.NUM_EPOCHS})")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE,
                        help=f"Batch size (default: {config.BATCH_SIZE})")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path ke checkpoint untuk evaluasi")
    parser.add_argument("--fold", type=int, default=1,
                        help="Fold untuk evaluasi (default: 1)")
    parser.add_argument("--gradcam_samples", type=int, default=20,
                        help="Jumlah sampel Grad-CAM++ Phase 6 (default: 20)")
    parser.add_argument("--shap_samples", type=int, default=10,
                        help="Jumlah sampel SHAP Phase 6 (default: 10)")

    return parser.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # Update config dengan args
    config.IMG_SIZE = args.img_size
    config.BATCH_SIZE = args.batch_size

    start_time = time.time()

    print("\n" + "#"*70)
    print("  [GON DETECTION PIPELINE]")
    print("  Hillel Yaffe Glaucoma Dataset (HYGD)")
    print("  Quality-Aware Multi-Task Deep Learning")
    print("#"*70)
    print(f"\n  Device: {config.DEVICE}")
    print(f"  Image Size: {args.img_size}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Max Epochs: {args.epochs}")

    if args.phase == 0 or args.phase == 1:
        run_phase1_preprocessing()

    if args.phase == 0 or args.phase == 2:
        run_phase2_dataset()

    if args.phase == 0 or args.phase == 3:
        run_phase3_model()

    if args.phase == 0 or args.phase == 4:
        run_phase4_training(img_size=args.img_size, num_epochs=args.epochs)

    if args.phase == 0 or args.phase == 5:
        run_phase5_evaluation(checkpoint_path=args.checkpoint, fold=args.fold)

    if args.phase == 0 or args.phase == 6:
        run_phase6_interpretability(
            checkpoint_path=args.checkpoint,
            fold=args.fold,
            gradcam_samples=args.gradcam_samples,
            shap_samples=args.shap_samples,
        )

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  Total waktu: {total_time/60:.1f} menit")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
