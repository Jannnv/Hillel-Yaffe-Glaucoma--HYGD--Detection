"""
=============================================================================
dataset.py — Phase 2: Dataset, Augmentation & Patient-Level Splitting
=============================================================================
Fitur:
  - Patient-level stratified K-Fold split (no data leakage)
  - Albumentations-based augmentation pipeline
  - Class-balanced sampling (oversampling GON-)
  - Quality score sebagai auxiliary target
  - MixUp augmentation
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import StratifiedGroupKFold

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("[WARNING] albumentations not found. Using built-in fallback transforms.")

try:
    from torchvision import transforms as T
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

import config
from preprocessing import (
    preprocess_single_image,
    compute_quality_weights
)


# ===========================================================================
# Pure NumPy/CV2 Fallback Transforms (tanpa dependency external)
# ===========================================================================

class NumpyTransform:
    """
    Fallback transform menggunakan numpy/cv2 murni.
    Dipanggil dengan interface: transform(image=img) -> {"image": tensor}
    """
    def __init__(self, img_size, augment=False):
        self.img_size = img_size
        self.augment = augment
        self.mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)
        self.std = np.array(config.IMAGENET_STD, dtype=np.float32)

    def __call__(self, image=None, **kwargs):
        img = image.copy()

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)

        if self.augment:
            # Random horizontal flip
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()

            # Random vertical flip
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()

            # Random rotation (-30 to 30 degrees)
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-30, 30)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))

            # Random brightness/contrast
            if np.random.rand() < 0.5:
                alpha = np.random.uniform(0.8, 1.2)  # contrast
                beta = np.random.uniform(-25, 25)     # brightness
                img = np.clip(alpha * img.astype(np.float32) + beta,
                              0, 255).astype(np.uint8)

            # Random Gaussian blur
            if np.random.rand() < 0.3:
                ksize = np.random.choice([3, 5])
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

            # Random noise
            if np.random.rand() < 0.2:
                noise = np.random.normal(0, 10, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise,
                              0, 255).astype(np.uint8)

        # Normalize: [0, 255] -> [0, 1] -> ImageNet normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        # To tensor: [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float()

        return {"image": tensor}


# ===========================================================================
# Augmentation Pipelines
# ===========================================================================

def get_train_transforms(img_size: int = config.IMG_SIZE):
    """
    Augmentation pipeline untuk training.
    Menggunakan albumentations jika tersedia, fallback ke torchvision.
    """
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            # Spatial transforms
            A.HorizontalFlip(p=config.AUG_PARAMS["horizontal_flip_p"]),
            A.VerticalFlip(p=config.AUG_PARAMS["vertical_flip_p"]),
            A.Rotate(limit=config.AUG_PARAMS["rotation_limit"], p=0.5,
                     border_mode=cv2.BORDER_CONSTANT, value=0),
            A.ShiftScaleRotate(
                shift_limit=config.AUG_PARAMS["shift_limit"],
                scale_limit=config.AUG_PARAMS["scale_limit"],
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),

            # Color transforms
            A.ColorJitter(
                brightness=config.AUG_PARAMS["brightness_limit"],
                contrast=config.AUG_PARAMS["contrast_limit"],
                saturation=config.AUG_PARAMS["saturation_limit"],
                hue=config.AUG_PARAMS["hue_limit"],
                p=0.5
            ),

            # Blur & noise
            A.OneOf([
                A.GaussianBlur(blur_limit=config.AUG_PARAMS["gaussian_blur_limit"], p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
            ], p=0.3),

            A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),

            # Elastic deformation (fundus-specific)
            A.ElasticTransform(
                alpha=config.AUG_PARAMS["elastic_alpha"],
                sigma=config.AUG_PARAMS["elastic_sigma"],
                p=0.2
            ),

            # Random erasing / cutout
            A.CoarseDropout(
                max_holes=config.AUG_PARAMS["cutout_num_holes"],
                max_height=config.AUG_PARAMS["cutout_max_h_size"],
                max_width=config.AUG_PARAMS["cutout_max_w_size"],
                fill_value=0,
                p=0.3
            ),

            # Resize & normalize
            A.Resize(img_size, img_size),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        # Fallback: pure numpy/cv2 transforms
        return NumpyTransform(img_size, augment=True)


def get_val_transforms(img_size: int = config.IMG_SIZE):
    """Augmentation pipeline untuk validation/test (hanya resize + normalize)."""
    if HAS_ALBUMENTATIONS:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ])
    else:
        return NumpyTransform(img_size, augment=False)


def get_tta_transforms(img_size: int = config.IMG_SIZE) -> list:
    """
    Test-Time Augmentation (TTA) transforms.
    Returns list of transforms untuk multiple predictions.
    """
    if HAS_ALBUMENTATIONS:
        base_norm = [
            A.Resize(img_size, img_size),
            A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
            ToTensorV2(),
        ]

        tta_list = [
            A.Compose(base_norm),  # Original
            A.Compose([A.HorizontalFlip(p=1.0)] + base_norm),
            A.Compose([A.VerticalFlip(p=1.0)] + base_norm),
            A.Compose([A.Rotate(limit=(90, 90), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0)] + base_norm),
            A.Compose([A.Rotate(limit=(-90, -90), p=1.0, border_mode=cv2.BORDER_CONSTANT, value=0)] + base_norm),
        ]
    else:
        # Fallback: gunakan NumpyTransform tanpa augmentation untuk TTA sederhana
        tta_list = [NumpyTransform(img_size, augment=False)] * 5

    return tta_list


# ===========================================================================
# Dataset Class
# ===========================================================================

class HYGDDataset(Dataset):
    """
    HYGD Fundus Dataset untuk GON Detection.

    Multi-task output:
        - GON label (binary: 0 atau 1)
        - Quality score (regression: 1-10)
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 images_dir: Path = config.IMAGES_DIR,
                 transform=None,
                 preprocess: bool = True,
                 img_size: int = config.IMG_SIZE,
                 return_meta: bool = False):
        """
        Args:
            dataframe: DataFrame dengan kolom [Image Name, Patient, Label, Quality Score]
            images_dir: Path ke folder Images
            transform: Albumentations transform
            preprocess: Apakah terapkan preprocessing (CLAHE, mask, crop)
            img_size: Target image size
            return_meta: Apakah return metadata (image name, patient id)
        """
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.preprocess = preprocess
        self.img_size = img_size
        self.return_meta = return_meta

        # Encode label: GON+ = 1, GON- = 0
        self.labels = (self.df["Label"] == "GON+").astype(int).values
        self.quality_scores = self.df["Quality Score"].values.astype(np.float32)
        self.image_names = self.df["Image Name"].values
        self.patient_ids = self.df["Patient"].values

        # Hitung class weights untuk sampling
        n_pos = (self.labels == 1).sum()
        n_neg = (self.labels == 0).sum()
        self.class_weights = {0: len(self.labels) / (2 * n_neg),
                              1: len(self.labels) / (2 * n_pos)}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns: dict with keys:
            - 'image': tensor [C, H, W]
            - 'label': tensor [1] (GON classification)
            - 'quality_score': tensor [1] (quality regression target)
            - 'quality_weight': tensor [1] (sample weight berdasarkan quality)
            - 'image_name': str (jika return_meta=True)
            - 'patient_id': int (jika return_meta=True)
        """
        image_name = self.image_names[idx]
        image_path = self.images_dir / image_name
        label = self.labels[idx]
        quality_score = self.quality_scores[idx]

        # Load & preprocess
        if self.preprocess:
            image = preprocess_single_image(str(image_path), self.img_size)
        else:
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (self.img_size, self.img_size))

        # Apply augmentation
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Manual normalize & to tensor
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)

        # Quality-based sample weight
        quality_weight = quality_score / 10.0  # Normalize ke [0, 1]

        result = {
            "image": image,
            "label": torch.tensor([label], dtype=torch.float32),
            "quality_score": torch.tensor([quality_score / 10.0], dtype=torch.float32),
            "quality_weight": torch.tensor([quality_weight], dtype=torch.float32),
        }

        if self.return_meta:
            result["image_name"] = image_name
            result["patient_id"] = int(self.patient_ids[idx])

        return result


# ===========================================================================
# MixUp Augmentation (dilakukan pada batch level)
# ===========================================================================

def mixup_data(images: torch.Tensor,
               labels: torch.Tensor,
               quality_scores: torch.Tensor,
               alpha: float = config.AUG_PARAMS["mixup_alpha"]):
    """
    MixUp augmentation: mencampur dua gambar secara linear.

    Args:
        images: batch of images [B, C, H, W]
        labels: batch of labels [B, 1]
        quality_scores: batch of quality scores [B, 1]
        alpha: Beta distribution parameter

    Returns:
        mixed_images, mixed_labels, mixed_qs, lambda_val
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size).to(images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_labels = lam * labels + (1 - lam) * labels[index]
    mixed_qs = lam * quality_scores + (1 - lam) * quality_scores[index]

    return mixed_images, mixed_labels, mixed_qs, lam


# ===========================================================================
# Patient-Level Data Splitting
# ===========================================================================

def create_patient_level_splits(labels_csv: Path = config.LABELS_CSV,
                                n_folds: int = config.NUM_FOLDS,
                                quality_threshold: float = config.QUALITY_THRESHOLD,
                                seed: int = config.RANDOM_SEED
                                ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Buat patient-level stratified K-Fold splits.

    KRITIS: Semua gambar dari pasien yang sama HARUS berada
    di fold yang sama untuk menghindari data leakage.

    Args:
        labels_csv: Path ke Labels.csv
        n_folds: Jumlah folds
        quality_threshold: Filter kualitas minimum
        seed: Random seed

    Returns:
        List of (train_df, val_df) tuples, satu per fold
    """
    # Load data
    df = pd.read_csv(labels_csv)
    df.columns = [c.strip() for c in df.columns]

    # Hapus baris kosong
    df = df[df["Image Name"].notna() & (df["Image Name"].str.strip() != "")]

    # Quality filter
    if quality_threshold > 0:
        before = len(df)
        df = df[df["Quality Score"] >= quality_threshold]
        print(f"[SPLIT] Quality filter: {before} -> {len(df)} images")

    # Buat patient-level label (label per pasien, bukan per gambar)
    patient_labels = df.groupby("Patient")["Label"].first().reset_index()
    patient_labels["Label_encoded"] = (patient_labels["Label"] == "GON+").astype(int)

    print(f"\n[SPLIT] Dataset Summary:")
    print(f"  Total images: {len(df)}")
    print(f"  Total patients: {len(patient_labels)}")
    print(f"  GON+ patients: {(patient_labels['Label_encoded'] == 1).sum()}")
    print(f"  GON- patients: {(patient_labels['Label_encoded'] == 0).sum()}")

    # Stratified Group K-Fold
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    groups = df["Patient"].values
    labels = (df["Label"] == "GON+").astype(int).values

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, labels, groups)):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        train_patients = train_df["Patient"].unique()
        val_patients = val_df["Patient"].unique()

        # Verifikasi: tidak ada overlap pasien
        overlap = set(train_patients) & set(val_patients)
        assert len(overlap) == 0, f"Data leakage detected! Overlapping patients: {overlap}"

        train_gon_plus = (train_df["Label"] == "GON+").sum()
        val_gon_plus = (val_df["Label"] == "GON+").sum()

        print(f"\n  Fold {fold_idx + 1}:")
        print(f"    Train: {len(train_df)} images, {len(train_patients)} patients "
              f"(GON+: {train_gon_plus}, GON-: {len(train_df) - train_gon_plus})")
        print(f"    Val:   {len(val_df)} images, {len(val_patients)} patients "
              f"(GON+: {val_gon_plus}, GON-: {len(val_df) - val_gon_plus})")

        splits.append((train_df, val_df))

    return splits


# ===========================================================================
# DataLoader Factory
# ===========================================================================

def create_dataloaders(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       img_size: int = config.IMG_SIZE,
                       batch_size: int = config.BATCH_SIZE,
                       use_weighted_sampler: bool = True
                       ) -> Tuple[DataLoader, DataLoader]:
    """
    Buat train dan validation DataLoaders.

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        img_size: Target image size
        batch_size: Batch size
        use_weighted_sampler: Gunakan WeightedRandomSampler untuk class balance

    Returns:
        (train_loader, val_loader)
    """
    # Buat datasets
    train_dataset = HYGDDataset(
        dataframe=train_df,
        transform=get_train_transforms(img_size),
        preprocess=True,
        img_size=img_size,
    )

    val_dataset = HYGDDataset(
        dataframe=val_df,
        transform=get_val_transforms(img_size),
        preprocess=True,
        img_size=img_size,
        return_meta=True,
    )

    # Weighted sampler untuk class imbalance
    sampler = None
    shuffle = True

    if use_weighted_sampler:
        labels = train_dataset.labels
        class_counts = np.bincount(labels)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]

        # Faktor tambahan: quality-based weighting
        if config.USE_QUALITY_WEIGHTING:
            quality_weights = train_dataset.quality_scores / train_dataset.quality_scores.max()
            sample_weights = sample_weights * quality_weights

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler sudah handle randomization

    # Buat DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    print(f"\n[DATALOADER] Created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Weighted sampler: {use_weighted_sampler}")

    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset dan splits."""
    # Test patient-level splits
    splits = create_patient_level_splits()

    # Test dataset loading
    train_df, val_df = splits[0]
    train_loader, val_loader = create_dataloaders(train_df, val_df, img_size=224, batch_size=4)

    # Test satu batch
    batch = next(iter(train_loader))
    print(f"\n[TEST] Batch shapes:")
    print(f"  Image: {batch['image'].shape}")
    print(f"  Label: {batch['label'].shape} -> {batch['label'].squeeze().tolist()}")
    print(f"  QS:    {batch['quality_score'].shape} -> {batch['quality_score'].squeeze().tolist()}")
    print(f"  QW:    {batch['quality_weight'].shape}")
