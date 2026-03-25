"""
=============================================================================
preprocessing.py — Phase 1: Image Preprocessing Pipeline
=============================================================================
Pipeline:
  1. Load & decode image
  2. Circular mask (remove black borders)
  3. CLAHE enhancement (green channel)
  4. Resize ke target size
  5. Quality-based filtering/weighting
"""

import cv2
import numpy as np
from pathlib import Path

import config


def load_image(image_path: str) -> np.ndarray:
    """Load gambar fundus dari path."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Tidak dapat membaca gambar: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def apply_circular_mask(image: np.ndarray) -> np.ndarray:
    """
    Terapkan circular mask untuk menghilangkan area hitam
    di sekitar fundus image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    radius = min(h, w) // 2

    # Buat mask lingkaran
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)

    # Terapkan mask
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def apply_clahe(image: np.ndarray,
                clip_limit: float = config.CLAHE_CLIP_LIMIT,
                grid_size: tuple = config.CLAHE_GRID_SIZE) -> np.ndarray:
    """
    Terapkan CLAHE (Contrast Limited Adaptive Histogram Equalization)
    pada green channel untuk meningkatkan kontras struktur retina.

    Green channel dipilih karena menampilkan struktur retina
    (pembuluh darah, optic disc) dengan kontras terbaik.
    """
    # Split channels
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Terapkan CLAHE pada green channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    g_enhanced = clahe.apply(g)

    # Gabungkan kembali
    result = np.stack([r, g_enhanced, b], axis=2)
    return result


def resize_image(image: np.ndarray,
                 target_size: int = config.IMG_SIZE) -> np.ndarray:
    """Resize gambar ke target_size × target_size."""
    resized = cv2.resize(image, (target_size, target_size),
                         interpolation=cv2.INTER_LANCZOS4)
    return resized


def crop_to_fundus(image: np.ndarray, threshold: int = 10) -> np.ndarray:
    """
    Crop gambar ke bounding box dari area fundus (non-hitam),
    kemudian pad ke aspect ratio 1:1.
    """
    # Konversi ke grayscale untuk deteksi area non-hitam
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Threshold untuk menemukan area fundus
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # Temukan contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    # Ambil contour terbesar (fundus)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop
    cropped = image[y:y+h, x:x+w]

    # Pad ke aspect ratio 1:1
    max_dim = max(w, h)
    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    pad_y = (max_dim - h) // 2
    pad_x = (max_dim - w) // 2
    padded[pad_y:pad_y+h, pad_x:pad_x+w] = cropped

    return padded


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalisasi gambar ke [0, 1] range."""
    return image.astype(np.float32) / 255.0


def ben_graham_preprocessing(image: np.ndarray,
                              sigma: int = 30) -> np.ndarray:
    """
    Ben Graham's preprocessing untuk fundus images:
    Menghilangkan variasi pencahayaan lokal.

    Formula: result = α * image + β * gaussian_blur + γ
    """
    # Gaussian blur untuk estimasi background illumination
    blur = cv2.GaussianBlur(image, (0, 0), sigma)

    # Subtract background + add constant
    result = cv2.addWeighted(image, 4, blur, -4, 128)

    return result


def preprocess_single_image(image_path: str,
                            target_size: int = config.IMG_SIZE,
                            apply_clahe_flag: bool = True,
                            apply_ben_graham: bool = False,
                            apply_mask: bool = True) -> np.ndarray:
    """
    Pipeline preprocessing lengkap untuk satu gambar fundus.

    Args:
        image_path: Path ke file gambar
        target_size: Ukuran output (target_size × target_size)
        apply_clahe_flag: Apakah terapkan CLAHE
        apply_ben_graham: Apakah terapkan Ben Graham preprocessing
        apply_mask: Apakah terapkan circular mask

    Returns:
        Preprocessed image (numpy array, uint8, RGB)
    """
    # 1. Load image
    img = load_image(image_path)

    # 2. Crop ke area fundus
    img = crop_to_fundus(img)

    # 3. Circular mask (opsional)
    if apply_mask:
        img = apply_circular_mask(img)

    # 4. CLAHE enhancement (opsional)
    if apply_clahe_flag:
        img = apply_clahe(img)

    # 5. Ben Graham preprocessing (opsional, alternatif CLAHE)
    if apply_ben_graham:
        img = ben_graham_preprocessing(img)

    # 6. Resize
    img = resize_image(img, target_size)

    return img


def compute_quality_weights(quality_scores: list,
                            normalize: bool = True) -> np.ndarray:
    """
    Hitung sample weights berdasarkan quality scores.

    Gambar dengan kualitas tinggi mendapat bobot lebih besar.

    Args:
        quality_scores: List of quality scores
        normalize: Apakah normalize ke [0, 1]

    Returns:
        Array of weights
    """
    scores = np.array(quality_scores, dtype=np.float32)

    if normalize:
        weights = scores / scores.max()
    else:
        weights = scores

    return weights


def filter_by_quality(labels_df,
                      threshold: float = config.QUALITY_THRESHOLD):
    """
    Filter gambar berdasarkan quality score minimum.

    Args:
        labels_df: DataFrame dengan kolom 'Quality Score'
        threshold: Minimum quality score

    Returns:
        Filtered DataFrame
    """
    before_count = len(labels_df)
    filtered = labels_df[labels_df["Quality Score"] >= threshold].copy()
    after_count = len(filtered)

    print(f"[PREPROCESSING] Quality filter (>= {threshold}):")
    print(f"  Before: {before_count} images")
    print(f"  After:  {after_count} images")
    print(f"  Removed: {before_count - after_count} images")

    return filtered


# ===========================================================================
# Preprocessing untuk seluruh dataset (batch)
# ===========================================================================
def preprocess_dataset(labels_df,
                       images_dir: Path = config.IMAGES_DIR,
                       target_size: int = config.IMG_SIZE,
                       apply_quality_filter: bool = True):
    """
    Preprocess seluruh dataset.

    Args:
        labels_df: DataFrame dari Labels.csv
        images_dir: Direktori gambar
        target_size: Target size

    Returns:
        Tuple of (preprocessed_images_dict, filtered_labels_df)
    """
    # Quality filter
    if apply_quality_filter:
        labels_df = filter_by_quality(labels_df)

    preprocessed = {}
    errors = []

    for idx, row in labels_df.iterrows():
        image_name = row["Image Name"]
        image_path = images_dir / image_name

        try:
            img = preprocess_single_image(str(image_path), target_size)
            preprocessed[image_name] = img
        except Exception as e:
            errors.append((image_name, str(e)))
            print(f"[ERROR] {image_name}: {e}")

    print(f"\n[PREPROCESSING] Complete:")
    print(f"  Processed: {len(preprocessed)} images")
    print(f"  Errors: {len(errors)} images")

    return preprocessed, labels_df


if __name__ == "__main__":
    """Test preprocessing pada beberapa sampel."""
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load labels
    df = pd.read_csv(config.LABELS_CSV)
    df.columns = [c.strip() for c in df.columns]

    # Test pada beberapa sampel
    samples = ["0_0.jpg", "187_0.jpg", "139_1.jpg", "193_0.jpg"]

    fig, axes = plt.subplots(2, len(samples), figsize=(20, 10))

    for i, name in enumerate(samples):
        path = config.IMAGES_DIR / name
        row = df[df["Image Name"] == name].iloc[0]

        # Original
        original = load_image(str(path))
        original = resize_image(original, 512)
        axes[0, i].imshow(original)
        axes[0, i].set_title(f"Original\n{name} | {row['Label']} | QS={row['Quality Score']}")
        axes[0, i].axis("off")

        # Preprocessed
        processed = preprocess_single_image(str(path), 512)
        axes[1, i].imshow(processed)
        axes[1, i].set_title(f"Preprocessed (CLAHE)")
        axes[1, i].axis("off")

    plt.suptitle("Phase 1: Preprocessing Pipeline", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "preprocessing_samples.png", dpi=150)
    plt.show()
    print("[DONE] Preprocessing samples saved.")
