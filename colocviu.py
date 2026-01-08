"""
Background blur (SVD low-rank) + sharp person using MediaPipe Tasks API (mediapipe 0.10.30/0.10.31).

Install (in your venv):
  pip install mediapipe pillow numpy

You ALSO need a MediaPipe image segmentation model file (.tflite), e.g.:
  selfie_multiclass_256x256.tflite

Example run (PowerShell):
  python F:\Facultate\MN\colocviu.py `
    --input F:\Facultate\MN\input.jpg `
    --model F:\Facultate\MN\models\selfie_multiclass_256x256.tflite `
    --output F:\Facultate\MN\output.jpg `
    --k 40 --mask_blur 3
"""

import argparse
import numpy as np
from PIL import Image
import mediapipe as mp

def load(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def save(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(rgb).save(path)


# -----------------------------
# MediaPipe Tasks: person mask
# -----------------------------
def get_person_mask_tasks(rgb: np.ndarray, model_path: str) -> np.ndarray:
    """
    Returns:
      person_mask: HxW float32 in {0,1} (1 = person, 0 = background)

    Notes:
      - Uses output_category_mask=True => per-pixel category ID (uint8).
      - For the common "selfie_multiclass" model:
          0 = background
          1 = hair
          2 = body
          3 = face
        So we treat person as (cat > 0).
    """
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        output_category_mask=True,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image)
        cat = result.category_mask.numpy_view()  # usually HxW uint8

    # Person = anything not background
    person = (cat > 0).astype(np.float32)

    # Ensure 2D mask (HxW), just in case the backend returns HxWx1
    person = np.squeeze(person)
    if person.ndim != 2:
        raise ValueError(f"Expected 2D person mask, got shape {person.shape}")

    return person


def box_blur_2d(x: np.ndarray, r: int) -> np.ndarray:
    """
    Correct box blur using an integral image.
    x: HxW float
    r: radius (window size = 2r+1)
    """
    if r <= 0:
        return x
    if x.ndim != 2:
        raise ValueError(f"box_blur_2d expects 2D array, got shape {x.shape}")

    H, W = x.shape
    k = 2 * r + 1

    # Pad by edge values
    xp = np.pad(x, ((r, r), (r, r)), mode="edge")

    # Integral image with an extra zero row/col so indexing is clean
    ii = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    ii = np.pad(ii, ((1, 0), (1, 0)), mode="constant", constant_values=0)

    # Window sums for each output pixel (H x W)
    y0, y1 = 0, H
    x0, x1 = 0, W

    # Using the summed-area table:
    # sum = ii[y+k, x+k] - ii[y, x+k] - ii[y+k, x] + ii[y, x]
    out = (
        ii[y0 + k : y1 + k, x0 + k : x1 + k]
        - ii[y0 : y1,         x0 + k : x1 + k]
        - ii[y0 + k : y1 + k, x0 : x1]
        + ii[y0 : y1,         x0 : x1]
    )

    return out / (k * k)


def soften_mask(mask: np.ndarray, blur_radius: int = 3) -> np.ndarray:
    """Soft edges for nicer compositing."""
    m = np.clip(np.squeeze(mask).astype(np.float32), 0.0, 1.0)
    if m.ndim != 2:
        raise ValueError(f"soften_mask expects 2D mask, got shape {m.shape}")
    if blur_radius > 0:
        m = box_blur_2d(m, blur_radius)
    return np.clip(m, 0.0, 1.0)

# -----------------------------
# SVD blur (low-rank approximation)
# -----------------------------
def svd_blur_channel(A: np.ndarray, k: int) -> np.ndarray:
    """
    A: HxW float in [0,1]
    Returns low-rank approximation (blur-ish)
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k = max(1, min(k, s.shape[0]))
    return (U[:, :k] * s[:k]) @ Vt[:k, :]


def svd_blur_rgb(rgb: np.ndarray, k: int) -> np.ndarray:
    x = rgb.astype(np.float64) / 255.0
    out = np.empty_like(x)
    for c in range(3):
        out[..., c] = svd_blur_channel(x[..., c], k)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# -----------------------------
# Composite: sharp person + blurred background
# -----------------------------
def composite_person_over_blurred_bg(rgb: np.ndarray, person_mask: np.ndarray, k: int) -> np.ndarray:
    """
    rgb: HxWx3 uint8
    person_mask: HxW float 0..1 (soft mask ok)
    """
    m = np.clip(np.squeeze(person_mask), 0.0, 1.0)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D person_mask, got shape {m.shape}")

    m3 = m[..., None]  # HxWx1
    sharp = rgb.astype(np.float64) / 255.0
    blurred = svd_blur_rgb(rgb, k=k).astype(np.float64) / 255.0

    out = sharp * m3 + blurred * (1.0 - m3)
    out = np.clip(out, 0.0, 1.0)
    return (out * 255).astype(np.uint8)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input image (e.g., MN/input.jpg)")
    parser.add_argument("--model", required=True, help="Path to segmentation model .tflite")
    parser.add_argument("--output", default="output.jpg", help="Output image path.")
    parser.add_argument("--k", type=int, default=40, help="SVD rank (smaller = blurrier). Try 15..80.")
    parser.add_argument("--mask_blur", type=int, default=3, help="Mask edge softening radius (0 = off).")
    args = parser.parse_args()

    rgb = load(args.input)

    # Person mask via MediaPipe Tasks
    person = get_person_mask_tasks(rgb, args.model)

    # Optional: soften edges (looks better on hair)
    if args.mask_blur > 0:
        person = soften_mask(person, blur_radius=args.mask_blur)

    out = composite_person_over_blurred_bg(rgb, person, k=args.k)
    save(args.output, out)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
