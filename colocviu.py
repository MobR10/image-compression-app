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

def load(path):
    return np.array(Image.open(path).convert("RGB"))


def save(path,rgb):
    Image.fromarray(rgb).save(path)

def get_person_mask_tasks(rgb, model_path):
    #return-ul este de forma 0 pentru background, 1 pentru par, 2 pentru corp etc. 
    # Obtinem o persoana daca value-ul este >0 (adica nu este background)
    BaseOptions = mp.tasks.BaseOptions
    ImageSegmenter = mp.tasks.vision.ImageSegmenter
    ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path), #modelul folosit
        running_mode=RunningMode.IMAGE, # IMAGE, VIDEO, LIVESTREAM 
        output_category_mask=True,
    )

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with ImageSegmenter.create_from_options(options) as segmenter:
        result = segmenter.segment(mp_image) #proceseaza segmentarea
        cat = result.category_mask.numpy_view() #ia rezultatul sub forma de masca si il transforma intr-un array

    person = (cat > 0).astype(np.float32)

    person = np.squeeze(person)
    if person.ndim != 2:
        raise ValueError(f"Expected 2D person mask, got shape {person.shape}")

    return person


def box_blur_2d(x, r):
    if r <= 0:
        return x.copy()

    H, W = x.shape
    out = np.zeros((H, W))

    for i in range(H):
        for j in range(W):
            yMin = max(0, i - r)
            yMax = min(H, i + r + 1)
            xMin = max(0, j - r)
            xMax = min(W, j + r + 1)

            window = x[yMin:yMax, xMin:xMax]
            out[i, j] = window.mean()

    return out

def soften_mask(mask, blur_radius):
    m = np.clip(mask, 0.0, 1.0)
    if m.ndim != 2:
        raise ValueError(f"soften_mask expects 2D mask, got shape {m.shape}")
    if blur_radius > 0:
        m = box_blur_2d(m, blur_radius)
    return m

def svd_blur_channel(A, k):
    U, s, Vt = np.linalg.svd(A)
    minDim =  min(k, s.shape[0])
    k = max(1,minDim)
    return (U[:, :k] * s[:k]) @ Vt[:k, :]

def svd_blur_rgb(rgb, k):
    x = rgb.astype(np.float64)/255
    H,W,C = x.shape
    out = np.empty((H,W,C))
    for c in range(3):
        out[:,:,c] = svd_blur_channel(x[:,:, c], k)
    out = np.clip(out,0, 1)
    return (out*255).astype(np.uint8)

def composition(rgb, person_mask, k):
    m = np.clip(person_mask,0,1)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D person_mask, got shape {m.shape}")
    m3 = m[:,:, None]  
    sharp = rgb.astype(np.float64)/255
    blurred = svd_blur_rgb(rgb, k).astype(np.float64)/255

    out = sharp * m3 + blurred * (1- m3)
    out = np.clip(out,0,1)
    return (out * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="output.jpg")
    parser.add_argument("--k", type=int)
    parser.add_argument("--mask_blur", type=int)
    args = parser.parse_args()

    rgb = load(args.input)

    person = get_person_mask_tasks(rgb, args.model)

    if args.mask_blur > 0:
        person = soften_mask(person, blur_radius=args.mask_blur)

    out = composition(rgb, person, k=args.k)
    save(args.output, out)

if __name__ == "__main__":
    main()