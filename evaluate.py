#!/usr/bin/env python3
import os
import glob
import pandas as pd
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Optional SciPy (preferred)
try:
    from scipy.stats import wasserstein_distance
except Exception:
    wasserstein_distance = None

import cv2
from skimage import filters, measure, morphology


# ============================================================
# IO
# ============================================================

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def list_images(folder):
    paths = []
    for e in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{e}")))
    return sorted(paths)


def load_rgb01(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


# ============================================================
# Sampling utilities
# ============================================================

def sample_flat(x, max_samples, rng):
    x = np.asarray(x).ravel()
    if x.size == 0:
        return x
    n = min(max_samples, x.size)
    idx = rng.choice(x.size, size=n, replace=False)
    return x[idx]


def w1(u, v):
    if u.size == 0 or v.size == 0:
        return np.nan
    if wasserstein_distance is not None:
        return float(wasserstein_distance(u, v))
    qs = np.linspace(0, 1, 257)
    return float(np.mean(np.abs(np.quantile(u, qs) - np.quantile(v, qs))))


def quantile_error(u, v, qs):
    return [float(abs(np.quantile(u, q) - np.quantile(v, q))) for q in qs]


# ============================================================
# Macenko stain deconvolution (H&E)
# ============================================================

def rgb_to_od(x, eps=1e-6):
    return -np.log(np.clip(x, eps, 1.0))


def normalize_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def macenko_he(rgb):
    OD = rgb_to_od(rgb)
    flat = OD.reshape(-1, 3)

    fg = np.all(flat > 0.15, axis=1)
    if fg.sum() < 1000:
        return OD[..., 2], OD[..., 1]

    _, _, Vt = np.linalg.svd(np.cov(flat[fg].T))
    V = Vt[:2]

    proj = flat[fg] @ V.T
    ang = np.arctan2(proj[:, 1], proj[:, 0])
    lo, hi = np.percentile(ang, [1, 99])

    v1 = V.T @ [np.cos(lo), np.sin(lo)]
    v2 = V.T @ [np.cos(hi), np.sin(hi)]
    stains = normalize_rows(np.stack([v1, v2]))

    if stains[0, 2] < stains[1, 2]:
        stains = stains[::-1]

    C = np.linalg.lstsq(stains.T, flat.T, rcond=None)[0]
    H = C[0].reshape(rgb.shape[:2])
    E = C[1].reshape(rgb.shape[:2])
    return H, E


# ============================================================
# Edge structure
# ============================================================

def edge_features(rgb):
    gray = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ori = np.arctan2(gy, gx)
    return mag, ori


# ============================================================
# Nuclei detection (classical)
# ============================================================

def detect_nuclei(rgb):
    b = rgb[..., 2]
    b = cv2.GaussianBlur(b, (5, 5), 0)
    thr = filters.threshold_otsu(b)
    bin_ = b < thr
    bin_ = morphology.remove_small_objects(bin_, 50)
    bin_ = morphology.binary_closing(bin_, morphology.disk(2))
    return measure.label(bin_)


def nuclei_stats(labels):
    props = measure.regionprops(labels)
    return (
        len(props),
        np.array([p.area for p in props], np.float32),
        np.array([p.eccentricity for p in props], np.float32),
    )


# ============================================================
# Main (SET-LEVEL)
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Set-level evaluation for unregistered virtual staining"
    )
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--ref_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=300_000)
    ap.add_argument("--quantiles", default="0.05,0.25,0.5,0.75,0.95")
    ap.add_argument("--no_nuclei", action="store_true")
    ap.add_argument("--out_csv", default="metrics_set_level.csv")
    args = ap.parse_args()

    qs = [float(q) for q in args.quantiles.split(",")]
    rng = np.random.default_rng(0)

    pred_imgs = list_images(args.pred_dir)
    ref_imgs = list_images(args.ref_dir)

    assert pred_imgs and ref_imgs, "Empty pred_dir or ref_dir"

    # ---- pooled distributions ----
    Hp, Ep, Hr, Er = [], [], [], []
    Mp, Mr, Op, Or = [], [], [], []
    Ap, Ar, Xp, Xr = [], [], [], []
    Cp, Cr = [], []

    # ---- predicted set ----
    print('Processing predicted images')
    for p in tqdm(pred_imgs):
        img = load_rgb01(p)
        H, E = macenko_he(img)
        Hp.append(sample_flat(H, args.max_samples, rng))
        Ep.append(sample_flat(E, args.max_samples, rng))

        mag, ori = edge_features(img)
        Mp.append(sample_flat(mag, args.max_samples, rng))
        Op.append(sample_flat(ori, args.max_samples, rng))

        if not args.no_nuclei:
            lab = detect_nuclei(img)
            c, a, x = nuclei_stats(lab)
            Cp.append(c)
            Ap.append(a)
            Xp.append(x)
    # ---- reference set ----
    print('Processing reference images')
    for r in tqdm(ref_imgs):
        img = load_rgb01(r)
        H, E = macenko_he(img)
        Hr.append(sample_flat(H, args.max_samples, rng))
        Er.append(sample_flat(E, args.max_samples, rng))

        mag, ori = edge_features(img)
        Mr.append(sample_flat(mag, args.max_samples, rng))
        Or.append(sample_flat(ori, args.max_samples, rng))

        if not args.no_nuclei:
            lab = detect_nuclei(img)
            c, a, x = nuclei_stats(lab)
            Cr.append(c)
            Ar.append(a)
            Xr.append(x)
    print(Hp)
    Hp, Ep, Hr, Er = map(np.concatenate, [Hp, Ep, Hr, Er])
    Mp, Mr, Op, Or = map(np.concatenate, [Mp, Mr, Op, Or])
    print('############### Hi ###################')
    print(Hp)
    print('############### Hi ###################')
    print(Ep)
    print('############### Hi ###################')
    print(Mp)
    results = {
        "W1_H": w1(Hp, Hr),
        "W1_E": w1(Ep, Er),
        "W1_edge_mag": w1(Mp, Mr),
        "W1_edge_ori": w1(Op, Or),
    }
    print(results)
    for q, e in zip(qs, quantile_error(Hp, Hr, qs)):
        results[f"QErr_H_q{q}"] = e
    for q, e in zip(qs, quantile_error(Ep, Er, qs)):
        results[f"QErr_E_q{q}"] = e
    for q, e in zip(qs, quantile_error(Mp, Mr, qs)):
        results[f"QErr_edge_mag_q{q}"] = e
    for q, e in zip(qs, quantile_error(Op, Or, qs)):
        results[f"QErr_edge_ori_q{q}"] = e
    print(results)
    if not args.no_nuclei:
        Ap, Ar = np.concatenate(Ap), np.concatenate(Ar)
        Xp, Xr = np.concatenate(Xp), np.concatenate(Xr)

        results.update({
            "nuclei_count_mean_pred": np.mean(Cp),
            "nuclei_count_mean_ref": np.mean(Cr),
            "W1_nuclei_area": w1(Ap, Ar),
            "W1_nuclei_ecc": w1(Xp, Xr),
        })
    print(results)
    # with open(args.out_csv, "w", newline="") as f:
    #     w = csv.writer(f)
    #     w.writerow(["metric", "value"])
    #     for k, v in results.items():
    #         w.writerow([k, v])
    pd.DataFrame(results.items(), columns=["metric", "value"]).to_csv(
    args.out_csv, index=False)


    print("Set-level evaluation complete")
    for k, v in results.items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
