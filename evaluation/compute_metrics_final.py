"""
Cálculo de métricas Dice y clDice para experimentos nnUNet
Compara predicciones del fold_0 contra ground truth en labelsTr

Uso:
    python compute_metrics.py

Requiere:
    pip install nibabel numpy scipy pandas tqdm
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import convolve
from tqdm import tqdm

# ─────────────────────────────────────────────
# RUTAS — ajusta si es necesario
# ─────────────────────────────────────────────
GT_DIR = "/path/to/project/nnUnet_raw/Dataset002_Imagecas/labelsTr"

EXPERIMENTS = {
    "Baseline": "/path/to/project/nnUNet_results/Dataset002_Imagecas/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/validation",
    "Exp1_clDice_w020": "/path/to/project/nnUNet_results/Dataset002_Imagecas/nnUNetTrainerBaseclDice_w020__nnUNetPlans__3d_fullres/fold_0/validation",
    "Exp2_clCE": "/path/to/project/nnUNet_results/Dataset002_Imagecas/nnUNetTrainerBaseclCE__nnUNetPlans__3d_fullres/fold_0/validation",
}

OUTPUT_CSV = "metrics_fold0.csv"
OUTPUT_SUMMARY_CSV = "metrics_fold0_summary.csv"

# ─────────────────────────────────────────────
# ESQUELETIZACIÓN SUAVE (soft-skeleton 3D)
# Basada en: https://arxiv.org/abs/2003.07311
# ─────────────────────────────────────────────

def soft_erode_3d(img):
    """Erosión morfológica suave en 3D."""
    p1 = -np.inf * np.ones_like(img)
    p2 = -np.inf * np.ones_like(img)
    p3 = -np.inf * np.ones_like(img)

    # Kernel para cada eje
    kernel_z = np.zeros((3, 1, 1)); kernel_z[:, 0, 0] = [0, 1, 0]
    kernel_y = np.zeros((1, 3, 1)); kernel_y[0, :, 0] = [0, 1, 0]
    kernel_x = np.zeros((1, 1, 3)); kernel_x[0, 0, :] = [0, 1, 0]

    # Mínimo local en cada eje
    for k in [kernel_z, kernel_y, kernel_x]:
        padded = np.pad(img.astype(float), 1, mode='edge')
        shifted_pos = convolve(padded, k)[1:-1, 1:-1, 1:-1]
        shifted_neg = convolve(padded, k[::-1, ::-1, ::-1])[1:-1, 1:-1, 1:-1]
        p1 = np.maximum(p1, np.minimum(shifted_pos, shifted_neg))

    # Alternativa: usar scipy minimum_filter
    from scipy.ndimage import minimum_filter
    eroded = minimum_filter(img.astype(float), size=3)
    return eroded


def soft_dilate_3d(img):
    """Dilatación morfológica suave en 3D."""
    from scipy.ndimage import maximum_filter
    return maximum_filter(img.astype(float), size=3)


def soft_open_3d(img):
    return soft_dilate_3d(soft_erode_3d(img))


def soft_skel_3d(img, iterations=10):
    """
    Esqueletización suave iterativa en 3D.
    Implementación simplificada basada en operaciones morfológicas.
    """
    img = img.astype(float)
    skel = np.zeros_like(img)
    delta = img.copy()

    for _ in range(iterations):
        opened = soft_open_3d(delta)
        temp = np.clip(delta - opened, 0, 1)
        skel = np.clip(skel + temp, 0, 1)
        delta = soft_erode_3d(delta)
        if delta.max() == 0:
            break

    return skel


# ─────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────

def dice_score(pred, gt):
    """Dice clásico (binario)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0  # ambos vacíos → perfecto
    return 2.0 * intersection / denom


def cl_dice_score(pred, gt, iterations=10):
    """
    clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)
    donde:
        Tprec  = |skel(pred) ∩ gt|   / |skel(pred)|
        Tsens  = |skel(gt)  ∩ pred|  / |skel(gt)|
    """
    pred_f = pred.astype(float)
    gt_f = gt.astype(float)

    skel_pred = soft_skel_3d(pred_f, iterations=iterations)
    skel_gt = soft_skel_3d(gt_f, iterations=iterations)

    # Precisión de topología
    num_prec = (skel_pred * gt_f).sum()
    denom_prec = skel_pred.sum()
    tprec = num_prec / denom_prec if denom_prec > 0 else 0.0

    # Sensibilidad de topología
    num_sens = (skel_gt * pred_f).sum()
    denom_sens = skel_gt.sum()
    tsens = num_sens / denom_sens if denom_sens > 0 else 0.0

    if tprec + tsens == 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────

def load_mask(path):
    """Carga un .nii.gz y devuelve array binario."""
    img = nib.load(path)
    data = img.get_fdata()
    return (data > 0.5).astype(np.uint8)


def find_gt_file(case_name, gt_dir):
    """
    Busca el ground truth correspondiente al caso.
    nnUNet nombra predicciones igual que los labels (mismo stem).
    """
    # Intenta coincidencia exacta primero
    for fname in os.listdir(gt_dir):
        if not fname.endswith(".nii.gz"):
            continue
        gt_stem = fname.replace(".nii.gz", "")
        pred_stem = case_name.replace(".nii.gz", "")
        if gt_stem == pred_stem:
            return os.path.join(gt_dir, fname)
    return None


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────

def main():
    all_rows = []

    for exp_name, pred_dir in EXPERIMENTS.items():
        if not os.path.isdir(pred_dir):
            print(f"[AVISO] No se encontró la carpeta: {pred_dir}")
            continue

        pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])
        print(f"\n{'='*60}")
        print(f"Experimento: {exp_name}  ({len(pred_files)} casos)")
        print(f"{'='*60}")

        for pred_fname in tqdm(pred_files, desc=exp_name):
            pred_path = os.path.join(pred_dir, pred_fname)
            gt_path = find_gt_file(pred_fname, GT_DIR)

            if gt_path is None:
                print(f"  [OMITIDO] Sin GT para: {pred_fname}")
                continue

            pred_mask = load_mask(pred_path)
            gt_mask = load_mask(gt_path)

            d = dice_score(pred_mask, gt_mask)
            cld = cl_dice_score(pred_mask, gt_mask)

            all_rows.append({
                "experiment": exp_name,
                "case": pred_fname.replace(".nii.gz", ""),
                "dice": round(d, 6),
                "cl_dice": round(cld, 6),
            })

            tqdm.write(f"  {pred_fname:40s}  Dice={d:.4f}  clDice={cld:.4f}")

    if not all_rows:
        print("\n[ERROR] No se procesó ningún caso. Revisa las rutas.")
        return

    # ── Guardar CSV detallado
    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResultados guardados en: {OUTPUT_CSV}")

    # ── Guardar resumen por experimento
    summary = (
        df.groupby("experiment")[["dice", "cl_dice"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["n_cases"] = df.groupby("experiment")["case"].count()
    summary.to_csv(OUTPUT_SUMMARY_CSV)
    print(f"Resumen guardado en:    {OUTPUT_SUMMARY_CSV}")

    # ── Imprimir resumen en consola
    print("\n" + "="*60)
    print("RESUMEN POR EXPERIMENTO")
    print("="*60)
    print(summary.to_string())


if __name__ == "__main__":
    main()
