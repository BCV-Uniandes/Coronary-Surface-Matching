"""
Analisis de sensibilidad y tests controlados de la metrica de superficie.
Responde a los tres reviewers de STACOM 2026:
  - TF63 : tests controlados + analisis de sensibilidad de la metrica.
  - br76 : sensibilidad a la tolerancia (radio) y al muestreo de superficie.
  - gbam : dependencia del numero de puntos / muestreo.

Conectado a la implementacion real:
  contour_extraction.contour_descriptor / subsample / local_radius
  hungarian_metric.match_contours_full

Idea (perturbaciones controladas sobre datos REALES)
----------------------------------------------------
Tomamos una mascara de referencia real y la degradamos de forma CONTROLADA,
donde conocemos la respuesta esperada de la metrica. Si la metrica responde
monotona y en la direccion correcta, queda validada sobre anatomia real.

Experimentos
------------
E1 identidad     : pred = ref            -> F1=1, FP=FN=0 (sanity).
E2 barrido FN    : borrar ramas por radio -> Recall baja, FN sube, Prec ~const.
E3 barrido FP    : dilatar / masa espuria -> Prec baja, FP sube, Recall ~const.
E4 tolerancia    : escalar radio x[0.5..2] sobre las MISMAS preds de cada metodo
                   -> valores cambian pero el RANKING entre metodos se preserva.
E5 muestreo      : submuestrear puntos de superficie [100..25%] -> F1 estable.

Uso
---
# Tests controlados E1-E3 (solo necesitan las mascaras de referencia):
python metric_sensitivity.py --gt GT_DIR --experiment controlled --out controlled.csv

# Sensibilidad a tolerancia y muestreo E4-E5 (necesitan predicciones de un metodo):
python metric_sensitivity.py --gt GT_DIR --pred PRED_DIR --experiment tolerance --out tol.csv
python metric_sensitivity.py --gt GT_DIR --pred PRED_DIR --experiment sampling  --out samp.csv

# E4 multi-metodo (ranking): pasar varias carpetas separadas por coma con etiquetas
python metric_sensitivity.py --gt GT_DIR \
       --pred_multi "Ours=PRED_OURS,cbDice=PRED_CBDICE,clDice=PRED_CLDICE,Dice+CE=PRED_BASE" \
       --experiment tolerance_ranking --out tol_rank.csv

GT_DIR = nnUNet_raw/Dataset002_Imagecas/labelsTs (las etiquetas de test).
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_dilation, distance_transform_edt

# --- implementacion real de la metrica ---
from contour_extraction import contour_descriptor, subsample, extract_surface_points, local_radius
from hungarian_metric import match_contours_full


_STRUCT = np.ones((3, 3, 3), np.uint8)


def _load(path):
    """Carga .nii.gz -> (uint8 mask, spacing) igual que evaluate_contour._load."""
    img = nib.load(path)
    arr = (np.asanyarray(img.dataobj) > 0.5).astype(np.uint8)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing


# ============================================================ perturbaciones

def delete_thin_branches(mask, radius_thresh, spacing):
    """Borra voxeles con radio local (EDT) < radius_thresh (mm): quita ramas finas
    primero. Sube FN de forma controlada."""
    edt = distance_transform_edt(mask, sampling=spacing)
    out = mask.copy()
    out[edt < radius_thresh] = 0
    return out


def dilate_surface(mask, iters):
    """Dilata la mascara: masa espuria pegada al vaso (over-segmentation). Sube FP."""
    if iters <= 0:
        return mask.copy()
    return binary_dilation(mask, structure=_STRUCT, iterations=iters).astype(np.uint8)


# ============================================================ metrica (envoltura)

def metric_from_masks(gt_mask, pred_mask, spacing, radius_scale=1.0):
    """P/R/F1/FP/FN entre dos mascaras, usando tu metrica real.
    radius_scale escala la tolerancia por-punto (para E4)."""
    gp, gn, grad = contour_descriptor(gt_mask, max_points=None,
                                      with_radius=True, spacing=spacing)
    pp, pn = contour_descriptor(pred_mask, max_points=None, spacing=spacing)
    if radius_scale != 1.0:
        grad = grad * float(radius_scale)          # escala la tolerancia
    return match_contours_full(gp, pp, grad)


def metric_with_sampling(gt_mask, pred_mask, spacing, frac, rng):
    """Como metric_from_masks pero submuestrea AMBAS nubes a una fraccion (E5)."""
    gp, gn, grad = contour_descriptor(gt_mask, max_points=None,
                                      with_radius=True, spacing=spacing)
    pp, pn = contour_descriptor(pred_mask, max_points=None, spacing=spacing)
    if frac < 1.0:
        kg = max(1, int(len(gp) * frac))
        kp = max(1, int(len(pp) * frac))
        gi = rng.choice(len(gp), size=kg, replace=False)
        pi = rng.choice(len(pp), size=kp, replace=False)
        gp, grad = gp[gi], grad[gi]
        pp = pp[pi]
    return match_contours_full(gp, pp, grad)


# ============================================================ experimentos

def exp_identity(gt, spacing):
    r = metric_from_masks(gt, gt, spacing)
    ok = (abs(r["f1"] - 1.0) < 1e-6 and r["n_fp"] == 0 and r["n_fn"] == 0)
    return {"experiment": "E1_identity", "f1": r["f1"],
            "fp": r["n_fp"], "fn": r["n_fn"], "pass": bool(ok)}


def exp_fn_sweep(gt, spacing, thresholds):
    out = []
    for t in thresholds:
        pred = delete_thin_branches(gt, t, spacing)
        r = metric_from_masks(gt, pred, spacing)
        out.append({"experiment": "E2_delete_FN", "radius_thresh_mm": t,
                    "precision": r["precision"], "recall": r["recall"],
                    "f1": r["f1"], "fp": r["n_fp"], "fn": r["n_fn"]})
    return out


def exp_fp_sweep(gt, spacing, dilations):
    out = []
    for it in dilations:
        pred = dilate_surface(gt, it)
        r = metric_from_masks(gt, pred, spacing)
        out.append({"experiment": "E3_dilate_FP", "dilation_iters": it,
                    "precision": r["precision"], "recall": r["recall"],
                    "f1": r["f1"], "fp": r["n_fp"], "fn": r["n_fn"]})
    return out


def exp_tolerance(gt, pred, spacing, scales):
    out = []
    for s in scales:
        r = metric_from_masks(gt, pred, spacing, radius_scale=s)
        out.append({"experiment": "E4_tolerance", "radius_scale": s,
                    "precision": r["precision"], "recall": r["recall"],
                    "f1": r["f1"], "fp": r["n_fp"], "fn": r["n_fn"]})
    return out


def exp_sampling(gt, pred, spacing, fracs, rng, repeats=3):
    out = []
    for f in fracs:
        for rep in range(repeats):
            r = metric_with_sampling(gt, pred, spacing, f, rng)
            out.append({"experiment": "E5_sampling", "frac": f, "rep": rep,
                        "precision": r["precision"], "recall": r["recall"],
                        "f1": r["f1"], "fp": r["n_fp"], "fn": r["n_fn"]})
    return out


# ============================================================ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="carpeta de mascaras de referencia (labelsTs)")
    ap.add_argument("--pred", default=None, help="carpeta de predicciones (E4/E5)")
    ap.add_argument("--pred_multi", default=None,
                    help="para tolerance_ranking: 'Etiq1=DIR1,Etiq2=DIR2,...'")
    ap.add_argument("--experiment", default="controlled",
                    choices=["controlled", "tolerance", "sampling",
                             "tolerance_ranking", "all"])
    ap.add_argument("--out", default="metric_sensitivity.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    rng = np.random.default_rng(args.seed)

    gts = sorted(glob.glob(os.path.join(args.gt, "*.nii.gz")))
    if not gts:
        print("[error] no encontre mascaras en", args.gt)
        return

    rows = []

    # Determinar la lista de casos a recorrer.
    # Para experimentos que necesitan predicciones (tolerance, sampling,
    # tolerance_ranking) iteramos sobre los casos que EXISTEN en la carpeta de
    # predicciones, no sobre todo --gt: asi solo procesamos esos (p.ej. los 160
    # de validacion) y no perdemos tiempo recorriendo y saltando el resto.
    if args.experiment in ("tolerance", "sampling") and args.pred:
        pred_files = sorted(glob.glob(os.path.join(args.pred, "*.nii.gz")))
        cases = [os.path.basename(p) for p in pred_files]
        print(f"[info] {len(cases)} casos con prediccion en --pred")
    elif args.experiment == "tolerance_ranking" and args.pred_multi:
        # usar la interseccion de casos presentes en TODAS las carpetas de metodos
        dirs = [pair.split("=")[1] for pair in args.pred_multi.split(",")]
        sets = []
        for d in dirs:
            sets.append({os.path.basename(p) for p in glob.glob(os.path.join(d, "*.nii.gz"))})
        common = set.intersection(*sets) if sets else set()
        cases = sorted(common)
        print(f"[info] {len(cases)} casos presentes en las {len(dirs)} carpetas de metodos")
    else:
        # controlled (o all): recorrer todas las mascaras de --gt
        cases = [os.path.basename(p) for p in gts]

    for case in cases:
        gp_path = os.path.join(args.gt, case)
        if not os.path.exists(gp_path):
            print(f"   [skip] sin GT para {case} en {args.gt}")
            continue
        gt, sp = _load(gp_path)
        print(f"== {case} ==")

        if args.experiment in ("controlled", "all"):
            rows.append({"case": case, **exp_identity(gt, sp)})
            for r in exp_fn_sweep(gt, sp, thresholds=[0.5, 0.8, 1.2, 1.5]):
                rows.append({"case": case, **r})
            for r in exp_fp_sweep(gt, sp, dilations=[0, 1, 2, 3]):
                rows.append({"case": case, **r})

        if args.experiment in ("tolerance", "all") and args.pred:
            pp = os.path.join(args.pred, case)
            if os.path.exists(pp):
                pred, _ = _load(pp)
                for r in exp_tolerance(gt, pred, sp, scales=[0.5, 0.75, 1.0, 1.5, 2.0]):
                    rows.append({"case": case, **r})

        if args.experiment in ("sampling", "all") and args.pred:
            pp = os.path.join(args.pred, case)
            if os.path.exists(pp):
                pred, _ = _load(pp)
                for r in exp_sampling(gt, pred, sp, fracs=[1.0, 0.75, 0.5, 0.25], rng=rng):
                    rows.append({"case": case, **r})

        if args.experiment == "tolerance_ranking" and args.pred_multi:
            for pair in args.pred_multi.split(","):
                label, d = pair.split("=")
                pp = os.path.join(d, case)
                if not os.path.exists(pp):
                    continue
                pred, _ = _load(pp)
                for r in exp_tolerance(gt, pred, sp, scales=[0.5, 0.75, 1.0, 1.5, 2.0]):
                    rows.append({"case": case, "method": label, **r})

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nGuardado: {args.out}  ({len(df)} filas)")
    if len(df) == 0:
        print("[AVISO] 0 filas: probablemente los nombres de las predicciones no")
        print("        coinciden con los de --gt. Verifica que la carpeta de")
        print("        predicciones contenga los mismos nombres (ej. 1000.nii.gz)")
        print("        que las mascaras de --gt.")
    _print_summaries(df)


def _print_summaries(df):
    import pandas as pd
    if "experiment" not in df.columns:
        return

    e1 = df[df.experiment == "E1_identity"]
    if len(e1):
        print(f"\n[E1] identidad: {e1['pass'].mean()*100:.0f}% de casos con F1=1, FP=FN=0")

    e2 = df[df.experiment == "E2_delete_FN"]
    if len(e2):
        g = e2.groupby("radius_thresh_mm")[["recall", "fn"]].mean()
        mono_rec = all(g["recall"].values[i] >= g["recall"].values[i+1]
                       for i in range(len(g)-1))
        print(f"[E2] borrar ramas -> recall medio {list(g['recall'].round(3))}, "
              f"monotono-decreciente={mono_rec}")

    e3 = df[df.experiment == "E3_dilate_FP"]
    if len(e3):
        g = e3.groupby("dilation_iters")[["precision", "fp"]].mean()
        mono_prec = all(g["precision"].values[i] >= g["precision"].values[i+1]
                        for i in range(len(g)-1))
        print(f"[E3] dilatar -> precision media {list(g['precision'].round(3))}, "
              f"monotono-decreciente={mono_prec}")

    e4 = df[df.experiment == "E4_tolerance"]
    if len(e4) and "method" in e4.columns:
        piv = e4.groupby(["radius_scale", "method"])["f1"].mean().unstack()
        print("\n[E4] F1 medio por (escala de radio x metodo):")
        print(piv.round(3).to_string())
        rankings = piv.rank(axis=1, ascending=False)
        same = (rankings.nunique() == 1).all()
        print(f"     ranking identico en todas las escalas: {bool(same)}")

    e5 = df[df.experiment == "E5_sampling"]
    if len(e5):
        g = e5.groupby("frac")["f1"].agg(["mean", "std"])
        print("\n[E5] F1 por fraccion de muestreo (mean/std):")
        print(g.round(4).to_string())


if __name__ == "__main__":
    main()