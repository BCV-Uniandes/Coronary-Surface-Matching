"""
Evaluacion de contorno sobre predicciones de nnU-Net.

Corre la metrica humgara (con delta = radio local del vaso) sobre las predicciones
de un run, caso por caso, y agrega Precision / Recall / F1 de contorno + FP/FN.
Si le pasas DOS carpetas de predicciones (tu metodo y el control) arma la tabla
comparativa y un test de Wilcoxon pareado sobre el F1 -> esa es la tabla del paper.

NO entrena: opera sobre los .nii.gz ya inferidos. Usa contour_extraction.py y
hungarian_metric.py (deben estar en la misma carpeta o en el PYTHONPATH).

Uso
---
# un solo run
python evaluate_contour.py --pred PRED_DIR --gt GT_DIR --out metodo.csv

# comparar metodo vs control (tabla del paper)
python evaluate_contour.py --pred METODO_DIR --pred2 CONTROL_DIR \\
       --gt GT_DIR --out comparacion.csv \\
       --name1 ContourSinkhorn --name2 Control

Donde:
  GT_DIR    = nnUNet_raw/Dataset002_Imagecas/labelsTs   (las etiquetas de test)
  PRED_DIR  = carpeta con las predicciones .nii.gz de ese run
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import nibabel as nib

from contour_extraction import contour_descriptor
from hungarian_metric import match_contours_full


def _load(path):
    """Carga un .nii.gz -> (array uint8, spacing). spacing en el orden de los ejes."""
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    arr = (arr > 0.5).astype(np.uint8)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return arr, spacing


def evaluate_case(pred_path, gt_path):
    """Metrica de contorno de un caso, usando TODOS los puntos (sin submuestreo).

    delta = radio local del vaso (GT), por punto. FP/FN quedan desacoplados porque
    cada nube tiene su numero real de puntos.
    """
    gt, sp = _load(gt_path)
    pred, _ = _load(pred_path)

    gp, gn, grad = contour_descriptor(gt, max_points=None,
                                      with_radius=True, spacing=sp)
    pp, pn = contour_descriptor(pred, max_points=None, spacing=sp)

    return match_contours_full(gp, pp, grad)


def evaluate_dir(pred_dir, gt_dir):
    """Recorre todas las predicciones y devuelve una lista de dicts por caso."""
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    rows = []
    for p in preds:
        case = os.path.basename(p)
        g = os.path.join(gt_dir, case)
        if not os.path.exists(g):
            print(f"  [skip] sin GT para {case}")
            continue
        r = evaluate_case(p, g)
        r["case"] = case
        rows.append(r)
        print(f"  {case:<28} P={r['precision']:.3f} R={r['recall']:.3f} "
              f"F1={r['f1']:.3f}  FP={r['n_fp']} FN={r['n_fn']}")
    return rows


def _summary(rows):
    import numpy as np
    f1 = np.array([r["f1"] for r in rows])
    p = np.array([r["precision"] for r in rows])
    rc = np.array([r["recall"] for r in rows])
    return {
        "n": len(rows),
        "F1_mean": f1.mean(), "F1_std": f1.std(),
        "P_mean": p.mean(), "P_std": p.std(),
        "R_mean": rc.mean(), "R_std": rc.std(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="carpeta de predicciones (metodo)")
    ap.add_argument("--pred2", default=None, help="carpeta de predicciones (control)")
    ap.add_argument("--gt", required=True, help="carpeta de etiquetas (labelsTs)")
    ap.add_argument("--out", default="contour_eval.csv")
    ap.add_argument("--name1", default="metodo")
    ap.add_argument("--name2", default="control")
    args = ap.parse_args()

    import pandas as pd

    print(f"== {args.name1} ==")
    rows1 = evaluate_dir(args.pred, args.gt)
    df1 = pd.DataFrame(rows1)
    df1["run"] = args.name1
    s1 = _summary(rows1)
    print(f"  -> F1 = {s1['F1_mean']:.3f} +/- {s1['F1_std']:.3f}  "
          f"P = {s1['P_mean']:.3f}  R = {s1['R_mean']:.3f}  (n={s1['n']})")

    all_df = df1
    if args.pred2:
        print(f"\n== {args.name2} ==")
        rows2 = evaluate_dir(args.pred2, args.gt)
        df2 = pd.DataFrame(rows2)
        df2["run"] = args.name2
        s2 = _summary(rows2)
        print(f"  -> F1 = {s2['F1_mean']:.3f} +/- {s2['F1_std']:.3f}  "
              f"P = {s2['P_mean']:.3f}  R = {s2['R_mean']:.3f}  (n={s2['n']})")
        all_df = pd.concat([df1, df2], ignore_index=True)

        # --- test de Wilcoxon pareado sobre F1 (mismos casos) ---------------- #
        from scipy.stats import wilcoxon
        m = pd.merge(df1[["case", "f1"]], df2[["case", "f1"]],
                     on="case", suffixes=("_1", "_2"))
        if len(m) >= 1 and not np.allclose(m["f1_1"], m["f1_2"]):
            stat, pval = wilcoxon(m["f1_1"], m["f1_2"])
            print("\n== Comparacion (tabla del paper) ==")
            print(f"  {args.name1}: F1 = {s1['F1_mean']:.3f} +/- {s1['F1_std']:.3f}")
            print(f"  {args.name2}: F1 = {s2['F1_mean']:.3f} +/- {s2['F1_std']:.3f}")
            print(f"  delta-F1 = {s1['F1_mean'] - s2['F1_mean']:+.3f}  "
                  f"(Wilcoxon p = {pval:.4g}, n={len(m)} casos pareados)")
        else:
            print("\n  [aviso] no se pudo correr Wilcoxon (pocos casos o F1 iguales)")

    all_df.to_csv(args.out, index=False)
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()