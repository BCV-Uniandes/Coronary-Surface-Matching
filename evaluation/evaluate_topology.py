"""
Evaluacion topologica sobre predicciones de nnU-Net: clDice y beta0-err.

Complementa a evaluate_contour.py (que da FP/FN/F1 de superficie). Aqui se calculan
las DOS metricas topologicas de la tabla del paper:

  - clDice  (Shit et al., CVPR 2021): solapamiento de centerlines.
            clDice = 2 * (Tprec * Tsens) / (Tprec + Tsens)
            Tprec  = |skel(pred) dentro de GT|   / |skel(pred)|
            Tsens  = |skel(GT)   dentro de pred| / |skel(GT)|
            Se reporta en escala 0-100 (se multiplica por 100).

  - beta0-err: |#componentes_conexas(pred) - #componentes_conexas(GT)|,
            error absoluto en el numero 0-esimo de Betti (componentes 3D).

NO entrena: opera sobre los .nii.gz ya inferidos, igual que evaluate_contour.py.

Decisiones de implementacion (explicitas, para que sean reproducibles):
  - Skeletonizacion 3D: skimage.morphology.skeletonize (algoritmo de Lee 1994
    para 3D). skeletonize devuelve un volumen booleano del esqueleto.
  - Conectividad para componentes (beta0): 26-conectividad (full connectivity
    en 3D), via scipy.ndimage.label con estructura 3x3x3 de unos. Esto cuenta
    como una sola componente a voxeles que se tocan por cara, arista o esquina;
    es la convencion estandar para "esta el arbol conectado".
  - Binarizacion: mascara > 0.5, igual que evaluate_contour.py.

Uso
---
# un solo run
python evaluate_topology.py --pred PRED_DIR --gt GT_DIR --out topo.csv

Donde:
  GT_DIR    = nnUNet_raw/Dataset002_Imagecas/labelsTr (o labelsTs)
  PRED_DIR  = carpeta con las predicciones .nii.gz (p.ej. fold_0/validation)
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import nibabel as nib

from skimage.morphology import skeletonize
from scipy.ndimage import label


# 26-conectividad en 3D (cara + arista + esquina)
_STRUCT_26 = np.ones((3, 3, 3), dtype=np.uint8)


def _load(path):
    """Carga un .nii.gz -> array uint8 binarizado (>0.5). Mismo criterio que
    evaluate_contour.py para que pred y GT se traten igual."""
    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)
    arr = (arr > 0.5).astype(np.uint8)
    return arr


def _cl_score(skel, mask):
    """Fraccion de voxeles del esqueleto 'skel' que caen dentro de 'mask'.
    Es el Tprec (si skel=skel_pred, mask=gt) o Tsens (si skel=skel_gt, mask=pred).
    skeletonize devuelve bool; lo usamos como indice booleano directamente."""
    n = int(skel.sum())
    if n == 0:
        return 0.0
    inside = int((mask[skel] > 0).sum())
    return inside / n


def cldice_case(pred, gt):
    """clDice de un caso (0-1). Sigue la definicion de Shit et al. 2021.

    Usa skeletonize(method='lee'), el algoritmo 3D de Lee et al. 1994, que es el
    correcto para volumenes. Vasos reales (curvos, irregulares) se esqueletonizan
    bien; si un esqueleto sale vacio se avisa, porque indica un caso degenerado.

    IMPORTANTE: skeletonize(method='lee') devuelve uint8 con valores 0/255, no
    bool. Hay que convertir a bool con .astype(bool) antes de usarlo como indice,
    o NumPy lo interpreta como indexacion avanzada (con 0 y 255 como indices) e
    intenta crear un array gigante."""
    skel_pred = skeletonize(pred.astype(bool), method='lee').astype(bool)
    skel_gt = skeletonize(gt.astype(bool), method='lee').astype(bool)

    if skel_pred.sum() == 0 or skel_gt.sum() == 0:
        print(f"    [aviso] esqueleto vacio (pred={int(skel_pred.sum())}, "
              f"gt={int(skel_gt.sum())}); clDice puede ser 0 para este caso")

    tprec = _cl_score(skel_pred, gt)    # precision topologica
    tsens = _cl_score(skel_gt, pred)    # sensibilidad topologica

    if (tprec + tsens) == 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


def beta0(mask):
    """Numero de componentes conexas 3D (beta_0) con 26-conectividad."""
    _, n = label(mask, structure=_STRUCT_26)
    return n


def evaluate_case(pred_path, gt_path):
    """Devuelve clDice (0-1) y beta0-err de un caso."""
    gt = _load(gt_path)
    pred = _load(pred_path)

    cld = cldice_case(pred, gt)
    b0_err = abs(beta0(pred) - beta0(gt))

    return {"cldice": cld, "beta0_pred": beta0(pred),
            "beta0_gt": beta0(gt), "beta0_err": b0_err}


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
        print(f"  {case:<28} clDice={r['cldice']*100:.2f} "
              f"b0-err={r['beta0_err']} "
              f"(pred={r['beta0_pred']}, gt={r['beta0_gt']})")
    return rows


def _summary(rows):
    cld = np.array([r["cldice"] for r in rows]) * 100.0   # a escala 0-100
    b0 = np.array([r["beta0_err"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "clDice_mean": cld.mean(), "clDice_std": cld.std(),
        "beta0err_mean": b0.mean(), "beta0err_std": b0.std(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="carpeta de predicciones")
    ap.add_argument("--gt", required=True, help="carpeta de etiquetas")
    ap.add_argument("--out", default="topology_eval.csv")
    ap.add_argument("--name", default="run")
    args = ap.parse_args()

    import pandas as pd

    print(f"== {args.name} (clDice / beta0-err) ==")
    rows = evaluate_dir(args.pred, args.gt)
    if not rows:
        print("  [error] no se evaluo ningun caso (revisa rutas y nombres)")
        return

    df = pd.DataFrame(rows)
    df["run"] = args.name
    s = _summary(rows)
    print(f"\n  -> clDice   = {s['clDice_mean']:.2f} +/- {s['clDice_std']:.2f}")
    print(f"  -> beta0-err = {s['beta0err_mean']:.2f} +/- {s['beta0err_std']:.2f}  "
          f"(n={s['n']})")

    df.to_csv(args.out, index=False)
    print(f"\nGuardado: {args.out}")


if __name__ == "__main__":
    main()