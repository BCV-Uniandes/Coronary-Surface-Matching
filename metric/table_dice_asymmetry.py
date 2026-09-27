"""
Tabla 5.1 ("Dice Hides an Asymmetric Error"): caracteriza, sobre un modelo entrenado
con Dice puro, la asimetria entre lo que OMITE (FN) y lo que ALUCINA (FP), que el Dice
no puede expresar por ser simetrico.

Para cada caso reporta:
  - Dice (volumetrico, como referencia de que el Dice es "alto")
  - masa de contorno FP y FN (terminos crudos de la boundary loss, normalizados por
    voxeles activos): FP = vaso predicho lejos del GT, FN = vaso del GT no cubierto
  - ratio FN/FP
  - (opcional) error de Betti-0 = |#componentes(pred) - #componentes(GT)|

Y al final el resumen agregado que va a la tabla (media/mediana + dispersion).

IMPORTANTE: correr sobre las predicciones de un modelo Dice PURO (el baseline original
nnUNetTrainer, o el control FTControl), NO sobre un modelo con la boundary loss.

Uso:
  python table_dice_asymmetry.py --pred_dir PRED_DIR --gt_dir GT_DIR --n 0   # 0 = todos
  python table_dice_asymmetry.py --pred_dir PRED_DIR --gt_dir GT_DIR --betti # +Betti0
"""

import argparse
import glob
import os
import numpy as np
import nibabel as nib

from nnunetv2.training.loss.boundary_distance_loss import distance_fields

def dice_score(pred, gt):
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 2.0 * inter / denom if denom > 0 else 1.0


def betti0_err(pred, gt):
    """|#componentes(pred) - #componentes(GT)| (26-conectividad)."""
    from scipy.ndimage import label
    struct = np.ones((3, 3, 3), dtype=int)
    _, n_pred = label(pred, structure=struct)   # label devuelve (etiquetas, n_comp)
    _, n_gt = label(gt, structure=struct)
    return abs(int(n_pred) - int(n_gt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True, help="predicciones de un modelo Dice puro")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--n", type=int, default=0, help="numero de casos (0 = todos)")
    ap.add_argument("--clip_mm", type=float, default=20.0)
    ap.add_argument("--betti", action="store_true", help="calcular tambien Betti-0 err")
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    if args.n > 0:
        preds = preds[:args.n]

    dices, fps, fns, ratios, b0 = [], [], [], [], []
    hdr = f"{'caso':<26}{'Dice':>8}{'FP_mass':>12}{'FN_mass':>12}{'FN/FP':>9}"
    if args.betti:
        hdr += f"{'b0-err':>8}"
    print(hdr)
    for pth in preds:
        case = os.path.basename(pth)
        gpath = os.path.join(args.gt_dir, case)
        if not os.path.exists(gpath):
            continue
        gimg = nib.load(gpath)
        sp = tuple(float(z) for z in gimg.header.get_zooms()[:3])
        g = (np.asanyarray(gimg.dataobj) > 0.5).astype(np.float32)
        p = np.asanyarray(nib.load(pth).dataobj).astype(np.float32)
        if p.max() > 1.0:
            p = (p > 0.5).astype(np.float32)
        p = np.clip(p, 0.0, 1.0)

        d = dice_score(p, g)
        phi_out, phi_in = distance_fields(g, sp, clip_mm=args.clip_mm)
        n_fg = max(int(g.sum()), 1)
        n_bg = max(int(g.size - g.sum()), 1)
        fp = float((phi_out * p).sum() / n_bg)
        fn = float((phi_in * (1.0 - p)).sum() / n_fg)
        r = fn / (fp + 1e-12)
        dices.append(d); fps.append(fp); fns.append(fn); ratios.append(r)
        line = f"{case:<26}{d:>8.3f}{fp:>12.3e}{fn:>12.3e}{r:>9.1f}"
        if args.betti:
            bb = betti0_err((p > 0.5).astype(np.uint8), (g > 0.5).astype(np.uint8))
            b0.append(bb); line += f"{bb:>8d}"
        print(line)

    dices = np.array(dices); ratios = np.array(ratios)
    fps = np.array(fps); fns = np.array(fns)
    print("\n================ RESUMEN (para la tabla 5.1) ================")
    print(f"casos: {len(dices)}")
    print(f"Dice           : media={dices.mean():.3f}  +/- {dices.std():.3f}")
    print(f"FP mass        : media={fps.mean():.3e}  mediana={np.median(fps):.3e}")
    print(f"FN mass        : media={fns.mean():.3e}  mediana={np.median(fns):.3e}")
    print(f"ratio FN/FP    : media={ratios.mean():.1f}  mediana={np.median(ratios):.1f}"
          f"  (min={ratios.min():.1f}  max={ratios.max():.1f})")
    if b0:
        b0 = np.array(b0)
        print(f"Betti-0 err    : media={b0.mean():.1f}  +/- {b0.std():.1f}")
    print("\nLectura: Dice alto conviviendo con ratio FN/FP >> 1 = el modelo omite")
    print("mucho mas vaso del que alucina, asimetria que el Dice (simetrico) no expresa.")


if __name__ == "__main__":
    main()