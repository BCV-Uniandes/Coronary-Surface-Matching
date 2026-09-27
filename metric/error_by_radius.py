"""
Respaldo cuantitativo de la LIMITACION 1 del Dice: el error se concentra en las ramas
DISTALES (finas), pero estas aportan poco volumen, asi que el Dice apenas las ve.

Para cada caso, sobre las predicciones de un modelo Dice puro:
  1. extrae los puntos de superficie del GT y su radio local (distance transform)
  2. los estratifica por radio en: finas (<r1), medias (r1..r2), gruesas (>r2)
  3. en cada estrato mide la TASA DE FN = fraccion de puntos GT sin emparejar con la
     prediccion (usando el mismo matching por radio de la metrica)
  4. mide la FRACCION DE VOLUMEN del GT en cada estrato

La tesis: la tasa de FN CRECE al bajar el radio (mas error en ramas finas), mientras
que la fraccion de volumen CAE (las finas pesan poco) -> el Dice, dominado por el
volumen, no refleja el error de las distales.

Uso:
  python error_by_radius.py --pred_dir PRED_DIR --gt_dir GT_DIR --n 0 \
      --r1 1.0 --r2 2.0
"""

import argparse
import glob
import os
import numpy as np
import nibabel as nib
from scipy.spatial import cKDTree

from contour_extraction import contour_descriptor
from hungarian_metric import match_contours_full  # no usado directamente; ver abajo


def _load(path):
    img = nib.load(path)
    sp = tuple(float(z) for z in img.header.get_zooms()[:3])
    arr = (np.asanyarray(img.dataobj) > 0.5).astype(np.uint8)
    return arr, sp


def fn_by_stratum(gt, pred, sp, r1, r2):
    """Devuelve, por estrato de radio, (tasa_FN, n_puntos_GT) y la fraccion de volumen.

    Estratos: fina (<r1), media (r1..r2), gruesa (>r2). La tasa de FN es la fraccion
    de puntos de superficie del GT que NO tienen un punto predicho dentro de su radio.
    """
    # puntos de superficie del GT con su radio local por punto
    gpts, _gn, grad = contour_descriptor(gt, max_points=None, with_radius=True, spacing=sp)
    ppts, _pn = contour_descriptor(pred, max_points=None, spacing=sp)
    if len(gpts) == 0:
        return None

    # para cada punto GT, hay un punto predicho dentro de su radio? (= emparejado)
    if len(ppts) > 0:
        tree_p = cKDTree(ppts)
        dist, _ = tree_p.query(gpts)            # distancia al punto predicho mas cercano
        matched = dist <= grad                  # aceptacion por radio del punto GT
    else:
        matched = np.zeros(len(gpts), dtype=bool)

    # estratos por radio del punto GT
    fina = grad < r1
    media = (grad >= r1) & (grad < r2)
    gruesa = grad >= r2
    out = {}
    for name, mask in [("fina(<%.1f)" % r1, fina),
                       ("media(%.1f-%.1f)" % (r1, r2), media),
                       ("gruesa(>%.1f)" % r2, gruesa)]:
        n = int(mask.sum())
        fn_rate = float((~matched[mask]).mean()) if n > 0 else float("nan")
        out[name] = (fn_rate, n)

    # fraccion de VOLUMEN del GT por estrato. Cada voxel del vaso se clasifica por el
    # CALIBRE REAL del vaso al que pertenece = radio del eje central (ridge) mas
    # cercano. NO por su propia distance-transform: eso clasificaria los voxeles de
    # pared de un vaso grueso como "finos" (estan cerca del borde), inflando
    # artificialmente el estrato fino.
    from scipy.ndimage import distance_transform_edt, maximum_filter
    dt = distance_transform_edt(gt.astype(bool), sampling=sp)   # mm
    # ridge = medial axis: voxeles que son maximo local de la distance transform
    mx = maximum_filter(dt, size=3)
    ridge = (gt > 0) & (dt >= mx - 1e-6) & (dt > 0)
    ridge_idx = np.argwhere(ridge)
    if len(ridge_idx) == 0:
        return out, {"fina": float("nan"), "media": float("nan"), "gruesa": float("nan")}
    ridge_r = dt[ridge]                                         # radio en cada punto del eje
    # a cada voxel del vaso le asignamos el radio del punto de eje mas cercano
    vox_idx = np.argwhere(gt > 0).astype(np.float32) * np.array(sp, dtype=np.float32)
    tree = cKDTree(ridge_idx.astype(np.float32) * np.array(sp, dtype=np.float32))
    _, nn = tree.query(vox_idx)
    rv = ridge_r[nn]                                            # calibre real por voxel
    vol_total = max(len(rv), 1)
    vol_frac = {
        "fina": float((rv < r1).sum()) / vol_total,
        "media": float(((rv >= r1) & (rv < r2)).sum()) / vol_total,
        "gruesa": float((rv >= r2).sum()) / vol_total,
    }
    return out, vol_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--r1", type=float, default=1.0, help="umbral fina/media (mm)")
    ap.add_argument("--r2", type=float, default=2.0, help="umbral media/gruesa (mm)")
    args = ap.parse_args()

    preds = sorted(glob.glob(os.path.join(args.pred_dir, "*.nii.gz")))
    if args.n > 0:
        preds = preds[:args.n]

    strata = ["fina(<%.1f)" % args.r1, "media(%.1f-%.1f)" % (args.r1, args.r2),
              "gruesa(>%.1f)" % args.r2]
    fn_acc = {s: [] for s in strata}
    vol_acc = {"fina": [], "media": [], "gruesa": []}

    for pth in preds:
        case = os.path.basename(pth)
        gpath = os.path.join(args.gt_dir, case)
        if not os.path.exists(gpath):
            continue
        gt, sp = _load(gpath)
        pred, _ = _load(pth)
        res = fn_by_stratum(gt, pred, sp, args.r1, args.r2)
        if res is None:
            continue
        out, vol_frac = res
        for s in strata:
            if not np.isnan(out[s][0]):
                fn_acc[s].append(out[s][0])
        for k in vol_acc:
            vol_acc[k].append(vol_frac[k])
        print(f"{case:<24} " + "  ".join(f"{s.split('(')[0]}:FN={out[s][0]*100:4.1f}%" for s in strata))

    print("\n================ RESUMEN (limitacion 1) ================")
    print(f"casos: {len(vol_acc['fina'])}")
    print(f"{'estrato':<18}{'tasa FN media':>15}{'% del volumen':>16}")
    vol_keys = ["fina", "media", "gruesa"]
    for s, vk in zip(strata, vol_keys):
        fn_m = 100 * np.mean(fn_acc[s]) if fn_acc[s] else float("nan")
        vol_m = 100 * np.mean(vol_acc[vk]) if vol_acc[vk] else float("nan")
        print(f"{s:<18}{fn_m:>14.1f}%{vol_m:>15.1f}%")
    print("\nLectura esperada: la tasa de FN CRECE de gruesa -> fina (mas error en ramas")
    print("distales), mientras el % de volumen CAE de gruesa -> fina (las finas pesan")
    print("poco). El Dice, dominado por el volumen, no refleja el error de las distales.")


if __name__ == "__main__":
    main()