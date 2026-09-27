"""
Metrica de contorno via bipartite matching (algoritmo humgaro).

Esta es la PRUEBA DE CONCEPTO / EVALUACION. No necesita gradientes.

Dado el contorno del GT (puntos C) y el predicho (puntos C_hat), el matching
decide que punto predicho corresponde a que punto del GT, una sola vez cada uno.
De ahi salen tres conjuntos:
    - matched : parejas dentro del umbral delta  -> aciertos
    - GT sin pareja                              -> falsos negativos (ramas perdidas)
    - pred sin pareja                            -> falsos positivos (ramas fantasma)
y con eso Precision / Recall / F1, igual que en el benchmark de contornos de BSDS.

Costo de emparejar (configurable):
    d(p_i, q_j) = ||p_i - q_j||  +  lambda_n * (1 - n_i . n_j)
El termino de orientacion penaliza puntos que coinciden en posicion pero cuya
superficie apunta distinto (util para no premiar contornos "atravesados").
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def _pairwise_cost(P, Q, NP=None, NQ=None, lambda_n=0.0):
    """Matriz de costo (len(P), len(Q)). Distancia + termino de orientacion."""
    # distancia euclidiana
    diff = P[:, None, :] - Q[None, :, :]          # (n, m, 3)
    dist = np.sqrt((diff ** 2).sum(-1))           # (n, m)
    if lambda_n > 0 and NP is not None and NQ is not None:
        # 1 - coseno entre normales (0 si apuntan igual, 2 si opuestas)
        cos = NP @ NQ.T                            # (n, m)
        dist = dist + lambda_n * (1.0 - cos)
    return dist


def match_contours(
    gt_points,
    pred_points,
    gt_normals=None,
    pred_normals=None,
    delta=2.0,
    lambda_n=0.0,
):
    """Resuelve el bipartite matching y devuelve P/R/F1 + los indices.

    Parameters
    ----------
    gt_points, pred_points : (N,3),(M,3)
    delta : float o (N,) array
        Umbral de distancia para considerar valida una pareja. Si es array, es el
        umbral por punto del GT: en coronarias se pasa el radio local del vaso
        (un punto predicho esta bien si cae dentro del tubo del GT). Ver
        `contour_extraction.local_radius`.
    lambda_n : float
        Peso del termino de orientacion. 0 = solo posicion.

    Returns
    -------
    dict con precision, recall, f1, n_matched, n_fp, n_fn y los pares (i,j).

    Nota de escalado
    ----------------
    Esta version arma la matriz densa de costo y corre el humgaro O(n^3): ok para
    <= ~1500 puntos. Para superficies completas, sparsificar con KDTree (solo pares
    a distancia <= delta) y usar
    `scipy.sparse.csgraph.min_weight_full_bipartite_matching`, o min-cost-max-flow.
    """
    n, m = len(gt_points), len(pred_points)
    if n == 0 or m == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "n_matched": 0, "n_fp": m, "n_fn": n, "matches": [],
        }

    cost = _pairwise_cost(gt_points, pred_points, gt_normals, pred_normals, lambda_n)

    # delta escalar o por-punto del GT -> matriz de umbral (n, m) por broadcasting
    delta_arr = np.asarray(delta, dtype=np.float32)
    if delta_arr.ndim == 0:
        thr = np.full((n, 1), float(delta_arr), np.float32)
    else:
        thr = delta_arr.reshape(n, 1)                  # umbral por fila (punto GT)

    within = cost <= thr

    # Pares mas alla del umbral: costo alto (cap) para que el humgaro los evite.
    cap = float(thr.max()) * 10.0 + cost.max() + 1.0
    cost_capped = np.where(within, cost, cap)

    row, col = linear_sum_assignment(cost_capped)

    # Nos quedamos solo con las parejas dentro del umbral de su punto GT.
    valid = within[row, col]
    matched_pairs = list(zip(row[valid].tolist(), col[valid].tolist()))
    n_matched = int(valid.sum())

    n_fn = n - n_matched          # puntos GT sin pareja
    n_fp = m - n_matched          # puntos pred sin pareja

    precision = n_matched / m if m else 0.0
    recall = n_matched / n if n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_matched": n_matched,
        "n_fp": n_fp,
        "n_fn": n_fn,
        "matches": matched_pairs,
    }


def match_contours_fast(gt_points, pred_points, delta=2.0):
    """Variante escalable y aproximada para nubes grandes (solo posicion).

    En vez del humgaro, hace matching greedy via KDTree: para cada punto predicho
    busca el GT mas cercano dentro de delta y lo reclama si sigue libre. No es el
    optimo global, pero es O(n log n) y suele dar P/R/F1 muy parecidos. Util para
    sanity-check en superficies completas antes de submuestrear.
    """
    n, m = len(gt_points), len(pred_points)
    if n == 0 or m == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "n_matched": 0, "n_fp": m, "n_fn": n}

    tree = cKDTree(gt_points)
    dist, idx = tree.query(pred_points, distance_upper_bound=delta)
    used = np.zeros(n, dtype=bool)
    n_matched = 0
    # ordenar predichos por cercania para que los mejores reclamen primero
    order = np.argsort(dist)
    for j in order:
        if not np.isfinite(dist[j]):
            continue
        gi = idx[j]
        if not used[gi]:
            used[gi] = True
            n_matched += 1

    precision = n_matched / m
    recall = n_matched / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_matched": n_matched, "n_fp": m - n_matched, "n_fn": n - n_matched}


def match_contours_full(gt_points, pred_points, gt_radii, max_candidates=8):
    """Matching escalable sobre TODOS los puntos (sin submuestreo). Por KD-tree.

    Para cada punto predicho busca puntos GT cercanos; acepta el par (i,j) si el
    predicho cae dentro del radio del punto GT, ||x_i - xhat_j|| <= r_i. Resuelve
    el emparejamiento uno-a-uno de forma greedy por distancia creciente (cada punto
    se usa una sola vez), que es casi optimo para este criterio y corre en O(n log n).

    A diferencia de `match_contours`, NO iguala los tamanos de las nubes: GT y
    prediccion tienen su numero real de puntos, de modo que
        FN = puntos GT sin pareja      (ramas perdidas)
        FP = puntos predichos sin pareja (ramas fantasma)
    quedan DESACOPLADOS y miden cosas distintas.

    Parameters
    ----------
    gt_points  : (N,3)   puntos de superficie del GT (todos)
    pred_points: (M,3)   puntos de superficie de la prediccion (todos)
    gt_radii   : (N,)    radio local del vaso en cada punto GT (umbral por punto)
    max_candidates : vecinos GT a considerar por punto predicho (k del KD-tree)

    Returns
    -------
    dict con precision, recall, f1, n_matched, n_fp, n_fn, n_gt, n_pred.
    """
    n, m = len(gt_points), len(pred_points)
    if n == 0 or m == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_matched": 0,
                "n_fp": m, "n_fn": n, "n_gt": n, "n_pred": m}

    tree_gt = cKDTree(gt_points)
    rmax = float(np.max(gt_radii))
    k = min(max_candidates, n)
    # vecinos GT mas cercanos a cada punto predicho, hasta rmax (cota superior)
    dist, idx = tree_gt.query(pred_points, k=k, distance_upper_bound=rmax)
    dist = np.atleast_2d(dist.reshape(m, -1))
    idx = np.atleast_2d(idx.reshape(m, -1))

    # candidatos validos: dentro del radio del PROPIO punto GR (umbral por punto)
    pred_id = np.repeat(np.arange(m), dist.shape[1])
    gt_id = idx.reshape(-1)
    d = dist.reshape(-1)
    ok = np.isfinite(d) & (gt_id < n)
    pred_id, gt_id, d = pred_id[ok], gt_id[ok], d[ok]
    within = d <= gt_radii[gt_id]            # aceptacion por radio del punto GT
    pred_id, gt_id, d = pred_id[within], gt_id[within], d[within]

    # greedy por distancia creciente: cada punto GT y cada predicho se usan 1 vez
    order = np.argsort(d, kind="stable")
    used_gt = np.zeros(n, dtype=bool)
    used_pred = np.zeros(m, dtype=bool)
    matched = 0
    for e in order:
        gi, pj = gt_id[e], pred_id[e]
        if not used_gt[gi] and not used_pred[pj]:
            used_gt[gi] = True
            used_pred[pj] = True
            matched += 1

    n_fn = n - matched
    n_fp = m - matched
    precision = matched / m
    recall = matched / n
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_matched": matched, "n_fp": n_fp, "n_fn": n_fn,
            "n_gt": n, "n_pred": m}