"""
Loss Chamfer-hibrida para segmentacion de contornos (opcion B verificada).

Disenada tras descartar el campo de distancia (premia volumen interior) y el OT/Sinkhorn
(sin palanca donde p es plano). Combina las dos mitades que validamos por separado:

  - Termino RP (CREAR vaso ausente): D_pred(detached) * (1 - p), sumado SOLO sobre la
    superficie de referencia, normalizado por n_surf. D_pred = distancia de cada punto
    de ref a la masa predicha (p>0.5). Da gradiente que sube p en ramas perdidas, con
    palanca incluso donde p~0 (porque D_pred es un campo fijo por paso). En verificacion
    sobre datos reales: signo correcto en ~74% de los FN reales, foco en lo distal.

  - Termino FP (BORRAR vaso espurio): phi_out * p sobre el FONDO, normalizado por n_bg.
    phi_out = distancia al vaso (0 dentro, crece afuera). Penaliza MASA de probabilidad
    lejos del vaso (no el borde), corrigiendo el fallo de la version con saliencia |grad p|
    que solo penalizaba el contorno del FP. En verificacion: signo correcto en 100% de
    los FP reales.

Gradiente (ambos campos detached, fluye limpio por p):
    dL/dp = lam_pr * phi_out/n_bg  (fondo, sube->baja p: borra)
          - lam_rp * D_pred/n_surf (superficie ref, sube p: crea)

D_pred se recalcula en cada forward desde la p actual (un EDT por sample en CPU). Es
el coste dominante; comparable a la boundary loss (que hacia 2 EDT por sample).

Interfaz forward_prob(prob, gt_mask) identica a BoundaryDistanceLoss y ContourSinkhornLoss
para intercambiarlas en el wrapper _ContourFineTuneLoss.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt, binary_erosion


class ChamferContourLoss(nn.Module):
    """Loss Chamfer-hibrida: crea vaso ausente (RP) + borra vaso espurio (FP).

    Parameters
    ----------
    spacing   : spacing fisico (mm) para los campos de distancia.
    lam_pr    : peso del termino FP (borrar). Default 1.0.
    lam_rp    : peso del termino RP (crear). Default 1.0. (El barrido mostro que el
                balance no cambia el signo del gradiente; se exponen por completitud
                y para diagnostico de magnitud.)
    clip_mm   : recorte de phi_out y de D_pred (acota outliers lejanos).
    """

    def __init__(self, spacing=(1.0, 1.0, 1.0), lam_pr=1.0, lam_rp=1.0, clip_mm=20.0):
        super().__init__()
        self.spacing = spacing
        self.lam_pr = lam_pr
        self.lam_rp = lam_rp
        self.clip_mm = clip_mm
        # diagnostico: magnitud media de cada mitad (lo lee el trainer para el log)
        self.last_fp = 0.0
        self.last_rp = 0.0

    def forward_prob(self, prob, gt_mask):
        if gt_mask.sum() < 1:
            return prob.sum() * 0.0

        with torch.no_grad():
            g = gt_mask.detach().cpu().numpy().astype(np.float32)
            gb = g > 0.5
            # superficie de referencia (borde interior del vaso)
            surf = gb & ~binary_erosion(gb)
            # phi_out: distancia al vaso (0 dentro, crece afuera) -> termino FP
            phi_out = distance_transform_edt(~gb, sampling=self.spacing).astype(np.float32)
            phi_out = np.minimum(phi_out, self.clip_mm)
            # D_pred: distancia de cada voxel a la masa predicha (p>0.5) -> termino RP.
            # Se recalcula de la p ACTUAL (detached); fijo por paso pero con palanca.
            p_np = prob.detach().cpu().numpy()
            pred_mass = p_np > 0.5
            if pred_mass.any():
                D_pred = distance_transform_edt(~pred_mass,
                                                sampling=self.spacing).astype(np.float32)
                # NB: D_pred NO se recorta (a diferencia de phi_out), para coincidir
                # exactamente con la verificacion. Una rama profundamente perdida debe
                # poder recibir empuje grande proporcional a su lejania.
            else:
                # sin prediccion: empuje maximo uniforme a crear
                D_pred = np.full(g.shape, self.clip_mm, np.float32)

            n_bg = max(float((~gb).sum()), 1.0)
            n_surf = max(float(surf.sum()), 1.0)

            phi_out_t = torch.as_tensor(phi_out, device=prob.device, dtype=prob.dtype)
            bg_t = torch.as_tensor((~gb).astype(np.float32),
                                   device=prob.device, dtype=prob.dtype)
            surf_t = torch.as_tensor(surf.astype(np.float32),
                                     device=prob.device, dtype=prob.dtype)
            D_pred_t = torch.as_tensor(D_pred, device=prob.device, dtype=prob.dtype)

        # terminos (lineales en prob -> gradiente limpio)
        term_fp = (bg_t * phi_out_t * prob).sum() / n_bg          # borrar masa espuria
        term_rp = (surf_t * D_pred_t * (1.0 - prob)).sum() / n_surf  # crear vaso ausente

        self.last_fp = float(term_fp.detach())
        self.last_rp = float(term_rp.detach())
        return self.lam_pr * term_fp + self.lam_rp * term_rp

    def forward(self, pred_logits, gt_mask):
        return self.forward_prob(torch.sigmoid(pred_logits), gt_mask)
