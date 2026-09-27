"""
Sonda: fine-tuning con la loss Chamfer-hibrida (crear vaso ausente + borrar espurio).

Intento final de la loss, tras descartar campo de distancia y OT. La formulacion paso
verificacion de gradiente sobre datos reales: borrar(FP)=100%, crear(FN)~74% (el 74% es
el techo; el barrido de balance no lo sube -> lo zanjamos entrenando una vez).

Construye ChamferContourLoss y la envuelve con el wrapper _ContourFineTuneLoss heredado
(EMA de normalizacion + warm-up de lambda + aplicacion a la cabeza de mayor resolucion).

Config: lam_pr=lam_rp=1.0, lambda_max=0.05 (peso global intermedio), 50 epocas, warm-up 10.

    nnUNetv2_train DATASET_ID 3d_fullres FOLD \\
        -tr nnUNetTrainerChamferContour \\
        --npz -device cuda -pretrained_weights /ruta/checkpoint_best.pth
"""

from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerContourSinkhorn import (
    nnUNetTrainerContourSinkhorn, _ContourFineTuneLoss,
)
from nnunetv2.training.loss.chamfer_contour_loss import ChamferContourLoss

class nnUNetTrainerChamferContour(nnUNetTrainerContourSinkhorn):
    """Fine-tuning con Dice+CE + lambda * L_chamfer (hibrida)."""

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # --- hiperparametros de la loss Chamfer ---
        self.ch_lam_pr = 1.0          # peso termino FP (borrar)
        self.ch_lam_rp = 1.0          # peso termino RP (crear)
        self.ch_clip_mm = 20.0
        # --- peso global y warm-up (igual escala que las sondas de banda) ---
        self.contour_lambda_max = 0.05
        self.contour_warmup_epochs = 10
        # --- SONDA corta ---
        self.num_epochs = 50

    def _build_loss(self):
        # loss base deep-supervised de nnU-Net (del nnUNetTrainer abuelo)
        base = super(nnUNetTrainerContourSinkhorn, self)._build_loss()

        spacing = tuple(self.configuration_manager.spacing)
        contour = ChamferContourLoss(
            spacing=spacing,
            lam_pr=self.ch_lam_pr,
            lam_rp=self.ch_lam_rp,
            clip_mm=self.ch_clip_mm,
        )
        return _ContourFineTuneLoss(
            base=base,
            contour_loss=contour,
            weight_fn=lambda: self._contour_weight,
            vessel_channel=1,
            normalize=self.contour_normalize,
        )

    def on_train_epoch_end(self, train_outputs):
        # log adicional del balance crear/borrar de la ultima evaluacion de la loss
        super().on_train_epoch_end(train_outputs)
        c = getattr(self.loss, "contour", None)
        if c is not None and hasattr(c, "last_fp"):
            self.print_to_log_file(
                f"[chamfer] term_fp(borrar)={c.last_fp:.3e}  "
                f"term_rp(crear)={c.last_rp:.3e}")
