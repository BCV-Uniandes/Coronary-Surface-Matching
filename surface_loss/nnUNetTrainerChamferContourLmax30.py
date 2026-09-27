"""Sonda Chamfer con peso global lambda_max = 0.20 (20%).

Hereda toda la maquinaria de nnUNetTrainerChamferContour y SOLO sube el peso
global del termino de contorno a 0.20. Todo lo demas (lam_pr, lam_rp, clip,
warm-up, epocas) es identico a la corrida base, para que la comparacion aisle
el efecto del peso.

ADVERTENCIA: con la boundary loss, pesos altos (0.1) DESPLOMARON el F1 de
contorno (0.736) aunque otras metricas no lo reflejaran. Evaluar SIEMPRE con
el F1 de contorno (evaluate_contour.py), no con el Dice, antes de concluir.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerChamferContour import (
    nnUNetTrainerChamferContour,
)


class nnUNetTrainerChamferContourLmax30(nnUNetTrainerChamferContour):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.contour_lambda_max = 0.30   # <-- unico cambio: -> 20%
