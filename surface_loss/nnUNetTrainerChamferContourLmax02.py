"""Sonda Chamfer con peso global lambda_max = 0.02 (2%).

Hereda toda la maquinaria de nnUNetTrainerChamferContour y SOLO baja el peso
global del termino de contorno de 0.05 a 0.02. Todo lo demas (lam_pr, lam_rp,
clip, warm-up, epocas) es identico a la corrida base, para que la comparacion
aisle el efecto del peso.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerChamferContour import (
    nnUNetTrainerChamferContour,
)


class nnUNetTrainerChamferContourLmax02(nnUNetTrainerChamferContour):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.contour_lambda_max = 0.02   # <-- unico cambio: 5% -> 2%
