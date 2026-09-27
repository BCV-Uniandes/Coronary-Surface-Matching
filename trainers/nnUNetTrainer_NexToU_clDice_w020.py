import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_NexToU import nnUNetTrainer_NexToU
from nnunetv2.training.loss.cldice_loss import dice_cldice_loss


class nnUNetTrainer_NexToU_clDice_w020(nnUNetTrainer_NexToU):
    """NexToU (arquitectura) + clDice loss (peso 0.20), fine-tuning de 150 epocas.

    Hereda la arquitectura NexToU de nnUNetTrainer_NexToU y reemplaza la loss base
    Dice+CE por dice_cldice_loss (Dice+CE + 0.20 * clDice), identica a la usada en
    nnU-Net.

    Deep supervision DESACTIVADO (igual que en el clDice de nnU-Net): el termino
    clDice esqueletoniza, lo cual solo es valido a resolucion completa, asi que la
    red debe devolver una unica salida y no una lista por resolucion. Esto mantiene
    la comparacion clDice-vs-clDice consistente entre backbones.

    Fine-tuning: lanzar con -pretrained_weights apuntando al checkpoint del baseline
    NexToU de cada fold.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # Desactivar deep supervision -> red devuelve un unico tensor.
        # Asi _build_loss no envuelve con DeepSupervisionWrapper y el target no se
        # downsamplea a lista, que es lo que dice_cldice_loss espera.
        self.enable_deep_supervision = False
        # Fine-tuning: 150 epocas, lr 1e-3 (mismo protocolo que las otras corridas).
        self.num_epochs = 150
        self.initial_lr = 1e-3

    def _build_loss(self):
        print(">>> Using nnUNetTrainer_NexToU_clDice_w020 (clDice w=0.20, DS off)")
        return dice_cldice_loss(
            iter_=3,
            smooth=1.0,
            weight_dice=1,
            weight_cldice=0.20,
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
        )
