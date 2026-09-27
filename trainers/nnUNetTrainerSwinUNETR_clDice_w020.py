import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR import nnUNetTrainerSwinUNETR
from nnunetv2.training.loss.cldice_loss import dice_cldice_loss


class nnUNetTrainerSwinUNETR_clDice_w020(nnUNetTrainerSwinUNETR):
    """SwinUNETR (arquitectura) + clDice loss (peso 0.20), fine-tuning de 150 epocas.

    Hereda de nnUNetTrainerSwinUNETR, asi que conserva automaticamente:
      - build_network_architecture con img_size (arreglo MONAI 1.4.0)
      - set_deep_supervision_enabled no-op (evita el error de mod.decoder)
      - enable_deep_supervision = False (SwinUNETR tiene una sola cabeza)

    Solo reemplaza la loss base Dice+CE por dice_cldice_loss (Dice+CE + 0.20*clDice),
    identica a la usada en nnU-Net y NexToU, para una comparacion limpia entre
    backbones. Deep supervision queda off (heredado), que es justo lo que clDice
    necesita (esqueletoniza a resolucion completa).

    Fine-tuning: lanzar con -pretrained_weights apuntando al checkpoint del baseline
    SwinUNETR de cada fold.
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # enable_deep_supervision = False ya viene del super().__init__();
        # lo reafirmamos por claridad.
        self.enable_deep_supervision = False
        # Fine-tuning: mismo protocolo que las otras corridas.
        self.num_epochs = 150
        self.initial_lr = 1e-3

    def _build_loss(self):
        print(">>> Using nnUNetTrainerSwinUNETR_clDice_w020 (clDice w=0.20, DS off)")
        return dice_cldice_loss(
            iter_=3,
            smooth=1.0,
            weight_dice=1,
            weight_cldice=0.20,
            batch_dice=self.configuration_manager.batch_dice,
            ddp=self.is_ddp,
        )
