from nnunetv2.training.loss.cldice_loss import (
    dice_cldice_loss,
    CE_cldice_loss,
    CE_clCE_loss,
    dice_clCE_loss,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import torch


class nnUNetTrainerDiceclDiceLoss(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerDiceclDiceLoss")
        return dice_cldice_loss(iter_=3, smooth=1.0, weight_dice=1, weight_cldice=0.25)


class nnUNetTrainerDiceclDiceLoss025(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerDiceclDiceLoss")
        return dice_cldice_loss(iter_=3, smooth=1.0, weight_dice=1, weight_cldice=0.25)


class nnUNetTrainerDiceclDiceLoss075(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerDiceclDiceLoss")
        return dice_cldice_loss(iter_=3, smooth=1.0, weight_dice=1, weight_cldice=0.75)


class nnUNetTrainerDiceclDiceLoss010(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerDiceclDiceLoss")
        return dice_cldice_loss(iter_=3, smooth=1.0, weight_dice=1, weight_cldice=0.10)


class nnUNetTrainerDiceclDiceLoss100(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerDiceclDiceLoss")
        return dice_cldice_loss(iter_=3, smooth=1.0, weight_dice=1, weight_cldice=1)



class nnUNetTrainerDiceclCELoss(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerDiceclCELoss")
        return dice_clCE_loss(iter_=3, smooth=1.0, weight_dice=1, weight_clCE=1)


class nnUNetTrainerCEclDiceLoss(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerCEclDiceLoss")
        return CE_cldice_loss({},iter_=3, smooth=1.0, weight_ce=1, weight_cldice=1, ignore_label=self.label_manager.ignore_label,)



class nnUNetTrainerCEclDiceLoss050(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerCEclDiceLoss")
        return CE_cldice_loss({},iter_=3, smooth=1.0, weight_ce=1, weight_cldice=0.5, ignore_label=self.label_manager.ignore_label,)


class nnUNetTrainerCEclDiceLoss010(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerCEclDiceLoss")
        return CE_cldice_loss({},iter_=3, smooth=1.0, weight_ce=1, weight_cldice=0.10, ignore_label=self.label_manager.ignore_label,)



class nnUNetTrainerCEclCELoss(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False
        self.num_epochs = 1000

    def _build_loss(self):
        print(">>> Using nnUNetTrainerCEclCELoss")
        return CE_clCE_loss({}, iter_=3, weight_ce=1, weight_clCE=1)




