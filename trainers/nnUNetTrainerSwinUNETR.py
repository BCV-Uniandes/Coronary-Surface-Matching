import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:
    from monai.networks.nets import SwinUNETR
except ImportError as e:
    raise ImportError(
        "SwinUNETR requiere MONAI. Instalalo en lambda004 con: pip install monai"
    ) from e


class _SwinUNETRWrapper(nn.Module):
    """Envuelve SwinUNETR de MONAI para que encaje en el bucle de nnU-Net.

    nnU-Net invoca `set_deep_supervision_enabled(...)` sobre la red y, cuando DS
    esta activo, espera que la red devuelva una LISTA de mapas (uno por resolucion).
    SwinUNETR tiene una sola cabeza, asi que:
      - `set_deep_supervision_enabled` es un no-op (no hay nada que activar/desactivar)
      - `forward` siempre devuelve un unico tensor
    En el trainer desactivamos DS (self.enable_deep_supervision = False) para que la
    loss NO se envuelva con DeepSupervisionWrapper y el target NO se downsamplee a lista.
    """

    def __init__(self, **swin_kwargs):
        super().__init__()
        self.swin = SwinUNETR(**swin_kwargs)

    def set_deep_supervision_enabled(self, enabled: bool):
        # SwinUNETR no tiene deep supervision; no-op intencional.
        return

    def forward(self, x):
        return self.swin(x)


class nnUNetTrainerSwinUNETR(nnUNetTrainer):
    """nnU-Net + SwinUNETR (MONAI) como backbone, con loss estandar Dice+CE.

    Esta es la fila 'Dice+CE / SwinUNETR' de la tabla del paper. Para combinarlo
    con tu boundary-distance loss, subclasea este trainer y sobreescribe _build_loss
    igual que en nnUNetTrainerBoundaryDist (manteniendo enable_deep_supervision=False).
    """

    def __init__(self, plans, configuration, fold, dataset_json,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # SwinUNETR no soporta deep supervision -> apagarlo en todo el pipeline.
        # Esto hace que _build_loss no envuelva con DeepSupervisionWrapper y que
        # _get_deep_supervision_scales devuelva None (no se downsamplea el target).
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
        # SwinUNETR no tiene la estructura decoder.deep_supervision de nnU-Net.
        # DS ya esta desactivado (self.enable_deep_supervision = False), asi que
        # esto es un no-op intencional que evita que el metodo base busque
        # mod.decoder, que no existe en _SwinUNETRWrapper.
        return
        
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        feature_size = 48

        # MONAI 1.4.0 aun exige img_size en SwinUNETR (deprecado pero requerido por la firma).
        # Tomamos el patch size del plan; arch_init_kwargs lo expone como 'patch_size'
        # en la config 3d_fullres. Fallback al valor conocido si no estuviera.
        patch_size = arch_init_kwargs.get('patch_size', (96, 160, 160))

        return _SwinUNETRWrapper(
            img_size=tuple(patch_size),   # <-- LINEA AÑADIDA
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            feature_size=feature_size,
            spatial_dims=3,
            use_checkpoint=True,
        )