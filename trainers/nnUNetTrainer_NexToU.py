import os
import torch
from torch import autocast, nn
from typing import Union, Tuple, List
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.NexToU import NexToU
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels

class nnUNetTrainer_NexToU(nnUNetTrainer):
    def build_network_architecture(self,
                                   architecture_class_name,
                                   arch_init_kwargs,
                                   arch_init_kwargs_req_import,
                                   num_input_channels,
                                   num_output_channels,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        # Port a la firma v2.2+ del repo. NexToU necesita patch_size (no viene en
        # arch_init_kwargs) y usa nombres de kwargs distintos a los del estandar UNet.
        import pydoc
        from copy import deepcopy
        from nnunetv2.training.nnUNetTrainer.variants.network_architecture.NexToU import NexToU
        from dynamic_network_architectures.initialization.weight_init import InitWeights_He

        kw = deepcopy(dict(arch_init_kwargs))

        # resolver strings de import (conv_op, norm_op, dropout_op, nonlin)
        for ri in arch_init_kwargs_req_import:
            if kw.get(ri) is not None:
                kw[ri] = pydoc.locate(kw[ri])

        # renombrar kwargs: estandar UNet -> NexToU
        if 'n_conv_per_stage' in kw:
            kw['n_blocks_per_stage'] = kw.pop('n_conv_per_stage')
        if 'n_conv_per_stage_decoder' in kw:
            kw['n_blocks_per_stage_decoder'] = kw.pop('n_conv_per_stage_decoder')

        model = NexToU(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            patch_size=self.configuration_manager.patch_size,
            deep_supervision=enable_deep_supervision,
            **kw
        )
        model.apply(InitWeights_He(1e-2))
        return model