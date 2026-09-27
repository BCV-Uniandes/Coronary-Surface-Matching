import os
import torch
from torch import autocast, nn
from typing import Union, Tuple, List
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from nnunetv2.training.nnUNetTrainer.Cb_Variants.NexToU import NexToU
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_batchnorm
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0, InitWeights_He
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import \
    nnUNetTrainerNoDeepSupervision
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels

class nnUNetTrainer_NexToU_NoDeepSupervision(nnUNetTrainerNoDeepSupervision):
    def build_network_architecture(
            self,
            architecture_class_name: str,
            arch_init_kwargs: dict,
            arch_init_kwargs_req_import,
            num_input_channels: int,
            num_output_channels: int,
            enable_deep_supervision: bool = True
    ) -> nn.Module:

        import importlib

        def maybe_import(x):
            if x is None or not isinstance(x, str):
                return x
            module_name, class_name = x.rsplit(".", 1)
            return getattr(importlib.import_module(module_name), class_name)

        configuration_manager = self.configuration_manager

        # In this nnU-Net version, architecture parameters are in arch_init_kwargs
        kernel_sizes = arch_init_kwargs["kernel_sizes"]
        strides = arch_init_kwargs["strides"]
        features_per_stage = arch_init_kwargs["features_per_stage"]

        num_stages = arch_init_kwargs.get("n_stages", len(kernel_sizes))

        # conv_op may already be imported, or may be a string
        conv_op = maybe_import(arch_init_kwargs.get("conv_op", None))

        if conv_op is None:
            dim = len(kernel_sizes[0])
            conv_op = convert_dim_to_conv_op(dim)

        norm_op = maybe_import(arch_init_kwargs.get("norm_op", get_matching_batchnorm(conv_op)))
        dropout_op = maybe_import(arch_init_kwargs.get("dropout_op", None))
        nonlin = maybe_import(arch_init_kwargs.get("nonlin", nn.LeakyReLU))

        norm_op_kwargs = arch_init_kwargs.get("norm_op_kwargs", {"eps": 1e-5, "affine": True})
        dropout_op_kwargs = arch_init_kwargs.get("dropout_op_kwargs", None)
        nonlin_kwargs = arch_init_kwargs.get("nonlin_kwargs", {"inplace": True})
        conv_bias = arch_init_kwargs.get("conv_bias", True)

        n_conv_per_stage = arch_init_kwargs.get(
            "n_conv_per_stage",
            arch_init_kwargs.get("n_blocks_per_stage", None)
        )

        n_conv_per_stage_decoder = arch_init_kwargs.get("n_conv_per_stage_decoder", None)

        if n_conv_per_stage is None:
            n_conv_per_stage = [2] * num_stages

        if n_conv_per_stage_decoder is None:
            n_conv_per_stage_decoder = [2] * (num_stages - 1)

        # These are specific to NexToU. If your plans do not define them, use defaults.
        n_conv_stages = getattr(configuration_manager, "n_conv_stages", 2)
        n_swin_gnn_stages = getattr(configuration_manager, "n_swin_gnn_stages", 2)

        if n_conv_stages + n_swin_gnn_stages > num_stages:
            n_conv_stages = min(n_conv_stages, num_stages)
            n_swin_gnn_stages = max(0, num_stages - n_conv_stages)

        model = NexToU(
            input_channels=num_input_channels,
            patch_size=configuration_manager.patch_size,
            n_conv_stages=n_conv_stages,
            n_swin_gnn_stages=n_swin_gnn_stages,
            n_stages=num_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_conv_per_stage=n_conv_per_stage,
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            deep_supervision=enable_deep_supervision
        )

        model.apply(InitWeights_He(1e-2))

        return model