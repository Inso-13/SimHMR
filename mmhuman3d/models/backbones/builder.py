# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .hrnet import PoseHighResolutionNet, PoseHighResolutionNetExpose
from .resnet import ResNet, ResNetV1d
from .convnext import ConvNeXt

BACKBONES = Registry('backbones')

BACKBONES.register_module(name='ResNet', module=ResNet)
BACKBONES.register_module(name='ResNetV1d', module=ResNetV1d)
BACKBONES.register_module(
    name='PoseHighResolutionNet', module=PoseHighResolutionNet)
BACKBONES.register_module(
    name='PoseHighResolutionNetExpose', module=PoseHighResolutionNetExpose)

BACKBONES.register_module(name='ConvNeXt', module=ConvNeXt)


def build_backbone(cfg):
    """Build backbone."""
    if cfg is None:
        return None
    return BACKBONES.build(cfg)
