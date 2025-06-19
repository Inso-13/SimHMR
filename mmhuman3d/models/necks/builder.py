# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .temporal_encoder import TemporalGRUEncoder
from .downsampler import ConvDownsamplerX4, ConvDownsamplerX8

NECKS = Registry('necks')

NECKS.register_module(name='TemporalGRUEncoder', module=TemporalGRUEncoder)
NECKS.register_module(name='ConvDownsamplerX4', module=ConvDownsamplerX4)
NECKS.register_module(name='ConvDownsamplerX8', module=ConvDownsamplerX8)

def build_neck(cfg):
    """Build neck."""
    if cfg is None:
        return None
    return NECKS.build(cfg)
