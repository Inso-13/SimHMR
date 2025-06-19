# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.utils import Registry

from .simhmr_head import SimHMRHead
HEADS = Registry('heads')
HEADS.register_module(name='SimHMRHead', module=SimHMRHead)

def build_head(cfg):
    """Build head."""
    if cfg is None:
        return None
    return HEADS.build(cfg)
