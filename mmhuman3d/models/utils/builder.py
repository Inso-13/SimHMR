from mmcv.utils import Registry
from .simhmr_former import SimHMRFormer

TRANSFORMER = Registry('transformer')
TRANSFORMER.register_module(name='SimHMRFormer', module=SimHMRFormer)

def build_transformer(cfg):
    """Build head."""
    if cfg is None:
        return None
    return TRANSFORMER.build(cfg)