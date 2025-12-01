import numpy as np
from models import aggregators
from models import backbones
from torch import nn


def get_backbone(backbone_arch='dinov2_vitb14',
                 pretrained=True,
                 layer1=20,
                 use_cls=False,
                 norm_descs=True,
                 out_indices=[8, 9, 10, 11]):
    
    if 'dino' in backbone_arch.lower():
        return backbones.DinoV2_self(model_name=backbone_arch, layer1=layer1,  use_cls=use_cls, norm_descs=norm_descs, out_indices=out_indices)
    else:
        print("wrong input backbone type")
        exit()

def get_aggregator(agg_arch='GeM', agg_config={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    else:
        print("wrong input aggregator type")
        exit()
        return aggregators.GeMPool(**agg_config)

