import warnings

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import BasicStem, BasicBlock, Bottleneck

from .utils import _generic_resnet, Conv3DDepthwise, BasicStem_Pool, IPConv3DDepthwise


__all__ = ["ir_csn_152", "ip_csn_152"]


def ir_csn_152(pretraining="", use_pool1=True, progress=False, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]

    if pretraining in avail_pretrainings:
        arch = "ir_csn_152_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            f"Unrecognized pretraining dataset, continuing with randomly initialized network."
            " Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )
        arch = "ir_csn_152"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[Conv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool if use_pool1 else BasicStem,
        **kwargs,
    )

    return model


def ip_csn_152(pretraining="", use_pool1=True, progress=False, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]

    if pretraining in avail_pretrainings:
        arch = "ip_csn_152_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            f"Unrecognized pretraining dataset, continuing with randomly initialized network."
            " Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )
        arch = "ip_csn_152"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[IPConv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool if use_pool1 else BasicStem,
        **kwargs,
    )

    return model
