import warnings

import torch.hub
import torch.nn as nn
from torchvision.models.video.resnet import R2Plus1dStem, BasicBlock, Bottleneck


from .utils import _generic_resnet, R2Plus1dStem_Pool, Conv2Plus1D


__all__ = ["r2plus1d_34", "r2plus1d_152"]


def r2plus1d_34(pretraining="", use_pool1=False, progress=False, **kwargs):
    avail_pretrainings = [
        "kinetics_8frms",
        "kinetics_32frms",
        "ig65m_8frms",
        "ig65m_32frms",
    ]
    if pretraining in avail_pretrainings:
        arch = "r2plus1d_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            f"Unrecognized pretraining dataset, continuing with randomly initialized network."
            " Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )
        arch = "r2plus1d_34"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs,
    )
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)

    return model


def r2plus1d_152(pretraining="", use_pool1=True, progress=False, **kwargs):
    avail_pretrainings = [
        "ig65m_32frms",
        "ig_ft_kinetics_32frms",
        "sports1m_32frms",
        "sports1m_ft_kinetics_32frms",
    ]
    if pretraining in avail_pretrainings:
        arch = "r2plus1d_" + pretraining
        pretrained = True
    else:
        warnings.warn(
            f"Unrecognized pretraining dataset, continuing with randomly initialized network."
            " Available pretrainings: {avail_pretrainings}",
            UserWarning,
        )
        arch = "r2plus1d_34"
        pretrained = False

    model = _generic_resnet(
        arch,
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 8, 36, 3],
        stem=R2Plus1dStem_Pool if use_pool1 else R2Plus1dStem,
        **kwargs,
    )
    # We need exact Caffe2 momentum for BatchNorm scaling
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
            m.momentum = 0.9

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)

    return model
