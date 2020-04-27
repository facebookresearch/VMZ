import warnings

import torch.hub
import torch.nn as nn

from torchvision.models.video.resnet import VideoResNet

from .utils import R2Plus1dStem, Conv2Plus1D

# TODO: upload models and load them
model_urls = {
    "r2plus1d_34_kinetics_8frms": "",  # noqa: E501
    "r2plus1d_34_kinetics_32frms": "",  # noqa: E501
    "r2plus1d_34_ig65m_8frms": "",  # noqa: E501
    "r2plus1d_34_ig65m_32frms": "",  # noqa: E501
    "r2plus1d_152_ig65m_8frms": "",  # noqa: E501
    "r2plus1d_152_ig65m_32frms": "",  # noqa: E501
}


def _generic_r2pluls1d(arch, pretrained=False, progress=False, **kwargs):
    model = VideoResNet(**kwargs)

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


def r2plus1d_34(pretraining="", progress=False, **kwargs):
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

    model = _generic_r2pluls1d(
        arch
        pretrained,
        progress,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
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


def r2plus1d_152(pretraining="", progress=False, **kwargs):
    avail_pretrainings = [
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

    model = _generic_r2pluls1d(
        arch
        pretrained,
        progress,
        block=Bottleneck,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 8, 36, 3],
        stem=R2Plus1dStem,
        **kwargs
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
