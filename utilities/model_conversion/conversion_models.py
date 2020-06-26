import torch.nn as nn

from torchvision.models.video.resnet import (
    BasicBlock,
    Bottleneck,
    BasicStem,
    R2Plus1dStem,
    _video_resnet,
)


class BasicStem_Pool(nn.Sequential):
    def __init__(self):
        super(BasicStem_Pool, self).__init__(
            nn.Conv3d(
                3,
                64,
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),
                padding=(1, 3, 3),
                bias=False,
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )


__all__ = ["r2plus1d_34", "r2plus1d_152", "ir_csn_152", "ip_csn_152", "ip_csn_50"]


class Conv3DDepthwise(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):

        assert in_planes == out_planes
        super(Conv3DDepthwise, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            groups=in_planes,
            bias=False,
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class IPConv3DDepthwise(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        assert in_planes == out_planes
        super(IPConv3DDepthwise, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_planes),
            # nn.ReLU(inplace=True),
            Conv3DDepthwise(out_planes, out_planes, None, stride),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


class Conv2Plus1D(nn.Sequential):
    def __init__(self, in_planes, out_planes, midplanes, stride=1, padding=1):

        midplanes = (in_planes * out_planes * 3 * 3 * 3) // (
            in_planes * 3 * 3 + 3 * out_planes
        )
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(
                in_planes,
                midplanes,
                kernel_size=(1, 3, 3),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
            ),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                midplanes,
                out_planes,
                kernel_size=(3, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
            ),
        )

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)


def r2plus1d_34(pretrained=False, progress=False, **kwargs):
    model = _video_resnet(
        "r2plus1d_34",
        False,
        False,
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
        **kwargs
    )
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
    return model


def r2plus1d_152(pretrained=False, progress=False, **kwargs):
    model = _video_resnet(
        "r2plus1d_152",
        False,
        False,
        block=Bottleneck,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 8, 36, 3],
        stem=R2Plus1dStem,
        **kwargs
    )
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
    return model


def ir_csn_152(pretrained=False, progress=False, **kwargs):
    model = _video_resnet(
        "ir_csn_152",
        False,
        False,
        block=Bottleneck,
        conv_makers=[Conv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool,
        **kwargs
    )
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
    return model


def ip_csn_152(pretrained=False, progress=False, **kwargs):
    model = _video_resnet(
        "ip_csn_152",
        False,
        False,
        block=Bottleneck,
        conv_makers=[IPConv3DDepthwise] * 4,
        layers=[3, 8, 36, 3],
        stem=BasicStem_Pool,
        **kwargs
    )
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
    return model


def ip_csn_50(pretrained=False, progress=False, **kwargs):
    model = _video_resnet(
        "ip_csn_50",
        False,
        False,
        block=Bottleneck,
        conv_makers=[IPConv3DDepthwise] * 4,
        layers=[3, 8, 6, 3],
        stem=BasicStem_Pool,
        **kwargs
    )
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.eps = 1e-3
    return model
