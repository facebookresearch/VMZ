import torch.nn

# TODO: upload models and load them
model_urls = {
    "r2plus1d_34_kinetics_8frms": "",  # noqa: E501
    "r2plus1d_34_kinetics_32frms": "",  # noqa: E501
    "r2plus1d_34_ig65m_8frms": "",  # noqa: E501
    "r2plus1d_34_ig65m_32frms": "",  # noqa: E501
    "r2plus1d_152_ig65m_8frms": "",  # noqa: E501
    "r2plus1d_152_ig65m_32frms": "",  # noqa: E501
    "ir_csn_152_ig65m_32frms": "",  # noqa: E501
    "ip_csn_152_ig65m_32frms": "",  # noqa: E501
}


def _generic_resnet(arch, pretrained=False, progress=False, **kwargs):
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
