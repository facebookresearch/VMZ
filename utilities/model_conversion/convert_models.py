import os
import argparse
import sys
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.models.video.resnet import (
    BasicBlock,
    R2Plus1dStem,
    BasicStem,
    Bottleneck,
)

# TODO: add packaged version
import conversion_models as video_classification
from conversion_models import (
    ip_csn_152,
    ir_csn_152,
    Conv3DDepthwise,
    IPConv3DDepthwise,
    Conv2Plus1D,
    BasicStem_Pool,
)


def blobs_from_pkl(path):
    # tested and works
    path = Path(path)
    with path.open(mode="rb") as f:
        pkl = pickle.load(f, encoding="latin1")
        blobs = pkl["blobs"]
        return blobs


def copy_tensor(data, blobs, name):
    # works with restrictions
    try:
        tensor = torch.from_numpy(blobs[name])
    except KeyError as e:
        print(f"No blob {name}")
        search_key = name[0:6]
        res = [key for key, val in blobs.items() if search_key in key]
        print(f"Values for substring {search_key} : " + str(res))

    del blobs[name]  # enforce: use at most once
    assert data.size() == tensor.size()
    assert data.dtype == tensor.dtype
    data.copy_(tensor)


def copy_conv(module, blobs, prefix):
    # tested and works
    assert isinstance(module, nn.Conv3d)
    assert module.bias is None
    copy_tensor(module.weight.data, blobs, prefix + "_w")


def copy_bn(module, blobs, prefix):
    # tested and works
    assert isinstance(module, nn.BatchNorm3d)
    copy_tensor(module.weight.data, blobs, prefix + "_s")
    copy_tensor(module.running_mean.data, blobs, prefix + "_rm")
    copy_tensor(module.running_var.data, blobs, prefix + "_riv")
    copy_tensor(module.bias.data, blobs, prefix + "_b")


def copy_conv3d(module, blobs, i, j):
    # tested and works
    copy_conv(module[0], blobs, "comp_" + str(i) + "_conv_" + str(j))
    copy_bn(module[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j))
    assert isinstance(module[2], nn.ReLU)


def copy_separated(module, blobs, i, j):
    # tested and works
    if isinstance(module, Conv2Plus1D):
        assert len(module) == 4
        copy_conv(module[0], blobs, "comp_" + str(i) + "_conv_" + str(j) + "_middle")
        copy_bn(module[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j) + "_middle")
        assert isinstance(module[2], nn.ReLU)
        copy_conv(module[3], blobs, "comp_" + str(i) + "_conv_" + str(j))
    elif isinstance(module, IPConv3DDepthwise):
        assert len(module) == 3
        copy_conv(module[0], blobs, "comp_" + str(i) + "_conv_" + str(j) + "_middle")
        copy_bn(module[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j) + "_middle")
        copy_conv(module[2], blobs, "comp_" + str(i) + "_conv_" + str(j))


def copy_fc(module, blobs):
    # works and tested
    assert isinstance(module, nn.Linear)
    n = module.out_features
    copy_tensor(module.bias.data, blobs, "last_out_L" + str(n) + "_b")
    copy_tensor(module.weight.data, blobs, "last_out_L" + str(n) + "_w")


def copy_stem(module, blobs):
    if isinstance(module, R2Plus1dStem):
        copy_r25d_stem(module, blobs)
    else:
        copy_basic_stem(module, blobs)


def copy_r25d_stem(module, blobs):
    assert isinstance(module, R2Plus1dStem)
    assert len(module) == 6
    copy_conv(module[0], blobs, "conv1_middle")
    copy_bn(module[1], blobs, "conv1_middle_spatbn_relu")
    assert isinstance(module[2], nn.ReLU)
    copy_conv(module[3], blobs, "conv1")
    copy_bn(module[4], blobs, "conv1_spatbn_relu")
    assert isinstance(module[5], nn.ReLU)


def copy_basic_stem(module, blobs):
    # works and tested
    assert isinstance(module, (BasicStem, BasicStem_Pool))
    assert len(module) == 3 or len(module) == 4
    copy_conv(module[0], blobs, "conv1")
    copy_bn(module[1], blobs, "conv1_spatbn_relu")
    assert isinstance(module[2], nn.ReLU)
    if len(module) == 4:
        assert isinstance(module[3], nn.MaxPool3d)


def copy_bottleneck(module, blobs, i):
    assert isinstance(module, Bottleneck)
    # Bottleneck 1:
    assert len(module.conv1) == 3
    copy_conv3d(module.conv1, blobs, i, 1)

    # Bottleneck 2:
    assert len(module.conv2) == 3
    j = 2
    if isinstance(module.conv2[0], Conv2Plus1D) or isinstance(
        module.conv2[0], IPConv3DDepthwise
    ):
        copy_separated(module.conv2[0], blobs, i, 2)
        assert isinstance(module.conv2[1], nn.BatchNorm3d)
        copy_bn(module.conv2[1], blobs, "comp_" + str(i) + "_spatbn_" + str(2))
    else:
        if isinstance(module.conv2[0], Conv3DDepthwise):
            j = 3
        copy_conv3d(module.conv2, blobs, i, j)

    assert isinstance(module.conv2[2], nn.ReLU)

    # Bottleneck 3:
    assert len(module.conv1) == 3
    j = j + 1
    copy_conv(module.conv3[0], blobs, "comp_" + str(i) + "_conv_" + str(j))
    assert isinstance(module.conv3[1], nn.BatchNorm3d)
    copy_bn(module.conv3[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j))

    if module.downsample is not None:
        assert len(module.downsample) == 2
        assert isinstance(module.downsample[0], nn.Conv3d)
        assert isinstance(module.downsample[1], nn.BatchNorm3d)
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(
            module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn"
        )


def copy_basicblock(module, blobs, i):
    assert isinstance(module, BasicBlock)

    j = 1  # Conv1:
    assert len(module.conv1) == 3
    if isinstance(module.conv1[0], Conv2Plus1D) or isinstance(
        module.conv2[0], IPConv3DDepthwise
    ):
        copy_separated(module.conv1[0], blobs, i, j)
        assert isinstance(module.conv1[1], nn.BatchNorm3d)
        copy_bn(module.conv1[1], blobs, "comp_" + str(i) + "_spatbn_" + str(j))
    else:
        copy_conv3d(module.conv1, blobs, i, j)

    assert isinstance(module.conv1[2], nn.ReLU)

    j = 2  # Conv 2
    assert len(module.conv2) == 2
    if isinstance(module.conv2[0], Conv2Plus1D) or isinstance(
        module.conv2[0], IPConv3DDepthwise
    ):
        copy_separated(module.conv2[0], blobs, i, 2)
        assert isinstance(module.conv2[1], nn.BatchNorm3d)
        copy_bn(module.conv2[1], blobs, "comp_" + str(i) + "_spatbn_" + str(2))
    else:
        copy_conv3d(module.conv2, blobs, i, j)

    if module.downsample is not None:
        assert len(module.downsample) == 2
        assert isinstance(module.downsample[0], nn.Conv3d)
        assert isinstance(module.downsample[1], nn.BatchNorm3d)
        copy_conv(module.downsample[0], blobs, "shortcut_projection_" + str(i))
        copy_bn(
            module.downsample[1], blobs, "shortcut_projection_" + str(i) + "_spatbn"
        )


def copy_layer(module, blobs, i):
    for curr_block in module:
        if isinstance(curr_block, Bottleneck):
            copy_bottleneck(curr_block, blobs, i)
            i += 1
        elif isinstance(curr_block, BasicBlock):
            copy_basicblock(curr_block, blobs, i)
            i += 1
        else:
            raise NotImplementedError
    return i


def init_canary(model):
    nan = float("nan")
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            nn.init.constant_(m.weight, nan)
        elif isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.running_mean, nan)
            nn.init.constant_(m.running_var, nan)
            nn.init.constant_(m.bias, nan)
        elif isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, nan)
            nn.init.constant_(m.bias, nan)


def check_canary(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            assert m.bias is None
            assert not torch.isnan(m.weight).any()
        elif isinstance(m, nn.BatchNorm3d):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.running_mean).any()
            assert not torch.isnan(m.running_var).any()
            assert not torch.isnan(m.bias).any()
        elif isinstance(m, nn.Linear):
            assert not torch.isnan(m.weight).any()
            assert not torch.isnan(m.bias).any()


def main(args):
    blobs = blobs_from_pkl(args.pkl)

    if not "last_out_L{}_w".format(args.classes) in blobs:
        sys.exit(
            "Number of --classes does not match the last linear layer in .pkl blobs"
        )

    if not "last_out_L{}_b".format(args.classes) in blobs:
        sys.exit(
            "Number of --classes does not match the last linear layer in .pkl blobs"
        )

    available_models = {
        **torchvision.models.video.__dict__,
        **video_classification.__dict__,
    }
    model = available_models[args.model](num_classes=args.classes)

    init_canary(model)
    copy_fc(model.fc, blobs)
    copy_stem(model.stem, blobs)

    i = copy_layer(model.layer1, blobs, 0)
    i = copy_layer(model.layer2, blobs, i)
    i = copy_layer(model.layer3, blobs, i)
    i = copy_layer(model.layer4, blobs, i)

    assert not blobs, "{}".format(sorted(blobs.keys()))
    check_canary(model)

    batch = torch.rand(1, 3, args.frames, args.inputsize, args.inputsize)  # NxCxTxHxW
    torch.save(model.state_dict(), args.out.with_suffix(".pth"))

    # Check pth roundtrip into fresh model
    model = available_models[args.model](num_classes=args.classes)
    model.load_state_dict(torch.load(args.out.with_suffix(".pth")))
    out = model(batch)
    out_path = args.out.with_suffix(".pth ")
    print(f"Conversion finished: new model at {out_path}")
    print("Please verify sizes:")
    print("\tInput: ", batch.size(), " Output: ", out.size())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="Model conversion tool")

    parser.add_argument(
        "--pkl",
        type=str,
        default="",
        help=".pkl file to read the model layer weights from",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default="",
        help="prefix to save converted model layer weights to",
    )
    parser.add_argument(
        "--model", type=str, default="r2plus1d_34", help="model to create"
    )
    parser.add_argument(
        "--frames",
        type=int,
        choices=(8, 16, 32),
        default=32,
        help="frames per clip for video model (mostly to test downsampling)",
    )
    parser.add_argument(
        "--classes", type=int, default=400, help="Number of classes in last layer"
    )
    parser.add_argument(
        "--inputsize",
        type=int,
        choices=(112, 224),
        default=112,
        help="Input image size",
    )

    args = parser.parse_args()

    main(args)
