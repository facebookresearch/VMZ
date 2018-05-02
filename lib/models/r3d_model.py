# Copyright 2018-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models.video_model import VideoModelBuilder

# For more depths, add the block config here
BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 3, 4, 3),
    34: (3, 4, 6, 3),
}


def create_model(
    model,
    model_name,
    model_depth,
    num_labels,
    num_channels,
    crop_size,
    clip_length,
    data,
    is_test,
):
    if model_name == 'r2d' or model_name == 'r2df':
        if model_name == 'r2d':
            creator = create_r2d
            conv1_kernel_length = clip_length
            final_temporal_kernel = 1
        else:
            creator = create_r2df
            conv1_kernel_length = 1
            final_temporal_kernel = clip_length
        last_out = creator(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            no_bias=True,
            no_loss=True,
            final_spatial_kernel=7 if crop_size == 112 else 14,
            model_depth=model_depth,
            conv1_kernel_length=conv1_kernel_length,
            final_temporal_kernel=final_temporal_kernel,
        )
    elif model_name[0:2] == 'mc' or model_name[0:3] == 'rmc':
        if model_name[0:2] == 'mc':
            mc_level = int(model_name[2])
            temporal_kernel = [8, 8, 4, 2]
            creator = create_mcx
        else:
            mc_level = int(model_name[3])
            temporal_kernel = [1, 1, 2, 4]
            creator = create_rmcx
        last_out = creator(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            no_bias=True,
            no_loss=True,
            final_spatial_kernel=7 if crop_size == 112 else 14,
            final_temporal_kernel=int(clip_length / 8) *
            temporal_kernel[mc_level - 2],
            model_depth=model_depth,
            mc_level=mc_level,
        )
    elif model_name == 'r3d' or model_name == 'r2plus1d':
        last_out = create_r3d(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            no_bias=True,
            no_loss=True,
            final_spatial_kernel=7 if crop_size == 112 else 14,
            final_temporal_kernel=int(clip_length / 8),
            model_depth=model_depth,
            is_decomposed=(model_name == 'r2plus1d'),
        )
    return last_out


# 3d or (2+1)d resnets, input 3 x t*8 x 112 x 112
# the final conv output is 512 * t * 7 * 7
def create_r3d(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    is_decomposed=False,
    spatial_bn_mom=0.9,
):

    # conv1 + maxpool
    if not is_decomposed:
        model.ConvNd(
            data,
            'conv1',
            num_input_channels,
            64,
            [3, 7, 7],
            weight_init=("MSRAFill", {}),
            strides=[1, 2, 2],
            pads=[1, 3, 3] * 2,
            no_bias=no_bias
        )
    else:
        model.ConvNd(
            data,
            'conv1_middle',
            num_input_channels,
            45,
            [1, 7, 7],
            weight_init=("MSRAFill", {}),
            strides=[1, 2, 2],
            pads=[0, 3, 3] * 2,
            no_bias=no_bias
        )

        model.SpatialBN(
            'conv1_middle',
            'conv1_middle_spatbn_relu',
            45,
            epsilon=1e-3,
            momentum=spatial_bn_mom,
            is_test=is_test
        )
        model.Relu('conv1_middle_spatbn_relu', 'conv1_middle_spatbn_relu')

        model.ConvNd(
            'conv1_middle_spatbn_relu',
            'conv1',
            45,
            64,
            [3, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[1, 1, 1],
            pads=[1, 0, 0] * 2,
            no_bias=no_bias
        )

    model.SpatialBN(
        'conv1',
        'conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=spatial_bn_mom,
        is_test=is_test
    )
    model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(model, 'conv1_spatbn_relu', no_bias=no_bias,
                                is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        builder.add_simple_block(64, 64, is_decomposed=is_decomposed)

    # conv_3x
    builder.add_simple_block(
        64, 128, down_sampling=True, is_decomposed=is_decomposed)
    for _ in range(n2 - 1):
        builder.add_simple_block(128, 128, is_decomposed=is_decomposed)

    # conv_4x
    builder.add_simple_block(
        128, 256, down_sampling=True, is_decomposed=is_decomposed)
    for _ in range(n3 - 1):
        builder.add_simple_block(256, 256, is_decomposed=is_decomposed)

    # conv_5x
    builder.add_simple_block(
        256, 512, down_sampling=True, is_decomposed=is_decomposed)
    for _ in range(n4 - 1):
        builder.add_simple_block(512, 512, is_decomposed=is_decomposed)

    # Final layers
    final_avg = model.AveragePool(
        builder.prev_blob,
        'final_avg',
        kernels=[
            final_temporal_kernel,
            final_spatial_kernel,
            final_spatial_kernel
        ],
        strides=[1, 1, 1],
    )

    last_out = model.FC(
        final_avg, 'last_out_L{}'.format(num_labels), 512, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax(last_out, "softmax")


# 2d resnet18, input 3 x t*8 x 112 x 112
def create_r2df(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    conv1_kernel_length=1,
    spatial_bn_mom=0.9,
):
    assert conv1_kernel_length == 1
    # conv1 + maxpool
    model.ConvNd(
        data,
        'conv1',
        num_input_channels,
        64,
        [conv1_kernel_length, 7, 7],
        weight_init=("MSRAFill", {}),
        strides=[1, 2, 2],
        pads=[0, 3, 3] * 2,
        no_bias=no_bias
    )

    model.SpatialBN(
        'conv1',
        'conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=spatial_bn_mom,
        is_test=is_test
    )
    model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(model, 'conv1_spatbn_relu', no_bias=no_bias,
                                is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        builder.add_simple_block(64, 64, is_real_3d=False)

    # conv_3x
    builder.add_simple_block(
        64, 128, down_sampling=True, is_real_3d=False)
    for _ in range(n2 - 1):
        builder.add_simple_block(128, 128, is_real_3d=False)

    # conv_4x
    builder.add_simple_block(
        128, 256, down_sampling=True, is_real_3d=False)
    for _ in range(n3 - 1):
        builder.add_simple_block(256, 256, is_real_3d=False)

    # conv_5x
    builder.add_simple_block(
        256, 512, down_sampling=True, is_real_3d=False)
    for _ in range(n4 - 1):
        builder.add_simple_block(512, 512, is_real_3d=False)

    # Final layers
    final_avg = model.AveragePool(
        builder.prev_blob,
        'final_avg',
        kernels=[
            final_temporal_kernel,
            final_spatial_kernel,
            final_spatial_kernel
        ],
        strides=[1, 1, 1],
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = model.FC(
        final_avg, 'last_out_L{}'.format(num_labels), 512, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax(last_out, "softmax")


def create_mcx(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    mc_level=2,
    spatial_bn_mom=0.9,
):
    assert mc_level >= 2 and mc_level <= 5

    # conv1 + maxpool
    model.ConvNd(
        data,
        'conv1',
        num_input_channels,
        64,
        [3, 7, 7],
        weight_init=("MSRAFill", {}),
        strides=[1, 2, 2],
        pads=[1, 3, 3] * 2,
        no_bias=no_bias
    )

    model.SpatialBN(
        'conv1',
        'conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=spatial_bn_mom,
        is_test=is_test
    )
    model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(model, 'conv1_spatbn_relu', no_bias=no_bias,
                                is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        builder.add_simple_block(
            64, 64, is_real_3d=True if mc_level > 2 else False)

    # conv_3x
    builder.add_simple_block(
        64, 128, down_sampling=True,
        is_real_3d=True if mc_level > 3 else False)
    for _ in range(n2 - 1):
        builder.add_simple_block(
            128, 128, is_real_3d=True if mc_level > 3 else False)

    # conv_4x
    builder.add_simple_block(
        128, 256, down_sampling=True,
        is_real_3d=True if mc_level > 4 else False)
    for _ in range(n3 - 1):
        builder.add_simple_block(
            256, 256,
            is_real_3d=True if mc_level > 4 else False)

    # conv_5x
    builder.add_simple_block(256, 512, down_sampling=True, is_real_3d=False)
    for _ in range(n4 - 1):
        builder.add_simple_block(512, 512, is_real_3d=False)

    # Final layers
    final_avg = model.AveragePool(
        builder.prev_blob,
        'final_avg',
        kernels=[
            final_temporal_kernel,
            final_spatial_kernel,
            final_spatial_kernel
        ],
        strides=[1, 1, 1],
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = model.FC(
        final_avg, 'last_out_L{}'.format(num_labels), 512, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax(last_out, "softmax")


def create_r2d(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    final_spatial_kernel=7,
    model_depth=18,
    conv1_kernel_length=8,
    final_temporal_kernel=1,
    spatial_bn_mom=0.9,
):
    assert final_temporal_kernel == 1
    # conv1 + maxpool
    model.ConvNd(
        data,
        'conv1',
        num_input_channels,
        64,
        [conv1_kernel_length, 7, 7],
        weight_init=("MSRAFill", {}),
        strides=[1, 2, 2],
        pads=[0, 3, 3] * 2,
        no_bias=no_bias
    )

    model.SpatialBN(
        'conv1',
        'conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=spatial_bn_mom,
        is_test=is_test
    )
    model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(model, 'conv1_spatbn_relu', no_bias=no_bias,
                                is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        builder.add_simple_block(64, 64, is_real_3d=False)

    # conv_3x
    builder.add_simple_block(64, 128, down_sampling=True, is_real_3d=False)
    for _ in range(n2 - 1):
        builder.add_simple_block(128, 128, is_real_3d=False)

    # conv_4x
    builder.add_simple_block(128, 256, down_sampling=True, is_real_3d=False)
    for _ in range(n3 - 1):
        builder.add_simple_block(256, 256, is_real_3d=False)

    # conv_5x
    builder.add_simple_block(256, 512, down_sampling=True, is_real_3d=False)
    for _ in range(n4 - 1):
        builder.add_simple_block(512, 512, is_real_3d=False)

    # Final layers
    final_avg = model.AveragePool(
        builder.prev_blob,
        'final_avg',
        kernels=[
            final_temporal_kernel,
            final_spatial_kernel,
            final_spatial_kernel
        ],
        strides=[1, 1, 1],
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = model.FC(
        final_avg, 'last_out_L{}'.format(num_labels), 512, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax(last_out, "softmax")


def create_rmcx(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_loss=False,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    mc_level=2,
    spatial_bn_mom=0.9,
):
    assert mc_level >= 2 and mc_level <= 5

    # conv1 + maxpool
    model.ConvNd(
        data,
        'conv1',
        num_input_channels,
        64,
        [1, 7, 7],
        weight_init=("MSRAFill", {}),
        strides=[1, 2, 2],
        pads=[0, 3, 3] * 2,
        no_bias=no_bias
    )

    model.SpatialBN(
        'conv1',
        'conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=spatial_bn_mom,
        is_test=is_test
    )
    model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(model, 'conv1_spatbn_relu', no_bias=no_bias,
                                is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    # conv_2x
    for _ in range(n1):
        builder.add_simple_block(
            64, 64, is_real_3d=True if mc_level <= 2 else False)

    # conv_3x
    builder.add_simple_block(
        64, 128, down_sampling=True,
        is_real_3d=True if mc_level <= 3 else False)
    for _ in range(n2 - 1):
        builder.add_simple_block(
            128, 128, is_real_3d=True if mc_level <= 3 else False)

    # conv_4x
    builder.add_simple_block(
        128, 256, down_sampling=True,
        is_real_3d=True if mc_level <= 4 else False)
    for _ in range(n3 - 1):
        builder.add_simple_block(
            256, 256, is_real_3d=True if mc_level <= 4 else False)

    # conv_5x
    builder.add_simple_block(
        256, 512, down_sampling=True, is_real_3d=True)
    for _ in range(n4 - 1):
        builder.add_simple_block(512, 512, is_real_3d=True)

    # Final layers
    final_avg = model.AveragePool(
        builder.prev_blob,
        'final_avg',
        kernels=[
            final_temporal_kernel,
            final_spatial_kernel,
            final_spatial_kernel
        ],
        strides=[1, 1, 1],
    )

    # Final dimension of the "image" is reduced to 7x7
    last_out = model.FC(
        final_avg, 'last_out_L{}'.format(num_labels), 512, num_labels
    )

    if no_loss:
        return last_out

    # If we create model for training, use softmax-with-loss
    if (label is not None):
        (softmax, loss) = model.SoftmaxWithLoss(
            [last_out, label],
            ["softmax", "loss"],
        )

        return (softmax, loss)
    else:
        # For inference, we just return softmax
        return model.Softmax(last_out, "softmax")
