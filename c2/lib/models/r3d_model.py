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

import numpy as np
import logging
from caffe2.python import brew

logging.basicConfig()
log = logging.getLogger("r3d_model")
log.setLevel(logging.INFO)

from models.builder.video_model \
    import VideoModelBuilder

# For more depths, add the block config here
BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    101: (3, 4, 23, 3),
    152: (3, 8, 36, 3),
}

SHALLOW_FILTER_CONFIG = [
    [64, 64],
    [128, 128],
    [256, 256],
    [512, 512]
]

DEEP_FILTER_CONFIG = [
    [256, 64],
    [512, 128],
    [1024, 256],
    [2048, 512]
]

model_blocktype = {
    'r2plus1d': '2.5d',
    'r3d': '3d',
    'ir-csn': '3d-sep',
    'ip-csn': '0.3d'
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
    use_full_ft=True,
    channel_multiplier=1.0,
    bottleneck_multiplier=1.0,
    use_dropout=False,
    conv1_temporal_stride=1,
    conv1_temporal_kernel=3,
    use_convolutional_pred=False,
    use_pool1=False,
):
    if crop_size == 112 or crop_size == 128:
        assert use_convolutional_pred or crop_size == 112
        final_spatial_kernel = 7
    elif crop_size == 224 or crop_size == 256:
        assert use_convolutional_pred or crop_size == 224
        if use_pool1:
            final_spatial_kernel = 7
        else:
            final_spatial_kernel = 14
    elif crop_size == 320:
        assert use_convolutional_pred
        assert use_pool1
        final_spatial_kernel = 7
    else:
        print('unknown crop size')
        assert 0

    if model_name[0:3] == 'r2d':
        if model_name == 'r2df':
            creator = create_r2df
            conv1_kernel_length = 1
            final_temporal_kernel = clip_length
        else:
            creator = create_r2d
            conv1_kernel_length = clip_length
            final_temporal_kernel = 1

        last_out = creator(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            no_bias=True,
            final_spatial_kernel=final_spatial_kernel,
            model_depth=model_depth,
            conv1_kernel_length=conv1_kernel_length,
            final_temporal_kernel=final_temporal_kernel,
            use_pool1=use_pool1,
            block_type=('3d-sep' if 'sep' in model_name else '3d'),
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
            final_spatial_kernel=final_spatial_kernel,
            final_temporal_kernel=int(clip_length / 8)
            * temporal_kernel[mc_level - 2],
            model_depth=model_depth,
            mc_level=mc_level,
            use_pool1=use_pool1,
        )
    elif model_name == 'r3d' or model_name == 'r2plus1d' or \
            '-csn' in model_name:
        block_type = model_blocktype[model_name]
        if model_depth <= 18 or model_depth == 34:
            transformation_type = 'simple_block'
        else:
            transformation_type = 'bottleneck'
        creator = create_r3d
        if clip_length >= 8:
            final_temporal_kernel = int(clip_length / 8 / conv1_temporal_stride)
        else:
            final_temporal_kernel = 1
        last_out = creator(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            use_full_ft=use_full_ft,
            no_bias=True,
            final_spatial_kernel=final_spatial_kernel,
            final_temporal_kernel=final_temporal_kernel,
            model_depth=model_depth,
            block_type=block_type,
            transformation_type=transformation_type,
            channel_multiplier=channel_multiplier,
            bottleneck_multiplier=bottleneck_multiplier,
            use_dropout=use_dropout,
            conv1_temporal_stride=conv1_temporal_stride,
            conv1_temporal_kernel=conv1_temporal_kernel,
            clip_length=clip_length,
            use_convolutional_pred=use_convolutional_pred,
            use_pool1=use_pool1,
        )
    else:
        # you will unlikely to reach here
        log.info('Unknow model name {}'.format(model_name))
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
    use_full_ft=True,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    block_type='3d',
    transformation_type='simple_block',
    channel_multiplier=1.0,
    bottleneck_multiplier=1.0,
    use_dropout=False,
    conv1_temporal_stride=1,
    conv1_temporal_kernel=3,
    spatial_bn_mom=0.9,
    clip_length=8,
    use_shuffle=False,
    use_convolutional_pred=False,
    use_pool1=False,
):
    assert conv1_temporal_kernel == 3 or conv1_temporal_kernel == 5

    if not use_full_ft:
        is_test = True

    # conv1 + maxpool
    if block_type != '2.5d' and block_type != '2.5d-sep':
        model.ConvNd(
            data,
            'conv1',
            num_input_channels,
            64,
            [conv1_temporal_kernel, 7, 7],
            weight_init=("MSRAFill", {}),
            strides=[conv1_temporal_stride, 2, 2],
            pads=[1 if conv1_temporal_kernel == 3 else 2, 3, 3] * 2,
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
            [conv1_temporal_kernel, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[conv1_temporal_stride, 1, 1],
            pads=[1 if conv1_temporal_kernel == 3 else 2, 0, 0] * 2,
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
    last_conv1 = model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')

    if use_pool1:
        last_conv1 = model.MaxPool(
            'conv1_spatbn_relu',
            'pool1',
            kernels=[1, 3, 3],
            strides=[1, 2, 2],
            pads=[0, 1, 1] * 2,
        )

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(model, last_conv1, no_bias=no_bias,
                                is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    if transformation_type == 'simple_block':
        transformation = builder.add_simple_block
    elif transformation_type == 'bottleneck':
        transformation = builder.add_bottleneck
    else:
        print('Unknown transformation type...')

    if model_depth <= 34:
        filter_config = SHALLOW_FILTER_CONFIG
    else:
        filter_config = DEEP_FILTER_CONFIG
    filter_config = np.multiply(
        filter_config, channel_multiplier).astype(np.int)

    # conv_2x
    transformation(
        64, filter_config[0][0],
        int(filter_config[0][1] * bottleneck_multiplier),
        block_type=block_type,
        use_shuffle=use_shuffle)
    for _ in range(n1 - 1):
        transformation(
            filter_config[0][0], filter_config[0][0],
            int(filter_config[0][1] * bottleneck_multiplier),
            block_type=block_type,
            use_shuffle=use_shuffle)

    # conv_3x
    transformation(
        filter_config[0][0], filter_config[1][0],
        int(filter_config[1][1] * bottleneck_multiplier),
        down_sampling=True,
        block_type=block_type,
        use_shuffle=use_shuffle)
    for _ in range(n2 - 1):
        transformation(
            filter_config[1][0], filter_config[1][0],
            int(filter_config[1][1] * bottleneck_multiplier),
            block_type=block_type,
            use_shuffle=use_shuffle)

    # conv_4x
    if clip_length < 4:
        transformation(
            filter_config[1][0], filter_config[2][0],
            int(filter_config[2][1] * bottleneck_multiplier),
            down_sampling=True,
            down_sampling_temporal=False,
            block_type=block_type,
            use_shuffle=use_shuffle)
    else:
        transformation(
            filter_config[1][0], filter_config[2][0],
            int(filter_config[2][1] * bottleneck_multiplier),
            down_sampling=True,
            block_type=block_type,
            use_shuffle=use_shuffle)
    for _ in range(n3 - 1):
        transformation(
            filter_config[2][0], filter_config[2][0],
            int(filter_config[2][1] * bottleneck_multiplier),
            block_type=block_type,
            use_shuffle=use_shuffle)

    # conv_5x
    if clip_length < 8:
        transformation(
            filter_config[2][0], filter_config[3][0],
            int(filter_config[3][1] * bottleneck_multiplier),
            down_sampling=True,
            down_sampling_temporal=False,
            block_type=block_type,
            use_shuffle=use_shuffle)
    else:
        transformation(
            filter_config[2][0], filter_config[3][0],
            int(filter_config[3][1] * bottleneck_multiplier),
            down_sampling=True, block_type=block_type,
            use_shuffle=use_shuffle)
    for _ in range(n4 - 1):
        transformation(
            filter_config[3][0], filter_config[3][0],
            int(filter_config[3][1] * bottleneck_multiplier),
            block_type=block_type,
            use_shuffle=use_shuffle)

    # Final layers
    model.AveragePool(
        builder.prev_blob,
        'final_avg',
        kernels=[
            final_temporal_kernel,
            final_spatial_kernel,
            final_spatial_kernel
        ],
        strides=[1, 1, 1],
    )
    if use_dropout:
        dropout = brew.dropout(model, 'final_avg', 'dropout', is_test=is_test)
    else:
        dropout = 'final_avg'

    if not use_full_ft:
        dropout = model.StopGradient(dropout, dropout)

    if use_convolutional_pred:
        assert is_test
        last_out = model.ConvNd(
            dropout,
            'last_out_L{}'.format(num_labels),
            filter_config[3][0],
            num_labels,
            [1, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[1, 1, 1],
            pads=[0, 0, 0] * 2,
            no_bias=False
        )
    else:
        last_out = model.FC(
            dropout, 'last_out_L{}'.format(num_labels),
            filter_config[3][0],
            num_labels
        )

    return last_out


# 2d resnet18, input 3 x t*8 x 112 x 112
def create_r2df(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
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

    if model_depth <= 34:
        transformation = builder.add_simple_block
        filter_config = SHALLOW_FILTER_CONFIG
    else:
        transformation = builder.add_bottleneck
        filter_config = DEEP_FILTER_CONFIG

    # conv_2x
    transformation(
        64, filter_config[0][0], filter_config[0][1], is_real_3d=False)
    for _ in range(n1 - 1):
        transformation(
            filter_config[0][0], filter_config[0][0],
            filter_config[0][1], is_real_3d=False)

    # conv_3x
    transformation(
        filter_config[0][0], filter_config[1][0],
        filter_config[1][1], down_sampling=True, is_real_3d=False)
    for _ in range(n2 - 1):
        transformation(
            filter_config[1][0], filter_config[1][0],
            filter_config[1][1], is_real_3d=False)

    # conv_4x
    transformation(
        filter_config[1][0], filter_config[2][0], filter_config[2][1],
        down_sampling=True, is_real_3d=False)
    for _ in range(n3 - 1):
        transformation(
            filter_config[2][0], filter_config[2][0],
            filter_config[2][1], is_real_3d=False)

    # conv_5x
    transformation(
        filter_config[2][0], filter_config[3][0], filter_config[3][1],
        down_sampling=True, is_real_3d=False)
    for _ in range(n4 - 1):
        transformation(
            filter_config[3][0], filter_config[3][0],
            filter_config[3][1], is_real_3d=False)

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
        final_avg,
        'last_out_L{}'.format(num_labels),
        filter_config[3][0],
        num_labels
    )

    return last_out


def create_mcx(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
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
        64, 128, down_sampling=True, is_real_3d=True if mc_level > 3 else False)
    for _ in range(n2 - 1):
        builder.add_simple_block(
            128, 128, is_real_3d=True if mc_level > 3 else False)

    # conv_4x
    builder.add_simple_block(
        128, 256, down_sampling=True, is_real_3d=True if mc_level > 4 else False)
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

    return last_out


def create_r2d(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
    no_bias=0,
    final_spatial_kernel=7,
    model_depth=18,
    conv1_kernel_length=8,
    final_temporal_kernel=1,
    spatial_bn_mom=0.9,
    use_pool1=False,
    block_type='3d',
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
    last_conv1 = model.Relu('conv1_spatbn_relu', 'conv1_spatbn_relu')

    if use_pool1:
        last_conv1 = model.MaxPool(
            'conv1_spatbn_relu',
            'pool1',
            kernels=[1, 3, 3],
            strides=[1, 2, 2],
            pads=[0, 1, 1] * 2,
        )

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    builder = VideoModelBuilder(model, last_conv1, no_bias=no_bias,
                                is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    if model_depth <= 34:
        if model_depth == 26:
            transformation = builder.add_bottleneck
        else:
            transformation = builder.add_simple_block
        filter_config = SHALLOW_FILTER_CONFIG
    else:
        transformation = builder.add_bottleneck
        filter_config = DEEP_FILTER_CONFIG

    # conv_2x
    transformation(
        64, filter_config[0][0], filter_config[0][1],
        is_real_3d=False, block_type=block_type)
    for _ in range(n1 - 1):
        transformation(
            filter_config[0][0], filter_config[0][0],
            filter_config[0][1], is_real_3d=False, block_type=block_type)

    # conv_3x
    transformation(
        filter_config[0][0], filter_config[1][0],
        filter_config[1][1], down_sampling=True,
        is_real_3d=False, block_type=block_type)
    for _ in range(n2 - 1):
        transformation(
            filter_config[1][0], filter_config[1][0],
            filter_config[1][1], is_real_3d=False, block_type=block_type)

    # conv_4x
    transformation(
        filter_config[1][0], filter_config[2][0], filter_config[2][1],
        down_sampling=True, is_real_3d=False, block_type=block_type)
    for _ in range(n3 - 1):
        transformation(
            filter_config[2][0], filter_config[2][0],
            filter_config[2][1], is_real_3d=False, block_type=block_type)

    # conv_5x
    transformation(
        filter_config[2][0], filter_config[3][0], filter_config[3][1],
        down_sampling=True, is_real_3d=False, block_type=block_type)
    for _ in range(n4 - 1):
        transformation(
            filter_config[3][0], filter_config[3][0],
            filter_config[3][1], is_real_3d=False, block_type=block_type)

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
        final_avg,
        'last_out_L{}'.format(num_labels),
        filter_config[3][0],
        num_labels
    )

    return last_out


def create_rmcx(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=False,
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

    return last_out
