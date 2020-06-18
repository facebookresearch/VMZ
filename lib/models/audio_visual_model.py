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

from caffe2.python import brew

import logging
import numpy as np

logging.basicConfig()
log = logging.getLogger("av_model")
log.setLevel(logging.INFO)

from models.builder.video_model \
    import VideoModelBuilder
from models.builder.audio_model \
    import AudioModelBuilder

# For more depths, add the block config here
BLOCK_CONFIG = {
    18: (2, 2, 2, 2),
    26: (2, 2, 2, 2),
    34: (3, 4, 6, 3),
    50: (3, 4, 6, 3),
    68: (3, 4, 23, 3),
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
    acoustic_data="logmels",
    channel_multiplier=1.0,
    bottleneck_multiplier=1.0,
    conv1_temporal_stride=1,
    conv1_temporal_kernel=3,
    use_convolutional_pred=False,
    use_pool1=False,
    audio_input_3d=False,
    g_blend=False,
):
    if model_name[0:9] == 'av_resnet':
        model_type = model_name[10:]
        block_type = model_blocktype[model_type]
        if model_depth <= 18 or model_depth == 34:
            transformation_type = 'simple_block'
        else:
            transformation_type = 'bottleneck'
        if crop_size == 112 or crop_size == 128:
            assert use_convolutional_pred or crop_size == 112
            final_spatial_kernel = 7
        elif crop_size == 224 or crop_size == 256:
            assert use_convolutional_pred or crop_size == 224
            if use_pool1:
                final_spatial_kernel = 7
            else:
                final_spatial_kernel = 14
        else:
            print('unknown crop size')
            assert 0

        last_out = create_av_resnet(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            acoustic_data=acoustic_data,
            is_test=is_test,
            use_full_ft=use_full_ft,
            no_bias=True,
            final_spatial_kernel=final_spatial_kernel,
            final_temporal_kernel=int(clip_length / 8 / conv1_temporal_stride),
            model_depth=model_depth,
            block_type=block_type,
            transformation_type=transformation_type,
            bottleneck_multiplier=bottleneck_multiplier,
            channel_multiplier=channel_multiplier,
            conv1_temporal_stride=conv1_temporal_stride,
            conv1_temporal_kernel=conv1_temporal_kernel,
            clip_length=clip_length,
            use_convolutional_pred=use_convolutional_pred,
            use_pool1=use_pool1,
            audio_input_3d=audio_input_3d,
            g_blend=g_blend,
        )
    elif model_name == "a_resnet":
        if model_depth <= 18 or model_depth == 34:
            transformation_type = 'simple_block'
        else:
            transformation_type = 'bottleneck'
        last_out = create_acoustic_resnet(
            model=model,
            data=data,
            num_labels=num_labels,
            is_test=is_test,
            no_bias=True,
            model_depth=model_depth,
            transformation_type=transformation_type,
            audio_input_3d=audio_input_3d,
        )
    else:
        log.info('Unknown model name...')
    return last_out


# 3d or (2+1)d audio-visual resnets
# visual input: 3 x t*8 x 112(224) x 112(224)
# audio input: 1 x 100 x 40 (channel x time x frequency)
def create_av_resnet(
    model,
    data,
    num_input_channels,
    num_labels,
    acoustic_data="logmels",
    label=None,
    is_test=False,
    use_full_ft=True,
    no_bias=0,
    final_spatial_kernel=7,
    final_temporal_kernel=1,
    model_depth=18,
    block_type='3d',
    transformation_type='simple_block',
    bottleneck_multiplier=1.0,
    channel_multiplier=1.0,
    use_dropout=False,
    conv1_temporal_stride=1,
    conv1_temporal_kernel=3,
    spatial_bn_mom=0.9,
    clip_length=8,
    use_convolutional_pred=False,
    use_pool1=False,
    audio_input_3d=False,
    g_blend=False,
):
    # sanity checking of model params
    assert conv1_temporal_kernel == 3 or conv1_temporal_kernel == 5

    # conv1 + maxpool for visual model
    if block_type != '2.5d' and block_type != '2.5d-sep':
        model.ConvNd(
            data,
            'v_conv1',
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
            'v_conv1_middle',
            num_input_channels,
            45,
            [1, 7, 7],
            weight_init=("MSRAFill", {}),
            strides=[1, 2, 2],
            pads=[0, 3, 3] * 2,
            no_bias=no_bias
        )

        model.SpatialBN(
            'v_conv1_middle',
            'v_conv1_middle_spatbn_relu',
            45,
            epsilon=1e-3,
            momentum=spatial_bn_mom,
            is_test=is_test
        )
        model.Relu('v_conv1_middle_spatbn_relu', 'v_conv1_middle_spatbn_relu')

        model.ConvNd(
            'v_conv1_middle_spatbn_relu',
            'v_conv1',
            45,
            64,
            [conv1_temporal_kernel, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[conv1_temporal_stride, 1, 1],
            pads=[1 if conv1_temporal_kernel == 3 else 2, 0, 0] * 2,
            no_bias=no_bias
        )

    model.SpatialBN(
        'v_conv1',
        'v_conv1_spatbn_relu',
        64,
        epsilon=1e-3,
        momentum=spatial_bn_mom,
        is_test=is_test
    )
    v_conv1 = model.Relu('v_conv1_spatbn_relu', 'v_conv1_spatbn_relu')

    if use_pool1:
        v_conv1 = model.MaxPool(
            'v_conv1_spatbn_relu',
            'pool1',
            kernels=[1, 3, 3],
            strides=[1, 2, 2],
            pads=[0, 1, 1] * 2,
        )

    # conv1 equivalent of audio model
    # it is approximated by a bunch of 3x3 kernels
    if audio_input_3d:
        acoustic_data_swap = model.NCHW2NHWC(
            acoustic_data, acoustic_data + "_NHWC")
        acoustic_data_greyscale = model.ReduceBackMean(
            acoustic_data_swap,
            acoustic_data_swap + '_c_pool',
            num_reduce_dim=1)
        acoustic_data = acoustic_data_greyscale
    a_conv1 = model.Conv(acoustic_data, 'a_conv1', 1, 16, kernel=3, pad=1)

    a_builder = AudioModelBuilder(
        model, a_conv1, no_bias=no_bias,
        is_test=is_test, spatial_bn_mom=spatial_bn_mom, prefix='a_')

    a_builder.add_simple_block(16, 32, down_sampling=True)
    a_builder.add_simple_block(32, 32)
    a_builder.add_simple_block(32, 32)

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

    # Residual blocks...
    v_builder = VideoModelBuilder(
        model, v_conv1, no_bias=no_bias,
        is_test=is_test, spatial_bn_mom=spatial_bn_mom, prefix='v_')

    if transformation_type == 'simple_block':
        v_transformation = v_builder.add_simple_block
        a_transformation = a_builder.add_simple_block
    elif transformation_type == 'bottleneck':
        v_transformation = v_builder.add_bottleneck
        a_transformation = a_builder.add_bottleneck
    else:
        print('Unknown transformation type...')

    if model_depth <= 34:
        filter_config = SHALLOW_FILTER_CONFIG
    else:
        filter_config = DEEP_FILTER_CONFIG

    filter_config = np.multiply(
        filter_config, channel_multiplier).astype(np.int)

    # conv_2x
    v_transformation(
        64, filter_config[0][0], int(bottleneck_multiplier * filter_config[0][1]),
        block_type=block_type)
    a_transformation(32, filter_config[0][0], filter_config[0][1],
        down_sampling=True)
    for _ in range(n1 - 1):
        v_transformation(
            filter_config[0][0], filter_config[0][0],
            int(bottleneck_multiplier * filter_config[0][1]),
            block_type=block_type)
        a_transformation(
            filter_config[0][0], filter_config[0][0], filter_config[0][1])

    # conv_3x
    v_transformation(
        filter_config[0][0], filter_config[1][0],
        int(bottleneck_multiplier * filter_config[1][1]),
        down_sampling=True, block_type=block_type)
    a_transformation(
        filter_config[0][0], filter_config[1][0], filter_config[1][1],
        down_sampling=True)
    for _ in range(n2 - 1):
        v_transformation(
            filter_config[1][0], filter_config[1][0],
            int(bottleneck_multiplier * filter_config[1][1]),
            block_type=block_type)
        a_transformation(
            filter_config[1][0], filter_config[1][0], filter_config[1][1])

    # conv_4x
    v_transformation(
        filter_config[1][0], filter_config[2][0],
        int(bottleneck_multiplier * filter_config[2][1]),
        down_sampling=True, block_type=block_type)
    a_transformation(
        filter_config[1][0], filter_config[2][0], filter_config[2][1],
        down_sampling=True)
    for _ in range(n3 - 1):
        v_transformation(
            filter_config[2][0], filter_config[2][0],
            int(bottleneck_multiplier * filter_config[2][1]),
            block_type=block_type)
        a_transformation(
            filter_config[2][0], filter_config[2][0], filter_config[2][1])

    # conv_5x
    if clip_length < 8:
        v_transformation(
            filter_config[2][0], filter_config[3][0],
            int(bottleneck_multiplier * filter_config[3][1]),
            down_sampling=True, down_sampling_temporal=False,
            block_type=block_type)
    else:
        v_transformation(
            filter_config[2][0], filter_config[3][0],
            int(bottleneck_multiplier * filter_config[3][1]),
            down_sampling=True, block_type=block_type)
    a_transformation(
        filter_config[2][0], filter_config[3][0], filter_config[3][1],
        down_sampling=True)
    for _ in range(n4 - 1):
        v_transformation(
            filter_config[3][0], filter_config[3][0],
            int(bottleneck_multiplier * filter_config[3][1]),
            block_type=block_type)
        a_transformation(
            filter_config[3][0], filter_config[3][0], filter_config[3][1])

    # Final layers
    # final pool for visual model
    v_builder.prev_blob = model.AveragePool(
        v_builder.prev_blob,
        'v_final_avg',
        kernels=[
            final_temporal_kernel,
            final_spatial_kernel,
            final_spatial_kernel
        ],
        strides=[1, 1, 1],
    )

    # final pool for audio model
    a_builder.prev_blob = model.MaxPool(
        a_builder.prev_blob,
        'a_final_avg',
        kernels=[4, 2],
        stride=1
    )

    last_a_layer = a_builder.prev_blob
    last_v_layer = v_builder.prev_blob

    if use_convolutional_pred:
        assert is_test
        a_last_3D = model.ExpandDims(
            last_a_layer, last_a_layer + '_3D', dims=[4]
        )
        a_last_tile_1 = model.Tile(
            a_last_3D,
            a_last_3D + '_tiled_1',
            tiles=2,
            axis=3
        )
        a_last_tile_2 = model.Tile(
            a_last_tile_1,
            a_last_3D + '_tiled_2',
            tiles=2,
            axis=4
        )
        av_concat = model.Concat(
            [last_v_layer, a_last_tile_2],
            'av_concat',
            axis=1
        )
        if use_dropout:
            dropout = brew.dropout(
                model, av_concat, 'dropout', is_test=is_test
            )
        else:
            dropout = av_concat
        if not use_full_ft:
            dropout = model.StopGradient(dropout, dropout)

        dim = 2 * filter_config[3][0]
        fc_dim = int(dim / 2)
        fc1 = model.ConvNd(
            dropout,
            'av_fc1',
            dim,
            fc_dim,
            [1, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[1, 1, 1],
            pads=[0, 0, 0] * 2,
            no_bias=False
        )
        relu1 = brew.relu(model, fc1, fc1)
        fc2 = model.ConvNd(
            relu1,
            'av_fc2',
            fc_dim,
            fc_dim,
            [1, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[1, 1, 1],
            pads=[0, 0, 0] * 2,
            no_bias=False
        )
        relu2 = brew.relu(model, fc2, fc2)
        last_out = model.ConvNd(
            relu2,
            'last_out_L{}'.format(num_labels),
            fc_dim,
            num_labels,
            [1, 1, 1],
            weight_init=("MSRAFill", {}),
            strides=[1, 1, 1],
            pads=[0, 0, 0] * 2,
            no_bias=False
        )
        return last_out
    else:
        # reduce to 4D tensor
        v_builder.prev_blob = model.Squeeze(
            v_builder.prev_blob, 'v_final_avg_squeezed', dims=[4])
        last_v_layer = v_builder.prev_blob
        av_concat = model.Concat(
            [last_v_layer, last_a_layer],
            'av_concat',
            axis=1
        )
        if use_dropout:
            dropout = brew.dropout(
                model, av_concat, 'dropout', is_test=is_test
            )
        else:
            dropout = av_concat
        dim = 2 * filter_config[3][0]
        fc_dim = int(dim / 2)
        fc1 = brew.fc(model, dropout, 'av_fc1', dim, fc_dim)
        relu1 = brew.relu(model, fc1, fc1)
        fc2 = brew.fc(model, relu1, 'av_fc2', fc_dim, fc_dim)
        relu2 = brew.relu(model, fc2, fc2)
        last_out = brew.fc(
            model, relu2, 'last_out_L{}'.format(num_labels), fc_dim, num_labels)

        if g_blend:
            a_last_out = brew.fc(
                model,
                last_a_layer,
                'a_last_out_L{}'.format(num_labels),
                filter_config[3][0],
                num_labels,
            )
            v_last_out = brew.fc(
                model,
                last_v_layer,
                'v_last_out_L{}'.format(num_labels),
                filter_config[3][0],
                num_labels,
            )
            return [a_last_out, v_last_out, last_out]
        return last_out


# audio input: 1 x 100 x 40 (channel x time x frequency)
def create_acoustic_resnet(
    model,
    data,
    num_labels,
    label=None,
    is_test=False,
    no_bias=0,
    final_spatial_kernel=2,
    final_temporal_kernel=4,
    model_depth=50,
    transformation_type='simple_block',
    spatial_bn_mom=0.9,
    audio_input_3d=False,
):
    if audio_input_3d:
        if audio_input_3d:
            data_swap = model.NCHW2NHWC(
                data, data + "_NHWC")
            data_greyscale = model.ReduceBackMean(
                data_swap,
                'logmels',
                num_reduce_dim=1)
            data = data_greyscale
    conv1 = model.Conv(data, 'conv1', 1, 16, kernel=3, pad=1)

    builder = AudioModelBuilder(
        model, conv1, no_bias=no_bias,
        is_test=is_test, spatial_bn_mom=spatial_bn_mom)

    builder.add_simple_block(16, 32, down_sampling=True)
    builder.add_simple_block(32, 32)
    builder.add_simple_block(32, 32)

    (n1, n2, n3, n4) = BLOCK_CONFIG[model_depth]

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

    # conv_2x
    transformation(32, filter_config[0][0], filter_config[0][1],
        down_sampling=True)
    for _ in range(n1 - 1):
        transformation(
            filter_config[0][0], filter_config[0][0], filter_config[0][1])

    # conv_3x
    transformation(
        filter_config[0][0], filter_config[1][0], filter_config[1][1],
        down_sampling=True)
    for _ in range(n2 - 1):
        transformation(
            filter_config[1][0], filter_config[1][0], filter_config[1][1])

    # conv_4x
    transformation(
        filter_config[1][0], filter_config[2][0], filter_config[2][1],
        down_sampling=True)
    for _ in range(n3 - 1):
        transformation(
            filter_config[2][0], filter_config[2][0], filter_config[2][1])

    # conv_5x
    transformation(
        filter_config[2][0], filter_config[3][0], filter_config[3][1],
        down_sampling=True)
    for _ in range(n4 - 1):
        transformation(
            filter_config[3][0], filter_config[3][0], filter_config[3][1])

    # Final layers
    final_avg = model.MaxPool(
        builder.prev_blob,
        'final_avg',
        kernels=[final_temporal_kernel, final_spatial_kernel],
        stride=1
    )

    # final_avg = builder.prev_blob
    last_out = brew.fc(
        model,
        final_avg,
        'last_out_L{}'.format(num_labels),
        filter_config[3][0],
        num_labels
    )

    return last_out
