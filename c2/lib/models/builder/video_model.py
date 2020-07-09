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

import logging

logging.basicConfig()
log = logging.getLogger("video_model")
log.setLevel(logging.INFO)

from caffe2.python import brew


class VideoModelBuilder():
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(
        self,
        model,
        prev_blob,
        no_bias,
        is_test,
        spatial_bn_mom=0.9,
        prefix=''
    ):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
        self.spatial_bn_mom = spatial_bn_mom
        self.no_bias = 1 if no_bias else 0
        self.prefix = prefix

    def add_conv(
        self,
        in_filters,
        out_filters,
        kernels,
        strides=[1, 1, 1] * 1,
        pads=[0, 0, 0] * 2,
        block_type='3d',  # set this to be '3d', '2.5d', or 'track',
        group=1,
    ):
        self.comp_idx += 1
        if group > 1:
            assert block_type == '3d-group'
        log.info('in: %d out: %d' % (in_filters, out_filters))
        if block_type == '2.5d':
            i = 3 * in_filters * out_filters * kernels[1] * kernels[2]
            i /= in_filters * kernels[1] * kernels[2] + 3 * out_filters
            middle_filters = int(i)
            self.prev_blob = self.model.ConvNd(
                self.prev_blob,
                '%scomp_%d_conv_%d_middle' %
                (self.prefix, self.comp_count, self.comp_idx),
                in_filters,
                middle_filters,
                [1, kernels[1], kernels[2]],
                weight_init=("MSRAFill", {}),
                strides=[1, strides[1], strides[2]] * 1,
                pads=[0, pads[1], pads[2]] * 2,
                no_bias=self.no_bias,
            )
            self.add_spatial_bn(middle_filters, suffix='_middle')
            self.add_relu()
            self.prev_blob = self.model.ConvNd(
                self.prev_blob,
                '%scomp_%d_conv_%d' %
                (self.prefix, self.comp_count, self.comp_idx),
                middle_filters,
                out_filters,
                [kernels[0], 1, 1],
                weight_init=("MSRAFill", {}),
                strides=[strides[0], 1, 1] * 1,
                pads=[pads[0], 0, 0] * 2,
                no_bias=self.no_bias,
            )
        elif block_type == '0.3d' or block_type == '0.3d+relu':
            self.prev_blob = self.model.ConvNd(
                self.prev_blob,
                '%scomp_%d_conv_%d_middle' %
                (self.prefix, self.comp_count, self.comp_idx),
                in_filters,
                out_filters,
                [1, 1, 1],
                weight_init=("MSRAFill", {}),
                strides=[1, 1, 1] * 1,
                pads=[0, 0, 0] * 2,
                no_bias=self.no_bias,
            )
            self.add_spatial_bn(out_filters, suffix='_middle')
            if block_type == '0.3d+relu':
                self.add_relu()
            self.prev_blob = brew.conv_nd(
                self.model,
                self.prev_blob,
                '%scomp_%d_conv_%d' %
                (self.prefix, self.comp_count, self.comp_idx),
                out_filters,
                out_filters,
                weight_init=("MSRAFill", {}),
                kernel=kernels,
                strides=strides,
                pads=pads,
                group=out_filters,
                no_bias=self.no_bias,
                use_cudnn=False,
                engine="CHANNELWISE_3D",
            )
        elif block_type == '3d':
            self.prev_blob = self.model.ConvNd(
                self.prev_blob,
                '%scomp_%d_conv_%d' %
                (self.prefix, self.comp_count, self.comp_idx),
                in_filters,
                out_filters,
                kernels,
                weight_init=("MSRAFill", {}),
                strides=strides,
                pads=pads,
                no_bias=self.no_bias,
            )
        elif block_type == '3d-sep':  # channel_wise 3d conv block
            self.add_channelwise_conv(
                in_filters, out_filters,
                kernels,
                strides=strides,
                pads=pads,
            )
        elif block_type == '2.5d-sep':
            assert in_filters == out_filters
            middle_filters = out_filters
            self.add_channelwise_conv(
                in_filters,
                middle_filters,
                [1, kernels[1], kernels[2]],
                strides=[1, strides[1], strides[2]] * 1,
                pads=[0, pads[1], pads[2]] * 2,
                suffix='_middle'
            )
            self.add_spatial_bn(middle_filters, suffix='_middle')
            self.add_relu()
            self.prev_blob = self.model.ConvNd(
                self.prev_blob,
                '%scomp_%d_conv_%d_1x' %
                (self.prefix, self.comp_count, self.comp_idx),
                middle_filters,
                middle_filters,
                [1, 1, 1],
                weight_init=("MSRAFill", {}),
                strides=[1, 1, 1] * 1,
                pads=[0, 0, 0] * 2,
                no_bias=self.no_bias,
            )
            self.add_spatial_bn(middle_filters, suffix='_1x')
            self.add_relu()
            self.add_channelwise_conv(
                middle_filters,
                out_filters,
                [kernels[0], 1, 1],
                strides=[strides[0], 1, 1] * 1,
                pads=[pads[0], 0, 0] * 2
            )
        else:
            log.info('Unknown block type!')
        return self.prev_blob

    def add_channelwise_conv(
        self,
        in_filters,
        out_filters,
        kernels,
        strides=[1, 1, 1] * 1,
        pads=[0, 0, 0] * 2,
        suffix=''
    ):
        self.comp_idx += 1

        assert in_filters == out_filters
        self.prev_blob = brew.conv_nd(
            self.model,
            self.prev_blob,
            '%scomp_%d_conv_%d%s' %
            (self.prefix, self.comp_count, self.comp_idx, suffix),
            in_filters,
            out_filters,
            weight_init=("MSRAFill", {}),
            kernel=kernels,
            strides=strides,
            pads=pads,
            group=in_filters,
            no_bias=self.no_bias,
            use_cudnn=False,
            engine="CHANNELWISE_3D",
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = self.model.Relu(
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters, suffix='', bn_init=1.0):
        bn_name = ('%scomp_%d_spatbn_%d%s' %
            (self.prefix, self.comp_count, self.comp_idx, suffix))
        self.prev_blob = self.model.SpatialBN(
            self.prev_blob,
            bn_name,
            num_filters,
            epsilon=1e-3,
            momentum=self.spatial_bn_mom,
            is_test=self.is_test,
        )
        if bn_init != 1 and not self.is_test:
            self.model.param_init_net.ConstantFill(
                [bn_name + "_s"], bn_name + "_s", value=bn_init)
            log.info('{} is initialized with {}'.format(bn_name, bn_init))
        return self.prev_blob

    '''
    Add a "bottleneck" component which can be 2d, 3d, (2+1)d
    '''
    def add_bottleneck(
        self,
        input_filters,   # num of feature maps from preceding layer
        output_filters,  # num of feature maps to output
        base_filters,  # num of filters internally in the component
        down_sampling=False,
        down_sampling_temporal=None,
        spatial_batch_norm=True,
        block_type='3d',
        is_real_3d=True,
        gama_init=False,
        group=1,
        use_shuffle=False,
    ):
        if block_type == '2.5d':
            # decomposition can only be applied to 3d conv
            assert is_real_3d

        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 1x1x1
        self.add_conv(
            input_filters,
            base_filters,
            kernels=[1, 1, 1]
        )

        if spatial_batch_norm:
            self.add_spatial_bn(
                base_filters,
                bn_init=(0. if gama_init else 1.)
            )

        self.add_relu()

        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling

        if down_sampling:
            if is_real_3d and down_sampling_temporal:
                use_striding = [2, 2, 2]
            else:
                use_striding = [1, 2, 2]
        else:
            use_striding = [1, 1, 1]

        # 3x3x3 (note the pad, required for keeping dimensions)
        self.add_conv(
            base_filters,
            base_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            strides=use_striding,
            pads=([1, 1, 1] * 2 if is_real_3d else [0, 1, 1] * 2),
            block_type=block_type,
            group=group,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)
        self.add_relu()

        # 1x1x1
        last_conv = self.add_conv(
            base_filters,
            output_filters,
            kernels=[1, 1, 1]
        )
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(output_filters)

        # Summation with input signal (shortcut)
        # If we need to increase dimensions (feature maps), need to
        # do do a projection for the short cut
        if (output_filters != input_filters or down_sampling):
            shortcut_blob = self.model.ConvNd(
                shortcut_blob,
                '%sshortcut_projection_%d' % (self.prefix, self.comp_count),
                input_filters,
                output_filters,
                [1, 1, 1],
                weight_init=("MSRAFill", {}),
                strides=use_striding,
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = self.model.SpatialBN(
                    shortcut_blob,
                    '%sshortcut_projection_%d_spatbn' %
                    (self.prefix, self.comp_count),
                    output_filters,
                    epsilon=1e-3,
                    momentum=self.spatial_bn_mom,
                    is_test=self.is_test,
                )

        self.prev_blob = self.model.Sum(
            [shortcut_blob, last_conv],
            '%scomp_%d_sum_%d' % (self.prefix, self.comp_count, self.comp_idx)
        )

        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1

    '''
    Add a "simple_block" component which can be 2d, 3d, (2+1)d
    '''
    def add_simple_block(
        self,
        input_filters,
        num_filters,
        base_filters=0,
        down_sampling=False,
        down_sampling_temporal=None,
        spatial_batch_norm=True,
        block_type='3d',
        is_real_3d=True,
        group=1,
        use_shuffle=False,
    ):
        if block_type == '2.5d':
            # decomposition can only be applied to 3d conv
            assert is_real_3d

        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        if down_sampling_temporal is None:
            down_sampling_temporal = down_sampling

        if down_sampling:
            if is_real_3d and down_sampling_temporal:
                use_striding = [2, 2, 2]
            else:
                use_striding = [1, 2, 2]
        else:
            use_striding = [1, 1, 1]

        # 3x3x3
        self.add_conv(
            input_filters,
            num_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            strides=use_striding,
            pads=([1, 1, 1] * 2 if is_real_3d else [0, 1, 1] * 2),
            block_type=block_type,
            group=group,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        last_conv = self.add_conv(
            num_filters,
            num_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            pads=([1, 1, 1] * 2 if is_real_3d else [0, 1, 1] * 2),
            block_type=block_type,
            group=group,
        )
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(num_filters)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters or down_sampling):
            shortcut_blob = self.model.ConvNd(
                shortcut_blob,
                '%sshortcut_projection_%d' % (self.prefix, self.comp_count),
                input_filters,
                num_filters,
                [1, 1, 1],
                weight_init=("MSRAFill", {}),
                strides=use_striding,
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = self.model.SpatialBN(
                    shortcut_blob,
                    '%sshortcut_projection_%d_spatbn' %
                    (self.prefix, self.comp_count),
                    num_filters,
                    epsilon=1e-3,
                    is_test=self.is_test,
                )

        self.prev_blob = self.model.Sum(
            [shortcut_blob, last_conv],
            '%scomp_%d_sum_%d' % (self.prefix, self.comp_count, self.comp_idx)
        )
        self.add_relu()

        self.comp_idx += 1

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1
