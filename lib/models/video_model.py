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

# adopted from @package resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

logging.basicConfig()
log = logging.getLogger("video_model")
log.setLevel(logging.DEBUG)


class VideoModelBuilder():
    '''
    Helper class for constructing residual blocks.
    '''

    def __init__(self, model, prev_blob, no_bias, is_test, spatial_bn_mom=0.9):
        self.model = model
        self.comp_count = 0
        self.comp_idx = 0
        self.prev_blob = prev_blob
        self.is_test = is_test
        self.spatial_bn_mom = spatial_bn_mom
        self.no_bias = 1 if no_bias else 0

    def add_conv(
        self,
        in_filters,
        out_filters,
        kernels,
        strides=[1, 1, 1] * 1,
        pads=[0, 0, 0] * 2,
        is_decomposed=False,  # set this to be True for (2+1)D conv
    ):
        self.comp_idx += 1

        if is_decomposed:
            i = 3 * in_filters * out_filters * kernels[1] * kernels[2]
            i /= in_filters * kernels[1] * kernels[2] + 3 * out_filters
            middle_filters = int(i)

            log.info("Number of middle filters: {}".format(middle_filters))
            self.prev_blob = self.model.ConvNd(
                self.prev_blob,
                'comp_%d_conv_%d_middle' % (self.comp_count, self.comp_idx),
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
                'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
                middle_filters,
                out_filters,
                [kernels[0], 1, 1],
                weight_init=("MSRAFill", {}),
                strides=[strides[0], 1, 1] * 1,
                pads=[pads[0], 0, 0] * 2,
                no_bias=self.no_bias,
            )
        else:
            self.prev_blob = self.model.ConvNd(
                self.prev_blob,
                'comp_%d_conv_%d' % (self.comp_count, self.comp_idx),
                in_filters,
                out_filters,
                kernels,
                weight_init=("MSRAFill", {}),
                strides=strides,
                pads=pads,
                no_bias=self.no_bias,
            )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = self.model.Relu(
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters, suffix=''):
        self.prev_blob = self.model.SpatialBN(
            self.prev_blob,
            'comp_%d_spatbn_%d%s' % (self.comp_count, self.comp_idx, suffix),
            num_filters,
            epsilon=1e-3,
            momentum=self.spatial_bn_mom,
            is_test=self.is_test,
        )
        return self.prev_blob

    '''
    Add a "bottleneck" component which can be 2d, 3d, (2+1)d
    '''
    def add_bottleneck(
        self,
        input_filters,   # num of feature maps from preceding layer
        base_filters,    # num of filters internally in the component
        output_filters,  # num of feature maps to output
        down_sampling=False,
        spatial_batch_norm=True,
        is_decomposed=False,
        is_real_3d=True,
    ):
        if is_decomposed:
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
            self.add_spatial_bn(base_filters)

        self.add_relu()

        if down_sampling:
            if is_real_3d:
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
            is_decomposed=is_decomposed,
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
        if (output_filters > input_filters):
            shortcut_blob = self.model.ConvNd(
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
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
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    output_filters,
                    epsilon=1e-3,
                    momentum=self.spatial_bn_mom,
                    is_test=self.is_test,
                )

        self.prev_blob = self.model.Sum(
            [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components
        self.comp_count += 1

    '''
    Add a "simple_block" component which can be 2d, 3d, (2+1)d
    '''
    def add_simple_block(
        self,
        input_filters,
        num_filters,
        down_sampling=False,
        spatial_batch_norm=True,
        is_decomposed=False,
        is_real_3d=True,
        only_spatial_downsampling=False,
    ):
        if is_decomposed:
            # decomposition can only be applied to 3d conv
            assert is_real_3d

        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        if down_sampling:
            if is_real_3d:
                if only_spatial_downsampling:
                    use_striding = [1, 2, 2]
                else:
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
            is_decomposed=is_decomposed,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        last_conv = self.add_conv(
            num_filters,
            num_filters,
            kernels=[3, 3, 3] if is_real_3d else [1, 3, 3],
            pads=([1, 1, 1] * 2 if is_real_3d else [0, 1, 1] * 2),
            is_decomposed=is_decomposed,
        )
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(num_filters)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters) or down_sampling:
            shortcut_blob = self.model.ConvNd(
                shortcut_blob,
                'shortcut_projection_%d' % self.comp_count,
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
                    'shortcut_projection_%d_spatbn' % self.comp_count,
                    num_filters,
                    epsilon=1e-3,
                    is_test=self.is_test,
                )

        self.prev_blob = self.model.Sum(
            [shortcut_blob, last_conv],
            'comp_%d_sum_%d' % (self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components
        self.comp_count += 1
