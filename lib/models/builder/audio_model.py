# Copyright 2020-present, Facebook, Inc.
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
log = logging.getLogger("audio_model")
log.setLevel(logging.INFO)

from caffe2.python import brew


class AudioModelBuilder():
    '''
    Helper class for constructing 2D residual blocks.
    '''

    def __init__(
        self,
        model,
        prev_blob,
        no_bias,
        is_test,
        spatial_bn_mom=0.9,
        prefix='',
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
            kernel,
            stride=1,
            pad=0,
    ):
        self.comp_idx += 1

        self.prev_blob = brew.conv(
            self.model,
            self.prev_blob,
            '%scomp_%d_conv_%d' % (self.prefix, self.comp_count, self.comp_idx),
            in_filters,
            out_filters,
            kernel=kernel,
            weight_init=("MSRAFill", {}),
            stride=stride,
            pad=pad,
            no_bias=self.no_bias,
        )
        return self.prev_blob

    def add_relu(self):
        self.prev_blob = brew.relu(
            self.model,
            self.prev_blob,
            self.prev_blob,  # in-place
        )
        return self.prev_blob

    def add_spatial_bn(self, num_filters, suffix=''):
        self.prev_blob = brew.spatial_bn(
            self.model,
            self.prev_blob,
            '%scomp_%d_spatbn_%d%s' %
            (self.prefix, self.comp_count, self.comp_idx, suffix),
            num_filters,
            epsilon=1e-3,
            momentum=self.spatial_bn_mom,
            is_test=self.is_test,
        )
        return self.prev_blob

    def cross_gated_global_pool(
        self, blob_in, dim_in, prefix='', ratio=8, reduced_dim=2, res_gate=False,
    ):
        gp_blob = self.model.ReduceBackMean(
            blob_in, prefix + '_g_pool', num_reduce_dim=reduced_dim)
        fc1 = self.model.FC(gp_blob, prefix + '_fc1', dim_in, dim_in // ratio)
        fc1_relu = self.model.Relu(fc1, prefix + '_fc1_relu')
        fc2 = self.model.FC(fc1_relu, prefix + '_fc2', dim_in // ratio, dim_in)
        sig = self.model.Sigmoid(fc2, fc2 + '_sig')
        shortcut_blob = self.prev_blob
        self.prev_blob = self.model.Mul(
            [self.prev_blob, sig], [prefix + 'g_pool_out'],
            broadcast=1, axis=0
        )
        if res_gate:
            self.prev_blob = brew.sum(
                self.model,
                [shortcut_blob, self.prev_blob],
                prefix + 'res_pool_out'
            )
        return self.prev_blob

    def fused_gated_global_pool(
        self, blob_in, dim_in, prefix='', ratio=8, down_rate=2, res_gate=False,
    ):
        fc1 = self.model.FC(blob_in, prefix + '_fc1', dim_in, dim_in // ratio)
        fc1_relu = self.model.Relu(fc1, prefix + '_fc1_relu')
        fc2 = self.model.FC(
            fc1_relu, prefix + '_fc2', dim_in // ratio, dim_in // down_rate)
        sig = self.model.Sigmoid(fc2, fc2 + '_sig')
        shortcut_blob = self.prev_blob
        self.prev_blob = self.model.Mul(
            [self.prev_blob, sig], [prefix + 'g_pool_out'],
            broadcast=1, axis=0
        )
        if res_gate:
            self.prev_blob = brew.sum(
                self.model,
                [shortcut_blob, self.prev_blob],
                prefix + 'res_pool_out'
            )
        return self.prev_blob

    '''
    Add a "bottleneck" component which can be 2d, sep
    '''

    def add_bottleneck(
            self,
            input_filters,  # num of feature maps from preceding layer
            output_filters,  # num of feature maps to output
            base_filters,  # num of filters internally in the component
            down_sampling=False,
            spatial_batch_norm=True,
    ):

        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 1x1
        self.add_conv(
            input_filters,
            base_filters,
            kernel=1,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)

        self.add_relu()

        # 3x3
        self.add_conv(
            base_filters,
            base_filters,
            kernel=3,
            stride=(2 if down_sampling else 1),
            pad=1,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(base_filters)
        self.add_relu()

        # 1x1
        last_conv = self.add_conv(
            base_filters,
            output_filters,
            kernel=1,
        )
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(output_filters)

        # Summation with input signal (shortcut)
        # If we need to increase dimensions (feature maps), need to
        # do do a projection for the short cut
        if (output_filters > input_filters or down_sampling):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                '%sshortcut_projection_%d' % (self.prefix, self.comp_count),
                input_filters,
                output_filters,
                kernel=1,
                weight_init=("MSRAFill", {}),
                stride=(2 if down_sampling else 1),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    '%sshortcut_projection_%d_spatbn' %
                    (self.prefix, self.comp_count),
                    output_filters,
                    epsilon=1e-3,
                    momentum=self.spatial_bn_mom,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model,
            [shortcut_blob, last_conv],
            '%scomp_%d_sum_%d' % (self.prefix, self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        self.comp_count += 1

    '''
    Add a "simple_block" component which can be 2d, sep
    '''

    def add_simple_block(
            self,
            input_filters,
            num_filters,
            base_filters=0,  # num of filters internally in the component
            down_sampling=False,
            spatial_batch_norm=True,
    ):
        self.comp_idx = 0
        shortcut_blob = self.prev_blob

        # 3x3
        self.add_conv(
            input_filters,
            num_filters,
            kernel=3,
            stride=(2 if down_sampling else 1),
            pad=1,
        )

        if spatial_batch_norm:
            self.add_spatial_bn(num_filters)
        self.add_relu()

        # 3x3
        last_conv = self.add_conv(
            num_filters,
            num_filters,
            kernel=3,
            pad=1,
        )
        if spatial_batch_norm:
            last_conv = self.add_spatial_bn(num_filters)

        # Increase of dimensions, need a projection for the shortcut
        if (num_filters != input_filters or down_sampling):
            shortcut_blob = brew.conv(
                self.model,
                shortcut_blob,
                '%sshortcut_projection_%d' % (self.prefix, self.comp_count),
                input_filters,
                num_filters,
                kernel=1,
                weight_init=("MSRAFill", {}),
                stride=(2 if down_sampling else 1),
                no_bias=self.no_bias,
            )
            if spatial_batch_norm:
                shortcut_blob = brew.spatial_bn(
                    self.model,
                    shortcut_blob,
                    '%sshortcut_projection_%d_spatbn' %
                    (self.prefix, self.comp_count),
                    num_filters,
                    epsilon=1e-3,
                    is_test=self.is_test,
                )

        self.prev_blob = brew.sum(
            self.model,
            [shortcut_blob, last_conv],
            '%scomp_%d_sum_%d' % (self.prefix, self.comp_count, self.comp_idx)
        )
        self.comp_idx += 1
        self.add_relu()

        # Keep track of number of high level components if this ResNetBuilder
        self.comp_count += 1
