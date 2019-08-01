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


# implement C3D model
# input 3 x 16 x 112 x 112
# reference model is here https://fburl.com/cfzvuwbj
def create_model(
    model,
    data,
    num_input_channels,
    num_labels,
    label=None,
    is_test=0,
    no_bias=0,
    fc6_dim=4096,
    fc7_dim=4096,
):
    # first conv layers
    model.ConvNd(
        data,
        'conv1a',
        num_input_channels,
        64,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv1a',
        'conv1a_bn',
        64,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv1a_bn', 'conv1a_bn')

    model.MaxPool(
        'conv1a_bn',
        'pool1',
        kernels=[1, 2, 2],
        strides=[1, 2, 2]
    )

    # second conv layers
    model.ConvNd(
        'pool1',
        'conv2a',
        64,
        128,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv2a',
        'conv2a_bn',
        128,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv2a_bn', 'conv2a_bn')

    model.MaxPool(
        'conv2a_bn',
        'pool2',
        kernels=[2, 2, 2],
        strides=[2, 2, 2]
    )

    # third conv layers
    model.ConvNd(
        'pool2',
        'conv3a',
        128,
        256,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv3a',
        'conv3a_bn',
        256,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv3a_bn', 'conv3a_bn')

    model.ConvNd(
        'conv3a_bn',
        'conv3b',
        256,
        256,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv3b',
        'conv3b_bn',
        256,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv3b_bn', 'conv3b_bn')

    model.MaxPool(
        'conv3b_bn',
        'pool3',
        kernels=[2, 2, 2],
        strides=[2, 2, 2]
    )

    # fourth conv layers
    model.ConvNd(
        'pool3',
        'conv4a',
        256,
        512,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv4a',
        'conv4a_bn',
        512,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv4a_bn', 'conv4a_bn')

    model.ConvNd(
        'conv4a_bn',
        'conv4b',
        512,
        512,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv4b',
        'conv4b_bn',
        512,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv4b_bn', 'conv4b_bn')

    model.MaxPool(
        'conv4b_bn',
        'pool4',
        kernels=[2, 2, 2],
        strides=[2, 2, 2]
    )

    # fifth conv layers
    model.ConvNd(
        'pool4',
        'conv5a',
        512,
        512,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv5a',
        'conv5a_bn',
        512,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv5a_bn', 'conv5a_bn')

    model.ConvNd(
        'conv5a_bn',
        'conv5b',
        512,
        512,
        [3, 3, 3],
        weight_init=("GaussianFill", {'std': 0.01}),
        strides=[1, 1, 1],
        pads=[1, 1, 1] * 2,
        no_bias=no_bias
    )
    model.SpatialBN(
        'conv5b',
        'conv5b_bn',
        512,
        epsilon=1e-3,
        momentum=0.9,
        is_test=is_test,
    )
    model.Relu('conv5b_bn', 'conv5b_bn')

    model.MaxPool(
        'conv5b_bn',
        'pool5',
        kernels=[2, 2, 2],
        strides=[2, 2, 2]
    )

    model.FC(
        'pool5',
        'fc6',
        512 * 3 * 3,
        fc6_dim,
        weight_init=("GaussianFill", {'std': 0.005})
    )
    model.Relu('fc6', 'fc6')
    model.Dropout('fc6', 'fc6_dropout', is_test=is_test)

    model.FC(
        'fc6_dropout',
        'fc7',
        fc6_dim,
        fc7_dim,
        weight_init=("GaussianFill", {'std': 0.005})
    )

    model.Relu('fc7', 'fc7')
    model.Dropout('fc7', 'fc7_dropout', is_test=is_test)

    last_out = model.FC(
        'fc7_dropout', 'last_out_L{}'.format(num_labels), fc7_dim, num_labels
    )

    return last_out
