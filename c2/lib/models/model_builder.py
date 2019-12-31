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
from __future__ import unicode_literals

import logging

logging.basicConfig()
log = logging.getLogger("model_builder")
log.setLevel(logging.INFO)


from models import c3d_model
from models import r3d_model

from models.loss import loss_creator


video_models = [
    'r2d', 'r2df',
    'mc2', 'mc3', 'mc4', 'mc5',
    'rmc2', 'rmc3', 'rmc4', 'rmc5',
    'r3d', 'r2plus1d', 'c3d',
    'ir-csn', 'ip-csn',
]

model_depths = [
    10, 16, 18, 26, 34, 50, 101, 152
]


def model_validation(
    model_name,
    model_depth,
    clip_length,
    crop_size,
):
    if crop_size != 112 and crop_size != 224:
        log.info("Unsupported crop size...")
        return False
    elif model_name not in video_models and model_name != 'c3d':
        log.info("Unsupported model name...")
        return False
    elif model_depth not in model_depths:
        log.info("Unsupported model depth...")
        return False
    elif clip_length % 2 != 0 and model_name[0:3] != 'r2d':
        log.info("Unsupported clip length...")
        return False
    elif model_name[-4:] == '-sep' and (model_depth <= 18 or model_depth == 34):
        log.info("depthwise convolution works with bottleneck block only")
        return False
    else:
        log.info("Validated: {} with {} layers".format(model_name, model_depth))
        log.info("with input {}x{}x{}".format(
            clip_length, crop_size, crop_size)
        )
        return True


def build_model(
    model,
    model_name,
    model_depth,
    num_labels,
    batch_size,
    num_channels,
    crop_size,
    clip_length,
    loss_scale,
    data="data",
    is_test=0,
    pred_layer_name=None,
    multi_label=0,
    channel_multiplier=1.0,
    bottleneck_multiplier=1.0,
    use_dropout=False,
    conv1_temporal_stride=1,
    conv1_temporal_kernel=3,
    use_convolutional_pred=False,
    use_pool1=False,
):
    log.info('creating {}, depth={}...'.format(
        model_name,
        (model_depth if model_name != 'c3d' else 8))
    )
    if model_name == 'c3d':
        # c3d supports only 16 x 112 x 112
        assert crop_size == 112
        assert clip_length == 16
        last_out = c3d_model.create_model(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
        )
    elif model_name in video_models:
        last_out = r3d_model.create_model(
            model=model,
            model_name=model_name,
            model_depth=model_depth,
            num_labels=num_labels,
            num_channels=num_channels,
            crop_size=crop_size,
            clip_length=clip_length,
            data=data,
            is_test=is_test,
            channel_multiplier=channel_multiplier,
            bottleneck_multiplier=bottleneck_multiplier,
            use_dropout=use_dropout,
            conv1_temporal_stride=conv1_temporal_stride,
            conv1_temporal_kernel=conv1_temporal_kernel,
            use_convolutional_pred=use_convolutional_pred,
            use_pool1=use_pool1,
        )
    else:
        # unlikely to happen if we have used model validation
        log.info("Unknown architecture...")

    # adding a loss
    loss = loss_creator.add_loss(
        model,
        last_out,
        multi_label,
        use_convolutional_pred,
        num_labels,
        batch_size,
        loss_scale
    )
    return [loss]
