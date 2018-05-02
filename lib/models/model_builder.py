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

from models import c3d_model
from models import r3d_model
from caffe2.python import core, brew

import logging
import lmdb

logging.basicConfig()
log = logging.getLogger("model_builder")
log.setLevel(logging.DEBUG)

r3d_models = [
    'r2d', 'r2df',
    'mc2', 'mc3', 'mc4', 'mc5',
    'rmc2', 'rmc3', 'rmc4', 'rmc5',
    'r3d', 'r2plus1d', 'c3d'
]

model_depths = [10, 16, 18, 26, 34]


def model_validation(
    model_name,
    model_depth,
    clip_length,
    crop_size,
):
    if model_name not in r3d_models and model_name != 'c3d':
        log.info("Unsupported model name: {}".format(model_name))
        return False
    elif model_depth not in model_depths:
        log.info("Unsupported model depth...")
        return False
    elif clip_length % 8 != 0:
        log.info("Unsupported clip length...")
        return False
    elif crop_size != 112 and crop_size != 224:
        log.info("Unsupported crop size...")
        return False
    else:
        log.info("Validated: {} with {} layers".format(
            model_name, model_depth)
        )
        log.info("with input {}x{}x{}".format(
            clip_length, crop_size, crop_size)
        )
        return True


def create_data_reader(
    model,
    name,
    input_data,
):

    reader = model.param_init_net.CreateDB(
        [],
        name=name,
        db=input_data,
        db_type='lmdb',
    )

    lmdb_env = lmdb.open(input_data, readonly=True)
    stat = lmdb_env.stat()
    number_of_examples = stat["entries"]
    lmdb_env.close()
    return reader, number_of_examples


def build_model(
    model,
    model_name,
    model_depth,
    num_labels,
    num_channels,
    crop_size,
    clip_length,
    loss_scale,
    data="data",
    is_test=0,
    pred_layer_name=None,
    multi_label=0,
):
    log.info('creating {}, depth={}...'.format(
        model_name,
        (model_depth if model_name != 'c3d' else 8))
    )
    if model_name == 'c3d':
        assert crop_size == 112  # c3d supports only 16 x 112 x 112
        assert clip_length == 16
        last_out = c3d_model.create_model(
            model=model,
            data=data,
            num_input_channels=num_channels,
            num_labels=num_labels,
            is_test=is_test,
            no_loss=True
        )
    elif model_name in r3d_models:
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
        )
    else:
        # unlikely to happen if we have used model validation
        log.info("Unknown architecture...")

    if multi_label:
        pred_logit = brew.elementwise_linear(
            model, last_out, 'pred_logit_L{}'.format(num_labels), num_labels
        )
        model.Sigmoid(pred_logit, 'prob')
        label_float = model.Cast(
            'label', 'label_float', to=core.DataType.FLOAT
        )
        loss = model.SigmoidCrossEntropyWithLogits(
            [pred_logit, label_float],
            'loss',
        )
    else:
        [softmax, loss] = model.SoftmaxWithLoss(
            [last_out, 'label'],
            ["softmax", "loss"],
        )
        model.Accuracy([softmax, "label"], "accuracy")

    loss = model.Scale(loss, scale=loss_scale)
    return [loss]
