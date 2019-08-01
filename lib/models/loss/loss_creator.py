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
log = logging.getLogger("loss_creator")
log.setLevel(logging.INFO)

from caffe2.python import core
from caffe2.python import brew


def add_loss(
    model,
    last_out,
    multi_label,
    use_convolutional_pred,
    num_labels,
    batch_size,
    loss_scale,
    use_softmax_loss=False
):

    if multi_label:
        if use_convolutional_pred:
            # (N, num_labels, L, 2, 2) => (N, L, 2, 2, num_labels)
            pred_logit_transpose = model.net.Transpose(
                last_out, "pred_logit_transpose", axes=[0, 2, 3, 4, 1]
            )
            # (N, L, 2, 2, num_labels) => (NL * 4, num_labels)
            out_shape = [-1, num_labels]
            pred_logit_reshape, _ = model.net.Reshape(
                pred_logit_transpose, [
                    'pred_logit_reshape', 'pred_logit_shape'
                ],
                shape=out_shape,
            )
            pred_logit = brew.elementwise_linear(
                model, pred_logit_reshape, 'pred_logit', num_labels
            )
            prob_unreduced = model.Sigmoid(pred_logit, 'prob_unreduced')

            # (NL * 4, num_labels) => (N, L, 2, 2, num_labels)
            prob_shape = [batch_size, 1, 2, 2, num_labels]
            prob_unreduced_reshape, _ = model.net.Reshape(
                prob_unreduced, [
                    'prob_unreduced_reshape', 'prob_unreduced_shape'
                ],
                shape=prob_shape
            )
            # (N, L, 2, 2, num_labels) => (N, num_labels, L, 2, 2)
            prob_unreduced_reshape_transpose = model.net.Transpose(
                prob_unreduced_reshape, axes=[0, 4, 1, 2, 3]
            )
            model.ReduceBackMean(
                prob_unreduced_reshape_transpose, 'prob', num_reduce_dim=3
            )
            loss = None
        elif use_softmax_loss:
            prob = brew.softmax(model, last_out, "prob")
            label_float = model.Cast(
                'label', 'label_float', to=core.DataType.FLOAT)

            entropy_loss = model.net.CrossEntropy(
                [prob, label_float], "entropy_loss")
            unscaled_loss = model.AveragedLoss(entropy_loss, "unscaled_loss")
            loss = model.Scale(unscaled_loss, "loss", scale=loss_scale)

            return loss
        else:
            pred_logit = brew.elementwise_linear(
                model, last_out, 'pred_logit', num_labels
            )
            model.Sigmoid(pred_logit, 'prob')
            label_float = model.Cast(
                'label', 'label_float', to=core.DataType.FLOAT)
            loss = model.SigmoidCrossEntropyWithLogits(
                [pred_logit, label_float],
                'loss',
            )
    else:
        if use_convolutional_pred:
            softmax_unreduced = model.Softmax(last_out, 'softmax_unreduced')
            softmax = model.ReduceBackMean(
                softmax_unreduced, 'softmax', num_reduce_dim=3
            )
            loss = None
        else:
            [softmax, loss] = model.SoftmaxWithLoss(
                [last_out, 'label'],
                ["softmax", "loss"],
            )
        model.Accuracy([softmax, "label"], "accuracy")
    if loss is not None:
        loss = model.Scale(loss, scale=loss_scale)
    return loss
