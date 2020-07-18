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
    use_softmax_loss=False,
    loss_prefix="",
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
                [prob, label_float], loss_prefix + "entropy_loss")
            unscaled_loss = model.AveragedLoss(
                entropy_loss, loss_prefix + "unscaled_loss")
            loss = model.Scale(
                unscaled_loss, loss_prefix + "loss", scale=loss_scale)

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
                loss_prefix + 'loss',
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
                ["softmax", loss_prefix + "loss"],
            )
        model.Accuracy([softmax, "label"], "accuracy")
    if loss is not None:
        loss = model.Scale(loss, scale=loss_scale)
    return loss


def add_weighted_loss(
    model,
    last_out,
    multi_label,
    use_convolutional_pred,
    num_labels,
    batch_size,
    loss_scale,
    use_softmax_loss=False,
    audio_weight=0.0,
    visual_weight=0.0,
    av_weight=1.0,
):
    assert not use_convolutional_pred, "FCN not supported in loss compute"

    a_last_out = last_out[0]
    v_last_out = last_out[1]
    av_last_out = last_out[2]

    av_loss = add_loss(
        model,
        av_last_out,
        multi_label,
        use_convolutional_pred,
        num_labels,
        batch_size,
        loss_scale,
        use_softmax_loss,
        prefix="av_"
    )

    # construct additional per-modality loss
    if multi_label:
        if use_softmax_loss:
            a_prob = brew.softmax(model, a_last_out, "a_prob")
            v_prob = brew.softmax(model, v_last_out, "v_prob")
            a_entropy_loss = model.net.CrossEntropy(
                [a_prob, 'label_float'], "a_entropy_loss")
            v_entropy_loss = model.net.CrossEntropy(
                [v_prob, 'label_float'], "v_entropy_loss")
            a_unscaled_loss = model.AveragedLoss(
                a_entropy_loss, "a_unscaled_loss")
            v_unscaled_loss = model.AveragedLoss(
                v_entropy_loss, "v_unscaled_loss")
        else:
            a_pred_logit = brew.elementwise_linear(
                model, a_last_out, 'a_pred_logit', num_labels
            )
            model.Sigmoid(a_pred_logit, 'a_prob')
            a_unscaled_loss = model.SigmoidCrossEntropyWithLogits(
                [a_pred_logit, 'label_float'],
                "a_unscaled_loss",
            )
            v_pred_logit = brew.elementwise_linear(
                model, v_last_out, 'v_pred_logit', num_labels
            )
            model.Sigmoid(v_pred_logit, 'v_prob')
            v_unscaled_loss = model.SigmoidCrossEntropyWithLogits(
                [v_pred_logit, 'label_float'],
                "v_unscaled_loss",
            )
    else:
        [a_softmax, a_unscaled_loss] = model.SoftmaxWithLoss(
            [a_last_out, 'label'],
            ["a_softmax",  "a_unscaled_loss"],
        )
        [v_softmax, v_unscaled_loss] = model.SoftmaxWithLoss(
            [v_last_out, 'label'],
            ["v_softmax",  "v_unscaled_loss"],
        )
    a_loss = model.Scale(
        a_unscaled_loss, "a_loss", scale=loss_scale)
    v_loss = model.Scale(
        v_unscaled_loss, "v_loss", scale=loss_scale)

    # G-Blend weighting loss
    weighted_a_loss = model.Scale(a_loss, a_loss + '_w', scale=audio_weight)
    weighted_v_loss = model.Scale(v_loss, v_loss + '_w', scale=visual_weight)
    weighted_av_loss = model.Scale(av_loss, av_loss + '_w', scale=av_weight)
    loss = model.Sum(
        [weighted_a_loss, weighted_v_loss, weighted_av_loss], "loss")
    return loss
