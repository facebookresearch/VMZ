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

import sklearn.metrics as metrics
import numpy as np


def accuracy_metric(softmax, label, k=5):
    sorted_preds = np.argsort(softmax)
    sorted_preds[:] = sorted_preds[::-1]
    c1 = sorted_preds[0] == label
    c5 = label in sorted_preds[0:k]
    return c1, c5


def mean_ap_metric(predicts, targets):

    predict = predicts[:, ~np.all(targets == 0, axis=0)]
    target = targets[:, ~np.all(targets == 0, axis=0)]

    mean_auc = 0
    aps = [0]
    try:
        mean_auc = metrics.roc_auc_score(target, predict)
    except ValueError:
        print(
            'The roc_auc curve requires a sufficient number of classes \
            which are missing in this sample.'
        )
    try:
        aps = metrics.average_precision_score(target, predict, average=None)
    except ValueError:
        print(
            'Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample.'
        )

    mean_ap = np.mean(aps)
    weights = np.sum(target.astype(float), axis=0)
    weights /= np.sum(weights)
    mean_wap = np.sum(np.multiply(aps, weights))
    all_aps = np.zeros((1, targets.shape[1]))
    all_aps[:, ~np.all(targets == 0, axis=0)] = aps

    return mean_auc, mean_ap, mean_wap, all_aps.flatten()
