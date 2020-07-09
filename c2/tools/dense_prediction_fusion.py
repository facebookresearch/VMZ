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
from __future__ import unicode_literals
from __future__ import print_function

import logging
import numpy as np
import os
import argparse
import cPickle as pickle

import utils.metric as metric

logging.basicConfig()
log = logging.getLogger("dense_prediction_fusion")
log.setLevel(logging.INFO)


def load_prediction_and_label(pkl_file, blob_name):
    with open(pkl_file, 'r') as fopen:
        blobs = pickle.load(fopen)
    video_id = blobs['video_id']
    label = blobs['label']
    feature = blobs[blob_name]
    n = np.max(video_id) + 1
    m = np.prod(feature.shape[1:])
    if len(label.shape) > 1:
        k = label.shape[1]
    else:
        k = 1

    # place blobs in order of video_id
    sorted_features = np.zeros((n, m), dtype=np.float)
    sorted_labels = np.zeros((n, k), dtype=np.int)
    prediction_counts = np.zeros((n, 1), dtype=np.int)
    for i in range(feature.shape[0]):
        if prediction_counts[video_id[i], 0] > 0:
            assert sorted_labels[video_id[i], :] == label[i]
            sorted_features[video_id[i], :] += feature[i]
        else:
            sorted_labels[video_id[i], :] = label[i]
            sorted_features[video_id[i], :] = feature[i]
        prediction_counts[video_id[i], 0] += 1
    return sorted_features, sorted_labels, prediction_counts


def evaluate_two_feature_bag(pkl_file1, pkl_file2, blob_name, weights):
    feat1, label1, count1 = load_prediction_and_label(pkl_file1, blob_name)
    feat2, label2, count2 = load_prediction_and_label(pkl_file2, blob_name)

    total = 0
    top1 = 0
    top5 = 0
    for i in range(feat1.shape[0]):
        if count1[i, 0] == 0 or count2[i, 0] == 0:
            continue
        assert label1[i, 0] == label2[i, 0]
        predict = weights[0] * feat1[i, :] + weights[1] * feat2[i, :]
        c1, c5 = metric.accuracy_metric(predict, label1[i, 0], 5)
        top1 += c1
        top5 += c5
        total += 1
    return top1, top5, total


def evaluate_dense_prediction(
    input_dir1, input_dir2, file_count=10, alpha=0.5
):
    correct1 = 0
    correct5 = 0
    total = 0
    for idx in range(file_count):
        suffix = '_{}.pkl'.format(idx + 1)

        for file1 in os.listdir(input_dir1):
            if file1.endswith(suffix):
                for file2 in os.listdir(input_dir2):
                    if file2.endswith(suffix):
                        fn_pkl1 = os.path.join(input_dir1, file1)
                        fn_pkl2 = os.path.join(input_dir2, file2)
                        c1, c5, n = evaluate_two_feature_bag(
                            fn_pkl1, fn_pkl2, 'softmax', [alpha, 1 - alpha])
                        correct1 += c1
                        correct5 += c5
                        total += n

    log.info('Alpha {}: top1 {}, top5 {}'.format(
        alpha,
        float(correct1) / total,
        float(correct5) / total)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dense prediction aggregation"
    )
    parser.add_argument("--input_dir1", type=str, default=None,
                        help="Path to test predictions")
    parser.add_argument("--input_dir2", type=str, default=None,
                        help="Path to test predictions")
    parser.add_argument("--file_count", type=int, default=10,
                        help="Path to test predictions")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="The weight ratio of feature 1")
    args = parser.parse_args()
    assert args.alpha >= 0 and args.alpha <= 1
    evaluate_dense_prediction(
        args.input_dir1, args.input_dir2, args.file_count, args.alpha
    )


if __name__ == '__main__':
    main()
