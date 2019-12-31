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
log = logging.getLogger("dense_prediction_aggregation")
log.setLevel(logging.INFO)


def evaluate_feature_bag(
    pkl_file, blob_name, output_clip_accuracy=0, max_clip=0
):
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
    clip = 0.
    for i in range(feature.shape[0]):
        if output_clip_accuracy:
            c0, _ = metric.accuracy_metric(feature[i], label[i], 5)
            clip += c0
        if max_clip == 0 or prediction_counts[video_id[i], 0] < max_clip:
            if prediction_counts[video_id[i], 0] > 0:
                assert sorted_labels[video_id[i], :] == label[i]
                sorted_features[video_id[i], :] += feature[i]
            else:
                sorted_labels[video_id[i], :] = label[i]
                sorted_features[video_id[i], :] = feature[i]
            prediction_counts[video_id[i], 0] += 1

    total = 0
    top1 = 0
    top5 = 0
    clip /= feature.shape[0]
    for i in range(sorted_features.shape[0]):
        if prediction_counts[i, 0] == 0:
            continue
        c1, c5 = metric.accuracy_metric(
            sorted_features[i, :], sorted_labels[i, 0], 5)
        top1 += c1
        top5 += c5
        total += 1
    log.info('{} has an accuracy of clip {}, top1 {}, top5 {}'.format(
        pkl_file,
        clip,
        float(top1) / total,
        float(top5) / total)
    )
    return top1, top5, total, clip


def evaluate_dense_prediction(input_dir, output_clip_accuracy=0, max_clip=0):
    correct1 = 0
    correct5 = 0
    total = 0
    clip_acc = 0
    for file in os.listdir(input_dir):
        if file.endswith(".pkl"):
            fn_pkl = os.path.join(input_dir, file)
            c1, c5, n, clip = evaluate_feature_bag(
                fn_pkl,
                'softmax',
                output_clip_accuracy=output_clip_accuracy,
                max_clip=max_clip)
            correct1 += c1
            correct5 += c5
            total += n
            clip_acc += n * clip
    log.info('Accuracy: clip {}, top1 {}, top5 {}'.format(
        float(clip_acc) / total,
        float(correct1) / total,
        float(correct5) / total)
    )


def main():
    parser = argparse.ArgumentParser(
        description="Dense prediction aggregation"
    )
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Path to test predictions")
    parser.add_argument("--output_clip_accuracy", type=int, default=1,
                        help="Turn this on to evaluate clip accuracy")
    parser.add_argument("--max_clip", type=int, default=0,
                        help="0: for all clip, >0: for max_clip to select "
                        + "from each video")

    args = parser.parse_args()
    evaluate_dense_prediction(
        args.input_dir,
        args.output_clip_accuracy,
        args.max_clip
    )


if __name__ == '__main__':
    main()
