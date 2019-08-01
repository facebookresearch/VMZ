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
import argparse
import math

from caffe2.python import workspace, cnn, core
from caffe2.python import data_parallel_model
from models import model_builder
from utils import model_helper
from utils import model_loader
from utils import metric
from utils import reader_utils
from caffe2.proto import caffe2_pb2


logging.basicConfig()
log = logging.getLogger("test_net_large")
log.setLevel(logging.INFO)


def PredictionAggregation(preds, method):
    if method == 0:  # average pooling
        return np.mean(preds, axis=0)
    elif method == 1:  # max pooling
        return np.max(preds, axis=0)
    else:
        log.info('Unknown aggregation method')
        return []


def Test(args):
    assert args.batch_size == 1  # large testing assume batch size one
    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(',')]
        num_gpus = len(gpus)
    else:
        gpus = range(args.num_gpus)
        num_gpus = args.num_gpus

    if num_gpus > 0:
        total_batch_size = args.batch_size * num_gpus
        log.info("Running on GPUs: {}".format(gpus))
        log.info("total_batch_size: {}".format(total_batch_size))
    else:
        total_batch_size = args.batch_size
        log.info("Running on CPU")
        log.info("total_batch_size: {}".format(total_batch_size))

    video_input_args = dict(
        batch_size=args.batch_size,
        clip_per_video=args.clip_per_video,
        decode_type=1,
        length_rgb=args.clip_length_rgb,
        sampling_rate_rgb=args.sampling_rate_rgb,
        scale_h=args.scale_h,
        scale_w=args.scale_w,
        crop_size=args.crop_size,
        video_res_type=args.video_res_type,
        short_edge=min(args.scale_h, args.scale_w),
        num_decode_threads=args.num_decode_threads,
        do_multi_label=args.multi_label,
        num_of_class=args.num_labels,
        random_mirror=False,
        random_crop=False,
        input_type=args.input_type,
        length_of=args.clip_length_of,
        sampling_rate_of=args.sampling_rate_of,
        frame_gap_of=args.frame_gap_of,
        do_flow_aggregation=args.do_flow_aggregation,
        flow_data_type=args.flow_data_type,
        get_rgb=(args.input_type == 0 or args.input_type >= 3),
        get_optical_flow=(args.input_type == 1 or args.input_type >= 4),
        use_local_file=args.use_local_file,
        crop_per_clip=args.crop_per_clip,
    )

    reader_args = dict(
        name="test_reader",
        input_data=args.test_data,
    )

    # Model building functions
    def create_model_ops(model, loss_scale):
        return model_builder.build_model(
            model=model,
            model_name=args.model_name,
            model_depth=args.model_depth,
            num_labels=args.num_labels,
            batch_size=args.batch_size * args.clip_per_video,
            num_channels=args.num_channels,
            crop_size=args.crop_size,
            clip_length=(
                args.clip_length_of if args.input_type == 1
                else args.clip_length_rgb
            ),
            loss_scale=loss_scale,
            is_test=1,
            pred_layer_name=args.pred_layer_name,
            multi_label=args.multi_label,
            channel_multiplier=args.channel_multiplier,
            bottleneck_multiplier=args.bottleneck_multiplier,
            use_dropout=args.use_dropout,
            conv1_temporal_stride=args.conv1_temporal_stride,
            conv1_temporal_kernel=args.conv1_temporal_kernel,
            use_convolutional_pred=args.use_convolutional_pred,
            use_pool1=args.use_pool1,
        )

    def empty_function(model, loss_scale=1):
        # null
        return

    test_data_loader = cnn.CNNModelHelper(
        order="NCHW",
        name="data_loader",
    )
    test_model = cnn.CNNModelHelper(
        order="NCHW",
        name="video_model",
        use_cudnn=(True if args.use_cudnn == 1 else False),
        cudnn_exhaustive_search=True,
    )

    test_reader, number_of_examples = reader_utils.create_data_reader(
        test_data_loader, **reader_args
    )

    if args.num_iter <= 0:
        num_iter = int(math.ceil(number_of_examples / total_batch_size))
    else:
        num_iter = args.num_iter

    def test_input_fn(model):
        model_helper.AddVideoInput(
            test_data_loader,
            test_reader,
            **video_input_args
        )

    if num_gpus > 0:
        data_parallel_model.Parallelize_GPU(
            test_data_loader,
            input_builder_fun=test_input_fn,
            forward_pass_builder_fun=empty_function,
            param_update_builder_fun=None,
            devices=gpus,
            optimize_gradient_memory=True,
        )
        data_parallel_model.Parallelize_GPU(
            test_model,
            input_builder_fun=empty_function,
            forward_pass_builder_fun=create_model_ops,
            param_update_builder_fun=None,
            devices=gpus,
            optimize_gradient_memory=True,
        )
    else:
        test_model._device_type = caffe2_pb2.CPU
        test_model._devices = [0]
        device_opt = core.DeviceOption(test_model._device_type, 0)
        with core.DeviceScope(device_opt):
            # Because our loaded models are named with "gpu_x",
            # keep the naming for now.
            # TODO: Save model using `data_parallel_model.ExtractPredictorNet`
            # to extract the model for "gpu_0". It also renames
            # the input and output blobs by stripping the "gpu_x/" prefix
            with core.NameScope("{}_{}".format("gpu", 0)):
                test_input_fn(test_data_loader)
                create_model_ops(test_model, 1.0)

    workspace.RunNetOnce(test_data_loader.param_init_net)
    workspace.CreateNet(test_data_loader.net)
    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    if args.db_type == 'minidb':
        if num_gpus > 0:
            model_helper.LoadModel(args.load_model_path, args.db_type)
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
                model_helper.LoadModel(args.load_model_path, args.db_type)
    elif args.db_type == 'pickle':
        if num_gpus > 0:
            model_loader.LoadModelFromPickleFile(
                test_model,
                args.load_model_path,
                use_gpu=True,
                root_gpu_id=gpus[0]
            )
        else:
            model_loader.LoadModelFromPickleFile(
                test_model,
                args.load_model_path,
                use_gpu=False
            )
    else:
        log.warning("Unsupported db_type: {}".format(args.db_type))

    data_parallel_model.FinalizeAfterCheckpoint(test_model)

    # metric couters for multilabel
    all_prob_for_map = np.empty(shape=[0, args.num_labels], dtype=np.float)
    all_label_for_map = np.empty(shape=[0, args.num_labels], dtype=np.int32)

    # metric counters for closed-world classification
    clip_acc = 0
    video_top1 = 0
    video_topk = 0
    video_count = 0
    clip_count = 0

    num_devices = 1  # default for cpu
    if num_gpus > 0:
        num_devices = num_gpus
    # actual_batch_size
    inference_batch_size = args.crop_per_inference
    num_crop_per_bag = args.clip_per_video * args.crop_per_clip
    # make sure you do your math correctly
    assert num_crop_per_bag % num_crop_per_bag == 0
    num_slice = int(num_crop_per_bag / inference_batch_size)

    for i in range(num_iter):
        # load one batch of data assume 1 video
        # which is (#clips x #crops) x 3 x crop_size x crop_size
        workspace.RunNet(test_data_loader.net.Proto().name)

        # get all data into a list, each per device (gpu)
        video_data = []
        label_data = []
        all_predicts = []
        for g in range(num_devices):
            data = workspace.FetchBlob("gpu_{}".format(gpus[g]) + '/data')
            video_data.append(data)
            label = workspace.FetchBlob("gpu_{}".format(gpus[g]) + '/label')
            label_data.append(label)
            all_predicts.append([])

        for slice in range(num_slice):
            for g in range(num_devices):
                data = video_data[g][
                    slice * inference_batch_size :
                    (slice + 1) * inference_batch_size,
                    :, :, :, :
                ]
                if args.multi_label:
                    label = label_data[g][
                        slice * inference_batch_size :
                        (slice + 1) * inference_batch_size,
                        :
                    ]
                else:
                    label = label_data[g][
                        slice * inference_batch_size :
                        (slice + 1) * inference_batch_size
                    ]
                workspace.FeedBlob("gpu_{}".format(gpus[g]) + '/data', data)
                workspace.FeedBlob("gpu_{}".format(gpus[g]) + '/label', label)

            # do one iteration of inference over one slice across devices
            workspace.RunNet(test_model.net.Proto().name)

            for g in range(num_devices):
                # get predictions
                if args.multi_label:
                    predicts = workspace.FetchBlob(
                        "gpu_{}".format(gpus[g]) + '/prob'
                    )
                else:
                    predicts = workspace.FetchBlob(
                        "gpu_{}".format(gpus[g]) + '/softmax'
                    )

                assert predicts.shape[0] == inference_batch_size

                # accumulate predictions
                if all_predicts[g] == []:
                    all_predicts[g] = predicts
                else:
                    all_predicts[g] = np.concatenate(
                        (all_predicts[g], predicts), axis=0
                    )

        for g in range(num_devices):
            # get clip accuracy
            predicts = all_predicts[g]
            if args.multi_label:
                sample_label = label_data[g][0, :]
            else:
                sample_label = label_data[g][0]
            for k in range(num_crop_per_bag):
                sorted_preds = np.argsort(predicts[k, :])
                sorted_preds[:] = sorted_preds[::-1]
                if sorted_preds[0] == sample_label:
                    clip_acc = clip_acc + 1

            # since batch_size == 1
            all_clips = predicts
            # aggregate predictions into one
            video_pred = PredictionAggregation(all_clips, args.aggregation)
            if args.multi_label:
                video_pred = np.expand_dims(video_pred, axis=0)
                sample_label = np.expand_dims(sample_label, axis=0)
                all_prob_for_map = np.concatenate(
                    (all_prob_for_map, video_pred), axis=0
                )
                all_label_for_map = np.concatenate(
                    (all_label_for_map, sample_label), axis=0
                )
            else:
                sorted_video_pred = np.argsort(video_pred)
                sorted_video_pred[:] = sorted_video_pred[::-1]
                if sorted_video_pred[0] == sample_label:
                    video_top1 = video_top1 + 1
                if sample_label in sorted_video_pred[0:args.top_k]:
                    video_topk = video_topk + 1

        video_count = video_count + num_devices
        clip_count = clip_count + num_devices * num_crop_per_bag

        if i > 0 and i % args.display_iter == 0:
            if args.multi_label:
                # mAP
                auc, ap, wap, aps = metric.mean_ap_metric(
                    all_prob_for_map, all_label_for_map
                )
                log.info(
                    'Iter {}/{}: mAUC: {}, mAP: {}, mWAP: {}, mAP_all: {}'.
                    format(i, num_iter, auc, ap, wap, np.mean(aps))
                )
            else:
                # accuracy
                log.info(
                    'Iter {}/{}: clip: {}, top1: {}, top 5: {}'.format(
                        i, num_iter, clip_acc / clip_count,
                        video_top1 / video_count, video_topk / video_count
                    )
                )

    if args.multi_label:
        # mAP
        auc, ap, wap, aps = metric.mean_ap_metric(
            all_prob_for_map, all_label_for_map)
        log.info(
            "Test mAUC: {}, mAP: {}, mWAP: {}, mAP_all: {}".format(
                auc, ap, wap, np.mean(aps)
            )
        )
        if args.print_per_class_metrics:
            log.info("Test mAP per class: {}".format(aps))
    else:
        # accuracy
        log.info(
            "Test accuracy: clip: {}, top 1: {}, top{}: {}".format(
                clip_acc / clip_count, video_top1 / video_count, args.top_k,
                video_topk / video_count
            )
        )

    if num_gpus > 0:
        flops, params, inters = model_helper.GetFlopsAndParams(test_model, gpus[0])
    else:
        flops, params, inters = model_helper.GetFlopsAndParams(test_model)
    log.info('FLOPs: {}, params: {}, inters: {}'.format(flops, params, inters))


def main():
    parser = argparse.ArgumentParser(
        description="Tool for testing large networks"
    )
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test data")
    parser.add_argument("--db_type", type=str, default='pickle',
                        help="Db type of the testing model")
    parser.add_argument("--model_depth", type=int, default=18,
                        help="Model depth")
    parser.add_argument("--model_name", type=str, default='r2plus1d',
                        help="Model name")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPU devices (instead of --gpus)")
    parser.add_argument("--scale_h", type=int, default=128,
                        help="Scale image height to")
    parser.add_argument("--scale_w", type=int, default=171,
                        help="Scale image width to")
    parser.add_argument("--num_iter", type=int, default=0,
                        help="Number of test iterations; 0: test the whole set")
    parser.add_argument("--crop_size", type=int, default=112,
                        help="Input image size (to crop to)")
    parser.add_argument("--clip_length_rgb", type=int, default=16,
                        help="Length of input clips")
    parser.add_argument("--sampling_rate_rgb", type=int, default=1,
                        help="Frame sampling rate")
    parser.add_argument("--num_labels", type=int, default=101,
                        help="Number of labels")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of channels")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size, total over all GPUs")
    parser.add_argument("--clip_per_video", type=int, default=10,
                        help="Number of clips to be sampled from a video")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top k video accuracy output")
    parser.add_argument("--aggregation", type=int, default=0,
                        help="0: avergage pool, 1: max pooling")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Load saved model for testing")
    parser.add_argument("--use_cudnn", type=int, default=1,
                        help="Use CuDNN")
    parser.add_argument("--print_per_class_metrics", type=int, default=0,
                        help="Log per class accuracy for multi-class setting")
    parser.add_argument("--pred_layer_name", type=str, default=None,
                        help="the prediction layer name")
    parser.add_argument("--multi_label", type=int, default=0,
                        help="Multiple label testing")
    parser.add_argument("--display_iter", type=int, default=10,
                        help="Display information every # of iterations.")
    parser.add_argument("--clip_length_of", type=int, default=8,
                        help="Frames of optical flow data")
    parser.add_argument("--sampling_rate_of", type=int, default=2,
                        help="")
    parser.add_argument("--frame_gap_of", type=int, default=2,
                        help="")
    parser.add_argument("--do_flow_aggregation", type=int, default=0,
                        help="whether to aggregate optical flow across"
                        + " multiple frames")
    parser.add_argument("--flow_data_type", type=int, default=0,
                        help="0=Flow2C, 1=Flow3C, 2=FlowWithGray, 3=FlowWithRGB")
    parser.add_argument("--input_type", type=int, default=0,
                        help="False=rgb, True=optical flow")
    parser.add_argument("--num_decode_threads", type=int, default=4,
                        help="number of decoding threads")
    parser.add_argument("--channel_multiplier", type=float, default=1.0,
                        help="Channel multiplier")
    parser.add_argument("--bottleneck_multiplier", type=float, default=1.0,
                        help="Bottleneck multiplier")
    parser.add_argument("--use_dropout", type=int, default=0,
                        help="Use dropout at the prediction layer")
    parser.add_argument("--conv1_temporal_stride", type=int, default=1,
                        help="Conv1 temporal striding")
    parser.add_argument("--conv1_temporal_kernel", type=int, default=3,
                        help="Conv1 temporal kernel")
    parser.add_argument("--use_convolutional_pred", type=int, default=0,
                        help="using convolutional predictions")
    parser.add_argument("--video_res_type", type=int, default=0,
                        help="Video frame scaling option, 0: scaled by "
                        + "height x width; 1: scaled by shorter edge")
    parser.add_argument("--use_pool1", type=int, default=0,
                        help="use pool1 layer")
    parser.add_argument("--use_local_file", type=int, default=0,
                        help="use local file")
    parser.add_argument("--crop_per_clip", type=int, default=1,
                        help="number of spatial crops per clip")
    parser.add_argument("--crop_per_inference", type=int, default=1,
                       help="number of spatial crops GPU memory can handle"
                       + "per one pass of inference")

    args = parser.parse_args()

    log.info(args)
    assert model_builder.model_validation(
        args.model_name,
        args.model_depth,
        args.clip_length_of if args.input_type == 1 else args.clip_length_rgb,
        args.crop_size if not args.use_convolutional_pred else 112
    )

    Test(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
