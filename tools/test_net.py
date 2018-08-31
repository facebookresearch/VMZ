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

from caffe2.python import workspace, cnn, core
from caffe2.python import data_parallel_model
import models.model_builder as model_builder
import utils.model_helper as model_helper
import utils.model_loader as model_loader
import utils.metric as metric

from caffe2.proto import caffe2_pb2

logging.basicConfig()
log = logging.getLogger("test_net")
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

    # Model building functions
    def create_model_ops(model, loss_scale):
        return model_builder.build_model(
            model=model,
            model_name=args.model_name,
            model_depth=args.model_depth,
            num_labels=args.num_labels,
            num_channels=args.num_channels,
            crop_size=args.crop_size,
            clip_length=(
                args.clip_length_of if args.input_type == 1
                else args.clip_length_rgb
            ),
            loss_scale=loss_scale,
            is_test=1,
            pred_layer_name=args.pred_layer_name,
        )

    test_model = cnn.CNNModelHelper(
        order="NCHW",
        name="video_model_test",
        use_cudnn=(True if args.use_cudnn == 1 else False),
        cudnn_exhaustive_search=True,
    )

    test_reader, number_of_examples = model_builder.create_data_reader(
        test_model,
        name="test_reader",
        input_data=args.test_data,
    )

    if args.num_iter <= 0:
        num_iter = int(number_of_examples / total_batch_size)
    else:
        num_iter = args.num_iter

    def test_input_fn(model):
        model_helper.AddVideoInput(
            test_model,
            test_reader,
            batch_size=args.batch_size,
            clip_per_video=args.clip_per_video,
            decode_type=1,
            length_rgb=args.clip_length_rgb,
            sampling_rate_rgb=args.sampling_rate_rgb,
            scale_h=args.scale_h,
            scale_w=args.scale_w,
            crop_size=args.crop_size,
            num_decode_threads=4,
            num_of_class=args.num_labels,
            random_mirror=False,
            random_crop=False,
            input_type=args.input_type,
            length_of=args.clip_length_of,
            sampling_rate_of=args.sampling_rate_of,
            frame_gap_of=args.frame_gap_of,
            do_flow_aggregation=args.do_flow_aggregation,
            flow_data_type=args.flow_data_type,
            get_rgb=(args.input_type == 0),
            get_optical_flow=(args.input_type == 1),
            get_video_id=args.get_video_id,
            use_local_file=args.use_local_file,
        )

    if num_gpus > 0:
        data_parallel_model.Parallelize_GPU(
            test_model,
            input_builder_fun=test_input_fn,
            forward_pass_builder_fun=create_model_ops,
            param_update_builder_fun=None,
            devices=gpus
        )
    else:
        test_model._device_type = caffe2_pb2.CPU
        test_model._devices = [0]
        device_opt = core.DeviceOption(test_model._device_type, 0)
        with core.DeviceScope(device_opt):
            # Because our loaded models are named with "gpu_x", keep the naming for now.
            # TODO: Save model using `data_parallel_model.ExtractPredictorNet`
            # to extract the model for "gpu_0". It also renames
            # the input and output blobs by stripping the "gpu_x/" prefix
            with core.NameScope("{}_{}".format("gpu", 0)):
                test_input_fn(test_model)
                create_model_ops(test_model, 1.0)

    workspace.RunNetOnce(test_model.param_init_net)
    workspace.CreateNet(test_model.net)

    if args.db_type == 'minidb':
        if num_gpus > 0:
            model_helper.LoadModel(args.load_model_path, args.db_type)
            data_parallel_model.FinalizeAfterCheckpoint(test_model)
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
            data_parallel_model.FinalizeAfterCheckpoint(test_model)
        else:
            model_loader.LoadModelFromPickleFile(
                test_model,
                args.load_model_path,
                use_gpu=False
            )
    else:
        log.warning("Unsupported db_type: {}".format(args.db_type))


    # metric counters for classification
    clip_acc = 0
    video_top1 = 0
    video_topk = 0
    video_count = 0
    clip_count = 0

    for i in range(num_iter):
        workspace.RunNet(test_model.net.Proto().name)
        num_devices = 1  # default for cpu
        if num_gpus > 0:
            num_devices = num_gpus

        for g in range(num_devices):
            # get labels
            label = workspace.FetchBlob(
                "gpu_{}".format(g) + '/label'
            )
            # get predictions
            predicts = workspace.FetchBlob("gpu_{}".format(g) + '/softmax')
            assert predicts.shape[0] == args.batch_size * args.clip_per_video

            for j in range(args.batch_size):
                # get label for one video
                sample_label = label[j * args.clip_per_video]
                # get clip accuracy
                for k in range(args.clip_per_video):
                    c1, _ = metric.accuracy_metric(
                        predicts[j * args.clip_per_video + k, :],
                        label[j * args.clip_per_video + k])
                    clip_acc = clip_acc + c1
                # get all clip predictions for one video
                all_clips = predicts[
                    j * args.clip_per_video:(j + 1) * args.clip_per_video, :]
                # aggregate predictions into one
                video_pred = PredictionAggregation(all_clips, args.aggregation)
                c1, ck = metric.accuracy_metric(
                    video_pred, sample_label, args.top_k)
                video_top1 = video_top1 + c1
                video_topk = video_topk + ck

            video_count = video_count + args.batch_size
            clip_count = clip_count + label.shape[0]

        if i > 0 and i % args.display_iter == 0:
            log.info('Iter {}/{}: clip: {}, top1: {}, top 5: {}'.format(
                i,
                num_iter,
                clip_acc / clip_count,
                video_top1 / video_count,
                video_topk / video_count))

    log.info("Test accuracy: clip: {}, top 1: {}, top{}: {}".format(
        clip_acc / clip_count,
        video_top1 / video_count,
        args.top_k,
        video_topk / video_count
    ))

    if num_gpus > 0:
        flops, params = model_helper.GetFlopsAndParams(test_model, gpus[0])
    else:
        flops, params = model_helper.GetFlopsAndParams(test_model)
    log.info('FLOPs: {}, params: {}'.format(flops, params))


def main():
    parser = argparse.ArgumentParser(
        description="test_net"
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
                        help="Number of test iterations; " +
                        "0: test the whole set")
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
    parser.add_argument("--batch_size", type=int, default=6,
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
    parser.add_argument("--pred_layer_name", type=str, default=None,
                        help="the prediction layer name")
    parser.add_argument("--display_iter", type=int, default=10,
                        help="Display information every # of iterations.")
    parser.add_argument("--clip_length_of", type=int, default=8,
                        help="Frames of optical flow data")
    parser.add_argument("--sampling_rate_of", type=int, default=2,
                        help="Optical flow sampling rate (in frames)")
    parser.add_argument("--frame_gap_of", type=int, default=2,
                        help="")
    parser.add_argument("--do_flow_aggregation", type=int, default=0,
                        help="whether to aggregate optical flow across " +
                        " multiple frames")
    parser.add_argument("--flow_data_type", type=int, default=0,
                        help="0=Flow2C, 1=Flow3C, 2=FlowWithGray, " +
                        "3=FlowWithRGB")
    parser.add_argument("--input_type", type=int, default=0,
                        help="False=rgb, True=optical flow")
    parser.add_argument("--get_video_id", type=int, default=0,
                        help="Output video id")
    parser.add_argument("--use_dropout", type=int, default=0,
                        help="Use dropout at the prediction layer")
    parser.add_argument("--use_local_file", type=int, default=0,
                        help="Use lmdb as a list of local filenames")

    args = parser.parse_args()

    log.info(args)
    assert model_builder.model_validation(
        args.model_name,
        args.model_depth,
        args.clip_length_of if args.input_type == 1 else args.clip_length_rgb,
        args.crop_size
    )

    Test(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
