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

from caffe2.python import workspace, cnn, data_parallel_model, core
from caffe2.proto import caffe2_pb2


from models import model_builder
from utils import model_helper
from utils import model_loader
from utils import metric
from utils import reader_utils


import numpy as np
import logging
import argparse
import os.path
import pickle
import h5py

logging.basicConfig()
log = logging.getLogger("feature_extractor")
log.setLevel(logging.INFO)


def ExtractFeatures(args):
    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(',')]
        num_gpus = len(gpus)
    else:
        gpus = range(args.num_gpus)
        num_gpus = args.num_gpus

    if num_gpus > 0:
        log.info("Running on GPUs: {}".format(gpus))
    else:
        log.info("Running on CPU")

    my_arg_scope = {
        'order': 'NCHW',
        'use_cudnn': True,
        'cudnn_exhaustive_search': True
    }

    model = cnn.CNNModelHelper(
        name="Extract Features",
        **my_arg_scope
    )

    video_input_args = dict(
        batch_size=args.batch_size,
        clip_per_video=args.clip_per_video,
        decode_type=args.decode_type,
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
        get_rgb=args.input_type == 0,
        get_optical_flow=args.input_type == 1,
        get_video_id=args.get_video_id,
        get_start_frame=args.get_start_frame,
        use_local_file=args.use_local_file,
        crop_per_clip=args.crop_per_clip,
    )

    reader_args = dict(
        name="extract_features" + '_reader',
        input_data=args.test_data,
    )

    reader, num_examples = reader_utils.create_data_reader(
        model,
        **reader_args
    )

    def input_fn(model):
        model_helper.AddVideoInput(
            model,
            reader,
            **video_input_args)

    def create_model_ops(model, loss_scale):
        return model_builder.build_model(
            model=model,
            model_name=args.model_name,
            model_depth=args.model_depth,
            num_labels=args.num_labels,
            batch_size=args.batch_size,
            num_channels=args.num_channels,
            crop_size=args.crop_size,
            clip_length=(
                args.clip_length_of if args.input_type == 1
                else args.clip_length_rgb
            ),
            loss_scale=loss_scale,
            is_test=1,
            multi_label=args.multi_label,
            channel_multiplier=args.channel_multiplier,
            bottleneck_multiplier=args.bottleneck_multiplier,
            use_dropout=args.use_dropout,
            use_convolutional_pred=args.use_convolutional_pred,
            use_pool1=args.use_pool1,
        )

    if num_gpus > 0:
        data_parallel_model.Parallelize_GPU(
            model,
            input_builder_fun=input_fn,
            forward_pass_builder_fun=create_model_ops,
            param_update_builder_fun=None,   # 'None' since we aren't training
            devices=gpus,
            optimize_gradient_memory=True,
        )
    else:
        model._device_type = caffe2_pb2.CPU
        model._devices = [0]
        device_opt = core.DeviceOption(model._device_type, 0)
        with core.DeviceScope(device_opt):
            # Because our loaded models are named with "gpu_x", keep the naming for now.
            # TODO: Save model using `data_parallel_model.ExtractPredictorNet`
            # to extract the model for "gpu_0". It also renames
            # the input and output blobs by stripping the "gpu_x/" prefix
            with core.NameScope("{}_{}".format("gpu", 0)):
                input_fn(model)
                create_model_ops(model, 1.0)

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    if args.db_type == 'pickle':
        model_loader.LoadModelFromPickleFile(model, args.load_model_path)
    elif args.db_type == 'minidb':
        if num_gpus > 0:
            model_helper.LoadModel(args.load_model_path, args.db_type)
        else:
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
                model_helper.LoadModel(args.load_model_path, args.db_type)
    else:
        log.warning("Unsupported db_type: {}".format(args.db_type))

    data_parallel_model.FinalizeAfterCheckpoint(model)

    def fetchActivations(model, outputs, num_iterations):

        all_activations = {}
        for counter in range(num_iterations):
            workspace.RunNet(model.net.Proto().name)

            num_devices = 1  # default for cpu
            if num_gpus > 0:
                num_devices = num_gpus
            for g in range(num_devices):
                for output_name in outputs:
                    blob_name = 'gpu_{}/'.format(g) + output_name
                    activations = workspace.FetchBlob(blob_name)
                    if output_name not in all_activations:
                        all_activations[output_name] = []
                    all_activations[output_name].append(activations)

            if counter % 20 == 0:
                log.info('{}/{} iterations'.format(counter, num_iterations))

        # each key holds a list of activations obtained from each minibatch.
        # we now concatenate these lists to get the final arrays.
        # concatenating during the loop requires a realloc and can get slow.
        for key in all_activations:
            all_activations[key] = np.concatenate(all_activations[key])

        return all_activations

    outputs = [name.strip() for name in args.features.split(',')]
    assert len(outputs) > 0

    if args.num_iterations > 0:
        num_iterations = args.num_iterations
    else:
        if num_gpus > 0:
            examples_per_iteration = args.batch_size * num_gpus
        else:
            examples_per_iteration = args.batch_size
        num_iterations = int(num_examples / examples_per_iteration)

    activations = fetchActivations(model, outputs, num_iterations)

    # saving extracted features
    for index in range(len(outputs)):
        log.info(
            "Read '{}' with shape {}".format(
                outputs[index],
                activations[outputs[index]].shape
            )
        )

    if args.output_path:
        output_path = args.output_path
    else:
        output_path = os.path.dirname(args.test_data) + '/features.pickle'

    log.info('Writing to {}'.format(output_path))
    if args.save_h5:
        with h5py.File(output_path, 'w') as handle:
            for name, activation in activations.items():
                handle.create_dataset(name, data=activation)
    else:
        with open(output_path, 'wb') as handle:
            pickle.dump(activations, handle)

    # perform sanity check
    if args.sanity_check == 1:  # check clip accuracy
        assert args.multi_label == 0
        clip_acc = 0
        softmax = activations['softmax']
        label = activations['label']
        for i in range(len(softmax)):
            sorted_preds = \
                np.argsort(softmax[i])
            sorted_preds[:] = sorted_preds[::-1]
            if sorted_preds[0] == label[i]:
                clip_acc += 1
        log.info('Sanity check --- clip accuracy: {}'.format(
            clip_acc / len(softmax))
        )
    elif args.sanity_check == 2:  # check auc
        assert args.multi_label == 1
        prob = activations['prob']
        label = activations['label']
        mean_auc, mean_ap, mean_wap, _ = metric.mean_ap_metric(prob, label)
        log.info('Sanity check --- AUC: {}, mAP: {}, mWAP: {}'.format(
            mean_auc, mean_ap, mean_wap)
        )


def main():
    parser = argparse.ArgumentParser(
        description="Simple feature extraction"
    )
    parser.add_argument("--db_type", type=str, default='pickle',
                        help="Db type of the testing model")
    parser.add_argument("--model_name", type=str, default='r2plus1d',
                        help="Model name")
    parser.add_argument("--model_depth", type=int, default=18,
                        help="Model depth")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPU devices (instead of --gpus)")
    parser.add_argument("--scale_h", type=int, default=128,
                        help="Scale image height to")
    parser.add_argument("--scale_w", type=int, default=171,
                        help="Scale image width to")
    parser.add_argument("--crop_size", type=int, default=112,
                        help="Input image size (to crop to)")
    parser.add_argument("--clip_length_rgb", type=int, default=4,
                        help="Length of input clips")
    parser.add_argument("--sampling_rate_rgb", type=int, default=1,
                        help="Frame sampling rate")
    parser.add_argument("--num_labels", type=int, default=101,
                        help="Number of labels")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of channels")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, per-GPU")
    parser.add_argument("--load_model_path", type=str, default='',
                        required=True,
                        help="Load saved model for testing")
    parser.add_argument("--test_data", type=str, default="", required=True,
                        help="Dataset on which we will extract features")
    parser.add_argument("--output_path", type=str, default="",
                        help="Path to output pickle; defaults to "
                        + "features.pickle next to <test_data>")
    parser.add_argument("--use_cudnn", type=int, default=1,
                        help="Use CuDNN")
    parser.add_argument("--features", type=str, default="final_avg",
                        help="Comma-separated list of blob names to fetch")
    parser.add_argument("--num_iterations", type=int, default=-1,
                        help="Run only this many iterations")
    parser.add_argument("--num_decode_threads", type=int, default=4,
                        help="")
    parser.add_argument("--clip_length_of", type=int, default=8,
                        help="Frames of optical flow data")
    parser.add_argument("--sampling_rate_of", type=int, default=2,
                        help="Sampling rate for optial flows")
    parser.add_argument("--frame_gap_of", type=int, default=2,
                        help="Frame gap of optical flows")
    parser.add_argument("--input_type", type=int, default=0,
                        help="0=rgb, 1=optical flow")
    parser.add_argument("--flow_data_type", type=int, default=0,
                        help="0=Flow2C, 1=Flow3C, 2=FlowWithGray, 3=FlowWithRGB")
    parser.add_argument("--do_flow_aggregation", type=int, default=0,
                        help="whether to aggregate optical flow across "
                        + "multiple frames")
    parser.add_argument("--clip_per_video", type=int, default=1,
                        help="When clips_per_video > 1, sample this many "
                        + "clips uniformly in time")
    parser.add_argument("--get_video_id", type=int, default=0,
                        help="Output video id")
    parser.add_argument("--multi_label", type=int, default=0,
                        help="Multiple label csv file input")
    parser.add_argument("--sanity_check", type=int, default=0,
                        help="Sanity check on the accuracy/auc")
    parser.add_argument("--decode_type", type=int, default=2,
                        help="0: random, 1: uniform sampling, "
                        + "2: use starting frame")
    parser.add_argument("--channel_multiplier", type=float, default=1.0,
                        help="Channel multiplier")
    parser.add_argument("--use_dropout", type=int, default=0,
                        help="Use dropout at the prediction layer")
    parser.add_argument("--save_h5", type=int, default=0,
                        help="whether save to h5 or pickle, default to pickle")
    parser.add_argument("--bottleneck_multiplier", type=float, default=1.0,
                        help="Bottleneck multiplier")
    parser.add_argument("--get_start_frame", type=int, default=0,
                        help="Output clip start frame")
    parser.add_argument("--use_convolutional_pred", type=int, default=0,
                        help="using convolutional predictions")
    parser.add_argument("--video_res_type", type=int, default=0,
                        help="Video frame scaling option, 0: scaled by "
                        + "height x width; 1: scaled by short edge")
    parser.add_argument("--use_pool1", type=int, default=0,
                        help="use pool1 layer")
    parser.add_argument("--use_local_file", type=int, default=0,
                        help="use local file")
    parser.add_argument("--crop_per_clip", type=int, default=1,
                        help="number of spatial crops per clip")

    args = parser.parse_args()
    log.info(args)

    assert model_builder.model_validation(
        args.model_name,
        args.model_depth,
        args.clip_length_of if args.input_type else args.clip_length_rgb,
        args.crop_size if not args.use_convolutional_pred else 112
    )

    ExtractFeatures(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
