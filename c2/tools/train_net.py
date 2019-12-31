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

# TODO: sync with latest workflow features (Deepti or Du should have a look)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import numpy as np
import time

from caffe2.python import workspace, cnn
from caffe2.python import timeout_guard, experiment_util, data_parallel_model
import caffe2.python.predictor.predictor_exporter as pred_exp
import models.model_builder as model_builder
import utils.model_helper as model_helper
import utils.model_loader as model_loader
from utils import reader_utils
from utils import metric

# Logger
log = logging.getLogger("train_net")
log.setLevel(logging.INFO)


def AddMomentumParameterUpdate(train_model, LR):
    '''
    Add the momentum-SGD update.
    '''
    params = train_model.GetParams()
    assert(len(params) > 0)

    for param in params:
        param_grad = train_model.param_to_grad[param]
        param_momentum = train_model.param_init_net.ConstantFill(
            [param], param + '_momentum', value=0.0
        )

        # Update param_grad and param_momentum in place
        train_model.net.MomentumSGDUpdate(
            [param_grad, param_momentum, LR, param],
            [param_grad, param_momentum, param],
            momentum=0.9,
            nesterov=1,
        )


def GetCheckpointParams(train_model):
    prefix = "gpu_{}".format(train_model._devices[0])
    params = [str(p) for p in train_model.GetParams(prefix)]
    params.extend([str(p) + "_momentum" for p in params])
    params.extend([str(p) for p in train_model.GetComputedParams(prefix)])

    assert len(params) > 0
    return params


def SaveModel(args, train_model, epoch):
    prefix = "gpu_{}".format(train_model._devices[0])
    predictor_export_meta = pred_exp.PredictorExportMeta(
        predict_net=train_model.net.Proto(),
        parameters=GetCheckpointParams(train_model),
        inputs=[prefix + "/data"],
        outputs=[prefix + "/softmax"],
        shapes={
            prefix + "/softmax": (1, args.num_labels),
            prefix + "/data": (
                args.num_channels,
                args.clip_length_of if args.input_type else args.clip_length_rgb,
                args.crop_size,
                args.crop_size
            )
        }
    )

    # save the train_model for the current epoch
    model_path = "%s/%s_%d.mdl" % (
        args.file_store_path,
        args.save_model_name,
        epoch,
    )

    # save the model
    pred_exp.save_to_db(
        db_type='minidb',
        db_destination=model_path,
        predictor_export_meta=predictor_export_meta,
    )


def RunEpoch(
    args,
    epoch,
    train_model,
    test_model,
    batch_size,
    num_shards,
    expname,
    explog,
):
    log.info("Starting epoch {}/{}".format(epoch, args.num_epochs))
    epoch_iters = int(args.epoch_size / batch_size / num_shards)
    if args.multi_label:
        accumulated_prob = np.empty(shape=[0, args.num_labels], dtype=np.float)
        accumulated_label = np.empty(shape=[0, args.num_labels], dtype=np.int32)
    for i in range(epoch_iters):
        # This timeout is required (temporarily) since CUDA-NCCL
        # operators might deadlock when synchronizing between GPUs.
        timeout = 6000.0 if i == 0 else 600.0
        with timeout_guard.CompleteInTimeOrDie(timeout):
            t1 = time.time()
            workspace.RunNet(train_model.net.Proto().name)
            t2 = time.time()
            dt = t2 - t1
        if args.multi_label:
            prefix = "gpu_{}".format(train_model._devices[0])
            prob = workspace.FetchBlob(prefix + '/prob')
            label = workspace.FetchBlob(prefix + '/label')
            accumulated_prob = np.concatenate((accumulated_prob, prob), axis=0)
            accumulated_label = np.concatenate(
                (accumulated_label, label), axis=0
            )

        if i % args.display_iter == 0:
            fmt = "Finished iteration {}/{} of epoch {} ({:.2f} clips/sec)"
            log.info(fmt.format(i, epoch_iters, epoch, batch_size / dt))
            prefix = "gpu_{}".format(train_model._devices[0])
            loss = workspace.FetchBlob(prefix + '/loss')
            if args.multi_label:
                mean_auc, mean_ap, _, _ = \
                    metric.mean_ap_metric(accumulated_prob, accumulated_label)
                train_msg = \
                    "Training loss: {}, AUC: {}, mAP: {}".format(
                        np.mean(loss), mean_auc, mean_ap
                    )
                if accumulated_label.shape[0] > 4096:
                    accumulated_prob = accumulated_prob[-4096:, :]
                    accumulated_label = accumulated_label[-4096:, :]
            else:
                accuracy = workspace.FetchBlob(prefix + '/accuracy')
                train_msg = "Training loss: {}, accuracy: {}".format(
                    loss, accuracy
                )

            log.info(train_msg)

    num_clips = epoch * epoch_iters * batch_size
    prefix = "gpu_{}".format(train_model._devices[0])
    loss = workspace.FetchBlob(prefix + '/loss')
    learning_rate = workspace.FetchBlob(prefix + '/LR')
    if args.multi_label:
        accuracy = -1
        loss = np.mean(loss)
    else:
        mean_ap = -1
        mean_auc = -1
    if (test_model is not None):
        # Run 100 iters of testing
        ntests = 0
        test_accuracy = 0
        test_mean_auc = 0
        test_mean_ap = 0
        all_prob = np.empty(shape=[0, args.num_labels], dtype=np.float)
        all_label = np.empty(shape=[0, args.num_labels], dtype=np.int32)
        for _ in range(0, 100):
            workspace.RunNet(test_model.net.Proto().name)
            for g in test_model._devices:
                prefix = "gpu_{}".format(g)
                if args.multi_label:
                    prob = workspace.FetchBlob(prefix + '/prob')
                    label = workspace.FetchBlob(prefix + '/label')
                    all_prob = np.concatenate((all_prob, prob), axis=0)
                    all_label = np.concatenate((all_label, label), axis=0)
                else:
                    accuracy = workspace.FetchBlob(prefix + '/accuracy')
                    test_accuracy += np.asscalar(accuracy)
                ntests += 1
        if args.multi_label:
            test_mean_auc, test_mean_ap, _, _ = \
                metric.mean_ap_metric(all_prob, all_label)
            log.info("Test AUC: {}, mAP: {}".format(mean_auc, mean_ap))
        else:
            test_accuracy /= ntests
            log.info("Test accuracy: {}".format(test_accuracy))
    else:
        test_accuracy = (-1)
        test_mean_auc = (-1)
        test_mean_ap = (-1)

    explog.log(
        input_count=num_clips,
        batch_count=(i + epoch * epoch_iters),
        additional_values={
            'accuracy': accuracy,
            'train_AUC': mean_auc,
            'train_mAP': mean_ap,
            'loss': loss,
            'learning_rate': learning_rate,
            'epoch': epoch,
            'test_accuracy': test_accuracy,
            'test_mean_auc': test_mean_auc,
            'test_mean_ap': test_mean_ap,
        }
    )
    assert loss < 40, "Exploded gradients :("

    return epoch + 1


def Train(args):
    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(',')]
        num_gpus = len(gpus)
    else:
        gpus = range(args.num_gpus)
        num_gpus = args.num_gpus

    log.info("Running on GPUs: {}".format(gpus))

    # Modify to make it consistent with the distributed trainer
    total_batch_size = args.batch_size * num_gpus
    batch_per_device = args.batch_size

    # Round down epoch size to closest multiple of batch size across machines
    epoch_iters = int(args.epoch_size / total_batch_size)
    args.epoch_size = epoch_iters * total_batch_size
    log.info("Using epoch size: {}".format(args.epoch_size))

    # Create CNNModeLhelper object
    train_model = cnn.CNNModelHelper(
        order="NCHW",
        name='{}_train'.format(args.model_name),
        use_cudnn=(True if args.use_cudnn == 1 else False),
        cudnn_exhaustive_search=True,
        ws_nbytes_limit=(args.cudnn_workspace_limit_mb * 1024 * 1024),
    )

    # Model building functions
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
                args.clip_length_of if args.input_type
                else args.clip_length_rgb
            ),
            loss_scale=loss_scale,
            pred_layer_name=args.pred_layer_name,
            multi_label=args.multi_label,
            channel_multiplier=args.channel_multiplier,
            bottleneck_multiplier=args.bottleneck_multiplier,
            use_dropout=args.use_dropout,
            conv1_temporal_stride=args.conv1_temporal_stride,
            conv1_temporal_kernel=args.conv1_temporal_kernel,
            use_pool1=args.use_pool1,
        )

    # SGD
    def add_parameter_update_ops(model):
        model.AddWeightDecay(args.weight_decay)
        ITER = model.Iter("ITER")
        stepsz = args.step_epoch * args.epoch_size / args.batch_size / num_gpus
        LR = model.net.LearningRate(
            [ITER],
            "LR",
            base_lr=args.base_learning_rate * num_gpus,
            policy="step",
            stepsize=int(stepsz),
            gamma=args.gamma,
        )
        AddMomentumParameterUpdate(model, LR)

    # Input. Note that the reader must be shared with all GPUS.
    train_reader, train_examples = reader_utils.create_data_reader(
        train_model,
        name="train_reader",
        input_data=args.train_data,
    )
    log.info("Training set has {} examples".format(train_examples))

    def add_video_input(model):
        model_helper.AddVideoInput(
            model,
            train_reader,
            batch_size=batch_per_device,
            length_rgb=args.clip_length_rgb,
            clip_per_video=1,
            random_mirror=True,
            decode_type=0,
            sampling_rate_rgb=args.sampling_rate_rgb,
            scale_h=args.scale_h,
            scale_w=args.scale_w,
            crop_size=args.crop_size,
            video_res_type=args.video_res_type,
            short_edge=min(args.scale_h, args.scale_w),
            num_decode_threads=args.num_decode_threads,
            do_multi_label=args.multi_label,
            num_of_class=args.num_labels,
            random_crop=True,
            input_type=args.input_type,
            length_of=args.clip_length_of,
            sampling_rate_of=args.sampling_rate_of,
            frame_gap_of=args.frame_gap_of,
            do_flow_aggregation=args.do_flow_aggregation,
            flow_data_type=args.flow_data_type,
            get_rgb=(args.input_type == 0),
            get_optical_flow=(args.input_type == 1),
            get_video_id=args.get_video_id,
            jitter_scales=[int(n) for n in args.jitter_scales.split(',')],
            use_local_file=args.use_local_file,
        )

    # Create parallelized model
    data_parallel_model.Parallelize_GPU(
        train_model,
        input_builder_fun=add_video_input,
        forward_pass_builder_fun=create_model_ops,
        param_update_builder_fun=add_parameter_update_ops,
        devices=gpus,
        rendezvous=None,
        net_type=('prof_dag' if args.profiling == 1 else 'dag'),
        optimize_gradient_memory=True,
    )

    # Add test model, if specified
    test_model = None
    if args.test_data is not None:
        log.info("----- Create test net ----")
        test_model = cnn.CNNModelHelper(
            order="NCHW",
            name='{}_test'.format(args.model_name),
            use_cudnn=(True if args.use_cudnn == 1 else False),
            cudnn_exhaustive_search=True
        )

        test_reader, test_examples = reader_utils.create_data_reader(
            test_model,
            name="test_reader",
            input_data=args.test_data,
        )

        log.info("Testing set has {} examples".format(test_examples))

        def test_input_fn(model):
            model_helper.AddVideoInput(
                model,
                test_reader,
                batch_size=batch_per_device,
                length_rgb=args.clip_length_rgb,
                clip_per_video=1,
                decode_type=0,
                random_mirror=False,
                random_crop=False,
                sampling_rate_rgb=args.sampling_rate_rgb,
                scale_h=args.scale_h,
                scale_w=args.scale_w,
                crop_size=args.crop_size,
                video_res_type=args.video_res_type,
                short_edge=min(args.scale_h, args.scale_w),
                num_decode_threads=args.num_decode_threads,
                do_multi_label=args.multi_label,
                num_of_class=args.num_labels,
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

        data_parallel_model.Parallelize_GPU(
            test_model,
            input_builder_fun=test_input_fn,
            forward_pass_builder_fun=create_model_ops,
            param_update_builder_fun=None,
            devices=gpus,
            optimize_gradient_memory=True,
        )
        workspace.RunNetOnce(test_model.param_init_net)
        workspace.CreateNet(test_model.net)

    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net)

    epoch = 0
    # load the pre-trained model and reset epoch
    if args.load_model_path is not None:
        if args.db_type == 'pickle':
            model_loader.LoadModelFromPickleFile(
                train_model,
                args.load_model_path,
                use_gpu=True,
                root_gpu_id=gpus[0]
            )
        else:
            model_helper.LoadModel(
                args.load_model_path, args.db_type
            )
        # Sync the model params
        data_parallel_model.FinalizeAfterCheckpoint(
            train_model,
            GetCheckpointParams(train_model),
        )

        if args.is_checkpoint:
            # reset epoch. load_model_path should end with *_X.mdl,
            # where X is the epoch number
            last_str = args.load_model_path.split('_')[-1]
            if last_str.endswith('.mdl'):
                epoch = int(last_str[:-4])
                log.info("Reset epoch to {}".format(epoch))
            else:
                log.warning("The format of load_model_path doesn't match!")

    expname = "%s_gpu%d_b%d_L%d_lr%.2f" % (
        args.model_name,
        args.num_gpus,
        total_batch_size,
        args.num_labels,
        args.base_learning_rate,
    )
    explog = experiment_util.ModelTrainerLog(expname, args)

    # Run the training one epoch a time
    while epoch < args.num_epochs:
        epoch = RunEpoch(
            args,
            epoch,
            train_model,
            test_model,
            total_batch_size,
            1,
            expname,
            explog
        )

        # Save the model for each epoch
        SaveModel(args, train_model, epoch)


def main():
    # TODO: use argv
    parser = argparse.ArgumentParser(
        description="Caffe2: simple video training"
    )
    parser.add_argument("--model_name", type=str, default='r2plus1d',
                        help="Name of the model")
    parser.add_argument("--model_depth", type=int, default=18,
                        help="Depth of the model")
    parser.add_argument("--train_data", type=str, default=None,
                        help="Path to train data",
                        required=True)
    parser.add_argument("--test_data", type=str, default=None,
                        help="Path to test data")
    parser.add_argument("--db_type", type=str, default="minidb",
                        help="Database type to save the training model")
    parser.add_argument("--gpus", type=str,
                        help="Comma separated list of GPU devices to use")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPU devices (instead of --gpus)")
    parser.add_argument("--scale_h", type=int, default=128,
                        help="Scale image height to")
    parser.add_argument("--scale_w", type=int, default=171,
                        help="Scale image width to")
    parser.add_argument("--crop_size", type=int, default=112,
                        help="Input image size (to crop to)")
    parser.add_argument("--num_decode_threads", type=int, default=4,
                        help="# of threads/GPU dedicated for video decoding")
    parser.add_argument("--clip_length_rgb", type=int, default=16,
                        help="Length of input clips")
    parser.add_argument("--sampling_rate_rgb", type=int, default=1,
                        help="Frame sampling rate")
    parser.add_argument("--num_labels", type=int, default=101,
                        help="Number of labels")
    parser.add_argument("--num_channels", type=int, default=3,
                        help="Number of channels")
    parser.add_argument("--clip_length_of", type=int, default=8,
                        help="Frames of optical flow data")
    parser.add_argument("--sampling_rate_of", type=int, default=2,
                        help="")
    parser.add_argument("--frame_gap_of", type=int, default=2,
                        help="")
    parser.add_argument("--input_type", type=int, default=0,
                        help="False=rgb, True=optical flow")
    parser.add_argument("--flow_data_type", type=int, default=0,
                        help="0=Flow2C, 1=Flow3C, 2=FlowWithGray, 3=FlowWithRGB")
    parser.add_argument("--do_flow_aggregation", type=int, default=0,
                        help="whether to aggregate optical flow across "
                        + "multiple frames")
    parser.add_argument("--get_video_id", type=int, default=0,
                        help="Output video id")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size, total over all GPUs")
    parser.add_argument("--epoch_size", type=int, default=110000,
                        help="Number of videos/epoch, total over all machines")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Num epochs.")
    parser.add_argument("--base_learning_rate", type=float, default=0.003,
                        help="Initial learning rate.")
    parser.add_argument("--step_epoch", type=int, default=10,
                        help="Reducing learning rate every step_epoch.")
    parser.add_argument("--gamma", type=float, default=0.1,
                        help="Learning rate decay factor.")
    parser.add_argument("--display_iter", type=int, default=10,
                        help="Display information every # of iterations.")
    parser.add_argument("--weight_decay", type=float, default=0.005,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--cudnn_workspace_limit_mb", type=int, default=64,
                        help="CuDNN workspace limit in MBs")
    parser.add_argument("--file_store_path", type=str, default="/tmp",
                        help="Path to directory to use for saving checkpoints")
    parser.add_argument("--save_model_name", type=str, default="simple_c3d",
                        help="Save the trained model to a given name")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Load previously saved model to continue training")
    parser.add_argument("--use_cudnn", type=int, default=1,
                        help="Use CuDNN")
    parser.add_argument("--profiling", type=int, default=0,
                        help="Profile training time")
    parser.add_argument("--pred_layer_name", type=str, default=None,
                        help="the prediction layer name")
    parser.add_argument("--multi_label", type=int, default=0,
                        help="Multiple label training")
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
    parser.add_argument("--video_res_type", type=int, default=1,
                        help="Video frame scaling option, 0: scaled by "
                        + "height x width; 1: scaled by short edge")
    parser.add_argument("--use_pool1", type=int, default=0,
                        help="use pool1 layer")
    parser.add_argument("--jitter_scales", type=str, default="128,160", required=True,
                        help="spatial scales jitter, separated by commas")
    parser.add_argument("--use_local_file", type=int, default=0,
                        help="use local file")
    parser.add_argument("--is_checkpoint", type=int, default=1,
                        help="0: pretrained_model is used as initalization"
                        + "1: pretrained_model is used as a checkpoint")
    args = parser.parse_args()

    log.info(args)

    assert model_builder.model_validation(
        args.model_name,
        args.model_depth,
        args.clip_length_of if args.input_type else args.clip_length_rgb,
        args.crop_size
    )

    Train(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
