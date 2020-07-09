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

import numpy as np
from caffe2.python import workspace, core
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants \
    as predictor_constants


import logging

logging.basicConfig()
log = logging.getLogger("model_helper")
log.setLevel(logging.INFO)


def GetFlopsAndParams(model, gpu_id=0):
    model_ops = model.net.Proto().op
    master_gpu = 'gpu_{}'.format(gpu_id)
    param_ops = []
    for idx in range(len(model_ops)):
        op_type = model.net.Proto().op[idx].type
        op_input = model.net.Proto().op[idx].input[0]
        if op_type in ['Conv', 'FC'] and op_input.find(master_gpu) >= 0:
            param_ops.append(model.net.Proto().op[idx])

    num_flops = 0
    num_params = 0
    num_interactions = 0
    for idx in range(len(param_ops)):
        op = param_ops[idx]
        op_type = op.type
        op_inputs = param_ops[idx].input
        op_output = param_ops[idx].output[0]
        layer_flops = 0
        layer_params = 0
        if op_type == 'Conv':
            for op_input in op_inputs:
                if '_w' in op_input and 'bias' not in op_input:
                    param_blob = op_input
                    param_shape = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = (
                        np.prod(param_shape)
                    )
                    output_shape = np.array(
                        workspace.FetchBlob(str(op_output))).shape
                    layer_flops = layer_params * np.prod(output_shape[2:])
                    layer_interactions = 0.5 * param_shape[0] * param_shape[1] * (param_shape[1] - 1)
                    # log.info('{} size {}x{}x{} FLOPs {} params {} inters {}'.format(
                    #     str(param_blob),
                    #     (param_shape[2] if len(param_shape) == 5 else 1),
                    #     (param_shape[3] if len(param_shape) == 5 else param_shape[2]),
                    #     (param_shape[4] if len(param_shape) == 5 else param_shape[3]),
                    #     layer_flops,
                    #     layer_params,
                    #     layer_interactions))
        elif op_type == 'FC':
            for op_input in op_inputs:
                if '_w' in op_input:
                    param_blob = op_input
                    param_shape = np.array(
                        workspace.FetchBlob(str(param_blob))).shape
                    layer_params = param_shape[0] * param_shape[1]
                    layer_flops = layer_params
                    layer_interactions = 0  # not count interactions on FC
        layer_params /= 1000000
        layer_flops /= 1000000000
        layer_interactions /= 1000000000
        num_flops += layer_flops
        num_params += layer_params
        num_interactions += layer_interactions
    return num_flops, num_params, num_interactions


def LoadModel(path, dbtype='minidb'):
    '''
    Load pretrained model from file
    '''
    log.info("Loading path: {}".format(path))
    meta_net_def = pred_exp.load_from_db(path, dbtype)
    init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.GLOBAL_INIT_NET_TYPE))
    predict_init_net = core.Net(pred_utils.GetNet(
        meta_net_def, predictor_constants.PREDICT_INIT_NET_TYPE))

    predict_init_net.RunAllOnGPU()
    init_net.RunAllOnGPU()
    assert workspace.RunNetOnce(predict_init_net)
    assert workspace.RunNetOnce(init_net)


def AddVideoInput(model, reader, **kwargs):
    if 'input_type' in kwargs:
        input_type = kwargs['input_type']
    else:
        input_type = False
    if 'get_video_id' in kwargs:
        get_video_id = kwargs['get_video_id']
    else:
        get_video_id = False
    if 'get_start_frame' in kwargs:
        get_start_frame = kwargs['get_start_frame']
    else:
        get_start_frame = False

    if input_type == 0:
        log.info('outputing rgb data')
    elif input_type == 1:
        log.info('outputing optical flow data')
    else:
        log.info('unknown input_type option')

    if get_video_id:
        if get_start_frame:
            data, label, video_id, start_frame = model.net.VideoInput(
                reader,
                ["data", "label", "video_id", "start_frame"],
                name="data",
                **kwargs
            )
        else:
            data, label, video_id = model.net.VideoInput(
                reader,
                ["data", "label", "video_id"],
                name="data",
                **kwargs
            )
    else:
        data, label = model.net.VideoInput(
            reader,
            ["data", "label"],
            name="data",
            **kwargs
        )

    data = model.StopGradient(data, data)


def GetModelWeights(model, gpu_id=0):
    '''
    function that returns all the model weights in a dict
    '''
    model_ops = model.net.Proto().op
    master_gpu = 'gpu_{}'.format(gpu_id)
    param_ops = []
    for idx in range(len(model_ops)):
        op_type = model.net.Proto().op[idx].type
        op_input = model.net.Proto().op[idx].input[0]
        if op_type in ['Conv', 'FC'] and op_input.find(master_gpu) >= 0:
            param_ops.append(model.net.Proto().op[idx])

    weight_dict = {}
    for idx in range(len(param_ops)):
        # op_type = op.type
        op_inputs = param_ops[idx].input
        # op_output = param_ops[idx].output[0]
        for op_input in op_inputs:
            param_blob = op_input
            weights = np.array(workspace.FetchBlob(str(param_blob)))
            weight_dict[param_blob] = weights
    return weight_dict


def getTrainingGPUs(path, dbtype):
    '''
    well, turns out that SaveModel savs the vars with the gpu_X/ prefix...
    this function returns the GPUs used during training.
    '''
    meta_net_def = pred_exp.load_from_db(path, dbtype)
    gpus = set()

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for kv in meta_net_def.nets:
        net = kv.value
        for op in net.op:
            if op.input and op.output:
                thisgpu = op.input[-1].split('/')[0].split('_')[-1]
                if is_number(thisgpu):
                    gpus.add(thisgpu)
    return gpus
