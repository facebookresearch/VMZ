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
from __future__ import unicode_literals

import logging

import numpy as np
import cPickle as pickle
from collections import OrderedDict

from caffe2.python import core, workspace, scope
from caffe2.proto import caffe2_pb2

logging.basicConfig()
log = logging.getLogger("model_loader")
log.setLevel(logging.INFO)


def unscope_name(blob_name):
    return blob_name[blob_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


def scoped_name(blob_name):
    return scope.CurrentNameScope() + blob_name


def FlipBGR2RGB(source_blob):
    return source_blob[:, ::-1, :, :, :]


def BroacastParameters(model, src_gpu, gpus):

    log.info("Broadcasting parameters from gpu {} to gpu: {}".format(
        src_gpu, ','.join([str(g) for g in gpus]))
    )

    for param in model.params:
        if 'gpu_{}'.format(gpus[0]) in str(param):
            for i in gpus:
                blob = workspace.FetchBlob(str(param))
                target_blob_name = str(param).replace(
                    'gpu_{}'.format(src_gpu),
                    'gpu_{}'.format(i)
                )
                log.info('broadcast {} -> {}'.format(
                    str(param), target_blob_name)
                )
                workspace.FetchBlob(str(param))
                with core.DeviceScope(
                        core.DeviceOption(caffe2_pb2.CUDA, i)):
                    workspace.FeedBlob(target_blob_name, blob)


def CollapseBlob(source_blob, target_shape, blob_name):
    original_shape = source_blob.shape
    assert len(target_shape) == 4
    assert target_shape[0] == original_shape[0]
    assert target_shape[1] == original_shape[1]

    assert source_blob.ndim == 5
    assert original_shape[2] == 1
    assert target_shape[2] == original_shape[3]
    assert target_shape[3] == original_shape[4]

    collapsed_blob = np.squeeze(
        source_blob,
        axis=2
    )
    log.info('collapsing {}'.format(blob_name))
    return collapsed_blob


def InflateBlob(source_blob, target_shape, blob_name, inflating_opt=1):
    original_shape = source_blob.shape
    # only inflate to 5D tensor
    assert len(target_shape) == 5
    assert target_shape[0] == original_shape[0]
    assert target_shape[1] == original_shape[1]

    assert source_blob.ndim == 4 or source_blob.ndim == 5
    if source_blob.ndim == 5:
        assert original_shape[2] == 1
        source_blob = np.squeeze(
            source_blob,
            axis=2
        )
        original_shape = source_blob.shape

    assert target_shape[3] == original_shape[2]
    assert target_shape[4] == original_shape[3]

    ncopies = target_shape[2]
    inflated_blob = np.repeat(
        np.expand_dims(
            source_blob,
            axis=-3
        ),
        ncopies,
        axis=-3
    )

    if inflating_opt == 1:
        inflated_blob = inflated_blob / ncopies
        log.info('inflating {} to T={}'.format(blob_name, ncopies))
    else:
        assert ncopies % 2 == 1
        inflated_blob[..., :int(ncopies / 2), :, :] = 0
        inflated_blob[..., int(ncopies / 2) + 1:, :, :] = 0
        log.info('center initializing {}'.format(blob_name))
    return inflated_blob


def LoadModelFromPickleFile(
    model,
    pkl_file,
    use_gpu=True,
    root_gpu_id=0,
    bgr2rgb=False,
    inflating=True,
    collapsing=True,
    center_init=False,
):

    ws_blobs = workspace.Blobs()
    with open(pkl_file, 'r') as fopen:
        blobs = pickle.load(fopen)

    if 'blobs' in blobs:
        blobs = blobs['blobs']

    unscoped_blob_names = OrderedDict()
    for blob in model.GetAllParams():
        unscoped_blob_names[unscope_name(str(blob))] = True

    if use_gpu:
        device_opt = caffe2_pb2.CUDA
    else:
        device_opt = caffe2_pb2.CPU

    with core.NameScope('gpu_{}'.format(root_gpu_id)):
        with core.DeviceScope(core.DeviceOption(device_opt, root_gpu_id)):
            for unscoped_blob_name in unscoped_blob_names.keys():
                scoped_blob_name = scoped_name(unscoped_blob_name)
                if unscoped_blob_name not in blobs:
                    log.info('{} not found'.format(unscoped_blob_name))
                    continue
                if scoped_blob_name in ws_blobs:
                    ws_blob = workspace.FetchBlob(scoped_blob_name)
                    target_shape = ws_blob.shape
                    if target_shape == blobs[unscoped_blob_name].shape:
                        log.info('copying {}'.format(unscoped_blob_name))
                        if bgr2rgb and unscoped_blob_name == 'conv1_w':
                            feeding_blob = FlipBGR2RGB(
                                blobs[unscoped_blob_name]
                            )
                        else:
                            feeding_blob = blobs[unscoped_blob_name]

                    elif ws_blob.ndim == 5:
                        # inflate from FC to 1x1x1 conv
                        if blobs[unscoped_blob_name].ndim == 2:
                            log.info('convolutionalize {}'.format(
                                unscoped_blob_name)
                            )
                            feeding_blob = blobs[unscoped_blob_name]
                            feeding_blob = np.reshape(
                                feeding_blob,
                                feeding_blob.shape + (1, 1, 1)
                            )
                        else:
                            # may need to inflate
                            if not inflating:
                                log.info(
                                    '{} found, but inflating is ignored'.format(
                                        unscoped_blob_name
                                    )
                                )
                                continue
                            feeding_blob = InflateBlob(
                                blobs[unscoped_blob_name],
                                target_shape,
                                unscoped_blob_name,
                                (0 if center_init else 1)
                            )

                    elif ws_blob.ndim == 4:
                        # may need to collapse
                        if not collapsing:
                            log.info(
                                '{} found, but collapsing is ignored'.format(
                                    unscoped_blob_name
                                )
                            )
                            continue
                        feeding_blob = CollapseBlob(
                            blobs[unscoped_blob_name],
                            target_shape,
                            unscoped_blob_name
                        )
                    # either copy, inflate, or collapse blob
                    workspace.FeedBlob(
                        scoped_blob_name,
                        feeding_blob.astype(np.float32, copy=False)
                    )
