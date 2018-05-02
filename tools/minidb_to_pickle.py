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

from caffe2.python import workspace, core, scope
import caffe2.python.predictor.predictor_py_utils as pred_utils
import caffe2.python.predictor.predictor_exporter as pred_exp
from caffe2.python.predictor_constants import predictor_constants

import argparse
import logging
import cPickle as pickle
logging.basicConfig()
log = logging.getLogger("minidb_to_pickle")
log.setLevel(logging.INFO)


def unscope_name(blob_name):
    return blob_name[blob_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


def ConvertModel(args):
    meta_net_def = pred_exp.load_from_db(args.load_model_path, args.db_type)
    net = core.Net(
        pred_utils.GetNet(meta_net_def, predictor_constants.PREDICT_NET_TYPE)
    )
    init_net = core.Net(
        pred_utils.
        GetNet(meta_net_def, predictor_constants.GLOBAL_INIT_NET_TYPE)
    )
    init_net.RunAllOnGPU()
    assert workspace.RunNetOnce(init_net)

    pred_params = list(set(net.Proto().external_input) - set(['gpu_0/data']))

    save_params = [str(param) for param in pred_params]
    save_blobs = {}
    for param in save_params:
        scoped_blob_name = str(param)
        unscoped_blob_name = unscope_name(scoped_blob_name)
        if unscoped_blob_name not in save_blobs:
            save_blobs[unscoped_blob_name] = workspace.FetchBlob(
                scoped_blob_name)
            log.info(
                '{:s} -> {:s}'.format(scoped_blob_name, unscoped_blob_name))
    log.info('saving weights to {}'.format(args.save_model_path))
    with open(args.save_model_path, 'w') as fwrite:
        pickle.dump(dict(blobs=save_blobs), fwrite, pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(
        description="Convert model to pickle"
    )

    parser.add_argument("--load_model_path", type=str,
                        default='', required=True, help="Saved model path")
    parser.add_argument("--save_model_path", type=str, default='',
                        required=True, help="Converted model path to save")
    parser.add_argument("--db_type", type=str, default='minidb',
                        help="Db type of the testing model")

    args = parser.parse_args()
    print(args)

    ConvertModel(args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
