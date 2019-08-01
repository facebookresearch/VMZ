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
from __future__ import absolute_import, division, print_function, unicode_literals

import lmdb
import pandas
import sys
import argparse

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace
import logging

logging.basicConfig()
log = logging.getLogger("create_video_db")
log.setLevel(logging.INFO)


def create_video_db(
    list_file,
    output_file,
    use_list=0,
    use_video_id=0,
    use_start_frame=0,
    num_epochs=1,
):

    # read csv list file
    list = pandas.read_csv(list_file)

    # checking necessary fields of the provided csv file
    assert 'org_video' in list, \
        "The input list does not have org_video column"
    assert 'label' in list, \
        "The input list does not have label column"
    if use_video_id:
        assert 'video_id' in list, \
            "The input list does not have video_id column"
    if use_start_frame:
        assert use_list == 1, "using starting frame is recommended only " + \
            "with using local file setting for feature extraction"
        assert 'start_frm' in list, \
            "The input list does not have start_frame column"

    if num_epochs > 1:
        assert use_list == 1, "using number of epochs > 1 " + \
            "is recommended only with using local file setting" + \
            "otherwise, there will be redundancy in data written into lmdb"

    # Write to lmdb database...
    LMDB_MAP_SIZE = 1 << 40   # MODIFY
    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)

    # index and size counters
    total_size = 0
    index = 0
    with env.begin(write=True) as txn:
        for epoch in range(num_epochs):
            # shuffle the data frame
            log.info('shuffling index for epoch {}'.format(epoch))
            list = list.sample(frac=1)
            for _, row in list.iterrows():
                file_name = row["org_video"]
                label = row["label"]

                if not use_list:
                    with open(file_name, mode='rb') as file:
                        video_data = file.read()
                else:
                    video_data = file_name

                tensor_protos = caffe2_pb2.TensorProtos()
                video_tensor = tensor_protos.protos.add()
                video_tensor.data_type = 4  # string data
                video_tensor.string_data.append(video_data)

                label_tensor = tensor_protos.protos.add()
                label_tensor.data_type = 2  # int32
                label_tensor.int32_data.append(label)

                if use_start_frame:
                    start_frame = row["start_frm"]
                    start_frame_tensor = tensor_protos.protos.add()
                    start_frame_tensor.data_type = 2  # int32
                    start_frame_tensor.int32_data.append(start_frame)

                if use_video_id:
                    video_id = row["video_id"]
                    video_id_tensor = tensor_protos.protos.add()
                    video_id_tensor.data_type = 10  # int64
                    video_id_tensor.int64_data.append(video_id)

                txn.put(
                    '{}'.format(index).encode('ascii'),
                    tensor_protos.SerializeToString()
                )

                index = index + 1
                if index % 1000 == 0:
                    log.info('processed {} videos'.format(index))
                total_size = total_size + len(video_data) + sys.getsizeof(int)
    return total_size


def main():
    parser = argparse.ArgumentParser(
        description="create video database"
    )
    parser.add_argument("--list_file", type=str, default=None,
                        help="Path to list file")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to output lmdb data")
    parser.add_argument("--use_list", type=int, default=0,
                        help="0: write video encoded data to lmdb, "
                        + "1: write only full path to local video files")
    parser.add_argument("--use_video_id", type=int, default=0,
                        help="0: does not use video_id, "
                        + "1: write also video_id to lmdb")
    parser.add_argument("--use_start_frame", type=int, default=0,
                        help="0: does not use start_frame, "
                        + "1: write also start_frame to lmdb")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Due to lmdb does not allow online shuffle"
                        + "we can write multiple shuffled list")
    args = parser.parse_args()
    create_video_db(
        args.list_file,
        args.output_file,
        args.use_list,
        args.use_video_id,
        args.use_start_frame,
        args.num_epochs
    )


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
