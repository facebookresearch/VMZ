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

import lmdb
import logging

logging.basicConfig()
log = logging.getLogger("reader_utils")
log.setLevel(logging.INFO)


def create_data_reader(
    model,
    name,
    input_data,
):
    reader = model.param_init_net.CreateDB(
        [],
        name=name,
        db=input_data,
        db_type='lmdb',
    )
    lmdb_env = lmdb.open(input_data, readonly=True)
    stat = lmdb_env.stat()
    number_of_examples = stat["entries"]
    lmdb_env.close()

    return reader, number_of_examples
