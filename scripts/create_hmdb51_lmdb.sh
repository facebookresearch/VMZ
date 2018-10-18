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

python data/create_video_db.py \
--list_file=data/list/hmdb51/hmdb51_test_01.csv \
--output_file=/data/users/trandu/datasets/hmdb51_test01

python data/create_video_db.py \
--list_file=data/list/hmdb51/hmdb51_train_01.csv \
--output_file=/data/users/trandu/datasets/hmdb51_train01
