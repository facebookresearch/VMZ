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

python tools/test_net.py \
--test_data=/data/users/joannahsu/datasets/kinetics_val_list/ \
--model_name=r2plus1d --model_depth=18 --num_gpus=2 \
--clip_length_rgb=8 --num_labels=400 --batch_size=1 \
--load_model_path=/mnt/homedir/trandu/video_models/kinetics/l8/r2.5d_d18_l8.pkl
