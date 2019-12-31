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

python tools/test_net_large.py \
--test_data=/data/users/trandu/kinetics/kinetics_val_high_qual_480_lmdb/ \
--model_name=ir-csn --model_depth=152 --gpus=0 \
--num_labels=400 --batch_size=1 --use_pool1=1 \
--clip_length_rgb=32 --sampling_rate_rgb=2 \
--scale_w=342 --scale_h=256 --crop_size=256 --video_res_type=1 \
--use_convolutional_pred=1 \
--crop_per_inference=1 --crop_per_clip=3 \
--clip_per_video=10 --use_local_file=1 \
--load_model_path=/data/users/trandu/models/irCSN_152_ft_kinetics_from_URU_f126851907.pkl
