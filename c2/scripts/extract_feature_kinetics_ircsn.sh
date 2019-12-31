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

# same as create lmdb feature extraction list
# we do feature extraction for 10 splits
python tools/extract_features.py \
--test_data=/mnt/vol/gfsdataswarm-oregon/users/trandu/kinetics_val_lists_480_lmdb_10crops/val_1 \
--model_name=ir-csn --model_depth=50 --clip_length_rgb=8 --sampling_rate_rgb=2 --use_pool1=1 \
--scale_w=342 --scale_h=256 --crop_size=224 --video_res_type=1 \
--load_model_path=/mnt/vol/gfsdataswarm-oregon/users/trandu/models/r3d-sep_50_from_scratch_kinetics_f100474898.pkl \
--gpus=0 \
--batch_size=8 --num_iterations=1000 \
--output_path=/data/users/trandu/test.pkl \
--features=softmax,label,video_id \
--sanity_check=1 --get_video_id=1 --sanity_check=1 --get_start_frame=1 --use_local_file=1 --num_labels=400
