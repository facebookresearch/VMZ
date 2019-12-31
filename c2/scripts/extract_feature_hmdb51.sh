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

for ((i=1;i<=2;i++)); \
do \
python tools/extract_features.py \
--test_data=/data/users/trandu/datasets/hmdb51_feature_extraction/hmdb51_test_01_video_id_dense_l32_$i \
--model_name=r2plus1d --model_depth=34 --clip_length_rgb=32 \
--gpus=0,1,2,3,4,5,6,7 \
--batch_size=4 \
--load_model_path=r2plus1d_8.mdl --db_type=minidb \
--output_path=/data/users/trandu/datasets/hmdb51_features/ft/hmdb51_test_01_video_id_dense_l32_$i.pkl \
--features=softmax,label,video_id \
--sanity_check=1 --get_video_id=1 --use_local_file=1 --num_labels=51; \
done
