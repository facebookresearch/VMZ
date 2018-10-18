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

python tools/train_net.py \
--train_data=/data/users/trandu/datasets/hmdb51_train01 \
--test_data=/data/users/trandu/datasets/hmdb51_test01 \
--model_name=r2plus1d --model_depth=34 \
--clip_length_rgb=32 --batch_size=4 \
--pretrained_model=/mnt/homedir/trandu/video_models/kinetics/l32/r2.5d_d34_l32.pkl \
--db_type='pickle' --is_checkpoint=0 \
--gpus=0,1,2,3,4,5,6,7 --base_learning_rate=0.0002 \
--epoch_size=40000 --num_epochs=8 --step_epoch=2 \
--weight_decay=0.005 --num_labels=51
