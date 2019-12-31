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
--train_data=/data/users/joannahsu/datasets/kinetics_train_list \
--test_data=/data/users/joannahsu/datasets/kinetics_val_list \
--model_name=r2plus1d --model_depth=18 \
--clip_length_rgb=8 --batch_size=32 \
--gpus=0,1 --base_learning_rate=0.01 \
--epoch_size=1000000 --num_epochs=35 --step_epoch=10 \
--weight_decay=0.0001 --num_labels=400 --use_local_file=1
