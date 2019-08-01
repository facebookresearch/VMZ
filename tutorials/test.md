# Testing

We provide some basic utilities for testing models by VMZ codebase. The ```test_net.py``` tool is used for testing small networks when GPU memory can fit at least 10 clips, .e.g shallow models or model with short clip input. The ```test_net_large.py``` tool is provided for testing large models with long input clips.

## Small Models

In example scripts, we provide an example of testing R(2+1)D-18 trained on 8-frame clips in ```scripts/scripts/test_r2plus1d_kinetics.sh```. The content of the script looks like below:

```
python tools/test_net.py \
--test_data=/data/users/trandu/datasets/kinetics_val_list/ \
--model_name=r2plus1d --model_depth=18 --gpus=0,1 \
--clip_length_rgb=8 --num_labels=400 --batch_size=1 \
--use_local_file=1 --clip_per_video=10 \
--load_model_path=/mnt/homedir/trandu/video_models/kinetics/l8/r2.5d_d18_l8.pkl
```
Note that you need to create an lmdb database of Kinetics using ```data/create_video_db.py``` and replace it to the ```test_data``` argument. You also need to download model (or train youself), in this case ```r2.5d_d18_l8.pkl```.

When ```test_data``` and ```load_model_path``` are properly adjusted, to run test, you simply laucnh it by:
```
sh scripts/scripts/test_r2plus1d_kinetics.sh
```

This script will evaluate the model using uniform sampling 10 clips per a testing video. In each clip, it uses only the center crop.

## Large Models

For some large model, such as ir-CSN-152 with longer and larger input i.e. 32x224x224, GPU memory cannot fit 10 clips into a single forward pass. Not counting some functionalities such as convolutional prediction and/or using multiple crops per clips (e.g. 3 crops per clip). If both multiple crop and convolutional prediction are used, it requires to evaluate upto 30 crops per videos and each can have larger input such as 32x256x256 (due to convolutional prediction).

We provide an example of using ```test_net_large.py``` to evaluate one large model (ir-CSN-152 with 32x224x224) with 30 crops testing in ```scripts/test_irCSN_152_kinetics.sh```. The content of the script looks like:

```
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
```

Again, here we assume you have your lmdb database of Kinetics-400 validation as well as download our ir-CSN model. You can run the testing by simply launch:

```
sh scripts/test_irCSN_152_kinetics.sh
```

That will output results as below:

```
Test accuracy: clip: 0.727954269527, top 1: 0.825677863213, top5: 0.953106029947
FLOPs: 96.715669504, params: 29.551744, inters: 11370.96928
```
