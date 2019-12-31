# Training Kinetics from scratch

This tutorial will help you, step-by-step, how to train a video model from scratch on Kinetics using VMZ codebase.

## Preparing data

* Download the [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset by following the steps provided [here](https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/README.md).


* You then need to pre-process the videos into clips (of 10 seconds). The Kinetics dataset is provided with temporal segments for each video, thus you can use some tool like ffmpeg to extract the clips from the download videos. For examples:

```
ffmpeg -y -ss 0:06:57 -t 00:00:10 -i raw_kinetics_videos/val/0wR5jVB-WPk.mp4 -r 30 -q:v 1 -vf scale=-2:320 trimmed_videos/val/0wR5jVB-WPk.mp4
ffmpeg -y -ss 0:00:15 -t 00:00:10 -i raw_kinetics_videos/val/InSYAA9cJOs.mp4 -r 30 -q:v 1 -vf scale=480:-2 trimmed_videos/val/InSYAA9cJOs.mp4
```

Note that we enforce the frame rate to 30 fps and scale the the videos to 320 pixel per shorter edge while keep the video original aspect ratio.

* You also need to prepare the list files for train and validation splits. You can download the list files [here](https://www.dropbox.com/s/fyz9fec72v7gbxj/list.tar.gz). You may want to adjust them (e.g. removing some rows if some videos are missing because of the expried urls).


* To create lmdb database for training, simple run the following scripts:
```
sh scripts/create_kinetics_lmdb.sh
```

## Training Kinetics

To run an example of training R(2+1)D on 8-frame clips on Kinetics, simply run:

```
sh scripts/train_r2plus1d_kinetics.sh
```

You can optionally train other models, such as ir-CSN, ip-CSN with different depth. For model names and depth, please check the available supporting model names and depth in `model_builder.py`.
Training this model may take a few days with 8 P100 GPUs. If you do not want to wait, you can download some of our pre-trained models [here](models.md).

## Testing models

You can test your model using `test_net.py` by simply running:

```
sh scripts/test_r2plus1d_kinetics.sh
```

Note that there are two model formats are supported: *pickle* and *minidb*. The model we provided in the model zoo are in pickle format. If you train model yourself, `train_net.py` will save model in minidb format. You can still use `test_net.py` to test your model with additional argument `--db_type=minidb`. You can also convert your model to pickle format using `minidb_to_pickle.py`.

## Testing large models

For deeper models (e.g. 101/152 layers) with longer clip input (e.g. 32-frame clips instead of 8), ```test_net.py``` tool may not work due to out of memory. We provide ```test_net_large.py``` to for such a usecase. This tool is also supported convolutional testing with multiple spatial crops which is used in non-local network paper and [codebase](https://github.com/facebookresearch/video-nonlocal-net).
