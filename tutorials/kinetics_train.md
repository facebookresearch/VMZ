# Tutorial 1: Training Kinetics from scratch

This tutorial will help you, step-by-step, how to train a video model from scratch on Kinetics using R2Plus1D codebase.

## Preparing data

* Download the [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) dataset by following the steps provided [here](https://github.com/activitynet/ActivityNet/blob/master/Crawler/Kinetics/README.md).


* You then need to pre-process the videos into clips (of 10 seconds). The Kinetics dataset is provided with temporal segments for each video, thus you can use some tool like ffmpeg to extract the clips from the download videos. For examples:

```
ffmpeg -y -ss 00:01:47.00 -t 00:00:10.00 -i kinetics/raw_video/train/-7kbO0v4hag.mp4 -vf scale=320:240 kinetics/clip/train/-7kbO0v4hag.mp4
```

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

You can optionaly train other models, such as R2D, f-R2D, R3D, MCx, rMCx with different depth. For model names and depth, please check the available supporting model names and depth in `model_builder.py`.
Training this model may take a few days with 8 M40 GPUs. If you do not want to wait, you can download some of our pre-trained models [here](models.md).

## Testing models

You can test your model using `test_net.py` by simply running:

```
sh scripts/test_r2plus1d_kinetics.sh
```

Note that there are two model formats are supported: *pickle* and *minidb*. The model we provided in the model zoo are in pickle format. If you train model yourself, `train_net.py` will save model in minidb format. You can still use `test_net.py` to test your model with addtional argument `--db_type=minidb`. You can also convert your model to pickle format using `minidb_to_pickle.py`.

## Dense prediction
The current implementation of `test_net.py` is to uniformly sample k clips across videos. These clips are then passed into the network for inference and aggregate them to make video-level prediction. Normally, 10 clips are sampled and evaluated per a video. However, for some large models our GPU does not have enough memory to handle a mini-batch of 10. For example, with R(2+1)D with 34 layers using 32-frame clips, M40 memory is enough for only a mini-batch of 4 examples (clips). Thus, for larger models (deep models applied on longer input clips), we use `extract_features.py` tool to extract predictions (e.g. softmax), then we aggregate these predictions to evaluate video-level accuracy of our model. Please check out [dense prediction](dense_prediction.md) for examples how to do dense prediction with R2Plus1D.
