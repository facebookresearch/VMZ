# Dense prediction

**THIS FUNCTIONALITY IS DEPRECATED, PLEASE USE LARGE MODEL TESTING TOOL**.

This tutorial will help you, step-by-step, how to do video dense prediction using VMZ.

## Example 1: dense prediction on HMDB51 split1

### Prepare you data
After you download the video and the list, to create an lmdb database for feature extraction, you can simply run:
```
sh scripts/create_hmdb51_lmdb_feature_extraction.sh
```

This script will create an lmdb database that contains only paths to local files, together their video_id for each video and also starting frame for each clip we want to extract feature. The temporal striding can be very dense, e.g. stride=1, or can be sparser, you can adjust this parameter to serve your purpose. For a reference, how this clip-sampling factor affects the video-level accuracy, please check our [paper](https://128.84.21.199/abs/1711.11248) Figure 5.b.

### Extracting predictions
Simply run:

```
sh scripts/extract_feature_hmdb51.sh
```

### Dense prediction
After you extract predictions for all clips, you can evaluate video-level accuracy by:

```
python tools/dense_prediction_aggregation.py \
--input_dir=/data/users/trandu/datasets/hmdb51_features/ft
```

## Example 2: dense prediction on Kinetics

Similar to HMDB51, you can create an lmdb database of Kinetics (paths to local files) for feature extraction by running:

```
sh scripts/create_kinetics_lmdb_feature_extraction.sh
```

then, extracting features with:
```
sh scripts/extract_feature_kinetics.sh
```
Finally, you can aggregate these predictions to get the video-level accuracy by:

```
python tools/dense_prediction_aggregation.py \
--input_dir=/data/users/trandu/datasets/kinetics_features/rgb_ft_45450620
```

If you want to extract clip predictions for model trained on optical flows, please check `extract_feature_kinetics_of.sh` for an example. You then need to run `sh scripts/fuse_prediction.py` to fuse the predictions from the RGB- and optical flow models.
