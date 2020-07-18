# Gradient-Blending: Additional notes to enable joint audio-visual training with gradient-blending.

This tutorial provides additional information on how to enable audio-visual learning with gradient-blending.

## Enable joint decoding of audio and visual
Current Caffe2 package/ binary does not support decoding audio from video; to solve this problem, we provide a suite of audio decoding functionalities in the ```ops``` folder. Please copy the files to ```caffe2/video/``` and re-build the Caffe2 package. Note that no overriding of existing video decoding operators is required. 

The ```AVInput``` operator provides temporally aligned video RGB clips and audio LogMels. Audio dimension by default is given by 100(time)x40(frequency). Alignment by default is perfect alignment with visual (both start and end of visual clip). Other options can be tuned inside ```av_input_op.h``` before compilation. 

## Audio decoding option in train/ test/ feature extraction
Audio decoding can be activated by changing parameter ```input_type``` in the workflows. Setting it to ```3``` gives decoding of both audio and visual. 

## Training with Gradient-Blending
Gradient-Blending requires one additional classifier (decoder) for audio stream and one additional for visual stream. It takes the loss of the three (audio, visual, jointAV) and optimally weights their loss. To set the weight, use parameter ```audio_weight```, ```visual_weight``` and ```av_weight```.

We here provide a few example weights calculated by Gradient-Blending (for more details, see the original [paper](https://arxiv.org/abs/1905.12681)):

| Dataset     | Pre-Train   | Model   | Depth | Audio Weight | Visual Weight | AV Weight |
| ----------- | ----------- | ------- | ----- | -----        | -----         | -----|
| Kinetics400 | N/A | R3D | 50 | 0.014 | 0.630 | 0.356 |
| Kinetics400 | None | ip-CSN | 152 | 0.009 | 0.485 | 0.506 |
| Kinetics400 | IG-65M | ip-CSN | 152 | 0.070 | 0.485 | 0.445 |
| AudioSet | None | R(2+1)D | 101 | 0.239 | 0.384 | 0.377 |
| EPIC-Kitchen Noun | IG-65M | ip-CSN | 152 | 0.175 | 0.460 | 0.364 |
| EPIC-Kitchen Verb | IG-65M | ip-CSN | 152 | 0.524 | 0.247 | 0.229 |
