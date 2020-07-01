# VMZ: Model Zoo for Video Modeling

VMZ is a Caffe2 codebase for video modeling developed by the Computer Vision team at Facebook AI. The aim of this codebase is to help other researchers and industry practitioners:
+ reproduce some of our research results and 
+ leverage our very strong pre-trained models. 

Currently, this codebase supports the following models:
+ R(2+1)D, MCx models [[1]](https://research.fb.com/wp-content/uploads/2018/04/a-closer-look-at-spatiotemporal-convolutions-for-action-recognition.pdf).
+ CSN models [[2]](https://arxiv.org/pdf/1904.02811.pdf).
+ R(2+1)D and CSN models pre-trained on large-scale (65 million!) weakly-supervised public Instagram videos (**IG-65M**) [[3]](https://research.fb.com/wp-content/uploads/2019/05/Large-scale-weakly-supervised-pre-training-for-video-action-recognition.pdf).

## Main Models

We provide our latest video models including R(2+1)D, ir-CSN, ip-CSN (all with 152 layers) which are pre-trained on Sports-1M or **IG-65M**, then fine-tuned on Kinetics-400. Both pre-trained and fine-tuned models are provided in the table below. We hope these models will serve as valuable baselines and feature extractors for the related video modeling tasks such as action detection, video captioning, and video Q&A.

For your convenience, all models are provided in torch hub. Pretrainings available with each respective model definition. Most models
allow  following pre-trainings which correspond to their equivalents of caffe2 pretrainings:
```
avail_pretrainings = [
    "ig65m_32frms",
    "ig_ft_kinetics_32frms",
    "sports1m_32frms",
    "sports1m_ft_kinetics_32frms",
]
```

This allows the models to be loaded using their respective pre-trainings, using torchhub. If you want to use the model direcly, you can simply import it from `vmz` package.

```
from vmz.models import ir_csn_152
model = ir_csn_152(pretraining="ig65m_32frms")
```


## References
1. D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun and M. Paluri. **A Closer Look at Spatiotemporal Convolutions for Action Recognition.** CVPR 2018.
2. D. Tran, H. Wang, L. Torresani and M. Feiszli. **Video Classification with Channel-Separated Convolutional Networks.** ICCV 2019.
3. D. Ghadiyaram, M. Feiszli, D. Tran, X. Yan, H. Wang and D. Mahajan, **Large-scale weakly-supervised pre-training for video action recognition.** CVPR 2019.


## License
VMZ is Apache 2.0 licensed, as found in the LICENSE file.
