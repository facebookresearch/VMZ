# VMZ: Model Zoo for Video Modeling

VMZ is a Caffe2 and Pytorch codebase for video modeling developed by the Computer Vision team at Facebook AI. The aim of this codebase is to help other researchers and industry practitioners:
+ reproduce some of our research results and 
+ leverage our very strong pre-trained models. 


Currently, this codebase supports the following models:
+ R(2+1)D, MCx models [[1]](https://research.fb.com/wp-content/uploads/2018/04/a-closer-look-at-spatiotemporal-convolutions-for-action-recognition.pdf).
+ CSN models [[2]](https://arxiv.org/pdf/1904.02811.pdf) (**note:pytorch implementation is buggy**).
+ R(2+1)D and CSN models pre-trained on large-scale (65 million!) weakly-supervised public Instagram videos (**IG-65M**) [[3]](https://research.fb.com/wp-content/uploads/2019/05/Large-scale-weakly-supervised-pre-training-for-video-action-recognition.pdf).
+ Gradient-Blending for audio-visual modeling [[4]](https://arxiv.org/pdf/1905.12681.pdf) (Caffe2 Only)

## References
1. D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun and M. Paluri. **A Closer Look at Spatiotemporal Convolutions for Action Recognition.** CVPR 2018.
2. D. Tran, H. Wang, L. Torresani and M. Feiszli. **Video Classification with Channel-Separated Convolutional Networks.** ICCV 2019.
3. D. Ghadiyaram, M. Feiszli, D. Tran, X. Yan, H. Wang and D. Mahajan, **Large-scale weakly-supervised pre-training for video action recognition.** CVPR 2019.
4. W. Wang, D. Tran, M. Feiszli, **What Makes Training Multi-Modal Classification Networks Hard?** CVPR 2020.


## Suporting Team
This codebase is actively supported by Facebook AI computer vision: @CHJoanna, @weiyaowang, @hengcv, @deeptigp, @dutran, and community researchers @bjuncek (Quansight, Oxford VGG).


