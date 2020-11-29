# Model Zoo

The evaluation metrics are top-1 and top-5 video-level accuracy on Kinetics-400 validation set. The ```Finetuned model``` column provides models either trained from scratch or fine-tuned on Kinetics. The ```Pre-trained model``` column provides models pre-trained on Sports1M or IG-65M **without** further finetuning on Kinetics.

### C3D
| Input size | Pre-trained dataset | Pre-trained model  | Video@1 | Video@5 | Finetuned model | GFLOPs | params(M) |
| -----------| ----------- |-----   | ------- | ------- | -------- | -----   | --------- |
| 16x112x112 | None        | None   | 66.6 | 86.7 | [link](https://www.dropbox.com/s/z6w799mqet65mkq/c3d_kinetics_from_scratch_f129282105.pkl?dl=0) | 38.5 | 64.9 |
| 16x112x112 | Sports1M    | [link](https://www.dropbox.com/s/s2ooup80wal8o5d/c3d_sports1m_from_scratch_f129642950.pkl?dl=0)   | 67.4 | 87.2 | [link](https://www.dropbox.com/s/jadubfxhh5hdyp5/c3d_ft_kinetics_from_sports1m_f131638306.pkl?dl=0) | 38.5 | 64.9 |

### R(2+1)D-34
| Input size | Pre-trained dataset | Pre-trained model  | Video@1 | Video@5 | Finetuned model | GFLOPs | params(M) |
| -----------| ----------- |-----   | ------- | ------- | -------- | -----   | --------- |
| 8x112x122  |  IG-65M     | [link](https://www.dropbox.com/s/6xwyu1az6oy4ts7/r2plus1d_34_clip8_ig65m_from_scratch_f79708462.pkl?dl=0)   | 74.9    | 91.8    | [link](https://www.dropbox.com/s/p81twy88kwrrcop/r2plus1d_34_clip8_ft_kinetics_from_ig65m_%20f128022400.pkl?dl=0)     | 49.8    | 63.6      |
| 32x112x122 |  IG-65M     | [link](https://www.dropbox.com/s/eimo232tqw8mwi9/r2plus1d_34_clip32_ig65m_from_scratch_f102649996.pkl?dl=0)   | 80.0    | 94.2    | [link](https://www.dropbox.com/s/z41ff7vs0bzf6b8/r2plus1d_34_clip32_ft_kinetics_from_ig65m_%20f106169681.pkl?dl=0)     | 199.0   | 63.6      |

### R(2+1)D-152

| Input size | Pre-trained dataset | Pre-trained model  | Video@1 | Video@5 | Finetuned model | GFLOPs | params(M) |
| -----------| -------|----- | ------- | ------- | -------- | -----   | --------- |
| 32x112x122 | None   |    None  | 77.3     | 92.5     | [link](https://www.dropbox.com/s/9770y063u3z5bmb/r2plus1d_152_kinetics_from_scratch_f127858918.pkl?dl=0)     | 329.1    | 118.0 |
| 32x112x122 | Sports1M | [link](https://www.dropbox.com/s/w5cdqeyqukuaqt7/r2plus1d_152_sports1m_from_scratch_f127111290.pkl?dl=0)   | 79.5     | 94.0      | [link](https://www.dropbox.com/s/twvcpe30rxuaf45/r2plus1d_152_ft_kinetics_from_sports1m_f128957437.pkl?dl=0)   | 329.1 | 118.0 |
| 32x112x112 | IG-65M | [link](https://www.dropbox.com/s/oqdg176p7nqc84v/r2plus1d_152_ig65m_from_scratch_f106380637.pkl?dl=0) | 81.6    | 95.3    | [link](https://www.dropbox.com/s/tmxuae8ubo5gipy/r2plus1d_152_ft_kinetics_from_ig65m_f107107466.pkl?dl=0)      | 329.1 | 118.0 |

### ir-CSN-50
| Input size | Pre-trained dataset | Pre-trained model  | Video@1 | Video@5 | Finetuned model | GFLOPs | params(M) |
|------------| -----       | -------| ------- | ------- | -------   | -------| -----     |
| 32x224x224 | None        | None   | TBD    | TBD    | link      | TBD   | TBD      |
| 32x224x224  | IG-65M | [link](https://www.dropbox.com/s/1cqzndm1zobxvhh/irCSN_50_pretrained_ig65m_f231921860.pkl?dl=0)      | TBD       | TBD       | [link](https://www.dropbox.com/s/b6o2mvoqrvupatt/irCSN_50_ft_kinetics_from_ig65m_f233743920.pkl?dl=0)      | TBD | TBD |


### ir-CSN-152
| Input size | Pre-trained dataset | Pre-trained model  | Video@1 | Video@5 | Finetuned model | GFLOPs | params(M) |
|------------| -----       | -------| ------- | ------- | -------   | -------| -----     |
| 32x224x224 | None        | None   | 76.5    | 92.1    | [link](https://www.dropbox.com/s/46gcm7up60ssx5c/irCSN_152_kinetics_from_scratch_f98268019.pkl?dl=0)      | 96.7   | 29.6      |
| 32x224x224 | Sports1M    | [link](https://www.dropbox.com/s/woh99y2hll1mlqv/irCSN_152_sports1m_from_scratch_f99918785.pkl?dl=0) | 78.2    | 93.0    | [link](https://www.dropbox.com/s/zuoj1aqouh6bo6k/irCSN_152_ft_kinetics_from_sports1m_f101599884.pkl?dl=0) | 96.7 | 29.6 |
| 32x224x224  | IG-65M | [link](https://www.dropbox.com/s/r0kppq7ox6c57no/irCSN_152_ig65m_from_scratch_f125286141.pkl?dl=0)      | 82.6       | 95.3       | [link](https://www.dropbox.com/s/gmd8r87l3wmkn3h/irCSN_152_ft_kinetics_from_ig65m_f126851907.pkl?dl=0)      | 96.7 | 29.6 |

### ip-CSN-152
| Input size | Pre-trained dataset | Pre-trained model  | Video@1 | Video@5 | Finetuned model | GFLOPs | params(M) |
| -----------| ------------ | -- | ------- | ------- | -------- | ----- | --------- |
| 32x224x224 | None | None | 77.8    | 92.8    | [link](https://www.dropbox.com/s/3fihu6ti60047mu/ipCSN_152_kinetics_from_scratch_f129594342.pkl?dl=0)   | 108.8 | 32.8 |
| 32x224x224 | Sports1M | [link](https://www.dropbox.com/s/70di7o7qz6gjq6x/ipCSN_152_sports1m_from_scratch_f111018543.pkl?dl=0) | 78.8    | 93.5    | [link](https://www.dropbox.com/s/ir7cr0hda36knux/ipCSN_152_ft_kinetics_from_sports1m_f111279053.pkl?dl=0)      | 108.8 | 32.8 |
| 32x224x224 | IG-65M | [link](https://www.dropbox.com/s/1ryvx8k7kzs8od6/ipCSN_152_ig65m_from_scratch_f130601052.pkl?dl=0) | 82.5    | 95.3    | [link](https://www.dropbox.com/s/zpp3p0vn2i7bibl/ipCSN_152_ft_kinetics_from_ig65m_f133090949.pkl?dl=0) | 108.8 | 32.8 |

### Gradient-Blending: Audio-Visual Model: ip-CSN-152
| Input size | Pre-trained dataset | Pre-trained model  | Video@1 | Video@5 | Finetuned model | GFLOPs |
| -----------| ------------ | -- | ------- | ------- | -------- | ----- |
| 32x224x224 | None | None | 79.1    | 93.9    | [link](https://www.dropbox.com/s/13d5jgq65nd5sn6/g_b_ip_csn_152_kinetics.pkl?dl=0)      | 110.1 | 
| 32x224x224 | Sports1M+AudioSet | [link](https://www.dropbox.com/s/u4dgz0z09aaim9x/sports1m_audioset_ip_csn_152.pkl?dl=0) | 80.4    | 94.8    | [link](https://www.dropbox.com/s/ubcy09t8ghes9m7/g_b_ip_csn_152_sports_audioset_ft_kinetics.pkl?dl=0)      | 110.1 | 
| 32x224x224 | IG-65M+AudioSet | [link](https://www.dropbox.com/s/bgxf62o9yzx7zfy/ig_audioset_ip_csn_152.pkl) | 83.3    | 96.0    | [link](https://www.dropbox.com/s/9ja3rlmr568qemt/g_b_ip_csn_152_ig_audioset_ft_kinetics.pkl?dl=0)      | 110.1 |

We additionally provide two audio-visual models:
- R(2+1)D-101 with 16x224x224 as input trained from scratch on AudioSet: [link](https://www.dropbox.com/s/7e36vogqyea152p/g_b_r2_plus_1d_101_audioset.pkl?dl=0)
- ip-CSN-152 pretrained on IG-65M and ImageNet, which is used for our entry in EPIC-Kitchen Action Recognition Challenge: [link](https://www.dropbox.com/s/63oy0vn4gkyc3xq/ig_imagenet_ip_csn_152.pkl?dl=0)
