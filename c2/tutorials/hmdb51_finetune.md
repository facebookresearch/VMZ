# Tutorial 2: Finetune R(2+1)D on HMDB51

This tutorial will help you step-by-step, how to fine-tune our R(2+1)D model on HMDB51.

## Preparing data
* Download [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Prepare a list file, which can be downloaded [here](https://www.dropbox.com/s/fyz9fec72v7gbxj/list.tar.gz).
* Create lmdb database for training/finetuning, simply run the following script:

```
sh scripts/create_hmdb51_lmdb.sh
```

## Fine-tuning

To fine-tune R(2+1)D on HMDB51, simply run:

```
sh scripts/finetune_hmdb51.sh
```
For a reference, the fine-tuning R(2+1)D-34 model on HMDB51 will give a clip accuracy of 66.1% on split1. If you do not have time to fine-tune, [here](https://www.dropbox.com/s/f5iby7th62fki2t/r2plus1d_d34_l32_ft_hmdb51_epoch8.mdl?dl=0) we provide you the fine-tuned model resulted from the above script. We also provide an example of fine-tuning on UCF101 in `scripts/finetune_ucf101.sh`.

## Evaluating the fine-tuned model.
You then can use [dense prediction](dense_prediction.md) to evaluate your fine-tuned model.
