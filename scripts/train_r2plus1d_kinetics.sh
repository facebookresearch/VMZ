python tools/train_net.py \
--train_data=/data/users/trandu/datasets/kinetics_train \
--test_data=/data/users/trandu/datasets/kinetics_val \
--model_name=r2plus1d --model_depth=18 \
--clip_length_rgb=8 --batch_size=32 \
--gpus=0,1,2,3,4,5,6,7 --base_learning_rate=0.01 \
--epoch_size=1000000 --num_epochs=45 --step_epoch=10 \
--weight_decay=0.0001 --num_labels=400
