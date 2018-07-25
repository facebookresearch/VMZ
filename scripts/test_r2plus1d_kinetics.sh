python tools/test_net.py \
--test_data=/data/users/joannahsu/datasets/kinetics_val_list/ \
--model_name=r2plus1d --model_depth=18 --num_gpus=2 \
--clip_length_rgb=8 --num_labels=400 --batch_size=1 \
--load_model_path=/mnt/homedir/trandu/video_models/kinetics/l8/r2.5d_d18_l8.pkl
