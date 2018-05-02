python tools/test_net.py \
--test_data=/data/users/trandu/datasets/kinetics_val/ \
--model_name=r2plus1d --model_depth=18 --gpus=0,1,2,3,4,5,6,7 \
--clip_length_rgb=8 --num_labels=400 --batch_size=3 \
--load_model_path=/mnt/homedir/trandu/video_models/kinetics/l8/r2.5d_d18_l8.pkl
