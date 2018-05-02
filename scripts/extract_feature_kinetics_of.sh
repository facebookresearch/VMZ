# extract optical flow model predictions
# we do feature extraction for 10 splits
for ((i=1;i<=10;i++)); \
do \
python tools/extract_features.py \
--test_data=/data/users/trandu/datasets/kinetics_feature_extraction/kinetics_val_video_id_dense_l32_$i \
--model_name=r2plus1d --model_depth=34 \
--clip_length_rgb=33 --sampling_rate_rgb=1 \
--clip_length_of=32 --sampling_rate_of=1 \
--flow_data_type=0 --frame_gap_of=1 --do_flow_aggregation=1 \
--num_channels=2 --input_type=1 \
--gpus=0,1,2,3,4,5,6,7 \
--batch_size=4 \
--load_model_path=/mnt/homedir/trandu/video_models/kinetics/l32/r2.5d_d34_l32_ft_sports1m_optical_flow.pkl \
--output_path=/data/users/trandu/datasets/kinetics_features/of_ft_50030948/kinetics_val_video_id_dense_l32_$i.pkl \
--features=softmax,label,video_id \
--sanity_check=1 --get_video_id=1 --use_local_file=1 --num_labels=400; \
done
