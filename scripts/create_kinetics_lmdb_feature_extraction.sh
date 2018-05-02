# create lmdb database (of video filenames, start_frame, videos) for feature extraction and dense prediction
# because of the large size of predictions to make, we split into 10 splits
for ((i=1;i<=10;i++)); \
do python data/create_video_db.py \
--list_file=data/list/kinetics/kinetics_val_full_video_id_dense_l32_$i.csv \
--output_file=/data/users/trandu/datasets/kinetics_feature_extraction/kinetics_val_video_id_dense_l32_$i \
--use_list=1 --use_video_id=1 --use_start_frame=1; \
done
