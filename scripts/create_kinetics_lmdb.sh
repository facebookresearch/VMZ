# create lmdb database for training and testing purpose
python data/create_video_db.py \
--list_file=data/list/kinetics/kinetics_val_full.csv \
--output_file=/data/users/trandu/datasets/kinetics_val

python data/create_video_db.py \
--list_file=data/list/kinetics/kinetics_train_full.csv \
--output_file=/data/users/trandu/datasets/kinetics_train
