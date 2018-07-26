# create lmdb database for training and testing purpose
python data/create_video_db.py \
--list_file=/home/joannahsu/local/R2Plus1D/process_data/kinetics/kinetics_val_full.csv \
--output_file=/data/users/joannahsu/datasets/kinetics_val_list \

python data/create_video_db.py \
--list_file=/home/joannahsu/local/R2Plus1D/process_data/kinetics/kinetics_train_full.csv \
--output_file=/data/users/joannahsu/datasets/kinetics_train_list \
--num_epochs=100
