# Feature extraction

This tutorial will help you, step-by-step, how to extract features using our pre-trained models.

## Preparing data

In this example, we assume that you would like to extract features for the following two videos:

```
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi
/your_path/kiss/Moviekissmontage_kiss_u_cm_np2_fr_goo_5.avi
```

You can create a list file as below. Let us name this file `my_list.csv`.

```
org_video,label,start_frm,video_id
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,0,1
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,8,2
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,16,3
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,24,4
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,32,5
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,48,6
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,56,7
/your_path/climb/rope_climb_Olivia_Age_4_climb_f_cm_np1_ba_med_1.avi,5,64,8
/your_path/kiss/Moviekissmontage_kiss_u_cm_np2_fr_goo_5.avi,4,0,9
/your_path/kiss/Moviekissmontage_kiss_u_cm_np2_fr_goo_5.avi,4,8,10
/your_path/kiss/Moviekissmontage_kiss_u_cm_np2_fr_goo_5.avi,4,16,11
/your_path/kiss/Moviekissmontage_kiss_u_cm_np2_fr_goo_5.avi,4,24,12
/your_path/kiss/Moviekissmontage_kiss_u_cm_np2_fr_goo_5.avi,4,32,13
/your_path/kiss/Moviekissmontage_kiss_u_cm_np2_fr_goo_5.avi,4,48,14
```

In this example, we will extract features for the two above-mentioned videos using their clips with a striding of 8 frames. It is OK if you do not provide labels for them, e.g. just filling that column by all zeros. The video_id column is used to keep track of your features, e.g. you want to know which feature belongs to which clip. You then create an lmdb database of this list by:

```
python data/create_video_db.py \
--list_file=my_list.csv \
--output_file=my_lmdb_data \
--use_list=1 --use_video_id=1 --use_start_frame=1
```

After creating the lmdb database, you can then extract features by:

```
python tools/extract_features.py \
--test_data=my_lmdb_data \
--model_name=r2plus1d --model_depth=34 --clip_length_rgb=32 \
--gpus=0,1 \
--batch_size=4 \
--load_model_path=/mnt/homedir/trandu/video_models/kinetics/l32/r2.5d_d34_l32_ft_sports1m.pkl \
--output_path=my_features.pkl \
--features=softmax,final_avg,video_id \
--sanity_check=0 --get_video_id=1 --use_local_file=1 --num_labels=400
```

You features are then extracted and saved into a pickle file, `my_features.pkl`. In this example, we extract `softmax` and `final_avg` features, but in general you can extract other features as well.
