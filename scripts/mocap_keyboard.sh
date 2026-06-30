# bash scripts/mocap_keyboard.sh

output_dir="data/raw_data/raw_data_human/human_test"
mp4_crop_w=1280
mp4_crop_h=720
mp4_downsample_ratio=2
camera_exposure=50

python human_data_collection_keyboard.py \
    --output_dir ${output_dir} \
    --mp4_crop_w ${mp4_crop_w} \
    --mp4_crop_h ${mp4_crop_h} \
    --mp4_downsample_ratio ${mp4_downsample_ratio} \
    --camera_exposure ${camera_exposure} \
    "$@"
