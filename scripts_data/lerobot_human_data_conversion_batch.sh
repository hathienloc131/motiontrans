# bash scripts_data/lerobot_human_data_conversion_batch.sh

input_dir="/Users/lochathien/Documents/data/raw_data/raw_data_human"
num_use_source=-1                # how many sources to use per task, -1 means all
n_demos=""                       # max demos to convert per task, empty = full dataset
output_dir="data/human_stack_cup"
calib_quest2camera_file="camera_params/quest_realsense/calib_result_quest2camera.npy"
adapt_config_file="scripts_data/human_data_adapt.json"
default_speed_downsample_ratio=2.25
default_hand_shrink_coef=1.0     # how much to shrink the grasping hand, 1.0 means no shrinking
gripper_type="gripper_f285"      # inspire_hand | gripper_f285
f285_close_ramp_k=4             # frames to ramp 0→0.8 before each close event
mode="o"                         # o: origin (recommended for LeRobot), s: stereo, d: depth, p: pointclouds, a: all
network_delay_checking=0.5
num_points_final=1024            # if save pointclouds, how many points to sample
points_max_distance_final=1.0    # if save pointclouds, the max distance to keep points
repo_id="human_demo/task"        # LeRobot repo_id prefix; per-task instruction suffix appended automatically
fps=30                           # dataset fps (should match Quest recording rate / downsample_ratio)
lerobot_src_path="/Users/lochathien/Documents/Code/vr_lfd/src"

# Use the lerobot conda env Python (has lerobot + motiontrans deps).
# One-time setup — install missing motiontrans deps into lerobot env:
#   conda run -n lerobot pip install pin==3.3.1 dex_retargeting==0.4.6 fpsample imageio imageio-ffmpeg
PYTHON="/opt/homebrew/Caskroom/miniconda/base/envs/lerobot/bin/python"

${PYTHON} -m scripts_data.entry.lerobot_human_data_conversion_batch \
  --input_dir ${input_dir} \
  --output ${output_dir} \
  --calib_quest2camera_file ${calib_quest2camera_file} \
  --adapt_config_file ${adapt_config_file} \
  --default_speed_downsample_ratio ${default_speed_downsample_ratio} \
  --default_hand_shrink_coef ${default_hand_shrink_coef} \
  --gripper_type ${gripper_type} \
  --f285_close_ramp_k ${f285_close_ramp_k} \
  --mode ${mode} \
  --resolution_resize 1280x720 \
  --resolution_crop 960x600 \
  --resolution_image_final 960x600 \
  --num_use_source ${num_use_source} \
  ${n_demos:+--n_demos ${n_demos}} \
  --num_points_final ${num_points_final} \
  --points_max_distance_final ${points_max_distance_final} \
  --network_delay_checking ${network_delay_checking} \
  --repo_id ${repo_id} \
  --fps ${fps} \
  --lerobot_src_path ${lerobot_src_path}
