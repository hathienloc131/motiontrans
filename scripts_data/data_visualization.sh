# bash scripts_data/data_visualization.sh

data_path="data/zarr_data/zarr_data_human/human_test++.zarr"
episode_idx=4
downsample_factor=1

python -m scripts_data.entry.data_visualization \
  --data_path "${data_path}" \
  --episode_idx ${episode_idx} \
  --downsample_factor ${downsample_factor} \
  --disable_pointclouds
  # --rgb_only
  --rgb_only --project_hands            # ZED: RGB + projected hand overlay (no pointcloud window)
  # --disable_pointclouds --rgb_only --project_hands   # RealSense: same, no pointcloud available