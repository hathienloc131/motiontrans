# bash scripts_data/streamlit_visualize.sh

data_dir="${1:-$HOME/Documents/data/raw_data/raw_data_human}"

streamlit run scripts_data/entry/streamlit_hand_visualizer.py -- --data_dir "${data_dir}"
