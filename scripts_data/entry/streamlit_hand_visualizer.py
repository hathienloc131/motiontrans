"""
Streamlit visualizer for raw human VR hand pose data.

Shows RGB video frames alongside a 3D hand pose visualization
with XYZ rotation axes for each joint.

Usage (from repo root):
    streamlit run scripts_data/entry/streamlit_hand_visualizer.py
"""
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def euler_pose_to_mat(pose: np.ndarray) -> np.ndarray:
    """(T,6) or (6,) euler pose [x,y,z,rx,ry,rz] → (T,4,4) or (4,4) SE(3)."""
    position = pose[..., :3]
    rotation = Rotation.from_euler("xyz", pose[..., 3:])
    shape = pose.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rotation.as_matrix()
    mat[..., :3, 3] = position
    mat[..., 3, 3] = 1.0
    return mat


def fast_mat_inv(mat: np.ndarray) -> np.ndarray:
    """Fast SE(3) inverse (assumes rigid/orthonormal rotation)."""
    ret = np.eye(4, dtype=np.float64)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


# VR coordinate (y-forward, x-right, z-up) → camera convention (x-right, y-down, z-forward)
VR2CAM = np.array([
    [1, 0,  0, 0],
    [0, 0, -1, 0],   # cam y = -vr_z  (z-up → -y → y-down)
    [0, 1,  0, 0],   # cam z =  vr_y  (y-forward → z-depth)
    [0, 0,  0, 1],
], dtype=np.float64)


def world_to_cam(head_mat_vr_t: np.ndarray) -> np.ndarray:
    """World-to-camera transform at timestep t (no calibration — uses head as camera)."""
    return VR2CAM @ fast_mat_inv(head_mat_vr_t)


# ---------------------------------------------------------------------------
# Hand skeleton topology
# ---------------------------------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (0, 6), (0, 11), (0, 16), (0, 21),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10),
    (11, 12), (12, 13), (13, 14), (14, 15),
    (16, 17), (17, 18), (18, 19), (19, 20),
    (21, 22), (22, 23), (23, 24), (24, 25),
]

JOINT_NAMES = (
    ["wrist"]
    + [f"thumb_{s}"  for s in ["cmc", "mcp", "ip",  "tip", "aux"]]
    + [f"index_{s}"  for s in ["mcp", "pip", "dip", "tip", "aux"]]
    + [f"middle_{s}" for s in ["mcp", "pip", "dip", "tip", "aux"]]
    + [f"ring_{s}"   for s in ["mcp", "pip", "dip", "tip", "aux"]]
    + [f"pinky_{s}"  for s in ["mcp", "pip", "dip", "tip", "aux"]]
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

ROOT_DATA = Path(__file__).resolve().parents[2] / "data" / "raw_data" / "raw_data_human"


@st.cache_data(show_spinner=False)
def load_episode(episode_dir: str):
    ep_dir = Path(episode_dir)
    with open(ep_dir / "episode.pkl", "rb") as f:
        ep = pickle.load(f)
    K = np.load(ep_dir / "camera_intrinsic.npy").astype(np.float64)

    cap = cv2.VideoCapture(str(ep_dir / "rgb.mp4"))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    return ep, np.array(frames), K


@st.cache_data(show_spinner=False)
def precompute_joints(episode_dir: str):
    """
    Pre-compute per-frame joint SE(3) matrices in two spaces:
      head_rel: relative to T=0 head (for 3-D visualisation)
      cam_t:    in camera frame at time t  (for 2-D projection)
    """
    ep_dir = Path(episode_dir)
    with open(ep_dir / "episode.pkl", "rb") as f:
        ep = pickle.load(f)

    T = len(ep["timestamp"])
    n_joints = ep["left_hand_mat"].shape[1] // 6

    head_mat = euler_pose_to_mat(ep["head_pose_mat"].astype(np.float64))  # (T,4,4) in VR world

    # Head-relative reference frame (centered on head at T=0)
    inv_head0 = fast_mat_inv(head_mat[0])  # brings T=0 head to origin

    left_head_rel  = np.zeros((T, n_joints, 4, 4))
    right_head_rel = np.zeros((T, n_joints, 4, 4))
    left_cam_t     = np.zeros((T, n_joints, 4, 4))
    right_cam_t    = np.zeros((T, n_joints, 4, 4))

    for t in range(T):
        W2C = world_to_cam(head_mat[t])          # (4,4) world → cam at t
        inv_head_t_from0 = inv_head0              # stays same (relative to T=0)

        for j in range(n_joints):
            sl = slice(j * 6, (j + 1) * 6)
            Lj = euler_pose_to_mat(ep["left_hand_mat"][t,  sl].astype(np.float64))
            Rj = euler_pose_to_mat(ep["right_hand_mat"][t, sl].astype(np.float64))

            left_head_rel[t, j]  = inv_head0 @ Lj
            right_head_rel[t, j] = inv_head0 @ Rj
            left_cam_t[t, j]     = W2C @ Lj
            right_cam_t[t, j]    = W2C @ Rj

    return left_head_rel, right_head_rel, left_cam_t, right_cam_t, n_joints

# ---------------------------------------------------------------------------
# 2-D projection onto RGB frame
# ---------------------------------------------------------------------------

def overlay_joints_on_frame(
    frame_rgb: np.ndarray,
    joints_cam: np.ndarray,   # (N, 4, 4) in camera frame
    K: np.ndarray,            # (3, 3)
    dot_color_bgr: tuple,
    axis_len_px: int = 30,
    axis_joint_idxs: set = frozenset(),
    draw_skeleton: bool = True,
) -> np.ndarray:
    img = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    N = len(joints_cam)

    pos3d = joints_cam[:, :3, 3]       # (N, 3) positions in cam frame
    z = pos3d[:, 2]
    valid = z > 0.01

    def proj(p3):
        if p3[2] <= 0:
            return None
        px = K @ p3 / p3[2]
        u, v = int(round(px[0])), int(round(px[1]))
        if 0 <= u < W and 0 <= v < H:
            return u, v
        return None

    px2d = {}
    for i in range(N):
        if valid[i]:
            pt = proj(pos3d[i])
            if pt:
                px2d[i] = pt

    # Skeleton bones
    if draw_skeleton:
        for (i, j) in HAND_CONNECTIONS:
            if i in px2d and j in px2d:
                cv2.line(img, px2d[i], px2d[j], dot_color_bgr, 1, cv2.LINE_AA)

    # Joints and axes
    for idx, (u, v) in px2d.items():
        cv2.circle(img, (u, v), 4, dot_color_bgr, -1, cv2.LINE_AA)
        if idx in axis_joint_idxs:
            rot = joints_cam[idx, :3, :3]
            p = pos3d[idx]
            z_d = max(p[2], 0.01)
            scale = axis_len_px * z_d / K[0, 0]
            for ai, bgr in enumerate([(0,0,255),(0,255,0),(255,0,0)]):  # X=R,Y=G,Z=B (BGR)
                tip3 = p + rot[:, ai] * scale
                tip = proj(tip3)
                if tip:
                    cv2.arrowedLine(img, (u, v), tip, bgr, 2, tipLength=0.3, line_type=cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# 3-D Plotly figure
# ---------------------------------------------------------------------------

def make_3d_figure(
    left_joints: np.ndarray,   # (N, 4, 4) in head-relative VR frame
    right_joints: np.ndarray,
    axis_joint_idxs: set,
    axis_scale: float = 0.05,
) -> go.Figure:
    fig = go.Figure()

    specs = [
        (right_joints, "Right", "#4488ff", "#2266dd"),
        (left_joints,  "Left",  "#ff4444", "#dd2222"),
    ]

    for joints, label, col_pt, col_bone in specs:
        pos = joints[:, :3, 3]

        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode="markers",
            marker=dict(size=5, color=col_pt, opacity=0.95),
            name=f"{label} joints",
            legendgroup=label,
        ))

        for (i, j) in HAND_CONNECTIONS:
            fig.add_trace(go.Scatter3d(
                x=[pos[i, 0], pos[j, 0]],
                y=[pos[i, 1], pos[j, 1]],
                z=[pos[i, 2], pos[j, 2]],
                mode="lines",
                line=dict(color=col_bone, width=3),
                showlegend=False,
                legendgroup=label,
            ))

        for idx in axis_joint_idxs:
            if idx >= len(joints):
                continue
            p = pos[idx]
            R = joints[idx, :3, :3]
            for ai, (ax_col, ax_name) in enumerate(zip(
                ["red", "green", "blue"], ["X", "Y", "Z"]
            )):
                tip = p + R[:, ai] * axis_scale
                fig.add_trace(go.Scatter3d(
                    x=[p[0], tip[0]], y=[p[1], tip[1]], z=[p[2], tip[2]],
                    mode="lines",
                    line=dict(color=ax_col, width=5),
                    name=f"{label} J{idx} {ax_name}",
                    showlegend=False,
                    legendgroup=label,
                ))

    # Coordinate axes at origin (head position at T=0)
    for length, col, name in [(0.05, "gray", "Head origin")]:
        for ai, (c, n) in enumerate(zip(["red","green","blue"],["X","Y","Z"])):
            tip = np.zeros(3); tip[ai] = length
            fig.add_trace(go.Scatter3d(
                x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
                mode="lines", line=dict(color=c, width=3, dash="dash"),
                name=f"Head {n}", showlegend=False,
            ))

    # VR coordinate frame note: x=right, y=forward, z=up
    fig.update_layout(
        scene=dict(
            xaxis_title="X — right (m)",
            yaxis_title="Y — forward (m)",
            zaxis_title="Z — up (m)",
            aspectmode="data",
            bgcolor="rgb(15, 15, 25)",
            xaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="#333"),
            yaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="#333"),
            zaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="#333"),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgb(15, 15, 25)",
        font_color="white",
        legend=dict(bgcolor="rgba(20,20,40,0.8)", font_color="white"),
        height=560,
    )
    return fig


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Hand Pose Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Hand Pose Visualizer")
st.caption("RGB video frames + 3D hand skeleton with XYZ rotation axes from raw VR data")

# ---- Sidebar ----
with st.sidebar:
    st.header("Episode")

    raw_dirs = []
    if ROOT_DATA.exists():
        for task_dir in sorted(ROOT_DATA.iterdir()):
            if task_dir.is_dir():
                for ep_dir in sorted(task_dir.iterdir()):
                    if ep_dir.is_dir() and (ep_dir / "episode.pkl").exists():
                        raw_dirs.append(str(ep_dir))

    if not raw_dirs:
        st.error(f"No episodes found under {ROOT_DATA}.")
        st.stop()

    selected_ep = st.selectbox(
        "Select episode",
        raw_dirs,
        format_func=lambda p: "/".join(Path(p).parts[-2:]),
    )

    st.divider()
    st.header("Display options")

    show_skeleton    = st.checkbox("Draw hand skeleton", value=True)
    axis_scale_3d    = st.slider("3D axis length (m)", 0.01, 0.20, 0.06, 0.005)
    axis_len_px_2d   = st.slider("2D axis arrow length (px)", 10, 80, 28, 2)

    st.subheader("Joints with rotation axes")
    preset = st.selectbox(
        "Preset",
        ["Wrist only", "Wrist + fingertips", "All joints", "None", "Custom"],
    )
    if preset == "Wrist only":
        axis_idxs = {0}
    elif preset == "Wrist + fingertips":
        axis_idxs = {0, 5, 10, 15, 20, 25}
    elif preset == "All joints":
        axis_idxs = set(range(26))
    elif preset == "None":
        axis_idxs = set()
    else:
        raw = st.text_input("Joint indices (comma-separated)", "0, 5, 10, 15, 20, 25")
        try:
            axis_idxs = {int(x.strip()) for x in raw.split(",") if x.strip()}
        except ValueError:
            axis_idxs = set()

# ---- Load data ----
with st.spinner("Loading episode…"):
    try:
        ep, frames, K = load_episode(selected_ep)
    except Exception as e:
        st.error(f"Failed to load episode: {e}")
        st.stop()

with st.spinner("Computing hand poses…"):
    try:
        L_rel, R_rel, L_cam, R_cam, n_joints = precompute_joints(selected_ep)
    except Exception as e:
        st.error(f"Failed to compute hand poses: {e}")
        st.stop()

T = min(len(frames), len(L_rel))
if T == 0:
    st.error("No frames found."); st.stop()

# ---- Frame slider ----
frame_idx = st.slider("Frame", 0, T - 1, 0, key="frame_slider")
t = frame_idx

# ---- Two-column layout ----
col_rgb, col_3d = st.columns([1, 1], gap="medium")

with col_rgb:
    st.subheader("RGB frame + projected joints")
    rgb = frames[t]
    rgb_out = overlay_joints_on_frame(
        rgb, L_cam[t], K,
        dot_color_bgr=(220, 80, 80),
        axis_len_px=axis_len_px_2d,
        axis_joint_idxs=axis_idxs,
        draw_skeleton=show_skeleton,
    )
    rgb_out = overlay_joints_on_frame(
        rgb_out, R_cam[t], K,
        dot_color_bgr=(80, 80, 220),
        axis_len_px=axis_len_px_2d,
        axis_joint_idxs=axis_idxs,
        draw_skeleton=show_skeleton,
    )
    st.image(rgb_out, use_container_width=True,
             caption=f"Frame {t} / {T-1}  |  Left=red  Right=blue")
    st.markdown(
        "<span style='color:#dd3333'>■</span> Left &nbsp;&nbsp;"
        "<span style='color:#3333dd'>■</span> Right &nbsp;&nbsp;"
        "<span style='color:red'>→</span> X &nbsp;"
        "<span style='color:green'>→</span> Y &nbsp;"
        "<span style='color:blue'>→</span> Z",
        unsafe_allow_html=True,
    )

with col_3d:
    st.subheader("3D hand pose (head-centred, VR coords: X=right, Y=fwd, Z=up)")
    fig = make_3d_figure(L_rel[t], R_rel[t], axis_idxs, axis_scale=axis_scale_3d)
    st.plotly_chart(fig, use_container_width=True)

# ---- Info expanders ----
with st.expander("Episode info"):
    st.json({
        "episode_dir": selected_ep,
        "n_frames": T,
        "n_joints_per_hand": n_joints,
        "camera_K": K.tolist(),
        "left_hand_mat_shape": list(ep["left_hand_mat"].shape),
        "right_hand_mat_shape": list(ep["right_hand_mat"].shape),
    })

with st.expander(f"Joint positions at frame {t}"):
    import pandas as pd
    for hand_label, joints in [("Right", R_rel[t]), ("Left", L_rel[t])]:
        st.markdown(f"**{hand_label} hand** (head-relative, VR frame in metres)")
        pos   = joints[:, :3, 3]
        rotvec = np.array([
            Rotation.from_matrix(joints[j, :3, :3]).as_rotvec()
            for j in range(n_joints)
        ])
        df = pd.DataFrame(
            np.concatenate([pos, rotvec], axis=1),
            columns=["x (right)", "y (fwd)", "z (up)", "rx", "ry", "rz"],
            index=JOINT_NAMES[:n_joints],
        )
        st.dataframe(df.style.format("{:.4f}"), height=280)
