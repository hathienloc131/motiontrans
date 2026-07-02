"""
Streamlit visualizer for raw human VR hand pose data.

Shows RGB video frames alongside a 3D hand pose visualization
with XYZ rotation axes for each joint.

Usage (from repo root):
    streamlit run scripts_data/entry/streamlit_hand_visualizer.py -- --data_dir /path/to/raw_data_human
"""
import sys as _sys
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

def euler_pose_to_mat(pose: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """(T,6) or (6,) euler pose [x,y,z,rx,ry,rz] → (T,4,4) or (4,4) SE(3).
    scale: multiply position by this factor (e.g. 1000 to convert m→mm)."""
    position = pose[..., :3] * scale
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


# Calibration-based projection constants (empirically determined):
#   VR positions must be in mm (scale=1000), yfxrzu2standard applied,
#   then C_NEGY (negate Y) applied to both head and joint matrices.
#   Camera convention has -Z pointing toward the scene, so depth = -z_cam.
_YFXRZU2STD = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=np.float64)
_C_NEGY     = np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]], dtype=np.float64)
_CALIB_SCALE = 1000.0  # VR meters → mm (same unit as calib_result_quest2camera.npy)
_D33 = np.diag([1.0, -1.0, -1.0])  # flip y and z: y-down image convention + depth convention


def compute_calib_vr2cam(T_q2c: np.ndarray, head_pose_6d: np.ndarray) -> np.ndarray:
    """Return a fixed world→camera matrix anchored at the given head pose (T=0)."""
    head_mat = euler_pose_to_mat(head_pose_6d, scale=_CALIB_SCALE) @ _YFXRZU2STD
    return T_q2c @ _C_NEGY @ fast_mat_inv(head_mat)


def calib_cam_joints(
    world_joints_m: np.ndarray,
    vr2cam: np.ndarray,
    dx_mm: float = 0.0,
    dy_mm: float = 0.0,
    depth_scale: float = 1.0,
) -> np.ndarray:
    """
    Project hand joints to camera frame using the calibration-derived transform.

    world_joints_m : (N, 4, 4) joint matrices with positions in VR meters
    vr2cam         : (4, 4) from compute_calib_vr2cam
    dx_mm / dy_mm  : camera-frame position offset in mm (compensate calibration residual)
    depth_scale    : multiply depth by this factor (<1 = skeleton appears larger)

    Returns (N, 4, 4) matrices in image-space convention:
      z > 0 = in front of camera, y > 0 = down in image (compatible with overlay_joints_on_frame).
    Left hand stays left, right hand stays right (T_q2c cam-X ≈ VR-X, no mirroring).
    """
    N = len(world_joints_m)
    result = np.zeros((N, 4, 4), dtype=np.float64)
    for i, wj in enumerate(world_joints_m):
        jmat_mm = wj.copy()
        jmat_mm[:3, 3] *= _CALIB_SCALE
        jmat_std = jmat_mm @ _YFXRZU2STD
        cam = vr2cam @ _C_NEGY @ jmat_std
        result[i] = cam
        result[i, 0, 3]   =  cam[0, 3] + dx_mm      # optional x nudge
        result[i, 1, 3]   = -cam[1, 3] + dy_mm      # negate y + optional y nudge
        result[i, 2, 3]   = -cam[2, 3] * depth_scale # negate z + depth scale
        result[i, :3, :3] = _D33 @ cam[:3, :3]       # rotate axes consistently with y+z flip
    return result


# VR world coordinate system (after quest_recorder.compute_rel_transform):
#   X = right,  Y = forward,  Z = up
# Camera convention: X = right, Y = down, Z = forward (depth)
#
# Two modes:
#   SAME_DIR: camera on user's head, looking in same direction (+Y = depth)
#   FACING:   camera on fixed stand, facing the user (-Y = depth, X mirrored)

VR2CAM_SAME = np.array([
    [ 1,  0,  0, 0],   # cam X =  VR_X  (right)
    [ 0,  0, -1, 0],   # cam Y = -VR_Z  (up → down)
    [ 0,  1,  0, 0],   # cam Z =  VR_Y  (forward → depth)
    [ 0,  0,  0, 1],
], dtype=np.float64)

VR2CAM_FACING = np.array([
    [-1,  0,  0, 0],   # cam X = -VR_X  (mirrored, camera faces user)
    [ 0,  0, -1, 0],   # cam Y = -VR_Z  (up → down)
    [ 0, -1,  0, 0],   # cam Z = -VR_Y  (depth toward user)
    [ 0,  0,  0, 1],
], dtype=np.float64)


def build_w2c(cam_pos_vr: np.ndarray, cam_euler_deg: np.ndarray, convention: str) -> np.ndarray:
    """
    Build world-to-camera matrix.
    cam_pos_vr:    camera position in VR world frame [x, y, z]
    cam_euler_deg: camera orientation as euler XYZ in degrees
    convention:    'same_dir' or 'facing'
    """
    R = Rotation.from_euler("xyz", cam_euler_deg, degrees=True).as_matrix()
    T_cam_world = np.eye(4, dtype=np.float64)
    T_cam_world[:3, :3] = R
    T_cam_world[:3, 3] = cam_pos_vr
    vr2cam = VR2CAM_SAME if convention == "same_dir" else VR2CAM_FACING
    return vr2cam @ fast_mat_inv(T_cam_world)


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
# Data root
# ---------------------------------------------------------------------------

_default_root = Path(__file__).resolve().parents[2] / "data" / "raw_data" / "raw_data_human"
_argv = _sys.argv[1:]
if "--data_dir" in _argv:
    ROOT_DATA = Path(_argv[_argv.index("--data_dir") + 1])
else:
    ROOT_DATA = _default_root

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
    Pre-compute per-frame joint SE(3) matrices:
      head_rel:  relative to head at T=0 (for 3-D visualisation)
      world:     in VR world frame (for re-projecting with any camera pose)
      head_mats: head pose in world frame at each timestep
    """
    ep_dir = Path(episode_dir)
    with open(ep_dir / "episode.pkl", "rb") as f:
        ep = pickle.load(f)

    T = len(ep["timestamp"])
    n_joints = ep["left_hand_mat"].shape[1] // 6

    head_mat = euler_pose_to_mat(ep["head_pose_mat"].astype(np.float64))  # (T,4,4)
    inv_head0 = fast_mat_inv(head_mat[0])

    left_head_rel  = np.zeros((T, n_joints, 4, 4))
    right_head_rel = np.zeros((T, n_joints, 4, 4))
    left_world     = np.zeros((T, n_joints, 4, 4))
    right_world    = np.zeros((T, n_joints, 4, 4))

    for t in range(T):
        for j in range(n_joints):
            sl = slice(j * 6, (j + 1) * 6)
            Lj = euler_pose_to_mat(ep["left_hand_mat"][t,  sl].astype(np.float64))
            Rj = euler_pose_to_mat(ep["right_hand_mat"][t, sl].astype(np.float64))
            left_world[t, j]      = Lj
            right_world[t, j]     = Rj
            left_head_rel[t, j]   = inv_head0 @ Lj
            right_head_rel[t, j]  = inv_head0 @ Rj

    return left_head_rel, right_head_rel, left_world, right_world, head_mat, n_joints


def apply_w2c(world_joints: np.ndarray, W2C: np.ndarray) -> np.ndarray:
    """world_joints: (N,4,4) in world frame → (N,4,4) in camera frame."""
    return W2C[None] @ world_joints  # broadcast (N,4,4)


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

    pos3d = joints_cam[:, :3, 3]   # (N, 3) positions in cam frame
    valid = pos3d[:, 2] > 0.01

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

    if draw_skeleton:
        for (i, j) in HAND_CONNECTIONS:
            if i in px2d and j in px2d:
                cv2.line(img, px2d[i], px2d[j], dot_color_bgr, 1, cv2.LINE_AA)

    for idx, (u, v) in px2d.items():
        cv2.circle(img, (u, v), 4, dot_color_bgr, -1, cv2.LINE_AA)
        if idx in axis_joint_idxs:
            rot = joints_cam[idx, :3, :3]
            p = pos3d[idx]
            z_d = max(p[2], 0.01)
            scale = axis_len_px * z_d / K[0, 0]
            for ai, bgr in enumerate([(0,0,255),(0,255,0),(255,0,0)]):
                tip3 = p + rot[:, ai] * scale
                tip = proj(tip3)
                if tip:
                    cv2.arrowedLine(img, (u, v), tip, bgr, 2, tipLength=0.3, line_type=cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# 3-D Plotly figure — static (single frame)
# ---------------------------------------------------------------------------

def make_3d_figure(
    left_joints: np.ndarray,
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
            name=f"{label} joints", legendgroup=label,
        ))
        for (i, j) in HAND_CONNECTIONS:
            fig.add_trace(go.Scatter3d(
                x=[pos[i, 0], pos[j, 0]], y=[pos[i, 1], pos[j, 1]], z=[pos[i, 2], pos[j, 2]],
                mode="lines", line=dict(color=col_bone, width=3),
                showlegend=False, legendgroup=label,
            ))
        for idx in axis_joint_idxs:
            if idx >= len(joints):
                continue
            p = pos[idx]
            R = joints[idx, :3, :3]
            for ai, (ax_col, ax_name) in enumerate(zip(["red","green","blue"], ["X","Y","Z"])):
                tip = p + R[:, ai] * axis_scale
                fig.add_trace(go.Scatter3d(
                    x=[p[0], tip[0]], y=[p[1], tip[1]], z=[p[2], tip[2]],
                    mode="lines", line=dict(color=ax_col, width=5),
                    name=f"{label} J{idx} {ax_name}", showlegend=False, legendgroup=label,
                ))
    for ai, (c, n) in enumerate(zip(["red","green","blue"],["X","Y","Z"])):
        tip = np.zeros(3); tip[ai] = 0.05
        fig.add_trace(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode="lines", line=dict(color=c, width=3, dash="dash"),
            name=f"Head {n}", showlegend=False,
        ))
    fig.update_layout(
        scene=dict(
            xaxis_title="X — right (m)", yaxis_title="Y — forward (m)", zaxis_title="Z — up (m)",
            aspectmode="data", bgcolor="rgb(15, 15, 25)",
            xaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="#333"),
            yaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="#333"),
            zaxis=dict(backgroundcolor="rgb(15,15,25)", gridcolor="#333"),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor="rgb(15, 15, 25)", font_color="white",
        legend=dict(bgcolor="rgba(20,20,40,0.8)", font_color="white"),
        height=560,
    )
    return fig


# ---------------------------------------------------------------------------
# 3-D Plotly figure — animated (all frames, client-side animation)
# ---------------------------------------------------------------------------

def make_3d_animated_figure(
    L_rel_all: np.ndarray,
    R_rel_all: np.ndarray,
    default_fps: int = 15,
) -> go.Figure:
    """Build one Plotly figure with all T frames embedded as animation frames.

    Plotly animates the figure client-side (JavaScript), so there is no Python
    rerun per frame and the chart never flickers from a full rebuild.

    Trace layout (stable across all frames):
      0: Right joint markers
      1: Right bones  (all segments in one trace, None-separated)
      2: Left joint markers
      3: Left bones
      4-6: Head X/Y/Z axes (static, not in animation frames)
    """
    T_ep = len(L_rel_all)

    def _hand_traces(joints, col_pt, col_bone):
        pos = joints[:, :3, 3]
        xs, ys, zs = [], [], []
        for (i, j) in HAND_CONNECTIONS:
            xs += [pos[i, 0], pos[j, 0], None]
            ys += [pos[i, 1], pos[j, 1], None]
            zs += [pos[i, 2], pos[j, 2], None]
        return (
            go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                         mode="markers", marker=dict(size=5, color=col_pt, opacity=0.95)),
            go.Scatter3d(x=xs, y=ys, z=zs,
                         mode="lines", line=dict(color=col_bone, width=3), showlegend=False),
        )

    # Frame-0 initial data
    rt_m, rt_b = _hand_traces(R_rel_all[0], "#4488ff", "#2266dd")
    lt_m, lt_b = _hand_traces(L_rel_all[0], "#ff4444", "#dd2222")
    rt_m.name = "Right joints"; rt_m.legendgroup = "Right"
    lt_m.name = "Left joints";  lt_m.legendgroup = "Left"
    rt_b.legendgroup = "Right"; lt_b.legendgroup = "Left"

    # Static head-origin axes (not animated)
    head_traces = []
    for ai, (c, n) in enumerate(zip(["red", "green", "blue"], ["X", "Y", "Z"])):
        tip = np.zeros(3); tip[ai] = 0.05
        head_traces.append(go.Scatter3d(
            x=[0, tip[0]], y=[0, tip[1]], z=[0, tip[2]],
            mode="lines", line=dict(color=c, width=3, dash="dash"),
            name=f"Head {n}", showlegend=False,
        ))

    fig = go.Figure(data=[rt_m, rt_b, lt_m, lt_b] + head_traces)

    # Build animation frames (only update the 4 hand traces)
    frames = []
    for t in range(T_ep):
        rm, rb = _hand_traces(R_rel_all[t], "#4488ff", "#2266dd")
        lm, lb = _hand_traces(L_rel_all[t], "#ff4444", "#dd2266")
        frames.append(go.Frame(data=[rm, rb, lm, lb], traces=[0, 1, 2, 3], name=str(t)))
    fig.frames = frames

    # ---- Fixed axis ranges: compute bbox across ALL frames so the view never
    # jumps as hands move. Use a cube so all 3 axes have the same scale. ----
    all_pos = np.concatenate([
        L_rel_all[:, :, :3, 3].reshape(-1, 3),
        R_rel_all[:, :, :3, 3].reshape(-1, 3),
    ])
    xc = float((all_pos[:, 0].min() + all_pos[:, 0].max()) / 2)
    yc = float((all_pos[:, 1].min() + all_pos[:, 1].max()) / 2)
    zc = float((all_pos[:, 2].min() + all_pos[:, 2].max()) / 2)
    half = float(max(
        all_pos[:, 0].max() - all_pos[:, 0].min(),
        all_pos[:, 1].max() - all_pos[:, 1].min(),
        all_pos[:, 2].max() - all_pos[:, 2].min(),
    ) / 2) * 1.15 + 0.04   # 15 % padding + 4 cm margin

    # ---- Initial camera: look from behind the head in the +Y (forward)
    # direction (VR: X=right, Y=forward, Z=up). eye is at -Y, slightly above. ----
    camera = dict(
        eye=dict(x=0.05, y=-2.0, z=0.4),
        center=dict(x=xc, y=yc, z=zc),
        up=dict(x=0, y=0, z=1),
    )

    _ax = dict(backgroundcolor="rgb(15,15,25)", gridcolor="#333", showspikes=False)
    ms_per_frame = max(1, int(1000 / default_fps))
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.08, x=0.0, xanchor="left", yanchor="top",
            buttons=[
                dict(label="▶", method="animate",
                     args=[None, dict(frame=dict(duration=ms_per_frame, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(frame=dict(duration=0), mode="immediate")]),
            ],
        )],
        sliders=[dict(
            currentvalue=dict(prefix="Frame: ", visible=True, xanchor="center"),
            pad=dict(t=10, b=0),
            steps=[dict(
                args=[[str(t)], dict(frame=dict(duration=0, redraw=True), mode="immediate")],
                label=str(t), method="animate",
            ) for t in range(T_ep)],
            active=0, x=0.08, len=0.92,
        )],
        scene=dict(
            xaxis_title="X right (m)", yaxis_title="Y fwd (m)", zaxis_title="Z up (m)",
            aspectmode="cube",
            xaxis=dict(**_ax, range=[xc - half, xc + half]),
            yaxis=dict(**_ax, range=[yc - half, yc + half]),
            zaxis=dict(**_ax, range=[zc - half, zc + half]),
            bgcolor="rgb(15, 15, 25)",
            camera=camera,
        ),
        margin=dict(l=0, r=0, t=50, b=60),
        paper_bgcolor="rgb(15, 15, 25)", font_color="white",
        legend=dict(bgcolor="rgba(20,20,40,0.8)", font_color="white"),
        height=580,
    )
    return fig


@st.cache_data(show_spinner="Building 3D animation (once per episode)…")
def _build_3d_animation(episode_dir: str) -> go.Figure:
    L, R, *_ = precompute_joints(episode_dir)
    return make_3d_animated_figure(L, R)


# ---------------------------------------------------------------------------
# F285 gripper: per-finger reach ratio (no URDF / dex_retargeting needed)
#
# Joint layout (from JOINT_NAMES / Unity XR Hands, 0-indexed):
#   wrist=0
#   thumb:  cmc=1, mcp=2, ip=3,  tip=4,  aux=5
#   index:  mcp=6, pip=7, dip=8,  tip=9,  aux=10
#   middle: mcp=11,pip=12,dip=13, tip=14, aux=15
#   ring:   mcp=16,pip=17,dip=18, tip=19, aux=20
#   pinky:  mcp=21,pip=22,dip=23, tip=24, aux=25
#
# Reach ratio = ||tip - MCP|| / (MCP→PIP + PIP→DIP + DIP→tip)
#   open (straight finger) ≈ 0.92-1.0
#   cup grasp (L-shape at PIP) ≈ 0.45-0.65
#   fist (fully curled) ≈ 0.15-0.35
# ---------------------------------------------------------------------------

_WRIST_IDX = 0
_FINGER_NAMES = ["index", "middle", "ring", "pinky"]  # thumb excluded (stays abducted)

# (MCP, PIP, DIP, TIP) indices per finger — used for reach ratio
_FINGER_JOINT_IDXS = [
    (6,  7,  8,  9),   # index
    (11, 12, 13, 14),  # middle
    (16, 17, 18, 19),  # ring
    (21, 22, 23, 24),  # pinky
]

# Thumb separately: (CMC, MCP, IP, TIP)
_THUMB_JOINT_IDXS = (1, 2, 3, 4)


def _jpos(inv_w, idx, data_t):
    """Position of joint `idx` in wrist frame."""
    return (inv_w @ euler_pose_to_mat(data_t[idx*6:(idx+1)*6]))[:3, 3]


def _reach_ratio_from_joints(inv_w, mcp_idx, pip_idx, dip_idx, tip_idx, data_t):
    """Wrist-relative reach ratio for one finger."""
    mcp, pip = _jpos(inv_w, mcp_idx, data_t), _jpos(inv_w, pip_idx, data_t)
    dip, tip = _jpos(inv_w, dip_idx, data_t), _jpos(inv_w, tip_idx, data_t)
    span = np.linalg.norm(pip - mcp) + np.linalg.norm(dip - pip) + np.linalg.norm(tip - dip)
    return np.linalg.norm(tip - mcp) / (span + 1e-6)


def _pip_angle_deg(inv_w, mcp_idx, pip_idx, dip_idx, data_t):
    """Angle at PIP joint in degrees. 180°=straight, 90°=L-shape (cup grasp)."""
    mcp, pip = _jpos(inv_w, mcp_idx, data_t), _jpos(inv_w, pip_idx, data_t)
    dip      = _jpos(inv_w, dip_idx, data_t)
    v1, v2 = mcp - pip, dip - pip
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


# MCP indices of the 4 non-thumb fingers (used for palm-plane fitting)
_FINGER_MCP_IDXS = [6, 11, 16, 21]   # index, middle, ring, pinky


def _palm_plane(inv_w, data_t):
    """
    Best-fit plane through the 4 non-thumb MCP joints (in wrist frame).
    Returns (normal, centroid) where normal points palmward (toward thumb_tip side).
    """
    mcps = np.array([_jpos(inv_w, i, data_t) for i in _FINGER_MCP_IDXS])
    centroid = mcps.mean(axis=0)
    _, _, Vt = np.linalg.svd(mcps - centroid)
    normal = Vt[-1]                             # smallest variance → plane normal
    thumb_tip = _jpos(inv_w, 4, data_t)         # thumb_tip idx=4
    if np.dot(normal, thumb_tip - centroid) < 0:
        normal = -normal                        # ensure normal points toward thumb side
    return normal, centroid


def _thumb_metrics(inv_w, data_t):
    """
    Two metrics for whether the thumb has swung into the finger plane (cup grasp check).

    elevation_angle_deg:
        Angle between thumb axis (CMC→tip) and the palm plane (= 4-finger MCP plane).
        0°  → thumb lies flat IN the palm plane  ("cùng mặt phẳng" ✓)
        90° → thumb sticks straight out of the palm (fully abducted/open)
        For cup grasp target: < 30°

    coplanar_dist_m:
        Signed distance of thumb_tip from the palm plane (metres).
        ≈ 0  → thumb tip in plane with MCPs
        > 0  → thumb tip above palm plane (still spread out)
        < 0  → thumb tip below palm plane (over-curled)
    """
    normal, centroid = _palm_plane(inv_w, data_t)
    thumb_cmc = _jpos(inv_w, 1, data_t)   # CMC idx=1
    thumb_tip = _jpos(inv_w, 4, data_t)   # tip idx=4

    # Elevation angle
    v = thumb_tip - thumb_cmc
    nv = np.linalg.norm(v)
    if nv > 1e-6:
        sin_elev = abs(np.dot(v / nv, normal))
        elevation = float(np.degrees(np.arcsin(np.clip(sin_elev, 0.0, 1.0))))
    else:
        elevation = 0.0

    # Coplanar distance (signed)
    coplanar = float(np.dot(thumb_tip - centroid, normal))

    return elevation, coplanar


@st.cache_data(show_spinner="Computing hand closure (reach ratios + thumb)…")
def compute_hand_closure(episode_dir: str):
    """
    Per-frame metrics for both hands.

    Returns:
      right_reach      : (T, 4)  reach ratios  [index, middle, ring, pinky]
      left_reach       : (T, 4)
      right_pip        : (T, 4)  PIP angles °   (180=straight, 90=L-shape/cup)
      left_pip         : (T, 4)
      right_mean       : (T,)    mean reach ratio (used for F285 mapping)
      left_mean        : (T,)
      right_thumb_elev : (T,)    thumb elevation angle ° from palm plane (0=in-plane)
      left_thumb_elev  : (T,)
      right_thumb_dist : (T,)    signed dist of thumb_tip from palm plane (m)
      left_thumb_dist  : (T,)
    """
    ep_dir = Path(episode_dir)
    with open(ep_dir / "episode.pkl", "rb") as f:
        ep = pickle.load(f)

    T = len(ep["timestamp"])
    right_reach = np.zeros((T, 4));  left_reach = np.zeros((T, 4))
    right_pip   = np.zeros((T, 4));  left_pip   = np.zeros((T, 4))
    right_elev  = np.zeros(T);       left_elev  = np.zeros(T)
    right_dist  = np.zeros(T);       left_dist  = np.zeros(T)

    for hand_key, reach_out, pip_out, elev_out, dist_out in [
        ("right_hand_mat", right_reach, right_pip, right_elev, right_dist),
        ("left_hand_mat",  left_reach,  left_pip,  left_elev,  left_dist),
    ]:
        data = ep[hand_key].astype(np.float64)
        for t in range(T):
            wrist = euler_pose_to_mat(data[t, _WRIST_IDX*6:(_WRIST_IDX+1)*6])
            inv_w = fast_mat_inv(wrist)
            for fi, (mcp, pip, dip, tip) in enumerate(_FINGER_JOINT_IDXS):
                reach_out[t, fi] = _reach_ratio_from_joints(inv_w, mcp, pip, dip, tip, data[t])
                pip_out[t, fi]   = _pip_angle_deg(inv_w, mcp, pip, dip, data[t])
            elev_out[t], dist_out[t] = _thumb_metrics(inv_w, data[t])

    return (right_reach, left_reach,
            right_pip,   left_pip,
            right_reach.mean(axis=1), left_reach.mean(axis=1),
            right_elev, left_elev,
            right_dist, left_dist)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Hand Pose Visualizer", layout="wide", initial_sidebar_state="expanded")
st.title("Hand Pose Visualizer")
st.caption("RGB video frames + 3D hand skeleton with XYZ rotation axes from raw VR data")

with st.sidebar:
    st.header("Episode")
    raw_dirs = []
    if ROOT_DATA.exists():
        for task_dir in sorted(ROOT_DATA.iterdir()):
            if not task_dir.is_dir():
                continue
            if (task_dir / "episode.pkl").exists():
                # Episodes are directly under ROOT_DATA (flat structure)
                raw_dirs.append(str(task_dir))
            else:
                # Episodes are one level deeper (task_dir → ep_dir)
                for ep_dir in sorted(task_dir.iterdir()):
                    if ep_dir.is_dir() and (ep_dir / "episode.pkl").exists():
                        raw_dirs.append(str(ep_dir))
    if not raw_dirs:
        st.error(f"No episodes found under {ROOT_DATA}.")
        st.stop()

    selected_ep = st.selectbox(
        "Select episode", raw_dirs,
        format_func=lambda p: "/".join(Path(p).parts[-2:]),
    )

    st.divider()
    st.header("Display options")
    show_skeleton  = st.checkbox("Draw hand skeleton", value=True)
    axis_scale_3d  = st.slider("3D axis length (m)", 0.01, 0.20, 0.06, 0.005)
    axis_len_px_2d = st.slider("2D axis arrow length (px)", 10, 80, 28, 2)

    st.subheader("Joints with rotation axes")
    preset = st.selectbox("Preset", ["Wrist only", "Wrist + fingertips", "All joints", "None", "Custom"])
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

    st.divider()
    st.header("Camera calibration")
    st.caption(
        "VR world coords: X=right, Y=forward, Z=up. "
        "Set camera pose to match your physical setup."
    )

    _default_calib = str(
        Path(__file__).resolve().parents[2]
        / "camera_params" / "quest_realsense" / "calib_result_quest2camera.npy"
    )
    use_calib_file = st.checkbox("Use calibration file (quest→camera)", value=Path(_default_calib).exists())
    calib_path = _default_calib
    calib_ok   = False
    if use_calib_file:
        calib_path = st.text_input("Calibration .npy path", _default_calib)
        calib_ok   = Path(calib_path).exists()
        if not calib_ok:
            st.warning(f"File not found: {calib_path}")
        else:
            st.success("Calibration file loaded.")

    cam_mode = st.radio(
        "Camera type (used when calibration file is OFF)",
        ["Fixed camera (on stand)", "Head-mounted (follows head)"],
        index=0,
    )

    convention = st.selectbox(
        "Camera facing direction",
        ["Facing user (-Y, X mirrored)", "Same as user (+Y, no mirror)"],
        index=0,
    )
    conv_key = "facing" if convention.startswith("Facing") else "same_dir"

    if cam_mode == "Fixed camera (on stand)":
        st.markdown("**Camera position in VR world (m)**")
        c1, c2, c3 = st.columns(3)
        cam_px = c1.number_input("X", value=0.0,  step=0.05, format="%.2f", key="cpx")
        cam_py = c2.number_input("Y", value=-1.0, step=0.05, format="%.2f", key="cpy")
        cam_pz = c3.number_input("Z", value=1.0,  step=0.05, format="%.2f", key="cpz")

        st.markdown("**Camera orientation — euler XYZ (deg)**")
        c1, c2, c3 = st.columns(3)
        cam_rx = c1.slider("Rx", -180, 180, 0,  key="crx")
        cam_ry = c2.slider("Ry", -180, 180, 0,  key="cry")
        cam_rz = c3.slider("Rz", -180, 180, 0,  key="crz")

        cam_pos_vr  = np.array([cam_px, cam_py, cam_pz], dtype=np.float64)
        cam_euler   = np.array([cam_rx, cam_ry, cam_rz], dtype=np.float64)
        fixed_cam   = True

    else:  # head-mounted
        st.markdown("**Offset from head center (m, VR frame)**")
        c1, c2, c3 = st.columns(3)
        off_x = c1.number_input("dX", value=0.0, step=0.01, format="%.3f", key="ox")
        off_y = c2.number_input("dY", value=0.0, step=0.01, format="%.3f", key="oy")
        off_z = c3.number_input("dZ", value=0.0, step=0.01, format="%.3f", key="oz")
        cam_offset = np.array([off_x, off_y, off_z], dtype=np.float64)
        fixed_cam  = False

    st.divider()
    st.header("Projection fine-tuning")
    st.caption(
        "Compensate residual calibration error. "
        "dx/dy shift the skeleton in camera-frame mm; "
        "depth scale < 1 makes the skeleton appear larger."
    )
    c1, c2 = st.columns(2)
    calib_dx = c1.slider("dx (mm, cam X)", -300, 300, 0, 5, key="cdx")
    calib_dy = c2.slider("dy (mm, cam Y)", -300, 300, 0, 5, key="cdy")
    calib_depth_scale = st.slider("Depth scale", 0.3, 2.0, 1.0, 0.05, key="cds")

    st.divider()
    st.header("F285 Gripper")
    st.caption(
        "Robotiq 2F-85 retargeting: maps thumb-to-index pinch distance "
        "to driver joint [0 = open, 0.8 = closed]."
    )
    st.caption(
        "reach_ratio = ‖tip − MCP‖ / finger_span. "
        "Open hand ≈ 0.92 · Cup grasp ≈ 0.50 · Fist ≈ 0.20"
    )
    st.markdown("**Right hand**")
    f285_open_ratio_right = st.slider(
        "R open ratio (reach_ratio → joint=0)",
        min_value=0.70, max_value=1.00, value=0.98, step=0.01,
        format="%.2f", key="f285_or_right",
    )
    st.markdown("**Left hand**")
    f285_open_ratio_left = st.slider(
        "L open ratio (reach_ratio → joint=0)",
        min_value=0.70, max_value=1.00, value=0.94, step=0.01,
        format="%.2f", key="f285_or_left",
    )
    thumb_dist_thresh = st.slider(
        "Thumb coplanar dist threshold (m) — dist ≤ this = 'cùng mặt phẳng' ✓",
        min_value=0.005, max_value=0.15, value=0.04, step=0.005,
        format="%.3f", key="thumb_thresh",
        help="Signed distance of thumb_tip from 4-finger MCP plane. ≤ 0.04 m = thumb in-plane.",
    )
    f285_close_ramp_k = st.number_input(
        "Close ramp k (frames before close event → interpolate 0→0.8)",
        min_value=0, max_value=50, value=5, step=1, key="f285_ramp_k",
        help="Số frame trước mỗi lần hand close sẽ được ramp tuyến tính 0 → 0.8. 0 = tắt.",
    )

    with st.expander("Scan episodes to calibrate open_dist"):
        n_scan = st.number_input("Episodes to scan", min_value=1, max_value=max(1, len(raw_dirs)), value=min(5, len(raw_dirs)), step=1, key="n_scan")
        if st.button("Run scan", key="run_scan"):
            scan_right, scan_left = [], []
            prog = st.progress(0.0, text="Scanning…")
            for si, ep_d in enumerate(raw_dirs[:int(n_scan)]):
                try:
                    rr, lr, rp, lp, rm, lm, re, le, rd, ld = compute_hand_closure(ep_d)
                    scan_right.append(rm)
                    scan_left.append(lm)
                except Exception:
                    pass
                prog.progress((si + 1) / int(n_scan), text=f"Scanned {si+1}/{int(n_scan)}")
            prog.empty()
            if scan_right:
                all_r = np.concatenate(scan_right)
                all_l = np.concatenate(scan_left)
                for hand, arr in [("Right mean reach ratio", all_r), ("Left mean reach ratio", all_l)]:
                    st.markdown(f"**{hand}** ({len(arr)} frames, {len(scan_right)} episodes)")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("min (fist)", f"{arr.min():.3f}")
                    c2.metric("p5", f"{np.percentile(arr, 5):.3f}")
                    c3.metric("p95 (open)", f"{np.percentile(arr, 95):.3f}")
                    c4.metric("max", f"{arr.max():.3f}")
                st.info(
                    "Set **open ratio** ≈ p95 (relaxed open hand). "
                    "Set **closed ratio** ≈ p5 (tightest grasp observed)."
                )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

with st.spinner("Loading episode…"):
    try:
        ep, frames, K = load_episode(selected_ep)
    except Exception as e:
        st.error(f"Failed to load episode: {e}"); st.stop()

with st.spinner("Computing hand poses…"):
    try:
        L_rel, R_rel, L_world, R_world, head_mats, n_joints = precompute_joints(selected_ep)
    except Exception as e:
        st.error(f"Failed to compute hand poses: {e}"); st.stop()

T = min(len(frames), len(L_rel))
if T == 0:
    st.error("No frames found."); st.stop()

# Head position at T=0 shown as reference
head0_pos = head_mats[0, :3, 3]
with st.sidebar:
    st.caption(f"Head at T=0: X={head0_pos[0]:.2f}, Y={head0_pos[1]:.2f}, Z={head0_pos[2]:.2f} m")

# ---------------------------------------------------------------------------
# Layout: two columns — RGB player (fragment) | 3D animation (static)
#
# The 3D chart is rendered by the MAIN SCRIPT (col_3d), so it is never
# touched by fragment reruns. Plotly animates all T frames client-side in JS;
# there is no Python rerun per 3D frame, so no rebuild and no flickering.
# The RGB player fragment uses st.rerun(scope="fragment") to update only
# the left column without touching the right column at all.
# ---------------------------------------------------------------------------

col_rgb, col_3d = st.columns([1, 1], gap="medium")

with col_3d:
    st.subheader("3D hand pose — all frames (VR head-centred)")
    fig_3d = _build_3d_animation(selected_ep)
    st.plotly_chart(fig_3d, use_container_width=True, key="hand_3d_chart")

@st.fragment
def _player():
    import time as _time

    # Session state init
    if "_pf" not in st.session_state:
        st.session_state._pf = None
    if "frame_slider" not in st.session_state:
        st.session_state["frame_slider"] = 0
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "fps_play" not in st.session_state:
        st.session_state.fps_play = 15

    # Apply pending frame BEFORE the slider widget is instantiated
    if st.session_state._pf is not None:
        st.session_state["frame_slider"] = st.session_state._pf
        st.session_state._pf = None
    if st.session_state["frame_slider"] >= T:
        st.session_state["frame_slider"] = 0

    # Controls
    c0, c1, c2, _ = st.columns([1, 1, 1, 4])
    with c0:
        if st.button("⏮ Reset"):
            st.session_state["frame_slider"] = 0
            st.session_state.playing = False
    with c1:
        if st.button("⏸ Pause" if st.session_state.playing else "▶ Play"):
            st.session_state.playing = not st.session_state.playing
    with c2:
        fps_opts = [5, 10, 15, 20, 30]
        fps_play = st.selectbox(
            "FPS", fps_opts,
            index=fps_opts.index(st.session_state.fps_play)
                  if st.session_state.fps_play in fps_opts else 2,
            label_visibility="collapsed",
        )
        st.session_state.fps_play = fps_play

    # Slider — on_change stops playback when user scrubs manually
    def _pause():
        st.session_state.playing = False
    t = st.slider("Frame", 0, T - 1, key="frame_slider", on_change=_pause)

    # Camera-frame joints for current frame
    if use_calib_file and calib_ok:
        T_q2c   = np.load(calib_path)
        vr2cam0 = compute_calib_vr2cam(T_q2c, ep["head_pose_mat"][0].astype(np.float64))
        L_cam_t = calib_cam_joints(L_world[t], vr2cam0,
                                   dx_mm=calib_dx, dy_mm=calib_dy, depth_scale=calib_depth_scale)
        R_cam_t = calib_cam_joints(R_world[t], vr2cam0,
                                   dx_mm=calib_dx, dy_mm=calib_dy, depth_scale=calib_depth_scale)
    else:
        if fixed_cam:
            W2C = build_w2c(cam_pos_vr, cam_euler, conv_key)
        else:
            T_offset = np.eye(4, dtype=np.float64)
            T_offset[:3, 3] = cam_offset
            T_cam_world = head_mats[t] @ T_offset
            vr2cam_mode = VR2CAM_SAME if conv_key == "same_dir" else VR2CAM_FACING
            W2C = vr2cam_mode @ fast_mat_inv(T_cam_world)
        L_cam_t = apply_w2c(L_world[t], W2C)
        R_cam_t = apply_w2c(R_world[t], W2C)

    n_L = int(np.sum(L_cam_t[:, 2, 3] > 0.01))
    n_R = int(np.sum(R_cam_t[:, 2, 3] > 0.01))

    # RGB image with skeleton overlay
    st.subheader("RGB frame + projected joints")
    rgb = frames[t]
    rgb_out = overlay_joints_on_frame(
        rgb, L_cam_t, K,
        dot_color_bgr=(80, 80, 220),
        axis_len_px=axis_len_px_2d, axis_joint_idxs=axis_idxs, draw_skeleton=show_skeleton,
    )
    rgb_out = overlay_joints_on_frame(
        rgb_out, R_cam_t, K,
        dot_color_bgr=(220, 80, 80),
        axis_len_px=axis_len_px_2d, axis_joint_idxs=axis_idxs, draw_skeleton=show_skeleton,
    )
    st.image(rgb_out, use_container_width=True,
             caption=f"Frame {t}/{T-1}  |  Left=red  Right=blue  |  "
                     f"Visible: L={n_L}/26, R={n_R}/26")
    st.markdown(
        "<span style='color:#dd3333'>■</span> Left &nbsp;&nbsp;"
        "<span style='color:#3333dd'>■</span> Right &nbsp;&nbsp;"
        "<span style='color:red'>→</span> X &nbsp;"
        "<span style='color:green'>→</span> Y &nbsp;"
        "<span style='color:blue'>→</span> Z",
        unsafe_allow_html=True,
    )

    # Auto-advance: scope="fragment" rerenders only this col_rgb section.
    # col_3d (animated 3D chart) is in the main script and is never touched.
    if st.session_state.playing:
        if t + 1 >= T:
            st.session_state.playing = False
        else:
            st.session_state._pf = t + 1
            _time.sleep(1.0 / st.session_state.fps_play)
            st.rerun(scope="fragment")

with col_rgb:
    _player()

# ---------------------------------------------------------------------------
# F285 Gripper — time-series chart + per-frame gauges (full-width)
# ---------------------------------------------------------------------------

t_dbg = min(st.session_state.get("frame_slider", 0), T - 1)

st.divider()
st.subheader("F285 Gripper (Robotiq 2F-85)")

try:
    (_right_reach, _left_reach,
     _right_pip,   _left_pip,
     _right_mean,  _left_mean,
     _right_elev,  _left_elev,
     _right_dist,  _left_dist) = compute_hand_closure(selected_ep)

    # Full-episode binary closed arrays (both reach AND thumb conditions)
    _r_closed_bin = (_right_mean <= f285_open_ratio_right) & (_right_dist <= thumb_dist_thresh)
    _l_closed_bin = (_left_mean  <= f285_open_ratio_left)  & (_left_dist  <= thumb_dist_thresh)

    def _apply_ramp_vis(closed_bin, k):
        """Binary 0/0.8 signal with optional k-frame linear ramp before each close event."""
        k = int(k)
        result = np.where(closed_bin, 0.8, 0.0)
        if k <= 0:
            return result
        rising = np.where((~closed_bin[:-1]) & closed_bin[1:])[0] + 1
        ramp_template = np.linspace(0.0, 0.8, k + 1)[1:]
        for t_c in rising:
            t_start = max(0, t_c - k)
            n = t_c - t_start
            ramp = ramp_template[-n:]
            mask = ~closed_bin[t_start:t_c]
            result[t_start:t_c][mask] = ramp[mask]
        return result

    _right_f285 = _apply_ramp_vis(_r_closed_bin, f285_close_ramp_k)
    _left_f285  = _apply_ramp_vis(_l_closed_bin,  f285_close_ramp_k)

    # ---- Combined grasp indicator at current frame ----
    _r_fingers_ok = bool(_right_mean[t_dbg] <= f285_open_ratio_right)
    _r_thumb_ok   = bool(_right_dist[t_dbg] <= thumb_dist_thresh)
    _l_fingers_ok = bool(_left_mean[t_dbg]  <= f285_open_ratio_left)
    _l_thumb_ok   = bool(_left_dist[t_dbg]  <= thumb_dist_thresh)

    _r_grasp = _r_fingers_ok and _r_thumb_ok
    _l_grasp = _l_fingers_ok and _l_thumb_ok

    ga1, ga2 = st.columns(2)
    with ga1:
        status = "🟢 GRASPING" if _r_grasp else ("🟡 fingers only" if _r_fingers_ok else ("🟡 thumb only" if _r_thumb_ok else "🔴 OPEN"))
        st.markdown(f"**Right hand** &nbsp; {status}")
    with ga2:
        status = "🟢 GRASPING" if _l_grasp else ("🟡 fingers only" if _l_fingers_ok else ("🟡 thumb only" if _l_thumb_ok else "🔴 OPEN"))
        st.markdown(f"**Left hand** &nbsp; {status}")

    # ---- Per-frame metric grid ----
    gc1, gc2, gc3, gc4, gc5, gc6 = st.columns(6)
    gc1.metric("R reach ratio", f"{_right_mean[t_dbg]:.3f}", help="open≈0.92  cup≈0.50  fist≈0.20")
    gc2.metric("R thumb dist (m)", f"{_right_dist[t_dbg]:.4f}",
               delta=f"{'✓ in-plane' if _r_thumb_ok else '✗ spread'}", delta_color="off",
               help=f"≤ {thumb_dist_thresh:.3f} m = thumb in-plane")
    gc3.metric("R thumb elev°", f"{_right_elev[t_dbg]:.1f}°")
    gc4.metric("L reach ratio", f"{_left_mean[t_dbg]:.3f}")
    gc5.metric("L thumb dist (m)", f"{_left_dist[t_dbg]:.4f}",
               delta=f"{'✓ in-plane' if _l_thumb_ok else '✗ spread'}", delta_color="off",
               help=f"≤ {thumb_dist_thresh:.3f} m = thumb in-plane")
    gc6.metric("L thumb elev°", f"{_left_elev[t_dbg]:.1f}°")

    gc1b, gc2b = st.columns(2)
    with gc1b:
        st.caption("Right F285")
        _rv = float(_right_f285[t_dbg] / 0.8)
        st.progress(_rv, text=f"{_right_f285[t_dbg]:.3f}  ({'CLOSED' if _rv > 0.75 else 'OPEN' if _rv < 0.25 else 'mid'})")
    with gc2b:
        st.caption("Left F285")
        _lv = float(_left_f285[t_dbg] / 0.8)
        st.progress(_lv, text=f"{_left_f285[t_dbg]:.3f}  ({'CLOSED' if _lv > 0.75 else 'OPEN' if _lv < 0.25 else 'mid'})")

    # ---- Chart tabs: reach ratios | PIP angles ----
    hand_choice = st.radio("Show hand", ["Right", "Left"], horizontal=True, key="f285_hand")
    _is_right   = hand_choice == "Right"
    _reach = _right_reach if _is_right else _left_reach
    _pip   = _right_pip   if _is_right else _left_pip
    _mean  = _right_mean  if _is_right else _left_mean
    _f285  = _right_f285  if _is_right else _left_f285
    _open_ratio = f285_open_ratio_right if _is_right else f285_open_ratio_left

    frames_idx = np.arange(len(_mean))
    _FINGER_COLORS = ["#44cc88", "#4488ff", "#cc44cc", "#ff6644"]  # index→pinky

    tab_reach, tab_pip, tab_thumb, tab_oc = st.tabs(["Reach ratio", "PIP angle (°)", "Thumb (in-plane check)", "Open / Close"])

    with tab_reach:
        fig_r = go.Figure()
        for fi, (fname, fcol) in enumerate(zip(_FINGER_NAMES, _FINGER_COLORS)):
            fig_r.add_trace(go.Scatter(
                x=frames_idx, y=_reach[:, fi],
                name=fname, line=dict(color=fcol, width=1.2, dash="dot"),
            ))
        fig_r.add_trace(go.Scatter(
            x=frames_idx, y=_mean,
            name="mean (→F285)", line=dict(color="white", width=2),
        ))
        fig_r.add_trace(go.Scatter(
            x=frames_idx, y=_f285 / 0.8,       # normalised to [0,1] for overlay
            name="F285 normalised", line=dict(color="#00e5ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,229,255,0.12)", yaxis="y2",
        ))
        fig_r.add_hline(y=_open_ratio, line=dict(color="orange", dash="dash", width=1.5),
                        annotation_text=f"open={_open_ratio:.2f}", annotation_position="top right")
        fig_r.add_vline(x=t_dbg, line=dict(color="yellow", dash="dash", width=2),
                        annotation_text=f"t={t_dbg}", annotation_position="top left")
        fig_r.update_layout(
            xaxis_title="Frame",
            yaxis=dict(title="Reach ratio [0–1]", range=[-0.05, 1.05]),
            yaxis2=dict(title="F285 norm.", overlaying="y", side="right", range=[-0.05, 1.05], showgrid=False),
            paper_bgcolor="rgb(15,15,25)", plot_bgcolor="rgb(25,25,40)", font_color="white",
            legend=dict(bgcolor="rgba(20,20,40,0.8)", orientation="h", y=1.10, x=0),
            height=320, margin=dict(l=0, r=60, t=50, b=30),
        )
        st.plotly_chart(fig_r, use_container_width=True, key="f285_reach_chart")

    with tab_pip:
        fig_p = go.Figure()
        for fi, (fname, fcol) in enumerate(zip(_FINGER_NAMES, _FINGER_COLORS)):
            fig_p.add_trace(go.Scatter(
                x=frames_idx, y=_pip[:, fi],
                name=fname, line=dict(color=fcol, width=1.5),
            ))
        fig_p.add_hline(y=180, line=dict(color="gray",   dash="dot",  width=1), annotation_text="straight (180°)")
        fig_p.add_hline(y=90,  line=dict(color="orange", dash="dash", width=1), annotation_text="L-shape / cup (90°)")
        fig_p.add_vline(x=t_dbg, line=dict(color="yellow", dash="dash", width=2),
                        annotation_text=f"t={t_dbg}", annotation_position="top left")
        fig_p.update_layout(
            xaxis_title="Frame",
            yaxis=dict(title="PIP angle (°)", range=[0, 200]),
            paper_bgcolor="rgb(15,15,25)", plot_bgcolor="rgb(25,25,40)", font_color="white",
            legend=dict(bgcolor="rgba(20,20,40,0.8)", orientation="h", y=1.10, x=0),
            height=320, margin=dict(l=0, r=60, t=50, b=30),
        )
        st.plotly_chart(fig_p, use_container_width=True, key="f285_pip_chart")

    with tab_thumb:
        _elev = _right_elev if hand_choice == "Right" else _left_elev
        _dist = _right_dist if hand_choice == "Right" else _left_dist

        fig_th = go.Figure()
        # Elevation angle — primary axis
        fig_th.add_trace(go.Scatter(
            x=frames_idx, y=_elev,
            name="thumb elevation (°)", line=dict(color="#f0a830", width=2),
        ))
        # Coplanar distance — secondary axis
        fig_th.add_trace(go.Scatter(
            x=frames_idx, y=_dist,
            name="thumb_tip dist from palm plane (m)", line=dict(color="#cc88ff", width=1.5, dash="dot"),
            yaxis="y2",
        ))
        # Threshold reference
        fig_th.add_hline(y=thumb_dist_thresh, yref="y2",
                         line=dict(color="tomato", dash="dash", width=2),
                         annotation_text=f"dist threshold {thumb_dist_thresh:.3f} m (in-plane ↓)",
                         annotation_position="top right")
        fig_th.add_hline(y=0, yref="y2",
                         line=dict(color="gray", dash="dot", width=1))
        fig_th.add_vline(x=t_dbg, line=dict(color="yellow", dash="dash", width=2),
                         annotation_text=f"t={t_dbg}", annotation_position="top left")
        fig_th.update_layout(
            xaxis_title="Frame",
            yaxis=dict(title="Elevation angle (°)", range=[-5, 95]),
            yaxis2=dict(title="Coplanar dist (m)", overlaying="y", side="right",
                        range=[float(_dist.min()) * 1.2 - 0.01, float(_dist.max()) * 1.2 + 0.01],
                        showgrid=False),
            paper_bgcolor="rgb(15,15,25)", plot_bgcolor="rgb(25,25,40)", font_color="white",
            legend=dict(bgcolor="rgba(20,20,40,0.8)", orientation="h", y=1.10, x=0),
            height=320, margin=dict(l=0, r=80, t=50, b=30),
        )
        st.plotly_chart(fig_th, use_container_width=True, key="f285_thumb_chart")
        st.caption(
            "**Elevation angle** = angle between thumb axis (CMC→tip) and the palm plane "
            "(plane through index/middle/ring/pinky MCPs). "
            "0° = thumb lies flat in same plane as fingers. "
            "**Coplanar dist** = signed distance of thumb tip from that plane."
        )

    with tab_oc:
        # Use the ramped F285 signal (same as process data)
        # Right: positive [0, 0.8], Left: negative [0, -0.8]
        _oc_right =  _right_f285            # 0=open, ramp up, 0.8=closed
        _oc_left  = -_left_f285             # 0=open, ramp down, -0.8=closed

        fig_oc = go.Figure()
        # Ramped signal — Right (up)
        fig_oc.add_trace(go.Scatter(
            x=frames_idx, y=_oc_right,
            name="Right (F285)", fill="tozeroy",
            fillcolor="rgba(68,136,255,0.25)",
            line=dict(color="#4488ff", width=2),
        ))
        # Ramped signal — Left (down)
        fig_oc.add_trace(go.Scatter(
            x=frames_idx, y=_oc_left,
            name="Left (F285)", fill="tozeroy",
            fillcolor="rgba(255,68,68,0.25)",
            line=dict(color="#ff4444", width=2),
        ))
        # Binary closed markers as step background
        fig_oc.add_trace(go.Scatter(
            x=frames_idx, y=_r_closed_bin.astype(float) * 0.8,
            name="R binary closed", line=dict(color="#aaccff", width=1, dash="dot", shape="hv"),
            opacity=0.5,
        ))
        fig_oc.add_trace(go.Scatter(
            x=frames_idx, y=-_l_closed_bin.astype(float) * 0.8,
            name="L binary closed", line=dict(color="#ffaaaa", width=1, dash="dot", shape="hv"),
            opacity=0.5,
        ))
        fig_oc.add_hline(y=0,    line=dict(color="rgba(255,255,255,0.3)", width=1))
        fig_oc.add_hline(y=0.8,  line=dict(color="#4488ff", dash="dash", width=1),
                         annotation_text="R max (0.8)", annotation_position="top right")
        fig_oc.add_hline(y=-0.8, line=dict(color="#ff4444", dash="dash", width=1),
                         annotation_text="L max (0.8)", annotation_position="bottom right")
        fig_oc.add_vline(x=t_dbg, line=dict(color="yellow", dash="dash", width=2),
                         annotation_text=f"t={t_dbg}", annotation_position="top left")
        fig_oc.update_layout(
            xaxis_title="Frame",
            yaxis=dict(
                tickvals=[-0.8, -0.4, 0, 0.4, 0.8],
                ticktext=["L 0.8", "L 0.4", "open", "R 0.4", "R 0.8"],
                range=[-0.95, 0.95],
            ),
            paper_bgcolor="rgb(15,15,25)", plot_bgcolor="rgb(25,25,40)", font_color="white",
            legend=dict(bgcolor="rgba(20,20,40,0.8)", orientation="h", y=1.12, x=0),
            height=320, margin=dict(l=0, r=80, t=50, b=30),
        )
        st.plotly_chart(fig_oc, use_container_width=True, key="f285_oc_chart")
        st.caption(
            "**Blue (up)** = right F285 qpos — ramp k={k} frames trước close, max 0.8 khi closed. "
            "**Red (down)** = left F285. Đường dot = binary closed step (không có ramp).".format(k=int(f285_close_ramp_k))
        )

    # ---- Per-finger table at current frame ----
    with st.expander(f"Per-finger values at frame {t_dbg}"):
        import pandas as pd
        rows_frame = []
        for hand_lbl, reach_arr, pip_arr, mean_arr, elev_arr, dist_arr, open_ratio in [
            ("Right", _right_reach, _right_pip, _right_mean, _right_elev, _right_dist, f285_open_ratio_right),
            ("Left",  _left_reach,  _left_pip,  _left_mean,  _left_elev,  _left_dist,  f285_open_ratio_left),
        ]:
            row = {"hand": hand_lbl}
            for fi, fname in enumerate(_FINGER_NAMES):
                row[f"{fname} reach"] = f"{reach_arr[t_dbg, fi]:.3f}"
                row[f"{fname} PIP°"]  = f"{pip_arr[t_dbg, fi]:.1f}"
            row["mean reach"] = f"{mean_arr[t_dbg]:.3f}"
            row[f"≤{open_ratio:.2f}?"] = "✓" if mean_arr[t_dbg] <= open_ratio else "✗"
            row["thumb dist m"] = f"{dist_arr[t_dbg]:.4f}"
            row["≤0.04m?"] = "✓" if dist_arr[t_dbg] <= thumb_dist_thresh else "✗"
            row["thumb elev°"] = f"{elev_arr[t_dbg]:.1f}"
            row["GRASP"] = "✓" if (mean_arr[t_dbg] <= open_ratio and dist_arr[t_dbg] <= thumb_dist_thresh) else "✗"
            rows_frame.append(row)
        st.dataframe(pd.DataFrame(rows_frame).set_index("hand"))

    with st.expander("Reach ratio stats (this episode)"):
        import pandas as pd
        pcts = [0, 5, 25, 50, 75, 95, 100]
        rows_stat = []
        for fi, fname in enumerate(_FINGER_NAMES):
            rows_stat.append({
                "finger": fname,
                **{f"R p{p}": float(np.percentile(_right_reach[:, fi], p)) for p in pcts},
                **{f"L p{p}": float(np.percentile(_left_reach[:,  fi], p)) for p in pcts},
            })
        rows_stat.append({
            "finger": "MEAN",
            **{f"R p{p}": float(np.percentile(_right_mean, p)) for p in pcts},
            **{f"L p{p}": float(np.percentile(_left_mean,  p)) for p in pcts},
        })
        st.dataframe(pd.DataFrame(rows_stat).set_index("finger").style.format("{:.3f}"))
        st.caption(
            "Set **open ratio** ≈ MEAN R p95 (relaxed open hand). "
            "Set **closed ratio** ≈ MEAN R p5 (tightest cup grasp)."
        )

except Exception as _e:
    st.warning(f"F285 computation failed: {_e}")

# ---------------------------------------------------------------------------
# Debug expanders — full-width below columns, reads current frame from state
# ---------------------------------------------------------------------------
if use_calib_file and calib_ok:
    _T_q2c   = np.load(calib_path)
    _vr2cam0 = compute_calib_vr2cam(_T_q2c, ep["head_pose_mat"][0].astype(np.float64))
    _L_dbg   = calib_cam_joints(L_world[t_dbg], _vr2cam0,
                                dx_mm=calib_dx, dy_mm=calib_dy, depth_scale=calib_depth_scale)
    _R_dbg   = calib_cam_joints(R_world[t_dbg], _vr2cam0,
                                dx_mm=calib_dx, dy_mm=calib_dy, depth_scale=calib_depth_scale)
else:
    if fixed_cam:
        _W2C = build_w2c(cam_pos_vr, cam_euler, conv_key)
    else:
        _T_off = np.eye(4, dtype=np.float64)
        _T_off[:3, 3] = cam_offset
        _T_cam = head_mats[t_dbg] @ _T_off
        _vm = VR2CAM_SAME if conv_key == "same_dir" else VR2CAM_FACING
        _W2C = _vm @ fast_mat_inv(_T_cam)
    _L_dbg = apply_w2c(L_world[t_dbg], _W2C)
    _R_dbg = apply_w2c(R_world[t_dbg], _W2C)

with st.expander("Debug — camera-frame positions (tune calibration here)"):
    st.markdown(
        "Joints should have **Z > 0** (in front of camera) and project to u∈[0,1280], v∈[0,720].\n\n"
        "Tune **Camera position** sliders until wrist joints land within the image."
    )
    for lbl, jmat in [("Left palm cam XYZ", _L_dbg[0]), ("Right palm cam XYZ", _R_dbg[0])]:
        xyz = jmat[:3, 3]
        if xyz[2] > 0.01:
            u = K[0, 0] * xyz[0] / xyz[2] + K[0, 2]
            v = K[1, 1] * xyz[1] / xyz[2] + K[1, 2]
            st.write(f"**{lbl}**: x={xyz[0]:.3f}, y={xyz[1]:.3f}, z={xyz[2]:.3f} → u={u:.0f}, v={v:.0f}")
        else:
            st.write(f"**{lbl}**: z={xyz[2]:.3f} (behind camera or invalid)")

with st.expander("Episode info"):
    st.json({
        "episode_dir": selected_ep,
        "n_frames": T,
        "n_joints_per_hand": n_joints,
        "camera_K": K.tolist(),
        "head_pos_t0": head0_pos.tolist(),
        "left_hand_mat_shape": list(ep["left_hand_mat"].shape),
    })

with st.expander(f"Joint positions at frame {t_dbg}"):
    import pandas as pd
    for hand_label, joints in [("Right", R_rel[t_dbg]), ("Left", L_rel[t_dbg])]:
        st.markdown(f"**{hand_label} hand** (head-relative at T=0, VR frame)")
        pos    = joints[:, :3, 3]
        rotvec = np.array([Rotation.from_matrix(joints[j, :3, :3]).as_rotvec()
                           for j in range(n_joints)])
        df = pd.DataFrame(
            np.concatenate([pos, rotvec], axis=1),
            columns=["x (right)", "y (fwd)", "z (up)", "rx", "ry", "rz"],
            index=JOINT_NAMES[:n_joints],
        )
        st.dataframe(df.style.format("{:.4f}"), height=280)
