# MCAP Human Data Collection

Hướng dẫn thu thập human VR data dùng format MCAP thay cho SVO2 + episode.pkl.

**Điểm khác biệt chính so với pipeline cũ:**
- Một file `.mcap` duy nhất chứa đầy đủ: hand pose, head pose, camera frames, và calibration
- Hỗ trợ bất kỳ camera nào có driver Python (RealSense, GoPro, webcam...) — không bắt buộc ZED
- Calibration được embed vào mỗi file, không cần quản lý file `.npy` riêng

---

## Tổng quan các bước

```
Step 1: Calibrate camera ← chỉ làm 1 lần khi đổi setup
Step 2: Thu data         ← làm mỗi session
Step 3: Convert sang zarr
Step 4: Train (giữ nguyên như cũ)
```

---

## Step 1: Camera Calibration

Mục đích: tính ma trận `quest2camera` (4×4) — ánh xạ tọa độ tay từ Quest space sang Camera frame.

**Chuẩn bị:**
- In checkerboard `human_data/calib.io_checker_270x180_6x9_30.pdf` (6×9 ô, ô vuông 30mm)
- Cắm camera vào máy tính
- Bật Quest và mở app Unity teleop

### Step 1.1 — Thu ảnh checkerboard + anchor Quest

```bash
python -m scripts_data.vr_calibration_device_data_collection \
    --camera realsense \
    --camera_params_dir camera_params \
    --checkboard_h 6 \
    --checkboard_w 9 \
    --square_size 30 \
    -n 1
```

Script này chạy 2 giai đoạn liên tiếp:

**Giai đoạn 1 — Anchor camera:**
1. Cửa sổ camera hiện lên
2. Đặt checkerboard vào vị trí cố định trong scene, đảm bảo camera nhìn thấy rõ toàn bộ checkerboard
3. Nhấn `S` để chụp ảnh → script tự detect các góc checkerboard
4. Nhấn `S` lần 2 để xác nhận

Output files trong `camera_params/quest_realsense/`:
```
camera_intrinsic.npy              ← intrinsic matrix K (3×3)
camera_distoration.npy            ← distortion coefficients
camera_resolution.npy             ← (width, height)
calib_camera_image_base_point.npy ← pixel coordinates của các góc
calib_camera_image_base_point_obj.npy ← 3D coordinates tương ứng
```

**Giai đoạn 2 — Anchor Quest:**
1. Đứng trước checkerboard, nhìn vào checkerboard qua Quest
2. Trong app Unity: nhấn nút **WorldFrame** khi Quest đang nhìn đúng vào checkerboard
3. Script nhận `n=1` anchor point rồi tự dừng

Output files:
```
calib_quest_base_point.json  ← vị trí checkerboard trong Quest space
calib_quest_head.json        ← vị trí headset tại thời điểm đó
```

> **Lưu ý:** Không di chuyển checkerboard giữa giai đoạn 1 và 2.

---

### Step 1.2 — Tính ma trận quest2camera

```bash
python -m scripts_data.vr_calibration_device_result_calculation \
    --camera realsense \
    --camera_params_dir camera_params \
    -i camera_intrinsic.npy \
    -b 1 \
    --resolution_resize 1280x720 \
    --resolution_crop 640x480
```

Giải thích các flag:
- `-b 1` — corner ID của checkerboard mà Quest nhìn vào (thường là `1` cho góc gần nhất)
- `--resolution_resize` — resolution camera chụp raw
- `--resolution_crop` — resolution sau khi crop (dùng trong training)

Output:
```
camera_params/quest_realsense/calib_result_quest2camera.npy  ← ma trận 4×4 cần dùng
```

Script in ra reprojection error — nếu error > 5 pixel thì nên làm lại Step 1.1.

---

## Step 2: Thu Data

### Chạy recorder

```bash
python -m human_data.mcap_data_recorder \
    --calib_path camera_params/quest_realsense/calib_result_quest2camera.npy \
    --output_dir data/mcap_sessions/human_se_pick_apple \
    --camera realsense \
    --frequency 20 \
    --jpeg_quality 90
```

Giải thích các flag:
| Flag | Mô tả | Default |
|---|---|---|
| `--calib_path` | File `calib_result_quest2camera.npy` từ Step 1.2 | bắt buộc |
| `--output_dir` | Thư mục lưu file `.mcap` | bắt buộc |
| `--camera` | `realsense` hoặc `zed` | `realsense` |
| `--frequency` | Hz thu data (Quest + camera) | 20 |
| `--jpeg_quality` | Chất lượng nén ảnh (1–100), thấp hơn = file nhỏ hơn | 90 |

**Quy trình mỗi episode:**

```
[Quest app]   Nhấn Start  →  recorder bắt đầu ghi vào file YYYY-MM-DD-HH-MM-SS.mcap
[thực hiện]   Làm demo task
[Quest app]   Nhấn Save   →  file đóng lại, sẵn sàng cho episode tiếp theo
              Nhấn Cancel →  file bị xóa (episode bị discard)
```

Mỗi episode = 1 file `.mcap` riêng biệt chứa toàn bộ data.

---

### Cấu trúc file MCAP

Mỗi file `.mcap` chứa:

```
/quest/left_hand   — JSON array (26 joints × 6 floats), ở Quest frequency
/quest/right_hand  — tương tự
/quest/head        — JSON array (6 floats: x,y,z,rx,ry,rz)
/camera/rgb        — JPEG bytes, ở camera frequency

[attachment] calib/quest2camera  — ma trận 4×4 float64 (embed 1 lần khi start)
[attachment] camera/intrinsics   — {"fx","fy","cx","cy"}
```

---

## Step 3: Convert sang Zarr

```bash
python -m scripts_data.entry.zarr_mcap_data_conversion \
    -i data/mcap_sessions/human_se_pick_apple \
    -o data/zarr_data/zarr_data_human \
    -ins "pick_apple" \
    -m o \
    --resolution_resize 640x480 \
    --resolution_crop 640x480 \
    --resolution_image_final 224x224 \
    --hand_shrink_coef 1.25
```

Giải thích các flag:
| Flag | Mô tả | Default |
|---|---|---|
| `-i` | Thư mục chứa file `.mcap` | bắt buộc |
| `-o` | Thư mục zarr output | bắt buộc |
| `-ins` | Tên task (dùng trong training) | bắt buộc |
| `-m` | `o` = chỉ RGB, `p` = có pointcloud, `s` = stereo | `o` |
| `--resolution_resize` | Resize ảnh từ raw resolution về | `640x480` |
| `--resolution_crop` | Crop sau resize | `640x480` |
| `--resolution_image_final` | Final size input vào model | `224x224` |
| `--hand_shrink_coef` | Scale finger movement (1.0 = raw, 1.25 = phóng to) | `1.25` |
| `-cf` | Fallback calib `.npy` nếu MCAP không có embedded calib | `None` |

**Về calib trong convert:**
- Script **tự đọc `quest2camera` từ mỗi file MCAP** — không cần `-cf` nếu file được thu đúng cách
- `-cf` chỉ cần khi convert file MCAP cũ không có embedded calib

Output:
```
data/zarr_data/zarr_data_human/human_se_pick_apple_640x480_640x480+pick_apple+.zarr
```

---

## Step 4: Training

Giữ nguyên như pipeline gốc:

```bash
bash scripts/dp_base_cotraining.sh
```

Trỏ `human_dataset_path` trong config về zarr path vừa tạo ở Step 3.

---

## Cấu trúc thư mục sau khi hoàn thành

```
motiontrans/
├── camera_params/
│   └── quest_realsense/
│       ├── camera_intrinsic.npy
│       ├── calib_result_quest2camera.npy   ← kết quả Step 1
│       └── ...
├── data/
│   ├── mcap_sessions/
│   │   └── human_se_pick_apple/
│   │       ├── 2026-06-26-10-00-00.mcap   ← mỗi file = 1 episode
│   │       ├── 2026-06-26-10-01-30.mcap
│   │       └── ...
│   └── zarr_data/
│       └── zarr_data_human/
│           └── human_se_pick_apple_640x480_640x480+pick_apple+.zarr
└── ...
```

---

## Troubleshooting

**Reprojection error cao (> 5 px) sau Step 1.2:**
- Checkerboard bị di chuyển giữa 2 giai đoạn → làm lại từ đầu
- Ánh sáng không đều → chụp ảnh ở chỗ sáng hơn
- Thử flag `-r` (reverse corner order) trong Step 1.2

**Quest không gửi data:**
- Kiểm tra IP trong `human_data/ip_config.py` khớp với máy tính
- Port mặc định: `12346`

**File MCAP quá lớn:**
- Giảm `--jpeg_quality` xuống `70–80` (chất lượng đủ cho training)
- Giảm `--frequency` xuống `15`

**Lỗi `No quest2camera in MCAP`:**
- File MCAP được thu mà không có `--calib_path` → cung cấp `-cf path/to/calib_result_quest2camera.npy` khi convert
