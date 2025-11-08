import cv2
import os, sys
import numpy as np
from pyquaternion import Quaternion

# Add NuScenes SDK
sys.path.append("nuscenes-devkit/python-sdk")
from nuscenes.nuscenes import NuScenes

# Camera keys
camera_keys = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"
]

# BEV 范围（在 ego 坐标系中，单位米）
# X: 前方(+x)，Y: 左右(+y)
x_min, x_max = 0.0, 50.0     # 前后方向范围（米）
y_min, y_max = -20.0, 20.0   # 左右方向范围（米）
mpp = 0.10                   # 每个像素代表的米数（分辨率）

data_dir = "/home/fzlcentral/BEVFormer/data/nuscenes"
nusc = NuScenes(version="v1.0-mini", dataroot=data_dir, verbose=True)

sample_token = nusc.scene[0]["first_sample_token"]
sample = nusc.get("sample", sample_token)

# 加载相机图像
print(f"Processing {camera_keys[1]}...")
cam_data = nusc.get("sample_data", sample["data"][camera_keys[1]])
calibrated_sensor_cam = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

img_path = os.path.join(data_dir, cam_data["filename"])
print("image--", img_path)
img = cv2.imread(img_path)
h, w, _ = img.shape
print(f"Camera {camera_keys[1]}: {img.shape}")

# 内参矩阵 (3x3)
K = np.array(calibrated_sensor_cam["camera_intrinsic"], dtype=np.float64)

# 外参：传感器->ego（在当前时刻）
R_se = Quaternion(calibrated_sensor_cam["rotation"]).rotation_matrix   #3x3
t_se = np.array(calibrated_sensor_cam["translation"], dtype=np.float64).reshape(3,1)

# 我们需要世界(ego)→相机的变换。
# 已知 x_ego = R_se x_cam + t_se，
# 则 x_cam = R_se^T (x_ego - t_se)。
R_we_to_cam = R_se.T
t_we_to_cam = -R_se.T @ t_se  # 3x1


# 对地面平面 Z=0：X_ego = [X, Y, 0]^T
# 平面→图像的单应矩阵：
# p ~ K [r1 r2 t] [X Y 1]^T，其中 r1,r2 为 R_we_to_cam 的前两列
H_plane_to_img = K @ np.hstack((R_we_to_cam[:, :2], t_we_to_cam))

# 取逆得到 图像→平面（单位米）
H_img_to_plane = np.linalg.inv(H_plane_to_img)

# 将地面平面坐标（米）映射到 BEV 像素坐标
bev_w = int(round((y_max - y_min) / mpp))   # 宽度（左右方向）
bev_h = int(round((x_max - x_min) / mpp))   # 高度（前后方向）

# 定义地面米→BEV 像素的变换：
# u' = (-Y - y_min)/mpp
# v' = (x_max - X)/mpp   （这样前方在图像顶部）
S_plane_to_bev = np.array([
    [ 0.0,        -1.0/mpp,  -y_min/mpp],
    [-1.0/mpp,    0.0,       x_max/mpp],
    [ 0.0,        0.0,       1.0      ]
], dtype=np.float64)

# 最终映射矩阵：图像像素 → BEV 像素
M_img_to_bev = S_plane_to_bev @ H_img_to_plane

ipm_image = cv2.warpPerspective(img, M_img_to_bev, (bev_w, bev_h), flags=cv2.INTER_LINEAR)

print(ipm_image.shape)

cv2.imwrite('ipm_fr.jpg', ipm_image)
cv2.imwrite('ipm1_fr.jpg', img)