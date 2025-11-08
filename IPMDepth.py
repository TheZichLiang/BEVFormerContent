import cv2
import os, sys
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

# Add NuScenes SDK
sys.path.append("nuscenes-devkit/python-sdk")
from nuscenes.nuscenes import NuScenes


# Helper: build BEV warp matrix for one calibrated camera
def build_ipm_matrix(calibrated_sensor, K):
    # Extrinsic parameters: sensor->ego (at current time)
    R_se = Quaternion(calibrated_sensor["rotation"]).rotation_matrix  # cam→ego
    t_se = np.array(calibrated_sensor["translation"], dtype=np.float64).reshape(3, 1)
    # We need the transformation from world (ego) to camera.
    # Given x_ego = R_se x_cam + t_se,
    # then x_cam = R_se^T (x_ego - t_se).
    R_we_to_cam = R_se.T
    t_we_to_cam = -R_se.T @ t_se
    # 对地面平面 Z=0：X_ego = [X, Y, 0]^T
    # 平面→图像的单应矩阵：
    # p ~ K [r1 r2 t] [X Y 1]^T，其中 r1,r2 为 R_we_to_cam 的前两列
    H_plane_to_img = K @ np.hstack((R_we_to_cam[:, :2], t_we_to_cam))
    H_img_to_plane = np.linalg.inv(H_plane_to_img)
    # Plane (meters) → BEV pixels
    S_plane_to_bev = np.array([
        [ 0.0,       -1.0/mpp,  -y_min/mpp],
        [-1.0/mpp,    0.0,       x_max/mpp],
        [ 0.0,        0.0,       1.0     ]
    ], dtype=np.float64)
    M_img_to_bev = S_plane_to_bev @ H_img_to_plane
    return M_img_to_bev

# Data Path
data_dir = "/home/fzlcentral/BEVFormer/data/nuscenes"
nusc = NuScenes(version="v1.0-mini", dataroot=data_dir, verbose=True)

# Output folder
output_dir = "/home/fzlcentral/BEVFormerProj/ipm_outputs"
os.makedirs(output_dir, exist_ok=True)

# Camera keys
camera_keys = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"
]

# BEV range in ego frame (meters)
x_min, x_max = 0.0, 50.0     # forward range
y_min, y_max = -20.0, 20.0   # left/right range
mpp = 0.10                   # meters per pixel (resolution)

# BEV size in pixels
bev_w = int(round((y_max - y_min) / mpp))   # width (left-right)
bev_h = int(round((x_max - x_min) / mpp))   # height (front-back)

# Main logic for 3 different scenes first sample and all 6 cameras
for scene_idx, scene in enumerate(tqdm(nusc.scene[:3], desc="Scenes")):
    sample_token = scene["first_sample_token"]
    sample = nusc.get("sample", sample_token)
    print(f"\nProcessing scene {scene_idx}: {scene['name']}")

    scene_folder = os.path.join(output_dir, f"scene_{scene_idx:03d}_{scene['name']}")
    os.makedirs(scene_folder, exist_ok=True)
    # Loop over cameras in this sample
    for camera_key in tqdm(camera_keys, desc=f"Scene {scene_idx} cameras"):
        print("Processing camera:", camera_key)
        # Get sample data for this camera
        cam_data = nusc.get("sample_data", sample["data"][camera_key])
        # Get calibrated sensor data
        calibrated_sensor_cam = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        # Intrinsic matrix (3x3)
        K = np.array(calibrated_sensor_cam["camera_intrinsic"], dtype=np.float64)

        # Load image
        img_path = os.path.join(data_dir, cam_data["filename"])
        if not os.path.exists(img_path):
            print(f"⚠️ Missing {img_path}")
            continue
        print("image--", img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image at {img_path}")
            continue
        h, w, _ = img.shape
        print(f"Camera {camera_key}: {img.shape}")
    
        # Build IPM matrix
        # Build IPM transform & warp
        M_img_to_bev = build_ipm_matrix(calibrated_sensor_cam, K)

        # Apply the warp
        ipm_image = cv2.warpPerspective(img, M_img_to_bev, (bev_w, bev_h), flags=cv2.INTER_LINEAR)
        print(ipm_image.shape)
        # Save output   
        cv2.imwrite(os.path.join(scene_folder, f"{camera_key}_ipm.jpg"), ipm_image)
        # save original image for reference
        cv2.imwrite(os.path.join(scene_folder, f"{camera_key}_orig.jpg"), img)
    print(f"✅ Finished scene {scene_idx}: saved to {scene_folder}")
