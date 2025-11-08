import sys
import os
import cv2
import numpy as np
from pyquaternion import Quaternion

sys.path.append("nuscenes-devkit/python-sdk")

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix

# Raw dataset
data_dir = "/home/fzlcentral/BEVFormer/data/nuscenes"
nusc = NuScenes(version='v1.0-mini', dataroot=data_dir, verbose=True)
# Get the first scene
my_scene = nusc.scene[0]
# Get the first sample in the scene
first_sample = nusc.get('sample', my_scene['first_sample_token'])

'''
Get Camera Calibration
    Retrieve the front camera sample data, its ego vehicle pose (global car position/orientation), and calibration (camera intrinsics & extrinsics).
    Build transform matrices:
        T_cam_to_ego: Camera -> Ego vehicle coordinates
        T_ego_to_global_cam: Ego -> Global coordinates
        T_cam_to_global: Camera -> Global coordinates
'''

# get the sample data for the front camera in the scene
cam_front_data = nusc.get('sample_data', first_sample['data']['CAM_FRONT'])
# get ego pose
ego_pose_cam = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
# Get calibrated sensor data
calibrated_sensor_cam = nusc.get('calibrated_sensor', cam_front_data['calibrated_sensor_token'])
# Transformation matrix from camera to ego coordinates
T_cam_to_ego = transform_matrix(calibrated_sensor_cam['translation'],
                                Quaternion(calibrated_sensor_cam['rotation']),
                                inverse=False)
# Transformation matrix from ego to global coordinates
T_ego_to_global_cam = transform_matrix(ego_pose_cam['translation'],
                                      Quaternion(ego_pose_cam['rotation']),
                                      inverse=False)
# Transformation matrix from camera to global coordinates
T_cam_to_global = T_ego_to_global_cam @ T_cam_to_ego

'''
Intrinsic camera matrix K (focal lengths + principal point).
Used for mapping between pixels (u,v) and camera rays.
'''
K = calibrated_sensor_cam['camera_intrinsic']

'''
Load image. Select a pixel (u,v) in the image.
'''
img_path = os.path.join(data_dir, cam_front_data['filename'])
img = cv2.imread(img_path)
h, w, _ = img.shape
u, v = 800,850
print(h, w)
print(u,v, K)

'''
Tranform 2D image coordinates to 3D Camera Coordinates
[x_c, y_c, z_c] = Z * K^-1 [u,v,1]

Compute ray = K^-1 [u,v,1] to get the direction of the 3D ray in camera space. Multiplying by depth
Z would give the actual 3D point. Since we don’t know
Z, I intersected this ray with the ground plane.
I could not directly compute Z from a single image pixel without forward-projecting 3D points or making an assumption.
'''
uv1 = np.array([u, v, 1.0])
ray = np.linalg.inv(np.array(K)) @ uv1
ray = ray / np.linalg.norm(ray)
print("Ray in camera coords:", ray)

'''
Transform the ray into world coordinates using rotation part of T_cam_to_global
t = camera position in world frame ray origin.
P(t)=O+t*D
'''
R = T_cam_to_global[:3, :3]
t = T_cam_to_global[:3, 3]
ray_world = R @ ray
cam_origin_world = t

print("Ray in world coords:", ray_world)

# Intersect ray with ground plane z=0
# Ray equation: P(t) = O + t*D
# Ground plane equation: z=0
# Solve for t when P_z = 0 and get the ground point P
o_z, d_z = cam_origin_world[2], ray_world[2]
if abs(d_z) < 1e-6:
    ground_point = None
else:
    t_hit = -o_z/ d_z
    if t_hit > 0:
        ground_point = cam_origin_world + t_hit * ray_world
    else:
        ground_point = None

if ground_point is None:
    print("Ray does not hit ground plane.")
else:
    print("Ground point in world coords:", ground_point[:2])
    # Map the ground point to BEV image coordinates
    ego_x, ego_y = ego_pose_cam['translation'][:2]

    # Define BEV region +-50m around ego, with resolution 0.1m/pixel.
    # Define BEV grid
    x_min, x_max = -50, 50    # forward/back range in meters
    y_min, y_max = -50, 50    # left/right range in meters
    res = 0.4                 # resolution (m per pixel)

    bev_w = int((y_max - y_min) / res)   # horizontal pixels
    bev_h = int((x_max - x_min) / res)   # vertical pixels

    bev = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # Ego car: always at bottom-center of BEV
    ego_u = bev_w // 2
    ego_v = bev_h - 1   # bottom row
    cv2.circle(bev, (ego_u, ego_v), 5, (0,255,0), -1)  # green dot

    # Convert projected world point into BEV coords
    # Reminder: in NuScenes ego frame: x = forward, y = left
    x_rel = ground_point[0] - ego_x
    y_rel = ground_point[1] - ego_y

    u_bev = int((y_rel - y_min) / res)
    v_bev = bev_h - int((x_rel - x_min) / res)   # flip so +x (forward) goes up

    # Draw projected point in red
    if 0 <= u_bev < bev_w and 0 <= v_bev < bev_h:
        cv2.circle(bev, (u_bev, v_bev), 5, (0,0,255), -1)

    # Draw grid lines every 10 m
    for x in range(0, bev_w, int(10/res)):
        cv2.line(bev, (x, 0), (x, bev_h), (50,50,50), 1)
    for y in range(0, bev_h, int(10/res)):
        cv2.line(bev, (0, y), (bev_w, y), (50,50,50), 1)

    cv2.imshow("BEV", bev)
    cv2.waitKey(0)