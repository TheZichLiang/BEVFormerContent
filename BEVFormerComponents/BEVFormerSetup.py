"""
Minimal BEVFormer demo:
- ResNet50+FPN backbone
- SpatialCrossAttention
- Projects BEV grid → world → camera → FPN feature maps
- Saves BEV heatmaps per camera
"""
import os, sys
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
from BEVFormerBackbone import ResNetFPNBackbone
import torch
from torchvision import transforms
import cv2

if not hasattr(np, "long"):
    np.long = np.int_

# BEVFormer package paths
sys.path.append("/home/fzlcentral/BEVFormer")
sys.path.append("/home/fzlcentral/BEVFormer/projects")
sys.path.append("/home/fzlcentral/BEVFormer/projects/mmdet3d_plugin")
from projects.mmdet3d_plugin.bevformer.modules.spatial_cross_attention import SpatialCrossAttention

# NuScenes
sys.path.append("nuscenes-devkit/python-sdk")
from nuscenes.nuscenes import NuScenes

# Dataset
data_dir = "/home/fzlcentral/BEVFormer/data/nuscenes"
nusc = NuScenes(version="v1.0-mini", dataroot=data_dir, verbose=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNetFPNBackbone(pretrained=True).to(device)
model.eval()

# ImageNet preproc
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 704)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
])

# Output folder
output_dir = "/home/fzlcentral/BEVFormerProj/bev_former_outputs"
os.makedirs(output_dir, exist_ok=True)

camera_keys = [
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"
]

# BEV bounds in ego (meters)
x_min, x_max = -20.0, 50.0     # forward/back distance
y_min, y_max = -40.0, 40.0     # left/right width
mpp = 0.1

bev_h = int(round((x_max - x_min) / mpp))
bev_w = int(round((y_max - y_min) / mpp))
bev_dim = 256

# Learnable BEV embeddings
bev_queries = torch.nn.Parameter(torch.randn(1, bev_h * bev_w, bev_dim, device=device))

# Loop scenes
for scene_idx, scene in enumerate(tqdm(nusc.scene[:1], desc="Scenes")):
    sample = nusc.get("sample", scene["first_sample_token"])
    print(f"\nProcessing scene {scene_idx}: {scene['name']}")

    scene_folder = os.path.join(output_dir, f"scene_{scene_idx:03d}_{scene['name']}")
    os.makedirs(scene_folder, exist_ok=True)

    for camera_key in camera_keys:
        cam_data = nusc.get("sample_data", sample["data"][camera_key])
        img_path = os.path.join(data_dir, cam_data["filename"])
        if not os.path.exists(img_path):
            print(f"⚠ Missing {img_path}")
            continue

        # Load image → preprocess
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img_rgb).unsqueeze(0).to(device)

        # FPN features
        with torch.no_grad():
            fpn_features = model(img_tensor)

        # Camera extrinsics (sensor→ego)
        calib = nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])
        K = torch.tensor(calib["camera_intrinsic"], dtype=torch.float32, device=device)
        R_cam2ego = torch.tensor(Quaternion(calib["rotation"]).rotation_matrix, dtype=torch.float32, device=device)
        t_cam2ego = torch.tensor(calib["translation"], dtype=torch.float32, device=device)

        # Ego pose: ego→world
        ego_pose = nusc.get("ego_pose", cam_data["ego_pose_token"])
        R_world_ego = torch.tensor(Quaternion(ego_pose["rotation"]).rotation_matrix, dtype=torch.float32, device=device)
        t_world_ego = torch.tensor(ego_pose["translation"], dtype=torch.float32, device=device)

        # Create BEV grid in ego coords
        xs = torch.linspace(x_min, x_max, bev_h, device=device)
        ys = torch.linspace(y_min, y_max, bev_w, device=device)
        ref_y, ref_x = torch.meshgrid(xs, ys)   # (bev_h, bev_w)
        ref_xyz = torch.stack((ref_x, ref_y, torch.zeros_like(ref_x)), dim=-1)  # (H,W,3)

        # Flatten
        pts_ego = ref_xyz.view(-1, 3)  # (HW,3)

        # Ego → World
        pts_world = (pts_ego @ R_world_ego.T) + t_world_ego

        # World → Camera
        R_ego2cam = R_cam2ego.T
        t_ego2cam = -(R_cam2ego.T @ t_cam2ego)
        pts_cam = (pts_world @ R_ego2cam.T) + t_ego2cam

        X, Y, Z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2].clamp(min=1e-6)

        # Project to pixels
        u = (K[0, 0] * X / Z) + K[0, 2]
        v = (K[1, 1] * Y / Z) + K[1, 2]

        H_img, W_img, _ = img_rgb.shape

        # === Build reference_points_cam per FPN level ===
        reference_points_cam_list = []
        for feat in [fpn_features[k] for k in ["C2","C3","C4","C5"]]:
            _, _, Hf, Wf = feat.shape
            
            # scale pixel coords to feature map coords
            u_feat = u * (Wf / W_img)
            v_feat = v * (Hf / H_img)

            # normalize to [0,1]
            u_feat_norm = (u_feat / (Wf - 1)).clamp(0, 1)
            v_feat_norm = (v_feat / (Hf - 1)).clamp(0, 1)

            reference_points_cam_list.append(torch.stack((u_feat_norm, v_feat_norm), dim=-1))  # (HW,2)

        # stack across levels → (1,1,HW,4,2)
        reference_points_cam = torch.stack(reference_points_cam_list, dim=1)
        reference_points_cam = reference_points_cam.view(1, 1, -1, 4, 2)

        # For SpatialCrossAttention, BEV reference_points (normalized)
        reference_points = torch.stack((
            (ref_x - x_min) / (x_max - x_min),
            (ref_y - y_min) / (y_max - y_min),
            torch.zeros_like(ref_x)
        ), dim=-1).view(1, -1, 1, 3)

        # FPN shapes and level start index
        mlvl_feats = [fpn_features[k] for k in ["C2","C3","C4","C5"]]
        mlvl_shapes = torch.tensor([feat.shape[-2:] for feat in mlvl_feats], dtype=torch.long, device=device)
        level_start_index = torch.cat((
            torch.zeros(1, dtype=torch.long, device=device),
            torch.cumsum(mlvl_shapes[:,0] * mlvl_shapes[:,1], dim=0)[:-1]
        ))

        query_pos = torch.zeros_like(bev_queries)
        bev_mask = [torch.zeros(bev_h * bev_w, dtype=torch.bool, device=device)]

        # Create attention module
        sca = SpatialCrossAttention(embed_dims=bev_dim, num_cams=1)

        # Forward attention
        with torch.no_grad():
            bev_output = sca(
                query=bev_queries,
                key=mlvl_feats,
                value=mlvl_feats,
                query_pos=query_pos,
                reference_points=reference_points,
                reference_points_cam=reference_points_cam,
                spatial_shapes=mlvl_shapes,
                level_start_index=level_start_index,
                bev_mask=bev_mask,
                flag='encoder'
            )

        # Visualize
        bev_img = bev_output[0].mean(1).reshape(bev_h, bev_w).detach().cpu().numpy()
        bev_img = (bev_img - bev_img.min()) / (bev_img.max() - bev_img.min() + 1e-6)
        bev_img = (bev_img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(scene_folder, f"{camera_key}_bev_attention.jpg"), bev_img)

print("\n✅ Done — BEV maps generated for all cameras!")