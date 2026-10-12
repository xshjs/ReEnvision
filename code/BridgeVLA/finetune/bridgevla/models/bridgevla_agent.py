'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
Adapted from https://github.com/NVlabs/RVT/blob/master/rvt/models/rvt_agent.py
Therefore, the code is also under the NVIDIA Source Code License

Author: Peiyan Li
Email: peiyan.li@cripac.ia.ac.cn
'''

import pprint
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import hashlib
from scipy.spatial.transform import Rotation
from torch.nn.parallel.distributed import DistributedDataParallel
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..."))
import RLBench.utils.peract_utils_rlbench as rlbench_utils
import GemBench.utils.peract_utils_gembench as gembench_utils
import bridgevla.mvt.utils as mvt_utils
import bridgevla.utils.rvt_utils as rvt_utils
from bridgevla.mvt.augmentation import apply_se3_aug_con, aug_utils
from yarr.agents.agent import ActResult
from PIL import Image, ImageDraw
import torch
import numpy as np
import os


def _hashed_text_vec(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    tokens = [tok for tok in str(text).replace("_", " ").lower().split() if tok]
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def _suffix_task_vec(text: str, dim: int) -> np.ndarray:
    vec = np.zeros((dim,), dtype=np.float32)
    digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=16).digest()
    for i, byte in enumerate(digest):
        vec[i % dim] += (float(byte) / 127.5) - 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def _language_to_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and value:
        return _language_to_text(value[0])
    return str(value)


def _cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32).reshape(-1)
    rhs = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(lhs, rhs) / denom)


class BridgeJointEnergy(nn.Module):
    def __init__(self, latent_dim: int, task_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim * 3 + task_dim),
            nn.Linear(latent_dim * 3 + task_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_latent, future_latent, motion_latent, task_vec):
        x = torch.cat([state_latent, future_latent, motion_latent, task_vec], dim=-1)
        return self.net(x).squeeze(-1)


class BridgeJointHypothesisRepairMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, future_dim: int, action_dim: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.future_head = nn.Linear(hidden_dim, future_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.future_head(h), self.action_head(h)


class BridgeJointAcceptanceMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def save_point_cloud_with_color(filename, points, colors, keypoint=None):
    """
    Save the point cloud and colors to a PLY file, automatically handling the color value range.
    :param filename: Output file name (e.g. 'point_cloud.ply')
    :param points: Point cloud coordinates (N,3) np.array
    :param colors: Color values (N,3) np.array (0-255 or 0-1)
    :param keypoint: Keypoint coordinates (3,) np.array (optional)
    """

    # Ensure data dimensions are correct
    assert points.shape[1] == 3 
    assert colors.shape[1] == 3
    
    # Automatically detect color value range and convert to 0-255
    if colors.max() <= 1.0:  # If color values are between 0-1
        colors = (colors * 255).astype(np.uint8)
    else:  # If color values are between 0-255
        colors = colors.astype(np.uint8)
    
    # Add keypoint (optional)
    if keypoint is not None:
        points = np.vstack([points, keypoint])
        colors = np.vstack([colors, np.array([255, 0, 0])])  # Mark keypoint in red

    # Write to PLY file
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for pt, clr in zip(points, colors):
            f.write(f"{pt[0]} {pt[1]} {pt[2]} {int(clr[0])} {int(clr[1])} {int(clr[2])}\n")


def visualize_images(
    color_tensor: torch.Tensor,  #  (3, 3, 224, 224) 
    gray_tensor: torch.Tensor,   #  (224, 224, 3) 
    save_dir: str = "/opt/tiger/3D_OpenVLA/3d_policy/RVT/rvt_our/debug"
) -> None:
    """
    1. original_0.png, original_1.png, original_2.png   (original image)
    2. gray_0.png, gray_1.png, gray_2.png              (gray image)
    3. overlay_0.png, overlay_1.png, overlay_2.png     (transparent image + annotation)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    color_imgs = color_tensor.cpu().numpy().transpose(0, 2, 3, 1) 
    gray_imgs = gray_tensor.cpu().numpy().transpose(2, 0, 1)     
    
    for i in range(3):

        original_img = np.clip(color_imgs[i], 0, 1) * 255
        original_img = original_img.astype(np.uint8)
        Image.fromarray(original_img).save(os.path.join(save_dir, f"original_{i}.png"))
        

        gray_img = np.clip(gray_imgs[i], 0, 1) * 255
        gray_img = gray_img.astype(np.uint8)
        Image.fromarray(gray_img, mode="L").save(os.path.join(save_dir, f"gray_{i}.png"))
        

        rgba = np.zeros((*original_img.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = original_img  
        rgba[..., 3] = 77            
        
    
        overlay_img = Image.fromarray(rgba, mode="RGBA")
        draw = ImageDraw.Draw(overlay_img)
        
        
        max_pos = np.unravel_index(gray_imgs[i].argmax(), gray_imgs[i].shape)
        x = max_pos[1]  
        y = max_pos[0]  
        
      
        point_radius = 5
        draw.ellipse(
            [x-point_radius, y-point_radius, x+point_radius, y+point_radius],
            fill=(255, 0, 0, 255)  
        )
        
        overlay_img.save(os.path.join(save_dir, f"overlay_{i}.png"))


def apply_channel_wise_softmax(gray_tensor):
    """
    Apply softmax normalization independently to each grayscale channel
    Input shape: (H, W, C) -> Output shape: (H, W, C)
    All elements in each channel are processed by softmax and sum to 1
    """
    # Convert to PyTorch tensor (if not already)
    if not isinstance(gray_tensor, torch.Tensor):
        gray_tensor = torch.tensor(gray_tensor, dtype=torch.float32)
    
    # Separate each channel (C, H, W)
    channels = gray_tensor.permute(2, 0, 1)
    
    # Apply softmax to each channel and flatten
    softmax_channels = []
    for c in range(channels.shape[0]):
        channel = channels[c].flatten()
        softmax_channel = torch.softmax(channel, dim=0)
        softmax_channels.append(softmax_channel.view_as(channels[c]))
    
    # Merge channels and restore original shape (H, W, C)
    return torch.stack(softmax_channels, dim=2)


def eval_con(gt, pred):
    assert gt.shape == pred.shape, print(f"{gt.shape} {pred.shape}")
    assert len(gt.shape) == 2
    dist = torch.linalg.vector_norm(gt - pred, dim=1)
    return {"avg err": dist.mean()}


def eval_con_cls(gt, pred, num_bin=72, res=5, symmetry=1):
    """
    Evaluate continuous classification where floating point values are put into
    discrete bins
    :param gt: (bs,)
    :param pred: (bs,)
    :param num_bin: int for the number of rotation bins
    :param res: float to specify the resolution of each rotation bin
    :param symmetry: degrees of symmetry; 2 is 180 degree symmetry, 4 is 90
        degree symmetry
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) in [0, 1], gt
    assert num_bin % symmetry == 0, (num_bin, symmetry)
    gt = torch.tensor(gt)
    pred = torch.tensor(pred)
    num_bin //= symmetry
    pred %= num_bin
    gt %= num_bin
    dist = torch.abs(pred - gt)
    dist = torch.min(dist, num_bin - dist)
    dist_con = dist.float() * res
    return {"avg err": dist_con.mean()}


def eval_cls(gt, pred):
    """
    Evaluate classification performance
    :param gt_coll: (bs,)
    :param pred: (bs,)
    """
    assert gt.shape == pred.shape
    assert len(gt.shape) == 1
    return {"per err": (gt != pred).float().mean()}


def eval_all(
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    eval_trans = []
    eval_rot_x = []
    eval_rot_y = []
    eval_rot_z = []
    eval_grip = []
    eval_coll = []

    for i in range(bs):
        eval_trans.append(
            eval_con(wpt[i : i + 1], pred_wpt[i : i + 1])["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        euler_gt = Rotation.from_quat(action_rot[i]).as_euler("xyz", degrees=True)
        euler_pred = Rotation.from_quat(pred_rot_quat[i]).as_euler("xyz", degrees=True)

        eval_rot_x.append(
            eval_con_cls(euler_gt[0], euler_pred[0], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_y.append(
            eval_con_cls(euler_gt[1], euler_pred[1], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )
        eval_rot_z.append(
            eval_con_cls(euler_gt[2], euler_pred[2], num_bin=360, res=1)["avg err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_grip.append(
            eval_cls(
                action_grip_one_hot[i : i + 1].argmax(-1),
                grip_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
            .item()
        )

        eval_coll.append(
            eval_cls(
                action_collision_one_hot[i : i + 1].argmax(-1),
                collision_q[i : i + 1].argmax(-1),
            )["per err"]
            .cpu()
            .numpy()
        )

    return eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll


def manage_eval_log(
    self,
    tasks,
    wpt,
    pred_wpt,
    action_rot,
    pred_rot_quat,
    action_grip_one_hot,
    grip_q,
    action_collision_one_hot,
    collision_q,
    reset_log=False,
):
    bs = len(wpt)
    assert wpt.shape == (bs, 3), wpt
    assert pred_wpt.shape == (bs, 3), pred_wpt
    assert action_rot.shape == (bs, 4), action_rot
    assert pred_rot_quat.shape == (bs, 4), pred_rot_quat
    assert action_grip_one_hot.shape == (bs, 2), action_grip_one_hot
    assert grip_q.shape == (bs, 2), grip_q
    assert action_collision_one_hot.shape == (bs, 2), action_collision_one_hot
    assert collision_q.shape == (bs, 2), collision_q

    if not hasattr(self, "eval_trans") or reset_log:
        self.eval_trans = {}
        self.eval_rot_x = {}
        self.eval_rot_y = {}
        self.eval_rot_z = {}
        self.eval_grip = {}
        self.eval_coll = {}

    (eval_trans, eval_rot_x, eval_rot_y, eval_rot_z, eval_grip, eval_coll,) = eval_all(
        wpt=wpt,
        pred_wpt=pred_wpt,
        action_rot=action_rot,
        pred_rot_quat=pred_rot_quat,
        action_grip_one_hot=action_grip_one_hot,
        grip_q=grip_q,
        action_collision_one_hot=action_collision_one_hot,
        collision_q=collision_q,
    )

    for idx, task in enumerate(tasks):
        if not (task in self.eval_trans):
            self.eval_trans[task] = []
            self.eval_rot_x[task] = []
            self.eval_rot_y[task] = []
            self.eval_rot_z[task] = []
            self.eval_grip[task] = []
            self.eval_coll[task] = []
        self.eval_trans[task].append(eval_trans[idx])
        self.eval_rot_x[task].append(eval_rot_x[idx])
        self.eval_rot_y[task].append(eval_rot_y[idx])
        self.eval_rot_z[task].append(eval_rot_z[idx])
        self.eval_grip[task].append(eval_grip[idx])
        self.eval_coll[task].append(eval_coll[idx])

    return {
        "eval_trans": eval_trans,
        "eval_rot_x": eval_rot_x,
        "eval_rot_y": eval_rot_y,
        "eval_rot_z": eval_rot_z,
    }


def print_eval_log(self):
    logs = {
        "trans": self.eval_trans,
        "rot_x": self.eval_rot_x,
        "rot_y": self.eval_rot_y,
        "rot_z": self.eval_rot_z,
        "grip": self.eval_grip,
        "coll": self.eval_coll,
    }

    out = {}
    for name, log in logs.items():
        for task, task_log in log.items():
            task_log_np = np.array(task_log)
            mean, std, median = (
                np.mean(task_log_np),
                np.std(task_log_np),
                np.median(task_log_np),
            )
            out[f"{task}/{name}_mean"] = mean
            out[f"{task}/{name}_std"] = std
            out[f"{task}/{name}_median"] = median

    pprint.pprint(out)

    return out


def manage_loss_log(
    agent,
    loss_log,
    reset_log,
):
    if not hasattr(agent, "loss_log") or reset_log:
        agent.loss_log = {}

    for key, val in loss_log.items():
        if key in agent.loss_log:
            agent.loss_log[key].append(val)
        else:
            agent.loss_log[key] = [val]


def print_loss_log(agent):
    out = {}
    for key, val in agent.loss_log.items():
        out[key] = np.mean(np.array(val))
    pprint.pprint(out)
    return out


class RVTAgent:
    def __init__(
        self,
        network: nn.Module,
        num_rotation_classes: int,
        stage_two: bool,
        move_pc_in_bound: bool,
        lr: float = 0.0001,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.1, 0.1, 0.1],
        transform_augmentation_rpy: list = [0.0, 0.0, 20.0],
        place_with_mean: bool = True,
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = "lamb",
        gt_hm_sigma: float = 1.5,
        img_aug: bool = False,
        add_rgc_loss: bool = False,
        scene_bounds: list = rlbench_utils.SCENE_BOUNDS,
        cameras: list = rlbench_utils.CAMERAS,
        rot_ver: int = 0,
        rot_x_y_aug: int = 2,
        log_dir="",
    ):
        self._network = network
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = 360 / self._num_rotation_classes
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._place_with_mean = place_with_mean
        self._transform_augmentation_xyz = torch.from_numpy(
            np.array(transform_augmentation_xyz)
        )
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = (
            transform_augmentation_rot_resolution
        )
        self._optimizer_type = optimizer_type
        self.gt_hm_sigma = gt_hm_sigma
        self.img_aug = img_aug
        self.add_rgc_loss = add_rgc_loss
        self.stage_two = stage_two
        self.log_dir = log_dir
        self.scene_bounds = scene_bounds
        self.cameras = cameras

        print("Cameras:",self.cameras)
        self.move_pc_in_bound = move_pc_in_bound
        self.rot_ver = rot_ver
        self.rot_x_y_aug = rot_x_y_aug

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        if isinstance(self._network, DistributedDataParallel):
            self._net_mod = self._network.module
        else:
            self._net_mod = self._network

        self.num_all_rot = self._num_rotation_classes * 3
        self._bridge_latent_heads = None
        self._bridge_latent_head_ckpt = ""
        self._bridge_standalone_gidm = None
        self._bridge_standalone_gidm_ckpt = ""
        self._joint_pair_selector = None
        self._suffix_td_bundle = None
        self._suffix_td_ckpt = ""
        self._suffix_td_apply_enabled = True
        self._suffix_td_persistent_future = None
        self._suffix_td_persistent_motion = None

    def set_suffix_td_apply_enabled(self, enabled: bool):
        self._suffix_td_apply_enabled = bool(enabled)

    def init_bridge_latent_heads(
        self,
        input_dim: int = 12288,
        latent_dim: int = 768,
        hidden_dim: int = 1024,
        ckpt_path: str = "",
    ):
        if ckpt_path:
            payload = torch.load(ckpt_path, map_location="cpu")
            meta = payload.get("meta", {})
            input_dim = int(meta.get("input_dim", input_dim))
            latent_dim = int(meta.get("latent_dim", latent_dim))
            hidden_dim = int(meta.get("hidden_dim", hidden_dim))
        else:
            payload = None
            meta = {}
        self._bridge_latent_heads = nn.ModuleDict(
            {
                "state_proj": nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "future_head": nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "motion_head": nn.Sequential(
                    nn.Linear(latent_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, latent_dim),
                ),
            }
        )
        self._bridge_latent_heads.to(self._device if hasattr(self, "_device") and self._device is not None else torch.device("cpu"))
        self._bridge_latent_heads.eval()
        self._bridge_latent_head_ckpt = ckpt_path or ""
        if ckpt_path:
            state_dict = payload.get("bridge_latent_heads", payload.get("state_dict", payload))
            self._bridge_latent_heads.load_state_dict(state_dict, strict=False)
            print(f"[bridge-latent-heads] loaded ckpt: {ckpt_path}")
            if meta:
                print(f"[bridge-latent-heads] meta: {meta}")

    def init_bridge_standalone_gidm(
        self,
        ckpt_path: str,
        t5_model_path: str,
    ):
        if not ckpt_path:
            return
        from policy_models.m_former_univla.latent_motion_tokenizer_univla import (
            UncontrolledDINOLatentActionModel,
        )

        payload = torch.load(ckpt_path, map_location="cpu")
        state_dict = payload.get("state_dict", payload)
        filtered_state = {}
        for key, value in state_dict.items():
            if not key.startswith("lam."):
                continue
            local_key = key[len("lam.") :]
            if local_key.startswith("dino_encoder."):
                continue
            filtered_state[local_key] = value
        gidm = UncontrolledDINOLatentActionModel(t5_model_path=t5_model_path)
        load_result = gidm.load_state_dict(filtered_state, strict=False)
        gidm.to(self._device if hasattr(self, "_device") and self._device is not None else torch.device("cpu"))
        gidm.eval()
        for param in gidm.parameters():
            param.requires_grad = False
        self._bridge_standalone_gidm = gidm
        self._bridge_standalone_gidm_ckpt = ckpt_path
        print(f"[bridge-standalone-gidm] loaded ckpt: {ckpt_path}")
        if getattr(load_result, "missing_keys", None):
            print(f"[bridge-standalone-gidm] missing keys: {len(load_result.missing_keys)}")
        if getattr(load_result, "unexpected_keys", None):
            print(f"[bridge-standalone-gidm] unexpected keys: {len(load_result.unexpected_keys)}")

    def init_joint_pair_selector(
        self,
        memory_npz: str,
        energy_ckpt: str = "",
        repair_ckpt: str = "",
        acceptance_ckpt: str = "",
        topk: int = 8,
        same_task_only: bool = True,
        success_only_memory: bool = True,
        min_similarity: float = 0.20,
        min_energy_improve: float = 0.05,
        max_future_shift: float = 1.50,
        max_motion_shift: float = 1.50,
        future_refine_alpha: float = 0.35,
        motion_refine_alpha: float = 0.50,
        verify_exp_max: float = 1e9,
        verify_latent_max: float = 1e9,
        verify_physical_max: float = 1e9,
        acceptance_threshold: float = 0.5,
    ):
        if not memory_npz:
            self._joint_pair_selector = None
            return
        with np.load(memory_npz, allow_pickle=True) as data:
            memory = {key: data[key] for key in data.files}
        required = ["state_latent", "future_latent", "motion_latent", "success"]
        missing = [key for key in required if key not in memory]
        if missing:
            raise ValueError(f"joint memory missing keys: {missing}")
        if success_only_memory:
            mask = np.asarray(memory["success"], dtype=np.float32) > 0.5
            for key, value in list(memory.items()):
                memory[key] = value[mask]
        energy_bundle = None
        if energy_ckpt:
            payload = torch.load(energy_ckpt, map_location="cpu")
            task_dim = int(payload["task_dim"])
            latent_dim = int(payload["latent_dim"])
            hidden_dim = int(payload["hidden_dim"])
            dropout = float(payload.get("dropout", 0.1))
            model = BridgeJointEnergy(latent_dim=latent_dim, task_dim=task_dim, hidden_dim=hidden_dim, dropout=dropout)
            model.load_state_dict(payload["model_state"])
            model.to(self._device if hasattr(self, "_device") and self._device is not None else torch.device("cpu"))
            model.eval()
            energy_bundle = {"model": model, "task_dim": task_dim, "latent_dim": latent_dim}
        repair_bundle = None
        if repair_ckpt:
            payload = torch.load(repair_ckpt, map_location="cpu")
            model = BridgeJointHypothesisRepairMLP(
                input_dim=int(payload["input_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                future_dim=int(payload["future_dim"]),
                action_dim=int(payload["action_dim"]),
                dropout=float(payload.get("dropout", 0.1)),
            )
            model.load_state_dict(payload["model_state"])
            model.to(self._device if hasattr(self, "_device") and self._device is not None else torch.device("cpu"))
            model.eval()
            repair_bundle = {
                "model": model,
                "input_dim": int(payload["input_dim"]),
                "future_dim": int(payload["future_dim"]),
                "action_dim": int(payload["action_dim"]),
                "num_tasks": int(payload.get("num_tasks", 0)),
            }
        acceptance_bundle = None
        if acceptance_ckpt:
            payload = torch.load(acceptance_ckpt, map_location="cpu")
            model = BridgeJointAcceptanceMLP(
                input_dim=int(payload["input_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                dropout=float(payload.get("dropout", 0.1)),
            )
            model.load_state_dict(payload["model_state"])
            model.to(self._device if hasattr(self, "_device") and self._device is not None else torch.device("cpu"))
            model.eval()
            acceptance_bundle = {
                "model": model,
                "input_dim": int(payload["input_dim"]),
            }
        self._joint_pair_selector = {
            "memory": memory,
            "energy": energy_bundle,
            "repair": repair_bundle,
            "acceptance": acceptance_bundle,
            "topk": int(topk),
            "same_task_only": bool(same_task_only),
            "success_only_memory": bool(success_only_memory),
            "min_similarity": float(min_similarity),
            "min_energy_improve": float(min_energy_improve),
            "max_future_shift": float(max_future_shift),
            "max_motion_shift": float(max_motion_shift),
            "future_refine_alpha": float(future_refine_alpha),
            "motion_refine_alpha": float(motion_refine_alpha),
            "verify_exp_max": float(verify_exp_max),
            "verify_latent_max": float(verify_latent_max),
            "verify_physical_max": float(verify_physical_max),
            "acceptance_threshold": float(acceptance_threshold),
        }
        print(
            f"[joint-pair-selector] loaded memory={memory_npz} "
            f"size={len(memory['success'])} energy_ckpt={energy_ckpt or '<none>'}"
        )

    def _ensure_suffix_td_bundle(self):
        if not bool(int(os.environ.get("BRIDGEVLA_ENABLE_SUFFIX_TD", "0"))):
            return None
        ckpt_path = os.environ.get("BRIDGEVLA_SUFFIX_TD_CKPT", "")
        if not ckpt_path:
            return None
        if self._suffix_td_bundle is not None and self._suffix_td_ckpt == ckpt_path:
            return self._suffix_td_bundle
        step3_path = os.environ.get("DEFI_STEP3_PATH", "/mnt/workspace/manipulation/DeFi/Step3_DeFI")
        if step3_path and step3_path not in sys.path:
            sys.path.insert(0, step3_path)
        from policy_evaluation.train_factored_belief_action_transition import (
            ActionAdaptationMLP,
            BeliefTransitionMLP,
            ExpectedTransitionMLP,
        )

        payload = torch.load(ckpt_path, map_location="cpu")
        device = self._device if hasattr(self, "_device") and self._device is not None else torch.device("cpu")
        t_model = BeliefTransitionMLP(
            state_dim=int(payload["state_dim"]),
            summary_dim=int(payload["summary_dim"]),
            action_dim=int(payload["action_dim"]),
            task_dim=int(payload["task_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload.get("dropout", 0.0)),
        )
        d_model = ActionAdaptationMLP(
            summary_dim=int(payload["summary_dim"]),
            action_dim=int(payload["action_dim"]),
            chunk_len=int(payload["chunk_len"]),
            task_dim=int(payload["task_dim"]),
            hidden_dim=int(payload["hidden_dim"]),
            dropout=float(payload.get("dropout", 0.0)),
            adaptation_input=str(payload.get("adaptation_input", "ht_hnext")),
        )
        t_model.load_state_dict(payload["t_model_state"])
        d_model.load_state_dict(payload["d_model_state"])
        f_model = None
        if bool(payload.get("has_expected_transition", False)) and "f_model_state" in payload:
            f_model = ExpectedTransitionMLP(
                state_dim=int(payload["state_dim"]),
                summary_dim=int(payload["summary_dim"]),
                action_dim=int(payload["action_dim"]),
                task_dim=int(payload["task_dim"]),
                hidden_dim=int(payload["hidden_dim"]),
                dropout=float(payload.get("dropout", 0.0)),
            )
            f_model.load_state_dict(payload["f_model_state"])
            f_model.to(device).eval()
        t_model.to(device).eval()
        d_model.to(device).eval()
        self._suffix_td_bundle = {
            "t_model": t_model,
            "d_model": d_model,
            "f_model": f_model,
            "state_dim": int(payload["state_dim"]),
            "summary_dim": int(payload["summary_dim"]),
            "action_dim": int(payload["action_dim"]),
            "chunk_len": int(payload["chunk_len"]),
            "task_dim": int(payload["task_dim"]),
        }
        self._suffix_td_ckpt = ckpt_path
        print(f"[bridge-suffix-td] loaded {ckpt_path}", flush=True)
        return self._suffix_td_bundle

    def _suffix_low_dim(self, value, state_dim: int):
        try:
            if torch.is_tensor(value):
                arr = value.detach().cpu().numpy()
            else:
                arr = np.asarray(value)
            arr = arr.astype(np.float32).reshape(-1)
        except Exception:
            arr = np.zeros((0,), dtype=np.float32)
        out = np.zeros((state_dim,), dtype=np.float32)
        if arr.size:
            out[: min(state_dim, arr.size)] = arr[: min(state_dim, arr.size)]
        return out

    def _suffix_latent_from_info(self, info, key, pooled_key, target_dim: int, device):
        value = None if info is None else info.get(key)
        if value is None and info is not None:
            value = info.get(pooled_key)
        if torch.is_tensor(value):
            tensor = value.detach().to(device=device, dtype=torch.float32)
        elif value is not None:
            tensor = torch.as_tensor(value, device=device, dtype=torch.float32)
        else:
            tensor = torch.zeros((1, max(1, target_dim // 2)), device=device)
        tensor = tensor.reshape(1, -1)
        if tensor.shape[1] > target_dim:
            tensor = tensor[:, :target_dim]
        elif tensor.shape[1] < target_dim:
            repeat = (target_dim + tensor.shape[1] - 1) // max(1, tensor.shape[1])
            tensor = tensor.repeat(1, repeat)[:, :target_dim]
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _apply_suffix_td_patch(self, action: np.ndarray, observation: dict, info: dict, language_goal: str):
        bundle = self._ensure_suffix_td_bundle()
        if bundle is None:
            return action, {}
        prev_info = observation.get("prev_transition_info", {}) if isinstance(observation, dict) else {}
        if not prev_info:
            return action, {"suffix_td_active": False, "suffix_td_reason": "no_prev_transition"}
        device = self._device if hasattr(self, "_device") and self._device is not None else torch.device("cpu")
        action_dim = int(bundle["action_dim"])
        if int(action_dim) != int(np.asarray(action).reshape(-1).shape[0]):
            return action, {"suffix_td_active": False, "suffix_td_reason": "action_dim_mismatch"}

        state_dim = int(bundle["state_dim"])
        prev_state = self._suffix_low_dim(prev_info.get("agent_low_dim_state_before"), state_dim)
        cur_state = self._suffix_low_dim(observation.get("low_dim_state"), state_dim)
        prev_action = self._suffix_low_dim(prev_info.get("agent_action"), action_dim)
        half_summary = max(1, int(bundle["summary_dim"]) // 2)
        prev_future = self._suffix_latent_from_info(prev_info, "agent_future_latent", "agent_pooled_latent", half_summary, device)
        prev_action_latent = self._suffix_latent_from_info(prev_info, "agent_motion_latent", "agent_pooled_latent", half_summary, device)
        h_t = torch.cat([prev_future, prev_action_latent], dim=-1)
        task_vec = torch.from_numpy(_suffix_task_vec(language_goal, int(bundle["task_dim"]))).to(device=device, dtype=torch.float32).unsqueeze(0)
        meta = torch.zeros((1, 3), device=device)
        state_t = torch.from_numpy(prev_state).to(device=device, dtype=torch.float32).unsqueeze(0)
        state_tp1 = torch.from_numpy(cur_state).to(device=device, dtype=torch.float32).unsqueeze(0)
        executed = torch.from_numpy(prev_action).to(device=device, dtype=torch.float32).unsqueeze(0)
        action_chunk = torch.zeros((1, int(bundle["chunk_len"]), action_dim), device=device)
        action_chunk[0, 0] = torch.from_numpy(np.asarray(action, dtype=np.float32).reshape(-1)).to(device=device)

        pred_delta = bundle["t_model"](h_t, state_t, executed, state_tp1, task_vec, meta)
        h_post = h_t + pred_delta
        if bool(int(os.environ.get("BRIDGEVLA_SUFFIX_PERSISTENT_BELIEF", "0"))):
            half_summary = h_post.shape[-1] // 2
            if half_summary > 0 and torch.isfinite(h_post).all():
                # Persist T_theta's posterior belief so the next action step can
                # condition GIDM on h_{t+1}, matching the recurrent belief view.
                self._suffix_td_persistent_future = h_post[:, :half_summary].detach().cpu()
                self._suffix_td_persistent_motion = h_post[:, half_summary:].detach().cpu()
        h_expected = None
        if bundle.get("f_model") is not None:
            expected_delta = bundle["f_model"](h_t, state_t, executed, task_vec, meta)
            h_expected = h_t + expected_delta
        d_out = bundle["d_model"](action_chunk, h_t, h_post, task_vec, meta, h_expected=h_expected)
        mode_prob = torch.softmax(d_out["mode_logits"], dim=-1)[0]
        mode = int(torch.argmax(mode_prob).item())
        patch_prob = float(mode_prob[1].item())
        replan_prob = float(mode_prob[2].item()) if mode_prob.numel() > 2 else 0.0
        min_prob = float(os.environ.get("BRIDGEVLA_SUFFIX_PATCH_MIN_PROB", "0.65"))
        mix = float(os.environ.get("BRIDGEVLA_SUFFIX_PATCH_MIX", "0.02"))
        max_abs = float(os.environ.get("BRIDGEVLA_SUFFIX_PATCH_MAX_ABS", "0.05"))
        apply_enabled = bool(getattr(self, "_suffix_td_apply_enabled", True))
        trigger_on_prob_only = bool(int(os.environ.get("BRIDGEVLA_SUFFIX_TRIGGER_ON_PROB_ONLY", "0")))
        trigger_on_replan = bool(int(os.environ.get("BRIDGEVLA_SUFFIX_PATCH_ON_REPLAN", "0")))
        force_replan_only = bool(int(os.environ.get("BRIDGEVLA_SUFFIX_FORCE_REPLAN_ONLY", "0")))
        original = np.asarray(action, dtype=np.float32).reshape(-1)
        patched = original.copy()
        requested_by_patch_mode = mode == 1 and patch_prob >= min_prob
        requested_by_prob = trigger_on_prob_only and patch_prob >= min_prob
        requested_by_replan = trigger_on_replan and mode == 2 and replan_prob >= min_prob
        # Replan-only is a trigger signal for the outer retry/replan logic; it
        # must not depend on patch mix, because no suffix action is written here.
        requested_by_replan_only = force_replan_only and (requested_by_patch_mode or requested_by_prob or requested_by_replan)
        patch_requested = (requested_by_patch_mode or requested_by_prob or requested_by_replan) and (not force_replan_only) and mix > 0
        requested = requested_by_replan_only or patch_requested
        if requested_by_replan_only:
            trigger_reason = "replan_only"
        elif requested_by_patch_mode:
            trigger_reason = "patch_mode"
        elif requested_by_prob:
            trigger_reason = "patch_prob"
        elif requested_by_replan:
            trigger_reason = "replan_as_patch"
        else:
            trigger_reason = "none"
        delta = d_out["action_delta"][0, 0].detach().cpu().numpy().astype(np.float32)
        if patch_requested and apply_enabled:
            delta = np.clip(delta, -max_abs, max_abs)
            patched = patched + mix * delta
            if patched.shape[0] >= 7:
                quat = patched[3:7]
                norm = float(np.linalg.norm(quat))
                if norm > 1e-6:
                    patched[3:7] = quat / norm
            if patched.shape[0] >= 9:
                patched[-2:] = np.clip(patched[-2:], 0.0, 1.0)
        applied = bool(patch_requested and np.linalg.norm(patched - original) > 1e-8)
        return patched, {
            "suffix_td_active": True,
            "suffix_td_requested": bool(requested),
            "suffix_td_applied": applied,
            "suffix_td_mode": mode,
            "suffix_td_patch_prob": patch_prob,
            "suffix_td_replan_prob": replan_prob,
            "suffix_td_trigger_reason": trigger_reason,
            "suffix_td_mix": mix,
            "suffix_td_delta_norm": float(np.linalg.norm(delta)),
            "suffix_td_apply_enabled": apply_enabled,
            "suffix_td_force_replan_only": force_replan_only,
        }

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
        params_to_optimize = filter(lambda p: p.requires_grad, self._network.parameters())

        self._optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=self._lr,
            weight_decay=self._lambda_weight_l2,
        )


    def _get_one_hot_expert_actions(
        self,
        batch_size,
        action_rot,
        action_grip,
        action_ignore_collisions,
        device,
    ):
        """_get_one_hot_expert_actions.

        :param batch_size: int
        :param action_rot: np.array of shape (bs, 4), quternion xyzw format
        :param action_grip: torch.tensor of shape (bs)
        :param action_ignore_collisions: torch.tensor of shape (bs)
        :param device:
        """
        bs = batch_size
        assert action_rot.shape == (bs, 4)
        assert action_grip.shape == (bs,), (action_grip, bs)

        action_rot_x_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_y_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_rot_z_one_hot = torch.zeros(
            (bs, self._num_rotation_classes), dtype=int, device=device
        )
        action_grip_one_hot = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device)

        # fill one-hots
        for b in range(bs):
            gt_rot = action_rot[b]
            gt_rot = aug_utils.quaternion_to_discrete_euler(
                gt_rot, self._rotation_resolution
            )
            action_rot_x_one_hot[b, gt_rot[0]] = 1
            action_rot_y_one_hot[b, gt_rot[1]] = 1
            action_rot_z_one_hot[b, gt_rot[2]] = 1

            # grip
            gt_grip = action_grip[b]
            action_grip_one_hot[b, gt_grip] = 1

            # ignore collision
            gt_ignore_collisions = action_ignore_collisions[b, :]
            action_collision_one_hot[b, gt_ignore_collisions[0]] = 1

        return (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,
            action_collision_one_hot,
        )


    def get_q(self, out, dims, only_pred=False, get_q_trans=True):
        """
        :param out: output of mvt
        :param dims: tensor dimensions (bs, nc, h, w)
        :param only_pred: some speedupds if the q values are meant only for
            prediction
        :return: tuple of trans_q, rot_q, grip_q and coll_q that is used for
            training and preduction
        """
        bs, nc, h, w = dims
        assert isinstance(only_pred, bool)

        if get_q_trans:
            pts = None
            # (bs, h*w, nc)
            q_trans = out["trans"].view(bs, nc, h * w).transpose(1, 2)
            if not only_pred:
                q_trans = q_trans.clone()

            # if two stages, we concatenate the q_trans, and replace all other
            if self.stage_two:
                out = out["mvt2"]
                q_trans2 = out["trans"].view(bs, nc, h * w).transpose(1, 2)
                if not only_pred:
                    q_trans2 = q_trans2.clone()
                q_trans = torch.cat((q_trans, q_trans2), dim=2)
        else:
            pts = None
            q_trans = None
            if self.stage_two:
                out = out["mvt2"]

        if self.rot_ver == 0:
            # (bs, 218)
            rot_q = out["feat"].view(bs, -1)[:, 0 : self.num_all_rot]
            grip_q = out["feat"].view(bs, -1)[:, self.num_all_rot : self.num_all_rot + 2]
            # (bs, 2)
            collision_q = out["feat"].view(bs, -1)[
                :, self.num_all_rot + 2 : self.num_all_rot + 4
            ]
        elif self.rot_ver == 1:
            rot_q = torch.cat((out["feat_x"], out["feat_y"], out["feat_z"]),
                              dim=-1).view(bs, -1)
            grip_q = out["feat_ex_rot"].view(bs, -1)[:, :2]
            collision_q = out["feat_ex_rot"].view(bs, -1)[:, 2:]
        else:
            assert False

        y_q = None

        return q_trans, rot_q, grip_q, collision_q, y_q, pts



    def update(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
    ) -> dict:
        assert replay_sample["rot_grip_action_indicies"].shape[1:] == (1, 4)
        assert replay_sample["ignore_collisions"].shape[1:] == (1, 1)
        assert replay_sample["gripper_pose"].shape[1:] == (1, 7)

        # sample
        action_rot_grip = replay_sample["rot_grip_action_indicies"][
            :, -1
        ].int()  # (b, 4) of int
        action_ignore_collisions = replay_sample["ignore_collisions"][
            :, -1
        ].int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"][:, -1]  # (b, 7)
        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3)
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4)
        action_grip = action_rot_grip[:, -1]  # (b,)
        tasks = replay_sample["tasks"]
        return_out = {}

        obs, pcd = rlbench_utils._preprocess_inputs(replay_sample, self.cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )

            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)

            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)  
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"]  
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )

        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )


        loss_log = {}
        if backprop:
            # cross-entropy loss
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()    # Soft-label cross-entropy loss. The target has the same shape as the input and is no longer one-hot encoded, but represented by class probabilities.
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )


            self._optimizer.zero_grad(set_to_none=True)
            
            total_loss.backward() 
            self._optimizer.step()


            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)


        return return_out



    def update_gembench(
        self,
        replay_sample: dict,
        backprop: bool = True,
        reset_log: bool = False,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
    ) -> dict:
        action_ignore_collisions = replay_sample["ignore_collisions"].unsqueeze(1).int()  # (b, 1) of int
        action_gripper_pose = replay_sample["gripper_pose"]  # (b, 8)  
        

        action_trans_con = action_gripper_pose[:, 0:3]  # (b, 3) 
        # rotation in quaternion xyzw
        action_rot = action_gripper_pose[:, 3:7]  # (b, 4) 

        action_grip = action_gripper_pose[:, -1].int()   # (b,)
        return_out = {}

        obs, pcd = gembench_utils._preprocess_inputs_gembench(replay_sample, cameras)
        
        with torch.no_grad():
            pc, img_feat = rvt_utils.get_pc_img_feat(
                obs,
                pcd,
            )
            import open3d as o3d
            def vis_pcd(pc, rgb,save_path):

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pc)  
                pcd.colors = o3d.utility.Vector3dVector(rgb) 
                o3d.io.write_point_cloud(save_path, pcd)
                # o3d.visualization.draw_geometries([pcd])
            if self._transform_augmentation and backprop:
                action_trans_con, action_rot, pc = apply_se3_aug_con(
                    pcd=pc,
                    action_gripper_pose=action_gripper_pose,
                    bounds=torch.tensor(self.scene_bounds),
                    trans_aug_range=self._transform_augmentation_xyz.clone().detach(),
                    rot_aug_range=torch.tensor(self._transform_augmentation_rpy),
                )
                action_trans_con = torch.tensor(action_trans_con).to(pc.device)
                action_rot = torch.tensor(action_rot).to(pc.device)
            
            # TODO: vectorize
            action_rot = action_rot.cpu().numpy()
            for i, _action_rot in enumerate(action_rot):
                _action_rot = aug_utils.normalize_quaternion(_action_rot)
                if _action_rot[-1] < 0:
                    _action_rot = -_action_rot
                action_rot[i] = _action_rot

            pc, img_feat = rvt_utils.move_pc_in_bound(
                pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
            )
            wpt = [x[:3] for x in action_trans_con]

            wpt_local = []
            rev_trans = []
            for _pc, _wpt in zip(pc, wpt):
                a, b = mvt_utils.place_pc_in_cube(
                    _pc,
                    _wpt,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )
                wpt_local.append(a.unsqueeze(0))
                rev_trans.append(b)

            wpt_local = torch.cat(wpt_local, axis=0)

            # TODO: Vectorize
            pc = [
                mvt_utils.place_pc_in_cube(
                    _pc,
                    with_mean_or_bounds=self._place_with_mean,
                    scene_bounds=None if self._place_with_mean else self.scene_bounds,
                )[0]
                for _pc in pc
            ]

            bs = len(pc)
            nc = self._net_mod.num_img
            h = w = self._net_mod.img_size

            if backprop and (self.img_aug != 0):
                img_aug = self.img_aug
            else:
                img_aug = 0

            dyn_cam_info = None

        (
            action_rot_x_one_hot,
            action_rot_y_one_hot,
            action_rot_z_one_hot,
            action_grip_one_hot,  # (bs, 2)
            action_collision_one_hot,  # (bs, 2)
        ) = self._get_one_hot_expert_actions(
            bs, action_rot, action_grip, action_ignore_collisions, device=self._device
        )

        if self.rot_ver == 1:
            rot_x_y = torch.cat(
                [
                    action_rot_x_one_hot.argmax(dim=-1, keepdim=True),
                    action_rot_y_one_hot.argmax(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            if self.rot_x_y_aug != 0:
                # add random interger between -rot_x_y_aug and rot_x_y_aug to rot_x_y
                rot_x_y += torch.randint(
                    -self.rot_x_y_aug, self.rot_x_y_aug, size=rot_x_y.shape
                ).to(rot_x_y.device)
                rot_x_y %= self._num_rotation_classes
        
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            lang_emb=None,
            img_aug=img_aug,
            wpt_local=wpt_local if self._network.training else None,
            rot_x_y=rot_x_y if self.rot_ver == 1 else None,
            language_goal=replay_sample["lang_goal"]  
        )
        
        q_trans, rot_q, grip_q, collision_q, y_q, pts = self.get_q(
            out, dims=(bs, nc, h, w)
        )

        action_trans = self.get_action_trans(
            wpt_local, pts, out, dyn_cam_info, dims=(bs, nc, h, w)
        )

        loss_log = {}
        if backprop:
            trans_loss = self._cross_entropy_loss(q_trans, action_trans).mean()  
            rot_loss_x = rot_loss_y = rot_loss_z = 0.0
            grip_loss = 0.0
            collision_loss = 0.0
            if self.add_rgc_loss:
                
                rot_loss_x = self._cross_entropy_loss(
                    rot_q[
                        :,
                        0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                    ],
                    action_rot_x_one_hot.argmax(-1),
                ).mean()

                rot_loss_y = self._cross_entropy_loss(
                    rot_q[
                        :,
                        1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                    ],
                    action_rot_y_one_hot.argmax(-1),
                ).mean()

                rot_loss_z = self._cross_entropy_loss(
                    rot_q[
                        :,
                        2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                    ],
                    action_rot_z_one_hot.argmax(-1),
                ).mean()
                
                grip_loss = self._cross_entropy_loss(
                    grip_q,
                    action_grip_one_hot.argmax(-1),
                ).mean()
                
                collision_loss = self._cross_entropy_loss(
                    collision_q, action_collision_one_hot.argmax(-1)
                ).mean()

            total_loss = (
                trans_loss
                + rot_loss_x
                + rot_loss_y
                + rot_loss_z
                + grip_loss
                + collision_loss
            )
            self._optimizer.zero_grad(set_to_none=True)
            
            total_loss.backward()
            self._optimizer.step()

            loss_log = {
                "total_loss": total_loss.item(),
                "trans_loss": trans_loss.item(),
                "rot_loss_x": rot_loss_x.item(),
                "rot_loss_y": rot_loss_y.item(),
                "rot_loss_z": rot_loss_z.item(),
                "grip_loss": grip_loss.item(),
                "collision_loss": collision_loss.item(),
                "lr": self._optimizer.param_groups[0]["lr"],
            }
            manage_loss_log(self, loss_log, reset_log=reset_log)
            return_out.update(loss_log)

        return return_out


    @torch.no_grad()
    def _flatten_pooled_feature(self, pooled_feature, device):
        def _collect_parts(value, out_parts):
            if value is None:
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    _collect_parts(item, out_parts)
                return
            if isinstance(value, np.ndarray) and value.dtype == object:
                for item in value.reshape(-1).tolist():
                    _collect_parts(item, out_parts)
                return
            try:
                tensor = value if torch.is_tensor(value) else torch.as_tensor(value, device=device)
            except Exception:
                return
            tensor = tensor.float()
            if tensor.numel() == 0:
                return
            if tensor.ndim == 0:
                tensor = tensor.reshape(1, 1)
            elif tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.reshape(tensor.shape[0], -1)
            if tensor.shape[-1] == 0:
                return
            out_parts.append(tensor)

        if pooled_feature is None:
            return None
        parts = []
        _collect_parts(pooled_feature, parts)
        if not parts:
            return None
        batch_size = max(part.shape[0] for part in parts)
        aligned_parts = []
        for part in parts:
            if part.shape[0] == batch_size:
                aligned_parts.append(part)
            elif part.shape[0] == 1:
                aligned_parts.append(part.repeat(batch_size, 1))
            else:
                aligned_parts.append(part[:batch_size])
        pooled_tensor = torch.cat(aligned_parts, dim=-1)
        if pooled_tensor.numel() == 0 or pooled_tensor.shape[1] == 0:
            return None
        pooled_tensor = torch.nan_to_num(pooled_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tanh(pooled_tensor)

    @torch.no_grad()
    def _resize_latent_dim(self, latent: torch.Tensor, target_dim: int) -> torch.Tensor:
        if latent.shape[1] > target_dim:
            return latent[:, :target_dim]
        if latent.shape[1] < target_dim and latent.shape[1] > 0:
            repeat = (target_dim + latent.shape[1] - 1) // latent.shape[1]
            return latent.repeat(1, repeat)[:, :target_dim]
        return latent

    @torch.no_grad()
    def _match_input_dim(self, tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
        if tensor.shape[1] > target_dim:
            return tensor[:, :target_dim]
        if tensor.shape[1] < target_dim and tensor.shape[1] > 0:
            repeat = (target_dim + tensor.shape[1] - 1) // tensor.shape[1]
            return tensor.repeat(1, repeat)[:, :target_dim]
        return tensor

    @torch.no_grad()
    def _repeat_token_grid(self, latent: torch.Tensor, num_tokens: int = 256, token_dim: int = 768) -> torch.Tensor:
        latent = self._resize_latent_dim(latent, token_dim)
        return latent.unsqueeze(1).repeat(1, num_tokens, 1)

    @torch.no_grad()
    def _run_bridge_standalone_gidm(self, current_latent: torch.Tensor, future_latent: torch.Tensor, language_goal):
        if self._bridge_standalone_gidm is None:
            return None
        if isinstance(language_goal, str):
            text = [language_goal]
        elif isinstance(language_goal, (list, tuple)):
            text = [str(x) for x in language_goal]
        else:
            text = [str(language_goal)]
        batch_size = current_latent.shape[0]
        if len(text) == 1 and batch_size > 1:
            text = text * batch_size
        elif len(text) != batch_size:
            text = [text[0] if text else ""] * batch_size
        current_tokens = self._repeat_token_grid(current_latent)
        future_tokens = self._repeat_token_grid(future_latent)
        gidm_tokens = torch.stack([current_tokens, future_tokens], dim=1)
        lam_out = self._bridge_standalone_gidm(gidm_tokens, text)
        video_action_patches = lam_out.get("video_action_patches", None)
        if video_action_patches is None:
            return None
        motion_latent = video_action_patches.reshape(batch_size, -1, video_action_patches.shape[-1]).mean(dim=1)
        motion_latent = torch.nan_to_num(motion_latent, nan=0.0, posinf=0.0, neginf=0.0)
        return motion_latent

    @torch.no_grad()
    def _joint_selector_task_mask(self, task_text: str):
        bundle = self._joint_pair_selector
        if bundle is None:
            return None
        memory = bundle["memory"]
        size = len(memory["success"])
        if not bundle["same_task_only"]:
            return np.ones((size,), dtype=bool)
        if "lang_goal" in memory:
            texts = [str(x) for x in memory["lang_goal"].tolist()]
        elif "task" in memory:
            texts = [str(x) for x in memory["task"].tolist()]
        else:
            return np.ones((size,), dtype=bool)
        query_tokens = {tok for tok in str(task_text).replace("_", " ").lower().split() if len(tok) > 1}
        if not query_tokens:
            return np.ones((size,), dtype=bool)
        mask = []
        for text in texts:
            toks = {tok for tok in text.replace("_", " ").lower().split() if len(tok) > 1}
            mask.append(bool(query_tokens & toks))
        mask = np.asarray(mask, dtype=bool)
        if mask.any():
            return mask
        return np.ones((size,), dtype=bool)

    @torch.no_grad()
    def _score_joint_energy(self, state_latent, future_latent, motion_latent, task_text: str):
        bundle = self._joint_pair_selector
        if bundle is None or bundle["energy"] is None:
            return None
        energy_bundle = bundle["energy"]
        device = state_latent.device
        task_vec = torch.from_numpy(_hashed_text_vec(task_text, energy_bundle["task_dim"])).to(device=device, dtype=state_latent.dtype).unsqueeze(0)
        model = energy_bundle["model"]
        return model(state_latent, future_latent, motion_latent, task_vec).reshape(-1)

    @torch.no_grad()
    def _repair_joint_pair(self, state_latent, future_latent, motion_latent, task_text: str):
        bundle = self._joint_pair_selector
        if bundle is None or bundle.get("repair") is None:
            return future_latent, motion_latent, {"repair_used": False, "repair_reason": "missing_repair_ckpt"}
        repair_bundle = bundle["repair"]
        task_dim = max(1, int(repair_bundle.get("num_tasks", 1)))
        task_vec = torch.from_numpy(_hashed_text_vec(task_text, task_dim)).to(
            device=state_latent.device,
            dtype=state_latent.dtype,
        ).unsqueeze(0)
        feat = torch.cat([state_latent, future_latent, motion_latent, task_vec], dim=-1)
        feat = self._match_input_dim(feat, int(repair_bundle["input_dim"]))
        pred_df, pred_da = repair_bundle["model"](feat)
        pred_df = self._match_input_dim(pred_df, future_latent.shape[-1])
        pred_da = self._match_input_dim(pred_da, motion_latent.shape[-1])
        repaired_future = future_latent + pred_df
        repaired_motion = motion_latent + pred_da
        return repaired_future, repaired_motion, {
            "repair_used": True,
            "repair_reason": "mlp_delta",
            "future_delta_norm": float(pred_df.norm(dim=-1).mean().item()),
            "motion_delta_norm": float(pred_da.norm(dim=-1).mean().item()),
        }

    @torch.no_grad()
    def _verify_joint_pair(
        self,
        state_latent,
        base_future,
        base_motion,
        final_future,
        final_motion,
        task_text: str,
        similarity: float,
    ):
        bundle = self._joint_pair_selector
        if bundle is None:
            return True, {"verify_used": False}
        c_exp_tensor = self._score_joint_energy(state_latent, final_future, final_motion, task_text)
        c_exp = float(c_exp_tensor.item()) if c_exp_tensor is not None else 0.0
        c_latent = float((final_future - base_future).norm(dim=-1).mean().item())
        c_physical = float((final_motion - base_motion).norm(dim=-1).mean().item())
        verify_info = {
            "verify_used": True,
            "verify_c_exp": c_exp,
            "verify_c_latent": c_latent,
            "verify_c_physical": c_physical,
            "verify_similarity": float(similarity),
        }
        reasons = []
        if c_exp > float(bundle["verify_exp_max"]):
            reasons.append(f"c_exp>{bundle['verify_exp_max']:g}")
        if c_latent > float(bundle["verify_latent_max"]):
            reasons.append(f"c_latent>{bundle['verify_latent_max']:g}")
        if c_physical > float(bundle["verify_physical_max"]):
            reasons.append(f"c_physical>{bundle['verify_physical_max']:g}")
        if bundle.get("acceptance") is not None:
            acc_bundle = bundle["acceptance"]
            feat_parts = [
                state_latent,
                final_future,
                final_motion,
                torch.tensor(
                    [[c_exp, c_latent, c_physical, float(similarity)]],
                    device=state_latent.device,
                    dtype=state_latent.dtype,
                ),
            ]
            acc_in = torch.cat(feat_parts, dim=-1)
            acc_in = self._match_input_dim(acc_in, int(acc_bundle["input_dim"]))
            prob = torch.sigmoid(acc_bundle["model"](acc_in)).reshape(-1)
            acc_prob = float(prob.item())
            verify_info["acceptance_prob"] = acc_prob
            if acc_prob < float(bundle["acceptance_threshold"]):
                reasons.append(f"accept<{bundle['acceptance_threshold']:g}")
        verify_info["verify_passed"] = len(reasons) == 0
        verify_info["verify_reason"] = "pass" if not reasons else ";".join(reasons)
        return len(reasons) == 0, verify_info

    @torch.no_grad()
    def _apply_joint_pair_selector(self, bridge_latent_bundle: dict, language_goal) -> tuple[dict, dict]:
        selector = self._joint_pair_selector
        debug = {"selector_used": False}
        if selector is None or bridge_latent_bundle is None:
            return bridge_latent_bundle, debug
        memory = selector["memory"]
        state = bridge_latent_bundle["state_latent"]
        future = bridge_latent_bundle["future_latent"]
        motion = bridge_latent_bundle["motion_latent"]
        if state is None or future is None or motion is None:
            return bridge_latent_bundle, debug
        task_text = str(language_goal[0][0][0] if isinstance(language_goal, list) else language_goal)
        task_mask = self._joint_selector_task_mask(task_text)
        candidate_ids = np.nonzero(task_mask)[0]
        if candidate_ids.size == 0:
            return bridge_latent_bundle, debug
        q_state = state[0].detach().cpu().numpy().astype(np.float32)
        q_future = future[0].detach().cpu().numpy().astype(np.float32)
        q_motion = motion[0].detach().cpu().numpy().astype(np.float32)
        sims = []
        for idx in candidate_ids.tolist():
            sim_s = _cosine_np(q_state, np.asarray(memory["state_latent"][idx], dtype=np.float32))
            sim_f = _cosine_np(q_future, np.asarray(memory["future_latent"][idx], dtype=np.float32))
            sim_a = _cosine_np(q_motion, np.asarray(memory["motion_latent"][idx], dtype=np.float32))
            sims.append(0.2 * sim_s + 0.5 * sim_f + 0.3 * sim_a)
        sims = np.asarray(sims, dtype=np.float32)
        order = np.argsort(-sims)[: max(1, min(selector["topk"], candidate_ids.size))]
        chosen = candidate_ids[order]
        base_energy = self._score_joint_energy(state, future, motion, task_text)
        best = None
        for rank, mem_idx in enumerate(chosen.tolist(), start=1):
            mem_future = torch.from_numpy(np.asarray(memory["future_latent"][mem_idx], dtype=np.float32)).to(device=future.device, dtype=future.dtype).unsqueeze(0)
            mem_motion = torch.from_numpy(np.asarray(memory["motion_latent"][mem_idx], dtype=np.float32)).to(device=motion.device, dtype=motion.dtype).unsqueeze(0)
            future_shift = float((mem_future - future).norm(dim=-1).mean().item())
            motion_shift = float((mem_motion - motion).norm(dim=-1).mean().item())
            if future_shift > selector["max_future_shift"] or motion_shift > selector["max_motion_shift"]:
                continue
            energy_val = self._score_joint_energy(state, mem_future, mem_motion, task_text)
            improve = None
            if base_energy is not None and energy_val is not None:
                improve = float((base_energy - energy_val).item())
                if improve < selector["min_energy_improve"]:
                    continue
            sim = float(sims[order[rank - 1]])
            if sim < selector["min_similarity"]:
                continue
            score = sim if improve is None else sim + 0.25 * improve
            rec = {
                "memory_index": int(mem_idx),
                "rank": int(rank),
                "similarity": sim,
                "energy_improve": improve,
                "future_shift": future_shift,
                "motion_shift": motion_shift,
                "future": mem_future,
                "motion": mem_motion,
                "score": score,
            }
            if best is None or rec["score"] > best["score"]:
                best = rec
        debug.update(
            {
                "selector_used": True,
                "num_candidates": int(candidate_ids.size),
                "num_topk": int(len(chosen)),
                "base_energy": None if base_energy is None else float(base_energy.item()),
            }
        )
        if best is None:
            debug["selector_applied"] = False
            return bridge_latent_bundle, debug
        init_future = torch.lerp(future, best["future"], selector["future_refine_alpha"])
        init_motion = torch.lerp(motion, best["motion"], selector["motion_refine_alpha"])
        repaired_future, repaired_motion, repair_info = self._repair_joint_pair(state, init_future, init_motion, task_text)
        final_future = repaired_future if bool(repair_info.get("repair_used", False)) else init_future
        final_motion = repaired_motion if bool(repair_info.get("repair_used", False)) else init_motion
        final_future_shift = float((final_future - future).norm(dim=-1).mean().item())
        final_motion_shift = float((final_motion - motion).norm(dim=-1).mean().item())
        verified, verify_info = self._verify_joint_pair(
            state,
            future,
            motion,
            final_future,
            final_motion,
            task_text,
            best["similarity"],
        )
        out = dict(bridge_latent_bundle)
        out["future_latent"] = final_future
        out["motion_latent"] = final_motion
        out["future_norm"] = float(final_future.norm(dim=-1).mean().item())
        out["motion_norm"] = float(final_motion.norm(dim=-1).mean().item())
        out["selector"] = {
            "memory_index": best["memory_index"],
            "rank": best["rank"],
            "similarity": best["similarity"],
            "energy_improve": best["energy_improve"],
            "future_shift": best["future_shift"],
            "motion_shift": best["motion_shift"],
            "final_future_shift": final_future_shift,
            "final_motion_shift": final_motion_shift,
        }
        debug["selector_applied"] = True
        debug.update(out["selector"])
        debug.update({f"repair_{k}": v for k, v in repair_info.items()})
        debug.update({f"verify_{k}": v for k, v in verify_info.items()})
        if not verified:
            debug["selector_applied"] = False
            debug["selector_blocked_by_verify"] = True
            return bridge_latent_bundle, debug
        return out, debug

    @torch.no_grad()
    def _build_bridge_gfdm_gidm_latent(self, pooled_feature, device, target_dim: int, language_goal=None):
        s_latent = self._flatten_pooled_feature(pooled_feature, device=device)
        if s_latent is None or s_latent.ndim != 2 or s_latent.shape[0] == 0 or s_latent.shape[1] == 0:
            return None
        if self._bridge_latent_heads is not None:
            state_in_dim = self._bridge_latent_heads["state_proj"][0].in_features
            future_in_dim = self._bridge_latent_heads["future_head"][0].in_features
            state_input = self._match_input_dim(s_latent, state_in_dim)
            future_input = self._match_input_dim(s_latent, future_in_dim)
            if (
                state_input.ndim != 2
                or future_input.ndim != 2
                or state_input.shape[0] == 0
                or future_input.shape[0] == 0
                or state_input.shape[1] == 0
                or future_input.shape[1] == 0
            ):
                return None
            state_proj = torch.tanh(self._bridge_latent_heads["state_proj"](state_input))
            zf_latent = torch.tanh(self._bridge_latent_heads["future_head"](future_input))
            if self._bridge_standalone_gidm is not None:
                za_latent = self._run_bridge_standalone_gidm(state_proj, zf_latent, language_goal)
                head_mode = "learned+standalone_gidm"
            else:
                za_input = torch.cat([state_proj, zf_latent], dim=-1)
                za_latent = torch.tanh(self._bridge_latent_heads["motion_head"](za_input))
                head_mode = "learned"
            if za_latent is None:
                za_input = torch.cat([state_proj, zf_latent], dim=-1)
                za_latent = torch.tanh(self._bridge_latent_heads["motion_head"](za_input))
                head_mode = "learned_fallback"
            return {
                "state_latent": state_proj,
                "future_latent": zf_latent,
                "motion_latent": za_latent,
                "state_norm": float(state_proj.norm(dim=-1).mean().item()),
                "future_norm": float(zf_latent.norm(dim=-1).mean().item()),
                "motion_norm": float(za_latent.norm(dim=-1).mean().item()),
                "head_mode": head_mode,
                "state_shape": tuple(state_input.shape),
                "future_shape": tuple(zf_latent.shape),
                "motion_shape": tuple(za_latent.shape),
            }
        # Heuristic Bridge-side GFDM proxy: smooth state features into a future-like latent.
        zf_full = F.avg_pool1d(s_latent.unsqueeze(1), kernel_size=9, stride=1, padding=4).squeeze(1)
        zf_full = torch.tanh(0.7 * zf_full + 0.3 * torch.roll(s_latent, shifts=11, dims=-1))
        # Heuristic Bridge-side GIDM proxy: combine current and future features into motion intent.
        delta = zf_full - s_latent
        coupled = 0.5 * (zf_full + s_latent)
        zf_latent = self._resize_latent_dim(zf_full, target_dim)
        if self._bridge_standalone_gidm is not None:
            current_latent = self._resize_latent_dim(s_latent, target_dim)
            za_latent = self._run_bridge_standalone_gidm(current_latent, zf_latent, language_goal)
            head_mode = "heuristic+standalone_gidm"
        else:
            za_full = torch.tanh(delta + 0.5 * torch.roll(coupled, shifts=23, dims=-1))
            za_latent = self._resize_latent_dim(za_full, target_dim)
            head_mode = "heuristic"
        if za_latent is None:
            za_full = torch.tanh(delta + 0.5 * torch.roll(coupled, shifts=23, dims=-1))
            za_latent = self._resize_latent_dim(za_full, target_dim)
            head_mode = "heuristic_fallback"
        return {
            "state_latent": s_latent,
            "future_latent": zf_latent,
            "motion_latent": za_latent,
            "state_norm": float(s_latent.norm(dim=-1).mean().item()),
            "future_norm": float(zf_latent.norm(dim=-1).mean().item()),
            "motion_norm": float(za_latent.norm(dim=-1).mean().item()),
            "head_mode": head_mode,
            "state_shape": tuple(s_latent.shape),
            "future_shape": tuple(zf_latent.shape),
            "motion_shape": tuple(za_latent.shape),
        }

    @torch.no_grad()
    def act(
        self,
        step: int,
        observation: dict,
        visualize=False,
        visualize_save_dir="",
        return_gembench_action=False,
        gidm_latent=None,
        gidm_gate: float = 1.0,
        gidm_from_pooled: bool = False,
        gidm_from_bridge_gfdm_gidm: bool = False,
    ) -> ActResult:
        collect_split_zf_za = bool(int(os.environ.get("BRIDGEVLA_COLLECT_SPLIT_ZF_ZA", "0")))
        force_bridge_gfdm_gidm = bool(int(os.environ.get("BRIDGEVLA_BRIDGE_GFDM_GIDM", "0")))
        use_split_za_in_action = bool(int(os.environ.get("BRIDGEVLA_USE_SPLIT_ZA_IN_ACTION", "0")))
        want_bridge_latents = (
            collect_split_zf_za
            or gidm_from_bridge_gfdm_gidm
            or force_bridge_gfdm_gidm
            or use_split_za_in_action
        )
        # Collection-only mode records z_f/z_a without changing the BridgeVLA action path.
        inject_bridge_latent = use_split_za_in_action or (
            gidm_from_bridge_gfdm_gidm and not collect_split_zf_za
        )
        language_goal =observation["language_goal"]
        obs, pcd = rlbench_utils._preprocess_inputs(observation, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(
            obs,
            pcd,
        )
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )
        pc_ori = pc[0].clone()
        img_feat_ori=img_feat[0].clone()
        # TODO: Vectorize
        pc_new = []
        rev_trans = []
        for _pc in pc:
            a, b = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
            rev_trans.append(b)
        pc = pc_new

        bs = len(pc)
        nc = self._net_mod.num_img
        h = w = self._net_mod.img_size
        dyn_cam_info = None
        out = self._network(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,  # no img augmentation while acting
            language_goal=language_goal,
        )
        pooled_feature = out.get("pooled_feature", None)
        if getattr(self, "_debug_gfdm_gidm", False):
            try:
                flat_dbg = self._flatten_pooled_feature(pooled_feature, device=pc[0].device)
                flat_shape = None if flat_dbg is None else tuple(flat_dbg.shape)
            except Exception as exc:
                flat_shape = f"flatten_error:{exc}"
            print(
                f"[bridge-gfdm-gidm-debug] step={step} "
                f"gidm_from_pooled={gidm_from_pooled} "
                f"gidm_from_bridge_gfdm_gidm={gidm_from_bridge_gfdm_gidm} "
                f"pooled_type={type(pooled_feature).__name__} "
                f"flat_shape={flat_shape}",
                flush=True,
            )
        gidm_source = "external" if gidm_latent is not None else "none"
        pooled_feature_norm = None
        bridge_latent_bundle = None
        selector_debug = None
        target_dim = getattr(self._net_mod, "gidm_latent_dim", 768)
        if gidm_latent is None and want_bridge_latents and pooled_feature is not None:
            bridge_latent_bundle = self._build_bridge_gfdm_gidm_latent(
                pooled_feature,
                device=pc[0].device,
                target_dim=target_dim,
                language_goal=language_goal,
            )
            if bridge_latent_bundle is not None:
                pooled_feature_norm = bridge_latent_bundle["state_norm"]
                if bool(int(os.environ.get("BRIDGEVLA_SUFFIX_PERSISTENT_BELIEF", "0"))):
                    persist_future = getattr(self, "_suffix_td_persistent_future", None)
                    persist_motion = getattr(self, "_suffix_td_persistent_motion", None)
                    if torch.is_tensor(persist_future) and torch.is_tensor(persist_motion):
                        persist_future = persist_future.to(device=pc[0].device, dtype=bridge_latent_bundle["future_latent"].dtype)
                        persist_motion = persist_motion.to(device=pc[0].device, dtype=bridge_latent_bundle["motion_latent"].dtype)
                        bridge_latent_bundle["future_latent"] = self._resize_latent_dim(persist_future, target_dim)
                        bridge_latent_bundle["motion_latent"] = self._resize_latent_dim(persist_motion, target_dim)
                        bridge_latent_bundle["future_norm"] = float(bridge_latent_bundle["future_latent"].norm(dim=-1).mean().item())
                        bridge_latent_bundle["motion_norm"] = float(bridge_latent_bundle["motion_latent"].norm(dim=-1).mean().item())
                        bridge_latent_bundle["head_mode"] = f"{bridge_latent_bundle.get('head_mode', 'unknown')}+persistent_belief"
                if inject_bridge_latent:
                    bridge_latent_bundle, selector_debug = self._apply_joint_pair_selector(bridge_latent_bundle, language_goal)
                    gidm_latent = bridge_latent_bundle["motion_latent"]
                    gidm_source = "bridge_gfdm_gidm_joint" if selector_debug and selector_debug.get("selector_applied") else "bridge_gfdm_gidm"
                else:
                    gidm_source = "bridge_gfdm_gidm_collect_only"
            if getattr(self, "_debug_gfdm_gidm", False):
                print(
                    f"[bridge-gfdm-gidm-debug] step={step} "
                    f"bundle={'yes' if bridge_latent_bundle is not None else 'no'} "
                    f"gidm_source={gidm_source} "
                    f"head_mode={None if bridge_latent_bundle is None else bridge_latent_bundle.get('head_mode')} "
                    f"state_shape={None if bridge_latent_bundle is None else bridge_latent_bundle.get('state_shape')} "
                    f"future_shape={None if bridge_latent_bundle is None else bridge_latent_bundle.get('future_shape')} "
                    f"motion_shape={None if bridge_latent_bundle is None else bridge_latent_bundle.get('motion_shape')}",
                    flush=True,
                )
        elif gidm_latent is None and gidm_from_pooled and pooled_feature is not None:
            pooled_tensor = self._flatten_pooled_feature(pooled_feature, device=pc[0].device)
            if pooled_tensor is not None:
                pooled_feature_norm = float(pooled_tensor.norm(dim=-1).mean().item())
                gidm_latent = self._resize_latent_dim(pooled_tensor, target_dim)
                gidm_source = "pooled_feature"
        if getattr(self, "_debug_gfdm_gidm", False):
            print(
                f"[bridge-gfdm-gidm-debug] step={step} final gidm_source={gidm_source} "
                f"gidm_latent_shape={None if gidm_latent is None else tuple(gidm_latent.shape)}",
                flush=True,
            )
        if visualize:
            q_trans, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=True
            )
        else:
            _, rot_q, grip_q, collision_q, y_q, _ = self.get_q(
                out, dims=(bs, nc, h, w), only_pred=True, get_q_trans=False
            )            
        pred_wpt, pred_rot_quat, pred_grip, pred_coll = self.get_pred(
            out,
            rot_q,
            grip_q,
            collision_q,
            y_q,
            rev_trans,
            dyn_cam_info,
            gidm_latent=gidm_latent,
            gidm_gate=gidm_gate,
        )
        if visualize:
            print("Visualizing")
            save_dir=visualize_save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_dir=os.path.join(save_dir,f"step{str(step)}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            mvt1_img=out["mvt1_ori_img"][0,:,3:6]
            mvt2_img=out["mvt2_ori_img"][0,:,3:6]
            q_trans_1=q_trans[0,:,:3].clone().view(224,224,3)
            q_trans_2=q_trans[0,:,3:6].clone().view(224,224,3)
            q_trans_1=apply_channel_wise_softmax(q_trans_1)*100
            q_trans_2=apply_channel_wise_softmax(q_trans_2)*100
            visualize_images(mvt1_img,q_trans_1,save_dir=os.path.join(save_dir,"mvt1"))
            visualize_images(mvt2_img,q_trans_2,save_dir=os.path.join(save_dir,"mvt2"))
            save_point_cloud_with_color(os.path.join(save_dir,"point_cloud.ply"), pc_ori.cpu().numpy(), img_feat_ori.cpu().numpy(), pred_wpt[0].cpu().numpy())
        continuous_action = np.concatenate(
            (
                pred_wpt[0].cpu().numpy(),
                pred_rot_quat[0],
                pred_grip[0].cpu().numpy(),
                pred_coll[0].cpu().numpy(),
                # [1.0],  # debug!!!!!!
            )
        )

        if return_gembench_action:
            continuous_action = np.concatenate(
                    [
                        pred_wpt[0].cpu().numpy(),
                        pred_rot_quat[0],
                        pred_grip[0].cpu().numpy(),
                    ], -1
                )
            return continuous_action
        else:
            info = {}
            pooled_latent = None
            if pooled_feature is not None:
                pooled_latent = self._flatten_pooled_feature(pooled_feature, device=pc[0].device)
                if pooled_latent is not None:
                    pooled_latent = self._resize_latent_dim(pooled_latent, target_dim)
                    info["pooled_latent"] = pooled_latent.detach().cpu()
                    info["pooled_latent_norm"] = float(pooled_latent.norm(dim=-1).mean().item())
            if pooled_feature is not None:
                if isinstance(pooled_feature, list):
                    info["pooled_feature"] = [
                        x.detach().cpu() if torch.is_tensor(x) else x for x in pooled_feature
                    ]
                elif torch.is_tensor(pooled_feature):
                    info["pooled_feature"] = pooled_feature.detach().cpu()
                else:
                    info["pooled_feature"] = pooled_feature
            if gidm_latent is not None:
                info["gidm_latent"] = gidm_latent.detach().cpu()
                info["gidm_latent_norm"] = float(gidm_latent.norm(dim=-1).mean().item())
            info["gidm_source"] = gidm_source
            info["gidm_gate"] = float(gidm_gate)
            if pooled_feature_norm is not None:
                info["pooled_feature_norm"] = pooled_feature_norm
            if bridge_latent_bundle is not None:
                info["state_latent"] = bridge_latent_bundle["state_latent"].detach().cpu()
                info["state_latent_norm"] = bridge_latent_bundle["state_norm"]
                info["future_latent"] = bridge_latent_bundle["future_latent"].detach().cpu()
                info["future_latent_norm"] = bridge_latent_bundle["future_norm"]
                info["motion_latent"] = bridge_latent_bundle["motion_latent"].detach().cpu()
                info["motion_latent_norm"] = bridge_latent_bundle["motion_norm"]
                info["bridge_latent_head_mode"] = bridge_latent_bundle.get("head_mode", "unknown")
            if selector_debug is not None:
                for key, value in selector_debug.items():
                    info[f"joint_selector_{key}"] = value
            info["low_dim_state_before"] = self._suffix_low_dim(
                observation.get("low_dim_state"), continuous_action.shape[0]
            )
            original_action = continuous_action.astype(np.float32, copy=True)
            patched_action, suffix_info = self._apply_suffix_td_patch(
                continuous_action,
                observation,
                info,
                _language_to_text(language_goal),
            )
            continuous_action = patched_action.astype(np.float32, copy=False)
            info["action"] = continuous_action.copy()
            info["original_action"] = original_action
            if suffix_info:
                info.update(suffix_info)
            return ActResult(continuous_action, info=info)

    @torch.no_grad()
    def extract_pooled_feature(self, observation: dict, language_goal=None):
        if language_goal is None:
            language_goal = observation.get("language_goal", None)
        if language_goal is None:
            language_goal = observation.get("lang_goal", None)
        if language_goal is None:
            language_goal = observation.get("lang", None)
        if language_goal is None:
            raise KeyError("language goal not found in observation")
        device = getattr(self, "_device", None)
        if device is None:
            try:
                device = next(self._network.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        prepped_obs = {}
        for k, v in observation.items():
            if k in {"language_goal", "lang_goal", "lang"}:
                continue
            if isinstance(v, torch.Tensor):
                t = v.detach().clone()
            else:
                t = torch.tensor(np.array([v]))
            if t.ndim <= 3:
                t = t.unsqueeze(0).unsqueeze(0)
            elif t.ndim == 4:
                t = t.unsqueeze(0)
            elif t.ndim == 1:
                t = t.unsqueeze(0).unsqueeze(0)
            prepped_obs[k] = t.to(device)

        net = self._network.module if hasattr(self._network, "module") else self._network
        prepped_obs["language_goal"] = [[[language_goal]]]

        obs, pcd = rlbench_utils._preprocess_inputs(prepped_obs, self.cameras)
        pc, img_feat = rvt_utils.get_pc_img_feat(obs, pcd)
        pc, img_feat = rvt_utils.move_pc_in_bound(
            pc, img_feat, self.scene_bounds, no_op=not self.move_pc_in_bound
        )
        pc_new = []
        for _pc in pc:
            a, _ = mvt_utils.place_pc_in_cube(
                _pc,
                with_mean_or_bounds=self._place_with_mean,
                scene_bounds=None if self._place_with_mean else self.scene_bounds,
            )
            pc_new.append(a)
        pc = pc_new

        img = net.render(
            pc=pc,
            img_feat=img_feat,
            img_aug=0,
            mvt1_or_mvt2=True,
            dyn_cam_info=None,
        )
        out = net.mvt1(
            img=img,
            wpt_local=None,
            rot_x_y=None,
            language_goal=[[[language_goal]]],
            forward_no_feat=False,
        )
        pooled = out.get("pooled_feature", None)
        if pooled is None:
            return None
        if isinstance(pooled, list):
            return [x.detach().cpu() if torch.is_tensor(x) else x for x in pooled]
        if torch.is_tensor(pooled):
            return pooled.detach().cpu()
        return pooled



    def get_pred(
        self,
        out,
        rot_q,
        grip_q,
        collision_q,
        y_q,
        rev_trans,
        dyn_cam_info,
        gidm_latent=None,
        gidm_gate: float = 1.0,
    ):
        if self.stage_two:
            assert y_q is None
            mvt1_or_mvt2 = False
        else:
            mvt1_or_mvt2 = True

        pred_wpt_local = self._net_mod.get_wpt(
            out,
            mvt1_or_mvt2,
            dyn_cam_info,
            y_q,
            gidm_latent=gidm_latent,
            gidm_gate=gidm_gate,
        )

        pred_wpt = []
        for _pred_wpt_local, _rev_trans in zip(pred_wpt_local, rev_trans):
            pred_wpt.append(_rev_trans(_pred_wpt_local))
        pred_wpt = torch.cat([x.unsqueeze(0) for x in pred_wpt])

        pred_rot = torch.cat(
            (
                rot_q[
                    :,
                    0 * self._num_rotation_classes : 1 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    1 * self._num_rotation_classes : 2 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
                rot_q[
                    :,
                    2 * self._num_rotation_classes : 3 * self._num_rotation_classes,
                ].argmax(1, keepdim=True),
            ),
            dim=-1,
        )
        pred_rot_quat = aug_utils.discrete_euler_to_quaternion(
            pred_rot.cpu(), self._rotation_resolution
        )
        pred_grip = grip_q.argmax(1, keepdim=True)
        pred_coll = collision_q.argmax(1, keepdim=True)

        return pred_wpt, pred_rot_quat, pred_grip, pred_coll


    @torch.no_grad()
    def get_action_trans(
        self,
        wpt_local,
        pts,
        out,
        dyn_cam_info,
        dims,
    ):
        bs, nc, h, w = dims
        wpt_img = self._net_mod.get_pt_loc_on_img(
            wpt_local.unsqueeze(1),
            mvt1_or_mvt2=True,
            dyn_cam_info=dyn_cam_info,
            out=None
        )
        assert wpt_img.shape[1] == 1
        if self.stage_two:
            wpt_img2 = self._net_mod.get_pt_loc_on_img(
                wpt_local.unsqueeze(1),
                mvt1_or_mvt2=False,
                dyn_cam_info=dyn_cam_info,
                out=out,
            )
            assert wpt_img2.shape[1] == 1

            # (bs, 1, 2 * num_img, 2)
            wpt_img = torch.cat((wpt_img, wpt_img2), dim=-2)
            nc = nc * 2

        # (bs, num_img, 2)
        wpt_img = wpt_img.squeeze(1)

        action_trans = mvt_utils.generate_hm_from_pt(
            wpt_img.reshape(-1, 2),
            (h, w),
            sigma=self.gt_hm_sigma,
            thres_sigma_times=3,
        )
        action_trans = action_trans.view(bs, nc, h * w).transpose(1, 2).clone()

        return action_trans



    def reset(self):
        self._suffix_td_persistent_future = None
        self._suffix_td_persistent_motion = None

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
