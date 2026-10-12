from __future__ import annotations

import numpy as np
from PIL import Image


DEFAULT_RGB_SCALE_FACTOR = float(2**24 - 1)


def image_to_float_array(image: Image.Image, scale_factor: float = DEFAULT_RGB_SCALE_FACTOR) -> np.ndarray:
    image_array = np.array(image)
    image_shape = image_array.shape
    channels = image_shape[2] if len(image_shape) > 2 else 1
    if channels == 3:
        float_array = np.sum(image_array * [65536, 256, 1], axis=2)
    else:
        float_array = image_array.astype(np.float32)
    return float_array.astype(np.float32) / float(scale_factor)


def create_uniform_pixel_coords_image(resolution: tuple[int, int] | list[int] | np.ndarray) -> np.ndarray:
    resolution = np.asarray(resolution, dtype=np.int64)
    pixel_x_coords = np.reshape(
        np.tile(np.arange(resolution[1]), [resolution[0]]),
        (resolution[0], resolution[1], 1),
    ).astype(np.float32)
    pixel_y_coords = np.reshape(
        np.tile(np.arange(resolution[0]), [resolution[1]]),
        (resolution[1], resolution[0], 1),
    ).astype(np.float32)
    pixel_y_coords = np.transpose(pixel_y_coords, (1, 0, 2))
    return np.concatenate((pixel_x_coords, pixel_y_coords, np.ones_like(pixel_x_coords)), axis=-1)


def _transform(coords: np.ndarray, trans: np.ndarray) -> np.ndarray:
    h, w = coords.shape[:2]
    coords = np.reshape(coords, (h * w, -1)).T
    transformed_coords_vector = np.matmul(trans, coords).T
    return np.reshape(transformed_coords_vector, (h, w, -1))


def pixel_to_world_coords(pixel_coords: np.ndarray, cam_proj_mat_inv: np.ndarray) -> np.ndarray:
    h, w = pixel_coords.shape[:2]
    pixel_coords = np.concatenate([pixel_coords, np.ones((h, w, 1), dtype=np.float32)], axis=-1)
    world_coords = _transform(pixel_coords, cam_proj_mat_inv)
    return np.concatenate([world_coords, np.ones((h, w, 1), dtype=np.float32)], axis=-1)


def pointcloud_from_depth_and_camera_params(
    depth_m: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    upc = create_uniform_pixel_coords_image(depth_m.shape)
    pc = upc * np.expand_dims(depth_m.astype(np.float32), axis=-1)
    extrinsics = np.asarray(extrinsics, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    c_vec = np.expand_dims(extrinsics[:3, 3], 0).T
    rot = extrinsics[:3, :3]
    rot_inv = rot.T
    rot_inv_c = np.matmul(rot_inv, c_vec)
    extrinsics_inv = np.concatenate((rot_inv, -rot_inv_c), axis=-1)
    cam_proj_mat = np.matmul(intrinsics, extrinsics_inv)
    cam_proj_mat_homo = np.concatenate([cam_proj_mat, [np.array([0, 0, 0, 1], dtype=np.float32)]], axis=0)
    cam_proj_mat_inv = np.linalg.inv(cam_proj_mat_homo)[0:3]
    world_coords_homo = np.expand_dims(pixel_to_world_coords(pc, cam_proj_mat_inv), axis=0)
    return world_coords_homo[..., :-1][0].astype(np.float32)
