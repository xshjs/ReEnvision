import os.path
from itertools import chain
import logging
from pathlib import Path
import pickle
from typing import Any, Dict, List, Tuple
import random

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
import numpy as np

from policy_models.datasets.base_dataset import BaseDataset
from policy_models.datasets.utils.episode_utils import lookup_naming_pattern

logger = logging.getLogger(__name__)


def load_pkl(filename: Path) -> Dict[str, np.ndarray]:
    with open(filename, "rb") as f:
        return pickle.load(f)


def load_npz(filename: Path) -> Dict[str, np.ndarray]:
    return np.load(filename.as_posix())


class DiskDataset(BaseDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.save_format = save_format  # 'npz'
        if self.save_format == "pkl":
            self.load_file = load_pkl
        elif self.save_format == "npz":  # True
            self.load_file = load_npz
        else:
            raise NotImplementedError
        self.pretrain = pretrain  # False
        self.skip_frames = skip_frames  # 1

        if self.with_lang:  # True
            self.episode_lookup, self.lang_lookup, self.lang_ann, self.lang_text = self._build_file_indices_lang(self.abs_datasets_dir)
        else:
            self.episode_lookup = self._build_file_indices(self.abs_datasets_dir)

        self.naming_pattern, self.n_digits = lookup_naming_pattern(self.abs_datasets_dir, self.save_format)

    def filter_by_tasks(self, task_names: List[str]) -> None:
        if not self.with_lang:
            return
        requested = set(task_names)
        keep_lang_ids = {
            idx
            for idx, task in enumerate(self.lang_data_tasks)
            if str(task) in requested or str(task).replace("_", " ") in requested
        }
        if not keep_lang_ids:
            raise ValueError(f"No language annotations matched requested tasks: {sorted(requested)}")
        keep_indices = [idx for idx, lang_idx in enumerate(self.lang_lookup) if lang_idx in keep_lang_ids]
        self.episode_lookup = self.episode_lookup[keep_indices]
        self.lang_lookup = [self.lang_lookup[idx] for idx in keep_indices]

    def _get_episode_name(self, file_idx: int) -> Path:
        """
        Convert file idx to file path.

        Args:
            file_idx: index of starting frame.

        Returns:
            Path to file.
        """
        return Path(f"{self.naming_pattern[0]}{file_idx:0{self.n_digits}d}{self.naming_pattern[1]}")

    def _build_file_indices_lang(self, abs_datasets_dir: Path) -> Tuple[np.ndarray, List, np.ndarray]:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the language dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
            lang_lookup: Mapping from training example to index of language instruction.
            lang_ann: Language embeddings.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        try:
            print("trying to load lang data from: ", abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / self.lang_folder / "auto_lang_ann.npy", allow_pickle=True).item()
        except Exception:
            print("Exception, trying to load lang data from: ", abs_datasets_dir / "auto_lang_ann.npy")
            lang_data = np.load(abs_datasets_dir / "auto_lang_ann.npy", allow_pickle=True).item()

        ep_start_end_ids = lang_data["info"]["indx"]  # each of them are 64
        lang_ann = lang_data["language"]["emb"]  # length total number of annotations
        lang_text = lang_data["language"]["ann"]  # length total number of annotations
        self.lang_data_tasks = lang_data["language"].get("task", [""] * len(lang_text))
        lang_lookup = []
        for i, (start_idx, end_idx) in enumerate(ep_start_end_ids):
            if self.pretrain:
                start_idx = max(start_idx, end_idx + 1 - self.min_window_size - self.aux_lang_loss_window)
            assert end_idx >= self.max_window_size
            cnt = 0
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                if cnt % self.skip_frames == 0:
                    lang_lookup.append(i)
                    episode_lookup.append(idx)
                cnt += 1

        return np.array(episode_lookup), lang_lookup, lang_ann, lang_text

    # no use
    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []
        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)


class ExtendedDiskDataset(DiskDataset):
    def __init__(
        self,
        *args: Any,
        obs_seq_len: int,
        action_seq_len: int,
        future_range: int,
        img_gen_frame_diff: int = 3,
        future_k_min: int = 1,
        future_k_max: int = 4,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.obs_seq_len = obs_seq_len  # 1
        self.action_seq_len = action_seq_len  # 10
        self.future_range = future_range  # Number of steps into the future to sample goals 29
        self.ep_start_end_ids = np.load(self.abs_datasets_dir / "ep_start_end_ids.npy")  # Load sequence boundaries (147, 2)
        self.img_gen_frame_diff = img_gen_frame_diff  # 3
        self.random_frame_diff = False if img_gen_frame_diff > -1 else True  # False
        self.future_k_min = future_k_min
        self.future_k_max = future_k_max
        self._trim_episode_lookup_to_valid_starts()

    def _trim_episode_lookup_to_valid_starts(self) -> None:
        required_window = self.obs_seq_len + self.action_seq_len - 1
        if required_window <= self.min_window_size:
            return

        valid_starts: list[int] = []
        valid_lang_lookup: list[int] = []
        lang_lookup = getattr(self, "lang_lookup", None)

        for dataset_idx, start_idx in enumerate(self.episode_lookup.tolist()):
            for ep_start, ep_end in self.ep_start_end_ids:
                if ep_start <= start_idx <= ep_end:
                    if start_idx + required_window - 1 <= ep_end:
                        valid_starts.append(int(start_idx))
                        if lang_lookup is not None:
                            valid_lang_lookup.append(lang_lookup[dataset_idx])
                    break

        self.episode_lookup = np.asarray(valid_starts, dtype=np.int64)
        if lang_lookup is not None:
            self.lang_lookup = valid_lang_lookup

    # no use
    def find_sequence_boundaries(self, idx: int) -> Tuple[int, int]:
        for start_idx, end_idx in self.ep_start_end_ids:
            if start_idx <= idx < end_idx:
                return start_idx, end_idx
        raise ValueError(f"Index {idx} does not belong to any sequence.")

    def _sample_future_offset(self, dataset_idx: int, current_idx: int, episode_end: int) -> int:
        max_valid = min(self.future_k_max, max(1, episode_end - current_idx))
        min_valid = min(self.future_k_min, max_valid)
        if self.validation:
            span = max_valid - min_valid + 1
            return min_valid + (hash((int(current_idx), int(dataset_idx))) % span)
        return random.randint(min_valid, max_valid)

    def _load_episode(self, idx: int, window_size: int) -> Dict[str, np.ndarray]:
        """
        Load consecutive frames saved as individual files on disk and combine to episode dict.

        Args:
            idx: Index of first frame.
            window_size: Length of sampled episode.

        Returns:
            episode: Dict of numpy arrays containing the episode where keys are the names of modalities.
        """
        start_idx = self.episode_lookup[idx]
        episode_start, episode_end = self.find_sequence_boundaries(start_idx)
        end_idx = start_idx + self.action_seq_len + self.obs_seq_len - 1
        keys = list(chain(*self.observation_space.values()))
        keys.remove("language")
        keys.append("scene_obs")
        episodes = [self.load_file(self._get_episode_name(file_idx)) for file_idx in range(start_idx, end_idx)]
        current_episode = episodes[0]
        next_idx = min(start_idx + 1, episode_end)
        next_episode = self.load_file(self._get_episode_name(next_idx))
        future_offset = self._sample_future_offset(idx, start_idx, episode_end)
        future_idx = min(start_idx + future_offset, episode_end)
        future_episode = self.load_file(self._get_episode_name(future_idx))
        if "stage" in episodes[0]:
            keys.append("stage")
        optional_keys = [
            "trans_action_indicies",
            "rot_grip_action_indicies",
            "ignore_collisions",
            "gripper_pose",
            "rlbench_target_index",
        ]
        for key in optional_keys:
            if key in episodes[0]:
                keys.append(key)

        episode = {}
        for key in keys:
            if 'gen' in key:
                continue
            stacked_data = np.stack([ep[key] for ep in episodes])
            if key in self.observation_space["actions"]:
                episode[key] = stacked_data[(self.obs_seq_len-1):((self.obs_seq_len-1) + self.action_seq_len), :]
            elif key == "stage":
                episode[key] = stacked_data[(self.obs_seq_len-1):((self.obs_seq_len-1) + self.action_seq_len)]
            elif key in optional_keys:
                episode[key] = stacked_data[(self.obs_seq_len-1):((self.obs_seq_len-1) + self.action_seq_len)]
            else:
                episode[key] = stacked_data[:self.obs_seq_len, :]

        episode["target_delta_xyz"] = (
            np.asarray(next_episode["robot_obs"][:3], dtype=np.float32)
            - np.asarray(current_episode["robot_obs"][:3], dtype=np.float32)
        ).astype(np.float32)
        episode["target_gripper"] = np.asarray([float(next_episode["robot_obs"][6] > 0.5)], dtype=np.int64)
        episode["target_collision"] = np.asarray([float(next_episode["robot_obs"][7] > 0.5)], dtype=np.int64)
        episode["future_offset"] = np.asarray([future_offset], dtype=np.int64)
        for key in self.observation_space["rgb_obs"]:
            episode[f"future_{key}"] = np.expand_dims(future_episode[key], axis=0)
        for key in self.observation_space.get("depth_obs", []):
            if key in future_episode:
                episode[f"future_{key}"] = np.expand_dims(future_episode[key], axis=0)
        for key in self.observation_space.get("point_cloud_obs", []):
            if key in future_episode:
                episode[f"future_{key}"] = np.expand_dims(future_episode[key], axis=0)
        for key in self.observation_space.get("camera_obs", []):
            if key in future_episode:
                episode[f"future_{key}"] = np.expand_dims(future_episode[key], axis=0)
        episode["future_robot_obs"] = np.expand_dims(future_episode["robot_obs"], axis=0)

        if self.with_lang:  # True
            episode["language"] = self.lang_ann[self.lang_lookup[idx]][0]
            episode["language_text"] = self.lang_text[self.lang_lookup[idx]]
        
        return episode
       
    # no use
    def merge_episodes(self, episode1: Dict[str, np.ndarray], episode2: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        merged_episode = {}
        all_keys = set(episode1.keys()).union(set(episode2.keys()))
        for key in all_keys:
            if key in episode1 and key in episode2:
                # Merge logic here, for example:
                merged_episode[key] = np.concatenate([episode1[key], episode2[key]], axis=0)
            elif key in episode1:
                merged_episode[key] = episode1[key]
            else:
                merged_episode[key] = episode2[key]
        return merged_episode
    
    # no use
    def _build_file_indices(self, abs_datasets_dir: Path) -> np.ndarray:
        """
        This method builds the mapping from index to file_name used for loading the episodes of the non language
        dataset.

        Args:
            abs_datasets_dir: Absolute path of the directory containing the dataset.

        Returns:
            episode_lookup: Mapping from training example index to episode (file) index.
        """
        assert abs_datasets_dir.is_dir()

        episode_lookup = []

        ep_start_end_ids = np.load(abs_datasets_dir / "ep_start_end_ids.npy")
        logger.info(f'Found "ep_start_end_ids.npy" with {len(ep_start_end_ids)} episodes.')
        for start_idx, end_idx in ep_start_end_ids:
            assert end_idx > self.max_window_size
            for idx in range(start_idx, end_idx + 1 - self.min_window_size):
                episode_lookup.append(idx)
        return np.array(episode_lookup)
