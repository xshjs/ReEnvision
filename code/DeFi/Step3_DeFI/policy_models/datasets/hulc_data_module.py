import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split, Subset
import torch
import torchvision
try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover - optional dependency fallback
    class _LightningDataModule:
        def __init__(self, *args, **kwargs):
            super().__init__()

    class _PLNamespace:
        LightningDataModule = _LightningDataModule

    pl = _PLNamespace()

import policy_models
from policy_models.datasets.utils.episode_utils import load_dataset_statistics
from policy_models.datasets.utils.shared_memory_utils import load_shm_lookup, save_shm_lookup, SharedMemoryLoader

logger = logging.getLogger(__name__)
DEFAULT_TRANSFORM = OmegaConf.create({"train": None, "val": None})
ONE_EP_DATASET_URL = "http://www.informatik.uni-freiburg.de/~meeso/50steps.tar.xz"


class HulcDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        root_data_dir: str = "data",
        num_workers: int = 8,
        transforms: DictConfig = DEFAULT_TRANSFORM,
        shuffle_val: bool = False,
        allow_auto_download_debug: bool = False,
        **kwargs: Dict,
    ):
        super().__init__()
        self.datasets_cfg = datasets
        self.train_datasets = None
        self.val_datasets = None
        self.train_sampler = None
        self.val_sampler = None
        self.num_workers = num_workers
        root_data_path = Path(root_data_dir)
        if not root_data_path.is_absolute():
            root_data_path = Path(policy_models.__file__).parent / root_data_path
        self.training_dir = root_data_path / "training"
        self.val_dir = root_data_path / "validation"
        self.shuffle_val = shuffle_val
        self.modalities: List[str] = []
        self.transforms = transforms
        self.use_shm = False
        self.allow_auto_download_debug = allow_auto_download_debug

    def prepare_data(self, *args, **kwargs):
        # check if files already exist
        dataset_exist = np.any([len(list(self.training_dir.glob(extension))) for extension in ["*.npz", "*.pkl"]])

        # download and unpack images
        if not dataset_exist:
            allow_debug_download = self.allow_auto_download_debug or os.environ.get("DEFI_ALLOW_DEBUG_DATASET_DOWNLOAD") == "1"
            if not allow_debug_download:
                raise FileNotFoundError(
                    f"No dataset found in {self.training_dir}. "
                    "Expected real dataset under <root_data_dir>/training and <root_data_dir>/validation. "
                    "If you intentionally want the tiny debug dataset, set DEFI_ALLOW_DEBUG_DATASET_DOWNLOAD=1."
                )
            logger.info(f"downloading dataset to {self.training_dir} and {self.val_dir}")
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.training_dir)
            torchvision.datasets.utils.download_and_extract_archive(ONE_EP_DATASET_URL, self.val_dir)

    def setup(self, stage=None):
        transforms = load_dataset_statistics(self.training_dir, self.val_dir, self.transforms)

        self.train_transforms = {}
        for cam in transforms.train:
            cam_transforms = []
            for transform in transforms.train[cam]:
                if transform._target_ == "torchvision.transforms.ColorJitter":
                    instantiated_transform = torchvision.transforms.ColorJitter(
                        brightness=transform.brightness,
                        contrast=tuple(transform.contrast),
                        saturation=tuple(transform.saturation),
                    )
                else:
                    instantiated_transform = hydra.utils.instantiate(transform)
                cam_transforms.append(instantiated_transform)
            self.train_transforms[cam] = cam_transforms

        self.val_transforms = {
            cam: [hydra.utils.instantiate(transform) for transform in transforms.val[cam]] for cam in transforms.val
        }
        self.train_transforms = {
            key: torchvision.transforms.Compose(val) for key, val in self.train_transforms.items()
        }
        self.val_transforms = {
            key: torchvision.transforms.Compose(val) for key, val in self.val_transforms.items()
        }
        
        self.train_datasets, self.train_sampler, self.val_datasets, self.val_sampler = {}, {}, {}, {}

        for _, dataset in self.datasets_cfg.items():
            if dataset == 'lang_paraphrase-MiniLM-L3-v2':
                continue
            else:
                train_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.training_dir, transforms=self.train_transforms
                )  # calvin:len 696473
                val_dataset = hydra.utils.instantiate(
                    dataset, datasets_dir=self.val_dir, transforms=self.val_transforms
                )  # calvin: len 41966

                key = dataset.key
                self.train_datasets[key] = train_dataset
                self.val_datasets[key] = val_dataset
                self.modalities.append(key)

    def train_dataloader(self):
        loaders = {}
        for key, dataset in self.train_datasets.items():
            kwargs = dict(
                dataset=dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
                shuffle=True,
            )
            if int(dataset.num_workers) > 0:
                kwargs["prefetch_factor"] = 2
            loaders[key] = DataLoader(**kwargs)
        return loaders

    def val_dataloader(self):
        return {
            key: DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                num_workers=dataset.num_workers,
                pin_memory=True,
            )
            for key, dataset in self.val_datasets.items()
        }
