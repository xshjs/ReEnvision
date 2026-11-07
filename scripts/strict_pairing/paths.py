"""Find bundled sources without assuming an RSI checkout location."""
import os
from pathlib import Path


def defi_root():
    override=os.environ.get('REENVISION_DEFI_DIR')
    if override:return Path(override).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        candidate=parent/'code/DeFi'
        if (candidate/'Step3_DeFI').is_dir():return candidate.resolve()
    return Path('/mnt/workspace/manipulation/DeFi')


def default_checkpoint(defi):
    explicit=os.environ.get('REENVISION_CHECKPOINT')
    if explicit:return Path(explicit).expanduser()
    for parent in Path(__file__).resolve().parents:
        candidate=parent/'checkpoints/calvin/last.pt'
        if candidate.is_file():return candidate
    return defi/'outputs/calvin_train/2026-10-12_20-24-17/saved_models/best_val.pt'
