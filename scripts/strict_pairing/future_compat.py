"""Scoped import bridge for legacy diffusers; never downgrade the original environment."""
from pathlib import Path

def install():
    import huggingface_hub as hub
    if not hasattr(hub,'cached_download'):
        def cached_download(url,*args,**kwargs):
            # Legacy diffusers imports this symbol for dynamic remote modules.
            # Experiment uses pre-existing local models only. Never silently
            # download different weights to make an ablation succeed.
            path=Path(str(url))
            if path.is_file():return str(path)
            raise RuntimeError('Legacy cached_download remote loading disabled: provide the original local model folders')
        hub.cached_download=cached_download
