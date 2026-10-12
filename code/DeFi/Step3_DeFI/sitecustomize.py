import os


def _patch_huggingface_hub() -> None:
    try:
        import huggingface_hub as hub

        if not hasattr(hub, "cached_download") and hasattr(hub, "hf_hub_download"):
            def _cached_download(*args, **kwargs):
                return hub.hf_hub_download(*args, **kwargs)

            hub.cached_download = _cached_download
    except Exception:
        pass


def _patch_transformers_utils() -> None:
    try:
        import transformers.utils as tf_utils

        if not hasattr(tf_utils, "FLAX_WEIGHTS_NAME"):
            tf_utils.FLAX_WEIGHTS_NAME = "flax_model.msgpack"
        if hasattr(tf_utils, "check_torch_load_is_safe"):
            tf_utils.check_torch_load_is_safe = lambda: None
    except Exception:
        pass


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_patch_huggingface_hub()
_patch_transformers_utils()
