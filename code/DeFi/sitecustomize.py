import shutil
import sys
import types

if "distutils.spawn" not in sys.modules:
    spawn_module = types.ModuleType("distutils.spawn")
    spawn_module.find_executable = shutil.which
    sys.modules["distutils.spawn"] = spawn_module
    try:
        import distutils  # type: ignore

        distutils.spawn = spawn_module  # type: ignore[attr-defined]
    except Exception:
        pass

try:
    import torch.utils._pytree as _pytree

    if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
        def _register_pytree_node_compat(node_type, flatten_fn, unflatten_fn, **kwargs):
            del kwargs
            return _pytree._register_pytree_node(node_type, flatten_fn, unflatten_fn)

        _pytree.register_pytree_node = _register_pytree_node_compat
except Exception:
    pass
