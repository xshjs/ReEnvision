from policy_models.module.rlbench_mvt import RLBenchMVT


class RLBenchVoxelEncoder(RLBenchMVT):
    """Backward-compatible alias.

    Older configs/code paths still import `RLBenchVoxelEncoder`. The actual
    implementation now lives in `RLBenchMVT`.
    """

    pass
