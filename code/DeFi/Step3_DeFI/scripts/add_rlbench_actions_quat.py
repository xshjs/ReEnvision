import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


def action_euler_to_quat(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    pos = action[:3]
    quat = R.from_euler("xyz", action[3:6], degrees=False).as_quat().astype(np.float32)
    quat = quat / np.linalg.norm(quat).clip(min=1e-6)
    return np.concatenate([pos, quat, action[6:7]], axis=0).astype(np.float32)


def rewrite_npz(path: Path) -> bool:
    data = np.load(path)
    if "actions_quat" in data.files:
        return False
    if "actions" not in data.files:
        raise KeyError(f"{path} does not contain actions")
    payload = {key: data[key] for key in data.files}
    payload["actions_quat"] = action_euler_to_quat(payload["actions"])
    tmp_path = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(tmp_path, **payload)
    tmp_path.replace(path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Add actions_quat to an existing RLBench DeFi npz dataset.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["training", "validation"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    total = 0
    updated = 0
    for split in args.splits:
        split_dir = args.root / split
        files = sorted(split_dir.glob("episode_*.npz"))
        if args.limit:
            files = files[: args.limit]
        for idx, path in enumerate(files, start=1):
            total += 1
            if rewrite_npz(path):
                updated += 1
            if idx % 10000 == 0:
                print(f"{split}: processed={idx} updated_total={updated}", flush=True)
    print({"processed": total, "updated": updated}, flush=True)


if __name__ == "__main__":
    main()
