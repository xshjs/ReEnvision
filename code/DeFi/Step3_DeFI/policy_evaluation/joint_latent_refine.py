from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from score_state_action_dynamics import load_probe as load_dynamics_probe, score_pair as score_dynamics_pair
from train_joint_future_action_energy import JointFutureActionEnergy


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def future_trace_from_row(row: dict[str, Any]) -> Path | None:
    for key in ("trace_path", "collection_trace_path", "target_proxy_trace_path"):
        raw = str(row.get(key, "") or "")
        if raw and Path(raw).exists():
            return Path(raw)
    return None


def action_trace_from_row(row: dict[str, Any], future_trace: Path | None) -> Path | None:
    for key in ("action_trace_path", "collection_action_trace_path"):
        raw = str(row.get(key, "") or "")
        if raw and Path(raw).exists():
            return Path(raw)
    if future_trace is None:
        return None
    candidate = future_trace.parent.parent / "action_traces" / future_trace.name.replace("future_trace_", "action_trace_")
    return candidate if candidate.exists() else None


def load_future(path: Path, key: str) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            if key in data:
                arr = np.asarray(data[key], dtype=np.float32)
            elif key == "target_proxy_future" and "final_future" in data:
                arr = np.asarray(data["final_future"], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    return arr if arr.ndim == 2 else None


def summarize_future_exec(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0), arr[0], arr[-1] - arr[0]], axis=0).astype(np.float32)


def load_action_chunk(path: Path, chunk_start: int, chunk_len: int) -> np.ndarray | None:
    try:
        with np.load(path) as data:
            actions = np.asarray(data["actions"], dtype=np.float32)
    except Exception:
        return None
    if actions.ndim != 2 or actions.shape[0] <= 0:
        return None
    sl = actions[max(0, int(chunk_start)) : max(0, int(chunk_start)) + int(chunk_len)]
    out = np.zeros((chunk_len, actions.shape[-1]), dtype=np.float32)
    take = min(chunk_len, sl.shape[0])
    if take > 0:
        out[:take] = sl[:take]
    return out


def hashed_task(task: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in str(task).replace("_", " ").split():
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(digest, "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-6)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return 0.0 if denom <= 1e-8 else float(np.dot(a, b) / denom)


def load_memory(npz_path: Path, success_only: bool):
    with np.load(npz_path, allow_pickle=True) as data:
        mem = {k: data[k] for k in data.files}
    if success_only:
        mask = mem["success"] > 0.5
        for key, value in list(mem.items()):
            mem[key] = value[mask]
    return mem


def load_joint_energy(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location="cpu")
    model = JointFutureActionEnergy(
        action_dim=int(ckpt["action_dim"]),
        future_dim=int(ckpt["future_dim"]),
        task_dim=int(ckpt["task_dim"]),
        hidden_dim=int(ckpt["hidden_dim"]),
        dropout=float(ckpt.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {
        "model": model,
        "task_dim": int(ckpt["task_dim"]),
        "future_dim": int(ckpt["future_dim"]),
        "action_dim": int(ckpt["action_dim"]),
    }


def row_state_start(row: dict[str, Any]) -> np.ndarray | None:
    robot_start = row.get("strong_gt_robot_obs_start")
    scene_start = row.get("strong_gt_scene_obs_start")
    if not isinstance(robot_start, list) or not isinstance(scene_start, list) or not robot_start or not scene_start:
        return None
    try:
        return np.concatenate(
            [
                np.asarray(robot_start, dtype=np.float32).reshape(-1),
                np.asarray(scene_start, dtype=np.float32).reshape(-1),
            ],
            axis=0,
        ).astype(np.float32)
    except Exception:
        return None


def energy_value(
    model: JointFutureActionEnergy,
    base_summary_exec: torch.Tensor,
    target_summary_exec: torch.Tensor,
    action_chunk: torch.Tensor,
    task_vec: torch.Tensor,
    subtask: torch.Tensor,
) -> torch.Tensor:
    target_delta_exec = target_summary_exec - base_summary_exec
    return model(
        base_summary_exec,
        target_summary_exec,
        target_delta_exec,
        action_chunk,
        task_vec,
        subtask,
    )


def retrieve_candidates(
    memory: dict[str, np.ndarray],
    task: str,
    base_summary_exec: np.ndarray,
    top_k: int,
    same_task_only: bool,
) -> list[dict[str, Any]]:
    task_mask = np.ones(len(memory["task"]), dtype=bool)
    if same_task_only:
        task_mask = np.asarray(memory["task"] == task)
    candidate_ids = np.where(task_mask)[0]
    if candidate_ids.size == 0:
        return []
    sims = np.asarray([cosine(base_summary_exec, np.asarray(memory["base_summary_exec"][idx], dtype=np.float32)) for idx in candidate_ids], dtype=np.float32)
    order = np.argsort(-sims)[: min(top_k, len(candidate_ids))]
    results = []
    for rank, local_idx in enumerate(order.tolist(), start=1):
        mem_idx = int(candidate_ids[local_idx])
        results.append(
            {
                "rank_retrieval": rank,
                "memory_index": mem_idx,
                "similarity": float(sims[local_idx]),
                "task": str(memory["task"][mem_idx]),
                "factor": str(memory["factor"][mem_idx]),
                "outcome": str(memory["outcome"][mem_idx]),
                "target_summary_exec": np.asarray(memory["target_summary_exec"][mem_idx], dtype=np.float32),
                "action_chunk": np.asarray(memory["action_chunk"][mem_idx], dtype=np.float32),
                "chunk_start": int(memory["chunk_start"][mem_idx]),
                "trace_path": str(memory["trace_path"][mem_idx]),
                "action_trace_path": str(memory["action_trace_path"][mem_idx]),
            }
        )
    return results


def weighted_mean(candidates: list[dict[str, Any]], key: str, temperature: float = 0.1) -> np.ndarray:
    if not candidates:
        raise ValueError("empty candidates")
    values = np.stack([np.asarray(c[key], dtype=np.float32) for c in candidates], axis=0)
    sims = np.asarray([float(c["similarity"]) for c in candidates], dtype=np.float32)
    sims = sims - sims.max()
    weights = np.exp(sims / max(float(temperature), 1e-6)).astype(np.float32)
    weights = weights / np.maximum(weights.sum(), 1e-8)
    shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return (values * weights.reshape(shape)).sum(axis=0).astype(np.float32)


def pair_distances(
    future_np: np.ndarray,
    action_np: np.ndarray,
    candidates: list[dict[str, Any]],
) -> dict[str, float]:
    if not candidates:
        return {
            "nearest_joint_l2": 0.0,
            "nearest_future_l2": 0.0,
            "nearest_action_l2": 0.0,
        }
    future_d = []
    action_d = []
    joint_d = []
    for cand in candidates:
        cf = np.asarray(cand["target_summary_exec"], dtype=np.float32)
        ca = np.asarray(cand["action_chunk"], dtype=np.float32)
        df = float(np.linalg.norm((future_np - cf).reshape(-1), ord=2))
        da = float(np.linalg.norm((action_np - ca).reshape(-1), ord=2))
        future_d.append(df)
        action_d.append(da)
        joint_d.append((df**2 + da**2) ** 0.5)
    return {
        "nearest_joint_l2": float(min(joint_d)),
        "nearest_future_l2": float(min(future_d)),
        "nearest_action_l2": float(min(action_d)),
    }


def optimize_pair(
    model: JointFutureActionEnergy,
    base_summary_exec_np: np.ndarray,
    init_future_np: np.ndarray,
    init_action_np: np.ndarray,
    prior_future_np: np.ndarray,
    prior_action_np: np.ndarray,
    task_vec_np: np.ndarray,
    subtask_scalar: float,
    optimize_za_only: bool,
    steps: int,
    lr_zf: float,
    lr_za: float,
    lambda_future_anchor: float,
    lambda_action_anchor: float,
    lambda_future_base: float,
    lambda_future_prior: float,
    lambda_action_prior: float,
    max_future_shift: float,
    max_action_shift: float,
    min_step_scale: float,
    backtrack_factor: float,
    max_backtracks: int,
    early_stop_patience: int,
    early_stop_min_improve: float,
    early_stop_grad_norm: float,
    device: torch.device,
) -> dict[str, Any]:
    base_summary_exec = torch.from_numpy(base_summary_exec_np).unsqueeze(0).to(device)
    init_future = torch.from_numpy(init_future_np).unsqueeze(0).to(device)
    init_action = torch.from_numpy(init_action_np).unsqueeze(0).to(device)
    prior_future = torch.from_numpy(prior_future_np).unsqueeze(0).to(device)
    prior_action = torch.from_numpy(prior_action_np).unsqueeze(0).to(device)
    task_vec = torch.from_numpy(task_vec_np).unsqueeze(0).to(device)
    subtask = torch.tensor([[subtask_scalar]], dtype=torch.float32, device=device)

    zf = init_future.clone().detach()
    za = init_action.clone().detach()

    history = []
    initial_energy = float(energy_value(model, base_summary_exec, init_future, init_action, task_vec, subtask).detach().cpu().item())
    best_objective = None
    no_improve_steps = 0
    stopped_early = False
    stop_reason = ""

    def project_pair(cur_future: torch.Tensor, cur_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_future = cur_future
        out_action = cur_action
        if not optimize_za_only:
            delta_zf = out_future - init_future
            zf_norm = delta_zf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            zf_scale = torch.clamp(torch.tensor(max_future_shift, device=device) / zf_norm, max=1.0)
            out_future = init_future + delta_zf * zf_scale
        delta_za = out_action - init_action
        za_norm = delta_za.reshape(delta_za.shape[0], -1).norm(dim=-1, keepdim=True).clamp_min(1e-6)
        za_scale = torch.clamp(torch.tensor(max_action_shift, device=device) / za_norm, max=1.0)
        out_action = init_action + delta_za * za_scale.view(-1, 1, 1)
        return out_future, out_action

    def objective_parts(cur_future: torch.Tensor, cur_action: torch.Tensor) -> dict[str, torch.Tensor]:
        e = energy_value(model, base_summary_exec, cur_future, cur_action, task_vec, subtask).mean()
        future_anchor = F.mse_loss(cur_future, init_future)
        action_anchor = F.mse_loss(cur_action, init_action)
        future_base = F.mse_loss(cur_future, base_summary_exec)
        future_prior = F.mse_loss(cur_future, prior_future)
        action_prior = F.mse_loss(cur_action, prior_action)
        loss = (
            e
            + float(lambda_future_anchor) * future_anchor
            + float(lambda_action_anchor) * action_anchor
            + float(lambda_future_base) * future_base
            + float(lambda_future_prior) * future_prior
            + float(lambda_action_prior) * action_prior
        )
        return {
            "loss": loss,
            "energy": e,
            "future_anchor": future_anchor,
            "action_anchor": action_anchor,
            "future_base": future_base,
            "future_prior": future_prior,
            "action_prior": action_prior,
        }

    for step_idx in range(1, steps + 1):
        zf_var = (init_future if optimize_za_only else zf.clone().detach()).requires_grad_(not optimize_za_only)
        za_var = za.clone().detach().requires_grad_(True)
        current_zf = init_future if optimize_za_only else zf_var
        current_za = za_var
        parts = objective_parts(current_zf, current_za)
        loss = parts["loss"]
        loss.backward()
        grad_future_norm = 0.0 if optimize_za_only else float(zf_var.grad.norm().detach().cpu().item())
        grad_action_norm = float(za_var.grad.norm().detach().cpu().item())
        grad_norm = (grad_future_norm**2 + grad_action_norm**2) ** 0.5

        current_objective = float(loss.detach().cpu().item())
        accepted = False
        step_scale = 1.0
        accept_objective = current_objective
        accept_energy = float(parts["energy"].detach().cpu().item())
        accept_future = init_future if optimize_za_only else zf
        accept_action = za

        for _ in range(max_backtracks):
            with torch.no_grad():
                cand_future = init_future if optimize_za_only else (zf - step_scale * float(lr_zf) * zf_var.grad)
                cand_action = za - step_scale * float(lr_za) * za_var.grad
                cand_future, cand_action = project_pair(cand_future, cand_action)
                cand_parts = objective_parts(cand_future, cand_action)
                cand_objective = float(cand_parts["loss"].detach().cpu().item())
                if cand_objective + 1e-8 < current_objective:
                    accepted = True
                    accept_objective = cand_objective
                    accept_energy = float(cand_parts["energy"].detach().cpu().item())
                    accept_future = cand_future.detach().clone()
                    accept_action = cand_action.detach().clone()
                    break
            step_scale *= float(backtrack_factor)
            if step_scale < float(min_step_scale):
                break

        if accepted:
            if not optimize_za_only:
                zf = accept_future
            za = accept_action
            improve = current_objective - accept_objective
            if best_objective is None or accept_objective < best_objective - float(early_stop_min_improve):
                best_objective = accept_objective
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            improve = 0.0
            no_improve_steps += 1

        if step_idx == 1 or step_idx == steps or step_idx % max(1, steps // 5) == 0:
            with torch.no_grad():
                current_energy = float(energy_value(model, base_summary_exec, init_future if optimize_za_only else zf, za, task_vec, subtask).detach().cpu().item())
                current_future_shift = 0.0 if optimize_za_only else float((zf - init_future).norm(dim=-1).mean().detach().cpu().item())
                current_action_shift = float((za - init_action).reshape(za.shape[0], -1).norm(dim=-1).mean().detach().cpu().item())
            history.append(
                {
                    "step": step_idx,
                    "energy": current_energy,
                    "objective": float(accept_objective if accepted else current_objective),
                    "future_shift": current_future_shift,
                    "action_shift": current_action_shift,
                    "future_prior_mse": float(F.mse_loss(init_future if optimize_za_only else zf, prior_future).detach().cpu().item()),
                    "action_prior_mse": float(F.mse_loss(za, prior_action).detach().cpu().item()),
                    "grad_norm": float(grad_norm),
                    "step_scale": float(step_scale),
                    "accepted": bool(accepted),
                    "objective_improve": float(improve),
                }
            )

        if grad_norm <= float(early_stop_grad_norm):
            stopped_early = True
            stop_reason = "grad_norm"
            break
        if improve <= float(early_stop_min_improve):
            if no_improve_steps >= int(early_stop_patience):
                stopped_early = True
                stop_reason = "no_improve"
                break

    with torch.no_grad():
        refined_zf = init_future if optimize_za_only else zf
        refined_za = za
        final_parts = objective_parts(refined_zf, refined_za)
        final_energy = float(final_parts["energy"].detach().cpu().item())
        future_shift = 0.0 if optimize_za_only else float((refined_zf - init_future).norm(dim=-1).mean().detach().cpu().item())
        action_shift = float((refined_za - init_action).reshape(refined_za.shape[0], -1).norm(dim=-1).mean().detach().cpu().item())
        future_prior_mse = 0.0 if optimize_za_only else float(F.mse_loss(refined_zf, prior_future).detach().cpu().item())
        action_prior_mse = float(F.mse_loss(refined_za, prior_action).detach().cpu().item())

    return {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "delta_energy": float(initial_energy - final_energy),
        "future_shift": future_shift,
        "action_shift": action_shift,
        "future_prior_mse": future_prior_mse,
        "action_prior_mse": action_prior_mse,
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "refined_target_summary_exec": refined_zf.detach().cpu().numpy()[0].astype(np.float32),
        "refined_action_chunk": refined_za.detach().cpu().numpy()[0].astype(np.float32),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint latent refinement over CALVIN future-action memory using learned joint energy.")
    parser.add_argument("--query-jsonl", type=Path, required=True)
    parser.add_argument("--memory-npz", type=Path, required=True)
    parser.add_argument("--joint-energy-ckpt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--future-key", default="base_future")
    parser.add_argument("--target-future-key", default="target_proxy_future")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--same-task-only", action="store_true")
    parser.add_argument("--success-only-memory", action="store_true")
    parser.add_argument("--dynamics-probe-ckpt", type=Path, default=None)
    parser.add_argument("--lambda-dynamics", type=float, default=0.0)
    parser.add_argument("--init-from-query-target", action="store_true")
    parser.add_argument("--optimize-za-only", action="store_true")
    parser.add_argument("--init-mode", choices=["best", "mean"], default="mean")
    parser.add_argument("--prior-top-k", type=int, default=4)
    parser.add_argument("--prior-temperature", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--lr-zf", type=float, default=0.05)
    parser.add_argument("--lr-za", type=float, default=0.05)
    parser.add_argument("--lambda-future-anchor", type=float, default=0.5)
    parser.add_argument("--lambda-action-anchor", type=float, default=0.5)
    parser.add_argument("--lambda-future-base", type=float, default=0.05)
    parser.add_argument("--lambda-future-prior", type=float, default=0.5)
    parser.add_argument("--lambda-action-prior", type=float, default=0.5)
    parser.add_argument("--max-future-shift", type=float, default=2.0)
    parser.add_argument("--max-action-shift", type=float, default=5.0)
    parser.add_argument("--min-step-scale", type=float, default=0.03125)
    parser.add_argument("--backtrack-factor", type=float, default=0.5)
    parser.add_argument("--max-backtracks", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=2)
    parser.add_argument("--early-stop-min-improve", type=float, default=1e-3)
    parser.add_argument("--early-stop-grad-norm", type=float, default=1e-3)
    parser.add_argument("--gate-min-delta-energy", type=float, default=0.05)
    parser.add_argument("--gate-max-future-shift", type=float, default=2.0)
    parser.add_argument("--gate-max-action-shift", type=float, default=5.0)
    parser.add_argument("--gate-min-novel-joint-l2", type=float, default=0.0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    bundle = load_joint_energy(args.joint_energy_ckpt, device)
    memory = load_memory(args.memory_npz, args.success_only_memory)
    model = bundle["model"]
    dynamics_bundle = None
    if args.dynamics_probe_ckpt is not None:
        dynamics_bundle = load_dynamics_probe(args.dynamics_probe_ckpt, device)

    results = []
    refined_applied = 0
    total = 0

    for row in iter_jsonl(args.query_jsonl):
        future_trace = future_trace_from_row(row)
        if future_trace is None:
            continue
        base = load_future(future_trace, args.future_key)
        if base is None:
            continue
        base_summary_exec = summarize_future_exec(base)
        task = str(row.get("task", "unknown"))
        subtask_scalar = float(row.get("subtask_index", 0) or 0.0) / 5.0
        task_vec = hashed_task(task, bundle["task_dim"]).astype(np.float32)
        query_state = row_state_start(row)

        candidates = retrieve_candidates(memory, task, base_summary_exec, args.top_k, args.same_task_only)
        if not candidates:
            continue
        if dynamics_bundle is not None and query_state is not None and "state_transition" in memory:
            d_model, d_mean, d_std = dynamics_bundle
            rescored = []
            for cand in candidates:
                mem_idx = int(cand["memory_index"])
                dyn = score_dynamics_pair(
                    d_model,
                    d_mean,
                    d_std,
                    query_state,
                    np.asarray(cand["action_chunk"], dtype=np.float32),
                    np.asarray(memory["state_transition"][mem_idx], dtype=np.float32),
                    device,
                )
                fused = float(cand["similarity"] + float(args.lambda_dynamics) * dyn["dynamics_score"])
                cand = {**cand, **dyn, "init_fused_score": fused}
                rescored.append(cand)
            rescored.sort(key=lambda x: float(x["init_fused_score"]), reverse=True)
            candidates = rescored
        best = candidates[0]
        prior_candidates = candidates[: max(1, min(int(args.prior_top_k), len(candidates)))]

        init_future = weighted_mean(prior_candidates, "target_summary_exec", args.prior_temperature) if args.init_mode == "mean" else np.asarray(best["target_summary_exec"], dtype=np.float32)
        if args.init_from_query_target:
            query_target = load_future(future_trace, args.target_future_key)
            if query_target is not None:
                init_future = summarize_future_exec(query_target)

        init_action = weighted_mean(prior_candidates, "action_chunk", args.prior_temperature) if args.init_mode == "mean" else np.asarray(best["action_chunk"], dtype=np.float32)
        query_action_trace = action_trace_from_row(row, future_trace)
        if query_action_trace is not None:
            query_chunk = load_action_chunk(query_action_trace, int(best["chunk_start"]), init_action.shape[0])
            if query_chunk is not None and query_chunk.shape == init_action.shape:
                init_action = query_chunk

        prior_future = weighted_mean(prior_candidates, "target_summary_exec", args.prior_temperature)
        prior_action = weighted_mean(prior_candidates, "action_chunk", args.prior_temperature)

        refine = optimize_pair(
            model=model,
            base_summary_exec_np=base_summary_exec,
            init_future_np=init_future,
            init_action_np=init_action,
            prior_future_np=prior_future,
            prior_action_np=prior_action,
            task_vec_np=task_vec,
            subtask_scalar=subtask_scalar,
            optimize_za_only=bool(args.optimize_za_only),
            steps=int(args.steps),
            lr_zf=float(args.lr_zf),
            lr_za=float(args.lr_za),
            lambda_future_anchor=float(args.lambda_future_anchor),
            lambda_action_anchor=float(args.lambda_action_anchor),
            lambda_future_base=float(args.lambda_future_base),
            lambda_future_prior=float(args.lambda_future_prior),
            lambda_action_prior=float(args.lambda_action_prior),
            max_future_shift=float(args.max_future_shift),
            max_action_shift=float(args.max_action_shift),
            min_step_scale=float(args.min_step_scale),
            backtrack_factor=float(args.backtrack_factor),
            max_backtracks=int(args.max_backtracks),
            early_stop_patience=int(args.early_stop_patience),
            early_stop_min_improve=float(args.early_stop_min_improve),
            early_stop_grad_norm=float(args.early_stop_grad_norm),
            device=device,
        )

        init_dist = pair_distances(init_future, init_action, candidates)
        refined_dist = pair_distances(refine["refined_target_summary_exec"], refine["refined_action_chunk"], candidates)
        novelty_gain = float(refined_dist["nearest_joint_l2"] - init_dist["nearest_joint_l2"])

        applied = bool(
            refine["delta_energy"] > float(args.gate_min_delta_energy)
            and refine["future_shift"] <= float(args.gate_max_future_shift)
            and refine["action_shift"] <= float(args.gate_max_action_shift)
            and refined_dist["nearest_joint_l2"] >= float(args.gate_min_novel_joint_l2)
        )
        refined_applied += int(applied)
        total += 1

        results.append(
            {
                "query_task": task,
                "query_trace_path": str(future_trace),
                "query_action_trace_path": str(query_action_trace) if query_action_trace is not None else "",
                "init_source": {
                    "memory_index": int(best["memory_index"]),
                    "memory_task": str(best["task"]),
                    "memory_factor": str(best["factor"]),
                    "memory_outcome": str(best["outcome"]),
                    "similarity": float(best["similarity"]),
                    "chunk_start": int(best["chunk_start"]),
                    "trace_path": str(best["trace_path"]),
                    "action_trace_path": str(best["action_trace_path"]),
                    "init_from_query_target": bool(args.init_from_query_target),
                    "optimize_za_only": bool(args.optimize_za_only),
                    "init_mode": str(args.init_mode),
                    "prior_top_k": int(len(prior_candidates)),
                    "lambda_dynamics": float(args.lambda_dynamics),
                    "init_dynamics_score": float(best.get("dynamics_score", 0.0)),
                    "init_fused_score": float(best.get("init_fused_score", 0.0)),
                },
                "refine": {
                    "applied": applied,
                    "initial_energy": float(refine["initial_energy"]),
                    "final_energy": float(refine["final_energy"]),
                    "delta_energy": float(refine["delta_energy"]),
                    "future_shift": float(refine["future_shift"]),
                    "action_shift": float(refine["action_shift"]),
                    "future_prior_mse": float(refine["future_prior_mse"]),
                    "action_prior_mse": float(refine["action_prior_mse"]),
                    "stopped_early": bool(refine["stopped_early"]),
                    "stop_reason": str(refine["stop_reason"]),
                    "init_nearest_joint_l2": float(init_dist["nearest_joint_l2"]),
                    "refined_nearest_joint_l2": float(refined_dist["nearest_joint_l2"]),
                    "init_nearest_future_l2": float(init_dist["nearest_future_l2"]),
                    "refined_nearest_future_l2": float(refined_dist["nearest_future_l2"]),
                    "init_nearest_action_l2": float(init_dist["nearest_action_l2"]),
                    "refined_nearest_action_l2": float(refined_dist["nearest_action_l2"]),
                    "novelty_gain_joint_l2": novelty_gain,
                    "generated_new_pair": bool(refined_dist["nearest_joint_l2"] > 1e-4),
                    "history": refine["history"],
                },
                "refined_target_summary_exec": refine["refined_target_summary_exec"].tolist(),
                "refined_action_chunk": refine["refined_action_chunk"].tolist(),
            }
        )
        if args.max_queries > 0 and total >= args.max_queries:
            break

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "query_jsonl": str(args.query_jsonl),
        "memory_npz": str(args.memory_npz),
        "joint_energy_ckpt": str(args.joint_energy_ckpt),
        "output_jsonl": str(args.output_jsonl),
        "num_queries": int(total),
        "top_k": int(args.top_k),
        "same_task_only": bool(args.same_task_only),
        "success_only_memory": bool(args.success_only_memory),
        "init_from_query_target": bool(args.init_from_query_target),
        "optimize_za_only": bool(args.optimize_za_only),
        "init_mode": str(args.init_mode),
        "prior_top_k": int(args.prior_top_k),
        "refined_applied": int(refined_applied),
        "refined_apply_rate": float(refined_applied / max(total, 1)),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
