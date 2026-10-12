#!/usr/bin/env python3
"""Visualize where a future-latent editor changes CALVIN future tokens/channels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(run_log: Path) -> list[dict]:
    rows = []
    with run_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "[collect-row]" not in line and "[joint-editor-row]" not in line:
                continue
            marker = "[collect-row]" if "[collect-row]" in line else "[joint-editor-row]"
            try:
                rows.append(json.loads(line.split(marker, 1)[1].strip()))
            except Exception:
                continue
    return rows


def load_edit(row: dict):
    path = row.get("trace") or row.get("trace_path") or row.get("collection_trace_path")
    if not path or not Path(path).exists():
        return None
    try:
        data = np.load(path)
    except Exception:
        return None
    if "edit_direction" in data:
        edit = np.asarray(data["edit_direction"], dtype=np.float32)
    elif "final_future" in data and ("base_future" in data or "original_base_future" in data):
        base_key = "base_future" if "base_future" in data else "original_base_future"
        edit = np.asarray(data["final_future"], dtype=np.float32) - np.asarray(data[base_key], dtype=np.float32)
    else:
        return None
    if edit.ndim != 2:
        return None
    return edit


def group_energy(edit: np.ndarray, group_size: int):
    token_energy = np.linalg.norm(edit, axis=1)
    groups = []
    for start in range(0, edit.shape[1], group_size):
        groups.append(float(np.linalg.norm(edit[:, start : start + group_size])))
    return token_energy, np.asarray(groups, dtype=np.float32)


def topk(values, k):
    if values.size == 0:
        return []
    ids = np.argsort(-values)[:k]
    return [{"idx": int(i), "value": float(values[i])} for i in ids]


def plot_heatmap(matrix, title, out_path, xlabel, ylabel):
    plt.figure(figsize=(12, 5))
    plt.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(label="mean |edit| energy")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_bar(values, labels, title, out_path, ylabel):
    plt.figure(figsize=(14, 4))
    plt.bar(np.arange(len(values)), values)
    plt.xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_int_list(text: str) -> list[int]:
    out = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def plot_focused_channels(channel_matrix, task_labels, channel_labels, title, out_path):
    plt.figure(figsize=(max(14, len(channel_labels) * 0.16), max(5, len(task_labels) * 0.35)))
    plt.imshow(channel_matrix, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(label="mean |edit|")
    plt.title(title)
    plt.xlabel("absolute channel id")
    plt.ylabel("task")
    stride = max(1, len(channel_labels) // 32)
    xticks = np.arange(0, len(channel_labels), stride)
    plt.xticks(xticks, [str(channel_labels[i]) for i in xticks], rotation=90, fontsize=6)
    plt.yticks(np.arange(len(task_labels)), task_labels, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--min-edit-norm", type=float, default=1e-6)
    parser.add_argument("--focus-groups", type=str, default="")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.run_log))
    examples = []
    by_task_tokens = defaultdict(list)
    by_task_groups = defaultdict(list)
    by_task_channel_abs = defaultdict(list)
    by_task_norms = defaultdict(list)
    by_task_success = defaultdict(list)
    all_tokens, all_groups, all_norms = [], [], []

    for row in rows:
        edit = load_edit(row)
        if edit is None:
            continue
        norm = float(np.linalg.norm(edit.reshape(-1)))
        if norm < args.min_edit_norm:
            continue
        token_e, group_e = group_energy(edit, args.group_size)
        channel_abs = np.mean(np.abs(edit), axis=0)
        task = str(row.get("task", "unknown"))
        by_task_tokens[task].append(token_e)
        by_task_groups[task].append(group_e)
        by_task_channel_abs[task].append(channel_abs)
        by_task_norms[task].append(norm)
        by_task_success[task].append(bool(row.get("success", False)))
        all_tokens.append(token_e)
        all_groups.append(group_e)
        all_norms.append(norm)
        examples.append(
            {
                "seq": row.get("seq", row.get("sequence_index")),
                "sub": row.get("sub", row.get("subtask_index")),
                "task": task,
                "success": bool(row.get("success", False)),
                "edit_norm": norm,
                "top_time_tokens": topk(token_e, args.topk),
                "top_channel_groups": topk(group_e, args.topk),
            }
        )

    if not examples:
        raise RuntimeError("no edited traces found")

    task_order = [task for task, _ in Counter(ex["task"] for ex in examples).most_common()]
    max_groups = max(len(g) for groups in by_task_groups.values() for g in groups)
    group_matrix = []
    token_matrix = []
    task_summary = {}
    for task in task_order:
        g = np.stack(by_task_groups[task], axis=0)
        t = np.stack(by_task_tokens[task], axis=0)
        group_matrix.append(g.mean(axis=0))
        token_matrix.append(t.mean(axis=0))
        task_summary[task] = {
            "n": len(by_task_norms[task]),
            "success_rate": float(np.mean(by_task_success[task])),
            "edit_norm_mean": float(np.mean(by_task_norms[task])),
            "edit_norm_median": float(np.median(by_task_norms[task])),
            "top_channel_groups": topk(g.mean(axis=0), args.topk),
            "top_time_tokens": topk(t.mean(axis=0), args.topk),
        }

    group_matrix = np.stack(group_matrix, axis=0)
    token_matrix = np.stack(token_matrix, axis=0)
    all_group_mean = np.stack(all_groups, axis=0).mean(axis=0)
    all_token_mean = np.stack(all_tokens, axis=0).mean(axis=0)

    plot_heatmap(group_matrix, "Edit energy by task and channel group", out_dir / "task_channel_group_heatmap.png", "channel group", "task")
    plot_heatmap(token_matrix, "Edit energy by task and time token", out_dir / "task_time_token_heatmap.png", "time token", "task")
    plot_bar(all_group_mean, [str(i) for i in range(len(all_group_mean))], "Overall edit energy by channel group", out_dir / "overall_channel_groups.png", "energy")
    plot_bar(all_token_mean, [str(i) for i in range(len(all_token_mean))], "Overall edit energy by time token", out_dir / "overall_time_tokens.png", "energy")

    focus_groups = parse_int_list(args.focus_groups)
    focused_summary = {}
    if focus_groups:
        channel_ids = []
        for group_id in focus_groups:
            start = group_id * args.group_size
            end = min(start + args.group_size, np.stack(all_groups, axis=0).shape[1] * args.group_size)
            channel_ids.extend(range(start, end))
        channel_ids = [c for c in channel_ids if c < np.stack(by_task_channel_abs[task_order[0]], axis=0).shape[1]]
        focus_matrix = []
        for task in task_order:
            ch = np.stack(by_task_channel_abs[task], axis=0).mean(axis=0)
            focus_matrix.append(ch[channel_ids])
        focus_matrix = np.stack(focus_matrix, axis=0)
        overall_focus = focus_matrix.mean(axis=0)
        plot_focused_channels(
            focus_matrix,
            task_order,
            channel_ids,
            f"Focused edit magnitude inside channel groups {focus_groups}",
            out_dir / "focused_channel_heatmap.png",
        )
        top_focus_ids = np.argsort(-overall_focus)[: min(args.topk * 4, len(overall_focus))]
        focused_summary = {
            "focus_groups": focus_groups,
            "num_focused_channels": len(channel_ids),
            "top_channels": [
                {"channel": int(channel_ids[i]), "value": float(overall_focus[i])}
                for i in top_focus_ids
            ],
            "per_group_top_channels": {},
            "plot": str(out_dir / "focused_channel_heatmap.png"),
        }
        for group_id in focus_groups:
            ids = [i for i, c in enumerate(channel_ids) if group_id * args.group_size <= c < (group_id + 1) * args.group_size]
            vals = overall_focus[ids]
            order = np.argsort(-vals)[: args.topk]
            focused_summary["per_group_top_channels"][str(group_id)] = [
                {"channel": int(channel_ids[ids[j]]), "value": float(vals[j])}
                for j in order
            ]

    with (out_dir / "edit_examples.jsonl").open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    summary = {
        "run_log": str(args.run_log),
        "num_rows": len(rows),
        "num_edited": len(examples),
        "group_size": args.group_size,
        "num_channel_groups": int(max_groups),
        "edit_norm_mean": float(np.mean(all_norms)),
        "edit_norm_median": float(np.median(all_norms)),
        "overall_top_channel_groups": topk(all_group_mean, args.topk),
        "overall_top_time_tokens": topk(all_token_mean, args.topk),
        "task_summary": task_summary,
        "focused_channels": focused_summary,
        "plots": {
            "task_channel_group_heatmap": str(out_dir / "task_channel_group_heatmap.png"),
            "task_time_token_heatmap": str(out_dir / "task_time_token_heatmap.png"),
            "overall_channel_groups": str(out_dir / "overall_channel_groups.png"),
            "overall_time_tokens": str(out_dir / "overall_time_tokens.png"),
            "focused_channel_heatmap": str(out_dir / "focused_channel_heatmap.png") if focus_groups else "",
        },
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
