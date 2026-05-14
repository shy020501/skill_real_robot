#!/usr/bin/env python
import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np


AXES = ["fx", "fy", "fz", "tx", "ty", "tz"]
FORCE_HISTORY_KEYS = ("right_force_history", "force_history")
EPS = 1e-8


def trim_zero_action_episode(ep, trailing_keep=5):
    if "actions" not in ep or len(ep["actions"]) == 0:
        return None

    actions = [np.asarray(a) for a in ep["actions"]]
    total_len = len(actions)

    start_idx = 0
    while start_idx < total_len and np.all(actions[start_idx] == 0):
        start_idx += 1
    if start_idx == total_len:
        return None

    end_idx = total_len - 1
    while end_idx >= 0 and np.all(actions[end_idx] == 0):
        end_idx -= 1

    trim_end = min(total_len, end_idx + 1 + trailing_keep)
    return {key: values[start_idx:trim_end] for key, values in ep.items()}


def get_force_history(obs):
    for key in FORCE_HISTORY_KEYS:
        if key in obs:
            return np.asarray(obs[key], dtype=np.float32).squeeze().reshape(-1, 6)
    raise KeyError(
        f"Observation is missing force history. Tried {FORCE_HISTORY_KEYS}; "
        f"available keys: {list(obs.keys())}"
    )


def load_task_episodes(task_dir):
    pkl_files = sorted(glob.glob(os.path.join(task_dir, "**/*.pkl"), recursive=True))
    episodes = []

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if len(data) == 0:
            continue

        cur = defaultdict(list)
        keys = list(data[0].keys())
        for step in data:
            for key in keys:
                cur[key].append(step.get(key))

            if step.get("dones", False):
                ep = trim_zero_action_episode({key: cur[key] for key in keys})
                if ep is not None and len(ep["actions"]) > 0:
                    episodes.append(ep)
                cur = defaultdict(list)

        if len(cur) > 0 and len(next(iter(cur.values()))) > 0:
            ep = trim_zero_action_episode({key: cur[key] for key in keys})
            if ep is not None and len(ep["actions"]) > 0:
                episodes.append(ep)

    return episodes


def update_stats(acc, values):
    values = np.asarray(values, dtype=np.float64)
    acc["count"] += values.shape[0]
    acc["sum"] += values.sum(axis=0)
    acc["sum_sq"] += np.square(values).sum(axis=0)


def finalize_stats(acc):
    if acc["count"] == 0:
        raise ValueError("Cannot finalize stats with zero samples")
    mean = acc["sum"] / acc["count"]
    var = np.maximum(acc["sum_sq"] / acc["count"] - np.square(mean), EPS)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def make_entry(num_episodes, num_steps, mean, std):
    return {
        "num_episodes": int(num_episodes),
        "num_steps": int(num_steps),
        "force": {
            "axis": AXES[:3],
            "mean": mean[:3].tolist(),
            "std": std[:3].tolist(),
        },
        "torque": {
            "axis": AXES[3:],
            "mean": mean[3:].tolist(),
            "std": std[3:].tolist(),
        },
        "force_torque_concat": {
            "axis": AXES,
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    }


def compute_task_stats(task_dir):
    episodes = load_task_episodes(task_dir)
    acc = {
        "count": 0,
        "sum": np.zeros(6, dtype=np.float64),
        "sum_sq": np.zeros(6, dtype=np.float64),
    }

    valid_episodes = 0
    for ep in episodes:
        histories = [
            get_force_history(obs)
            for obs in ep.get("observations", [])
        ]
        if len(histories) == 0:
            continue

        # Treat force history as one continuous 100hz stream:
        # (policy_steps, history_samples, 6) -> (policy_steps * history_samples, 6).
        continuous_history = np.stack(histories, axis=0).reshape(-1, 6)
        update_stats(acc, continuous_history)
        valid_episodes += 1

    mean, std = finalize_stats(acc)
    return make_entry(valid_episodes, acc["count"], mean, std)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-task 100hz force_history stats with continuous flattened histories."
    )
    parser.add_argument(
        "--data_prefix",
        default="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos",
        help="Directory containing one subdirectory per task.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <data_prefix>/force_history_stats.json.",
    )
    args = parser.parse_args()

    data_prefix = os.path.abspath(args.data_prefix)
    output_path = args.output or os.path.join(data_prefix, "force_history_stats.json")

    task_names = sorted(
        name
        for name in os.listdir(data_prefix)
        if os.path.isdir(os.path.join(data_prefix, name))
    )
    if len(task_names) == 0:
        raise ValueError(f"No task directories found under {data_prefix}")

    stats_by_task = {}
    for task_name in task_names:
        task_dir = os.path.join(data_prefix, task_name)
        print(f"[{task_name}] computing 100hz force_history stats")
        stats_by_task[task_name] = compute_task_stats(task_dir)
        print(
            f"[{task_name}] episodes={stats_by_task[task_name]['num_episodes']} "
            f"samples={stats_by_task[task_name]['num_steps']}"
        )

    with open(output_path, "w") as f:
        json.dump(stats_by_task, f, indent=4)
        f.write("\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
