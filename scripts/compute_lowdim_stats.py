#!/usr/bin/env python
import argparse
import json
import os

import numpy as np

from quest.utils.force_torque_utils import DEFAULT_FT_CONFIG
from quest.utils.real_robot_utils import (
    FORCE_HISTORY_KEYS,
    STATE_HISTORY_KEYS,
    get_obs_value_with_alias,
    load_task_episodes_from_pkls,
    smooth_force_history_sequence,
)


EPS = 1e-8
DEFAULT_KEYS = (
    "state",
    "force",
    "left_force_history",
    "left_state_history",
    "right_force_history",
    "right_state_history",
)
FORCE_STATE_IDXS = (20, 21, 22, 29, 30, 31)
FORCE_COMPONENTS = (
    "right/tcp_force_x",
    "right/tcp_force_y",
    "right/tcp_force_z",
    "right/tcp_torque_x",
    "right/tcp_torque_y",
    "right/tcp_torque_z",
)


def update_accumulator(acc, values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[None]
    else:
        values = values.reshape(-1, values.shape[-1])
    acc["count"] += values.shape[0]
    acc["sum"] += values.sum(axis=0)
    acc["sum_sq"] += np.square(values).sum(axis=0)


def finalize_accumulator(acc):
    if acc["count"] == 0:
        raise ValueError("Cannot finalize zero-count lowdim stats.")
    mean = acc["sum"] / acc["count"]
    var = np.maximum(acc["sum_sq"] / acc["count"] - np.square(mean), EPS)
    return {
        "mean": mean.astype(np.float32).tolist(),
        "std": np.sqrt(var).astype(np.float32).tolist(),
        "count": int(acc["count"]),
    }


def make_accumulator(dim):
    return {
        "count": 0,
        "sum": np.zeros(dim, dtype=np.float64),
        "sum_sq": np.zeros(dim, dtype=np.float64),
    }


def extract_force_from_state(obs):
    state = np.asarray(obs["state"], dtype=np.float32).squeeze()
    if state.shape[-1] <= max(FORCE_STATE_IDXS):
        raise ValueError(
            f"Cannot extract force stats from state with shape {state.shape}; "
            f"need indices up to {max(FORCE_STATE_IDXS)}."
        )
    return state[..., FORCE_STATE_IDXS]


def collect_episode_values(episode, key, ft_config):
    observations = episode.get("observations", [])
    if len(observations) == 0:
        return None
    if key == "force":
        if "state" not in observations[0]:
            return None
        return np.stack([extract_force_from_state(obs) for obs in observations], axis=0)
    if key not in observations[0] and key != "state":
        return None

    if key in FORCE_HISTORY_KEYS:
        values = [
            np.asarray(get_obs_value_with_alias(obs, key), dtype=np.float32).squeeze().reshape(-1, 6)
            for obs in observations
        ]
        values = np.stack(values, axis=0)
        return smooth_force_history_sequence(values, ft_config)

    values = [
        np.asarray(get_obs_value_with_alias(obs, key), dtype=np.float32).squeeze()
        for obs in observations
    ]
    return np.stack(values, axis=0)


def compute_lowdim_stats(data_prefix, keys, ft_config):
    task_names = sorted(
        name
        for name in os.listdir(data_prefix)
        if os.path.isdir(os.path.join(data_prefix, name))
    )
    if len(task_names) == 0:
        raise ValueError(f"No task directories found under {data_prefix}")

    accumulators = {}
    for task_name in task_names:
        task_dir = os.path.join(data_prefix, task_name)
        episodes = load_task_episodes_from_pkls(task_dir, load_obs=False)
        print(f"[{task_name}] episodes={len(episodes)}")

        for episode in episodes:
            for key in keys:
                values = collect_episode_values(episode, key, ft_config)
                if values is None:
                    continue
                dim = values.shape[-1]
                if key not in accumulators:
                    accumulators[key] = make_accumulator(dim)
                elif accumulators[key]["sum"].shape[0] != dim:
                    raise ValueError(
                        f"Inconsistent dim for key '{key}': expected "
                        f"{accumulators[key]['sum'].shape[0]}, got {dim}"
                    )
                update_accumulator(accumulators[key], values)

    stats = {}
    for key, acc in accumulators.items():
        if acc["count"] == 0:
            continue
        stats[key] = finalize_accumulator(acc)
        if key == "force":
            stats[key]["state_indices"] = list(FORCE_STATE_IDXS)
            stats[key]["components"] = list(FORCE_COMPONENTS)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Compute global lowdim mean/std for real-robot demos.")
    parser.add_argument(
        "--data_prefix",
        default="/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos",
        help="Directory containing one subdirectory per task.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <data_prefix>/lowdim_stats.json.",
    )
    parser.add_argument(
        "--keys",
        nargs="+",
        default=list(DEFAULT_KEYS),
        help="Lowdim keys to include.",
    )
    parser.add_argument("--history_filter", default=DEFAULT_FT_CONFIG["history_filter"])
    parser.add_argument("--history_sample_rate_hz", type=float, default=DEFAULT_FT_CONFIG["history_sample_rate_hz"])
    parser.add_argument("--history_cutoff_hz", type=float, default=DEFAULT_FT_CONFIG["history_cutoff_hz"])
    parser.add_argument("--history_filter_order", type=int, default=DEFAULT_FT_CONFIG["history_filter_order"])
    args = parser.parse_args()

    data_prefix = os.path.abspath(args.data_prefix)
    output_path = args.output or os.path.join(data_prefix, "lowdim_stats.json")
    ft_config = {
        **DEFAULT_FT_CONFIG,
        "history_filter": args.history_filter,
        "history_sample_rate_hz": args.history_sample_rate_hz,
        "history_cutoff_hz": args.history_cutoff_hz,
        "history_filter_order": args.history_filter_order,
    }

    stats = compute_lowdim_stats(data_prefix, args.keys, ft_config)
    with open(output_path, "w") as f:
        json.dump(stats, f, indent=4)
        f.write("\n")
    print(f"Wrote {output_path}")
    for key, value in stats.items():
        print(f"{key}: dim={len(value['mean'])} count={value['count']}")


if __name__ == "__main__":
    main()
