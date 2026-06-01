#!/usr/bin/env python
import argparse
import glob
import json
import os
import pickle
from collections import defaultdict

import numpy as np


DATA_PREFIX = "/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos"
PI_STATE_INDICES = (
    23, 24, 25, 26, 27, 28,
    19,
    20, 21, 22,
    29, 30, 31,
)
FORCE_HISTORY_LEN = 10
FORCE_DIM = 6
PKL_RIGHT_GRIPPER_STATE_IDX = 19
PKL_RIGHT_GRIPPER_ACTION_IDX = 13


class RunningStats:
    """OpenPI-style running stats: last axis is feature, all leading axes are samples."""

    def __init__(self, num_quantile_bins=5000):
        self.count = 0
        self.mean = None
        self.mean_of_squares = None
        self.min = None
        self.max = None
        self.histograms = None
        self.bin_edges = None
        self.num_quantile_bins = num_quantile_bins

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.size == 0:
            return
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape

        if self.count == 0:
            self.mean = np.mean(batch, axis=0)
            self.mean_of_squares = np.mean(batch ** 2, axis=0)
            self.min = np.min(batch, axis=0)
            self.max = np.max(batch, axis=0)
            self.histograms = [np.zeros(self.num_quantile_bins, dtype=np.float64) for _ in range(vector_length)]
            self.bin_edges = [
                self._make_edges(self.min[i], self.max[i])
                for i in range(vector_length)
            ]
        else:
            if vector_length != self.mean.size:
                raise ValueError(
                    f"Feature dim mismatch: expected {self.mean.size}, got {vector_length}"
                )
            new_min = np.min(batch, axis=0)
            new_max = np.max(batch, axis=0)
            min_changed = np.any(new_min < self.min)
            max_changed = np.any(new_max > self.max)
            self.min = np.minimum(self.min, new_min)
            self.max = np.maximum(self.max, new_max)
            if min_changed or max_changed:
                self._adjust_histograms()

        self.count += num_elements
        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch ** 2, axis=0)
        self.mean += (batch_mean - self.mean) * (num_elements / self.count)
        self.mean_of_squares += (
            (batch_mean_of_squares - self.mean_of_squares) * (num_elements / self.count)
        )
        self._update_histograms(batch)

    def get_statistics(self):
        if self.count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")
        variance = self.mean_of_squares - self.mean ** 2
        std = np.sqrt(np.maximum(0.0, variance))
        q01, q99 = self._compute_quantiles((0.01, 0.99))
        return {
            "mean": self.mean.astype(np.float32).tolist(),
            "std": std.astype(np.float32).tolist(),
            "q01": q01.astype(np.float32).tolist(),
            "q99": q99.astype(np.float32).tolist(),
        }

    def _make_edges(self, min_value, max_value):
        if min_value == max_value:
            min_value -= 1e-10
            max_value += 1e-10
        return np.linspace(min_value, max_value, self.num_quantile_bins + 1)

    def _adjust_histograms(self):
        for i in range(len(self.histograms)):
            old_edges = self.bin_edges[i]
            new_edges = self._make_edges(self.min[i], self.max[i])
            new_hist, _ = np.histogram(
                old_edges[:-1],
                bins=new_edges,
                weights=self.histograms[i],
            )
            self.histograms[i] = new_hist
            self.bin_edges[i] = new_edges

    def _update_histograms(self, batch):
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self.bin_edges[i])
            self.histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        results = []
        for quantile in quantiles:
            target_count = quantile * self.count
            values = []
            for hist, edges in zip(self.histograms, self.bin_edges):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                idx = min(idx, len(edges) - 1)
                values.append(edges[idx])
            results.append(np.asarray(values))
        return results


def normalize_done(done):
    if isinstance(done, np.ndarray):
        return bool(done.item())
    return bool(done)


def trim_zero_action_episode(ep, trailing_keep=5):
    if "actions" not in ep or len(ep["actions"]) == 0:
        return None

    actions = [np.asarray(action) for action in ep["actions"]]
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


def load_task_episodes(task_dir):
    pkl_files = sorted(glob.glob(os.path.join(task_dir, "**", "*.pkl"), recursive=True))
    episodes = []

    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            raw_data = pickle.load(f)
        if len(raw_data) == 0:
            continue

        current = defaultdict(list)
        keys = list(raw_data[0].keys())
        for step in raw_data:
            for key in keys:
                current[key].append(step.get(key))

            if normalize_done(step.get("dones", False)):
                episode = trim_zero_action_episode({key: current[key] for key in keys})
                if episode is not None and len(episode["actions"]) > 0:
                    episodes.append(episode)
                current = defaultdict(list)

        if len(current) > 0 and len(next(iter(current.values()))) > 0:
            episode = trim_zero_action_episode({key: current[key] for key in keys})
            if episode is not None and len(episode["actions"]) > 0:
                episodes.append(episode)

    return episodes


def extract_pi_state(obs):
    raw_state = np.asarray(obs["state"], dtype=np.float32).squeeze()
    if raw_state.shape[-1] <= max(PI_STATE_INDICES):
        raise ValueError(
            f"Cannot extract PI state from shape {raw_state.shape}; "
            f"need indices up to {max(PI_STATE_INDICES)}."
        )
    return raw_state[..., PI_STATE_INDICES].astype(np.float32)


def relabel_gripper_action(action, obs):
    state = np.asarray(obs["state"], dtype=np.float32).squeeze()
    action = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if state.shape[0] <= PKL_RIGHT_GRIPPER_STATE_IDX or action.shape[0] <= PKL_RIGHT_GRIPPER_ACTION_IDX:
        return action

    gripper_action = action[PKL_RIGHT_GRIPPER_ACTION_IDX]
    if np.abs(gripper_action) < 0.9:
        action[PKL_RIGHT_GRIPPER_ACTION_IDX] = (state[PKL_RIGHT_GRIPPER_STATE_IDX] > 0.95) * 2.0 - 1.0
    return action


def extract_pi_action(action, obs):
    action = relabel_gripper_action(action, obs)
    if action.shape[-1] < 14:
        raise ValueError(f"Expected raw action dim >= 14, got shape {action.shape}")
    return action[7:].astype(np.float32)


def extract_force_history(obs):
    force_history = np.asarray(obs["right_force_history"], dtype=np.float32).squeeze()
    if force_history.shape != (FORCE_HISTORY_LEN, FORCE_DIM):
        raise ValueError(f"Unexpected right_force_history shape: {force_history.shape}")
    return force_history.astype(np.float32)


def update_stats_from_episode(stats, episode):
    states = []
    actions = []
    force_histories = []

    for action, obs in zip(episode.get("actions", []), episode.get("observations", [])):
        states.append(extract_pi_state(obs))
        actions.append(extract_pi_action(action, obs))
        force_histories.append(extract_force_history(obs))

    if len(states) == 0:
        return 0

    stats["state"].update(np.stack(states, axis=0))
    stats["actions"].update(np.stack(actions, axis=0))
    stats["force_history"].update(np.stack(force_histories, axis=0))
    return len(states)


def compute_pi_stats(data_prefix):
    task_names = sorted(
        name
        for name in os.listdir(data_prefix)
        if os.path.isdir(os.path.join(data_prefix, name))
    )
    if len(task_names) == 0:
        raise ValueError(f"No task directories found under {data_prefix}")

    stats = {
        "state": RunningStats(),
        "actions": RunningStats(),
        "force_history": RunningStats(),
    }
    total_episodes = 0
    total_frames = 0

    for task_name in task_names:
        task_dir = os.path.join(data_prefix, task_name)
        episodes = load_task_episodes(task_dir)
        task_frames = 0
        for episode in episodes:
            task_frames += update_stats_from_episode(stats, episode)

        total_episodes += len(episodes)
        total_frames += task_frames
        print(f"[{task_name}] episodes={len(episodes)} frames={task_frames}")

    norm_stats = {
        key: stat.get_statistics()
        for key, stat in stats.items()
    }
    counts = {
        key: int(stat.count)
        for key, stat in stats.items()
    }
    return {
        "norm_stats": norm_stats,
        "counts": counts,
        "num_episodes": int(total_episodes),
        "num_frames": int(total_frames),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute PI/OpenPI-style state, action, and force_history stats from real-robot PKLs."
    )
    parser.add_argument(
        "--data_prefix",
        default=DATA_PREFIX,
        help="Directory containing one subdirectory per task.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to <data_prefix>/pi_stats.json.",
    )
    args = parser.parse_args()

    data_prefix = os.path.abspath(args.data_prefix)
    output_path = args.output or os.path.join(data_prefix, "pi_stats.json")

    result = compute_pi_stats(data_prefix)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(f"Wrote {output_path}")
    for key, count in result["counts"].items():
        dim = len(result["norm_stats"][key]["mean"])
        print(f"{key}: dim={dim} count={count}")


if __name__ == "__main__":
    main()
