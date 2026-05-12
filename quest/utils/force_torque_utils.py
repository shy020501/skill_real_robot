import json
import os

import numpy as np
from scipy.signal import butter, sosfiltfilt


STATE_KEY = "state"
FORCE_HISTORY_KEYS = ("right_force_history", "force_history")
RIGHT_FORCE_SLICE = slice(20, 23)
RIGHT_TORQUE_SLICE = slice(29, 32)
EPS = 1e-8


DEFAULT_FT_CONFIG = {
    "ft_source": "state",
    "stats_path": None,
    "use_threshold_mask": False,
    "ema_alpha": 0.3,
    "history_filter": "butterworth",
    "history_sample_rate_hz": 100.0,
    "history_cutoff_hz": 10.0,
    "history_filter_order": 2,
    "history_reduce": "none",
    "local_window": 7,
    "lambda_delta": 1.0,
    "t_high": 3.5,
    "t_low": 1.5,
    "min_seed_len": 2,
    "min_active_len": 3,
    "max_gap_len": 3,
}


def load_global_ft_stats_from_json(path):
    with open(path, "r") as f:
        stats_by_task = json.load(f)

    total_steps = 0
    weighted_sum = np.zeros(6, dtype=np.float64)
    weighted_second_moment = np.zeros(6, dtype=np.float64)

    for task_name, task_stats in stats_by_task.items():
        if "force_torque_concat" not in task_stats:
            continue
        n_steps = int(task_stats["num_steps"])
        mean = np.asarray(task_stats["force_torque_concat"]["mean"], dtype=np.float64)
        std = np.asarray(task_stats["force_torque_concat"]["std"], dtype=np.float64)
        total_steps += n_steps
        weighted_sum += n_steps * mean
        weighted_second_moment += n_steps * (std ** 2 + mean ** 2)

    if total_steps == 0:
        raise ValueError(f"No valid force_torque_concat stats found in {path}")

    mean = weighted_sum / total_steps
    var = np.maximum(weighted_second_moment / total_steps - mean ** 2, EPS)
    return {
        "mean": mean.astype(np.float32),
        "std": np.sqrt(var).astype(np.float32),
    }


def extract_right_ft_from_state(state):
    state = np.asarray(state, dtype=np.float32).squeeze()
    force = state[RIGHT_FORCE_SLICE]
    torque = state[RIGHT_TORQUE_SLICE]
    return np.concatenate([force, torque], axis=0).astype(np.float32)


def extract_ft_sequence(episode, state_key=STATE_KEY):
    obs_list = episode["observations"]
    ft = []
    for obs in obs_list:
        if state_key not in obs:
            raise KeyError(f"Observation does not contain '{state_key}'")
        ft.append(extract_right_ft_from_state(obs[state_key]))
    return np.stack(ft, axis=0).astype(np.float32)


def get_force_history_value(obs_t):
    for key in FORCE_HISTORY_KEYS:
        if key in obs_t:
            return obs_t[key]
    raise KeyError(
        f"None of force history keys {FORCE_HISTORY_KEYS} found. "
        f"Available keys: {list(obs_t.keys())}"
    )


def extract_force_history_sequence(episode):
    histories = []
    for obs in episode["observations"]:
        history = np.asarray(get_force_history_value(obs), dtype=np.float32).squeeze()
        histories.append(history.reshape(-1, 6))
    return np.stack(histories, axis=0).astype(np.float32)


def compute_ft_stats_for_episodes(episodes, state_key=STATE_KEY, config=None):
    config = {**DEFAULT_FT_CONFIG, **(config or {})}
    if config["ft_source"] == "force_history":
        all_ft = [extract_force_history_sequence(ep).reshape(-1, 6) for ep in episodes]
    elif config["ft_source"] == "state":
        all_ft = [extract_ft_sequence(ep, state_key=state_key) for ep in episodes]
    else:
        raise ValueError(f"Unsupported ft_source: {config['ft_source']}")
    all_ft = np.concatenate(all_ft, axis=0)
    mean = all_ft.mean(axis=0).astype(np.float32)
    std = np.maximum(all_ft.std(axis=0), EPS).astype(np.float32)
    return {"mean": mean, "std": std}


def ema_filter_1d(x, alpha=0.3):
    out = np.zeros_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = alpha * x[t] + (1.0 - alpha) * out[t - 1]
    return out


def smooth_ft_sequence(data, ema_alpha=0.3):
    smoothed = data.copy()
    for axis in range(data.shape[1]):
        x = data[:, axis]
        if ema_alpha is not None:
            x = ema_filter_1d(x, alpha=ema_alpha)
        smoothed[:, axis] = x
    return smoothed.astype(np.float32)


def butterworth_lowpass_sequence(
    data,
    sample_rate_hz=100.0,
    cutoff_hz=10.0,
    order=2,
):
    data = np.asarray(data, dtype=np.float32)
    if data.shape[0] < 2:
        return data.copy()
    nyquist = 0.5 * sample_rate_hz
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(
            f"cutoff_hz must be in (0, Nyquist). Got cutoff_hz={cutoff_hz}, "
            f"sample_rate_hz={sample_rate_hz}."
        )
    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")
    padlen = min(data.shape[0] - 1, max(0, 3 * (2 * len(sos) + 1)))
    return sosfiltfilt(sos, data, axis=0, padlen=padlen).astype(np.float32)


def reduce_force_history(history, mode="last"):
    if mode == "last":
        return history[-1]
    if mode == "mean":
        return history.mean(axis=0)
    if mode == "max":
        max_idx = np.abs(history).argmax(axis=0)
        return history[max_idx, np.arange(history.shape[1])]
    raise ValueError(f"Unsupported history_reduce: {mode}")


def build_force_history_ft_sequence(episode, stats, config=None):
    config = {**DEFAULT_FT_CONFIG, **(config or {})}
    histories = extract_force_history_sequence(episode)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(stats["std"], dtype=np.float32), EPS)
    normalized = (histories - mean) / std

    if config["history_filter"] == "butterworth":
        smoothed = np.stack([
            butterworth_lowpass_sequence(
                history,
                sample_rate_hz=config["history_sample_rate_hz"],
                cutoff_hz=config["history_cutoff_hz"],
                order=config["history_filter_order"],
            )
            for history in normalized
        ], axis=0)
    elif config["history_filter"] in (None, "none"):
        smoothed = normalized
    else:
        raise ValueError(f"Unsupported history_filter: {config['history_filter']}")

    if config["history_reduce"] in (None, "none"):
        mask = np.ones_like(smoothed, dtype=bool)
        return smoothed.astype(np.float32), mask.astype(np.float32), smoothed.astype(np.float32)

    reduced = np.stack([
        reduce_force_history(history, mode=config["history_reduce"])
        for history in smoothed
    ], axis=0)
    mask = np.ones_like(reduced, dtype=bool)
    return reduced.astype(np.float32), mask.astype(np.float32), smoothed.astype(np.float32)


def local_rms_1d(x, window=7):
    assert window % 2 == 1, "window must be odd"
    pad = window // 2
    x2 = x ** 2
    x2_pad = np.pad(x2, (pad, pad), mode="edge")
    out = np.zeros_like(x)
    for t in range(len(x)):
        out[t] = np.sqrt(np.mean(x2_pad[t:t + window]))
    return out


def compute_score_1d(x, local_window=7, lambda_delta=1.0):
    mag = np.abs(x)
    rms = local_rms_1d(x, window=local_window)
    delta = np.zeros_like(x)
    delta[1:] = np.abs(x[1:] - x[:-1])
    delta[0] = delta[1] if len(x) > 1 else 0.0
    return np.maximum.reduce([mag, rms, lambda_delta * delta])


def high_seed_low_expand_mask_1d(score, t_high=3.5, t_low=1.5, min_seed_len=2):
    T = len(score)
    mask = np.zeros(T, dtype=bool)
    high = score > t_high
    t = 0
    while t < T:
        if not high[t]:
            t += 1
            continue

        seed_start = t
        while t < T and high[t]:
            t += 1
        seed_end = t

        if seed_end - seed_start < min_seed_len:
            continue

        left = seed_start
        while left > 0 and score[left - 1] >= t_low:
            left -= 1

        right = seed_end - 1
        while right < T - 1 and score[right + 1] >= t_low:
            right += 1

        mask[left:right + 1] = True

    return mask


def remove_short_active(mask, min_len=3):
    mask = mask.copy()
    T = len(mask)
    t = 0
    while t < T:
        if not mask[t]:
            t += 1
            continue

        start = t
        while t < T and mask[t]:
            t += 1
        end = t
        if end - start < min_len:
            mask[start:end] = False

    return mask


def fill_short_gaps(mask, max_gap_len=3):
    mask = mask.copy()
    T = len(mask)
    t = 0
    while t < T:
        if mask[t]:
            t += 1
            continue

        start = t
        while t < T and not mask[t]:
            t += 1
        end = t

        left_active = start > 0 and mask[start - 1]
        right_active = end < T and mask[end]
        if left_active and right_active and (end - start) <= max_gap_len:
            mask[start:end] = True

    return mask


def compute_mask_sequence(data, config=None):
    config = {**DEFAULT_FT_CONFIG, **(config or {})}
    T, C = data.shape
    scores = np.zeros((T, C), dtype=np.float32)
    masks = np.zeros((T, C), dtype=bool)

    for axis in range(C):
        score = compute_score_1d(
            data[:, axis],
            local_window=config["local_window"],
            lambda_delta=config["lambda_delta"],
        )
        mask = high_seed_low_expand_mask_1d(
            score,
            t_high=config["t_high"],
            t_low=config["t_low"],
            min_seed_len=config["min_seed_len"],
        )
        mask = remove_short_active(mask, min_len=config["min_active_len"])
        mask = fill_short_gaps(mask, max_gap_len=config["max_gap_len"])
        scores[:, axis] = score
        masks[:, axis] = mask

    return scores, masks


def build_episode_masked_ft(episode, stats, config=None, state_key=STATE_KEY):
    config = {**DEFAULT_FT_CONFIG, **(config or {})}
    if config["ft_source"] == "force_history":
        return build_force_history_ft_sequence(episode, stats, config=config)
    if config["ft_source"] != "state":
        raise ValueError(f"Unsupported ft_source: {config['ft_source']}")

    ft = extract_ft_sequence(episode, state_key=state_key)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(stats["std"], dtype=np.float32), EPS)
    normalized = (ft - mean) / std
    smoothed = smooth_ft_sequence(
        normalized,
        ema_alpha=config["ema_alpha"],
    )
    if config["use_threshold_mask"]:
        _, mask = compute_mask_sequence(smoothed, config=config)
        masked_ft = smoothed * mask.astype(np.float32)
    else:
        mask = np.ones_like(smoothed, dtype=bool)
        masked_ft = smoothed
    return masked_ft.astype(np.float32), mask.astype(np.float32), smoothed.astype(np.float32)
