import os
import glob
import pickle
import json
import io
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from transformers import AutoModel, AutoTokenizer, logging
from PIL import Image
from quest.utils.force_torque_utils import (
    DEFAULT_FT_CONFIG,
    EPS,
    causal_butterworth_lowpass_sequence,
    compute_ft_stats_for_episodes,
    compute_mask_sequence,
    extract_force_history_sequence,
    extract_ft_sequence,
    extract_right_ft_from_state,
    RIGHT_FORCE_SLICE,
    RIGHT_TORQUE_SLICE,
    load_ft_stats_from_lowdim_json,
    load_global_ft_stats_from_json,
    reduce_force_history,
    smooth_ft_sequence,
)

# IMAGE_KEYS = {"front_cam", "head_cam", "left_wrist_cam", "right_wrist_cam"}
IMAGE_KEYS = {"front_cam", "right_wrist_cam"}
FORCE_HISTORY_KEYS = ("left_force_history", "right_force_history")
STATE_HISTORY_KEYS = ("left_state_history", "right_state_history")
RIGHT_STATE_KEYS = ("state",)
RIGHT_STATE_FORCE_TORQUE_IDXS = (1, 2, 3, 10, 11, 12)
PKL_RIGHT_GRIPPER_STATE_IDX = 19
PKL_RIGHT_GRIPPER_ACTION_IDX = 13

# -------------------------------------------------
# task embedding
# -------------------------------------------------
def get_task_embs(task_embedding_format, descriptions):
    logging.set_verbosity_error()

    if task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModel.from_pretrained("bert-base-cased")
        tokens = tz(
            text=descriptions,
            add_special_tokens=True,
            max_length=25,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])["pooler_output"].detach()

    elif task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,
            add_special_tokens=True,
            max_length=25,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]

    elif task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32",
            clean_up_tokenization_spaces=True
        )
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,
            add_special_tokens=True,
            max_length=25,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        task_embs = model.get_text_features(**tokens).detach()

    elif task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,
            add_special_tokens=True,
            max_length=25,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
        )
        task_embs = model(**tokens)["pooler_output"].detach()

    else:
        raise ValueError(f"Unsupported task_embedding_format: {task_embedding_format}")

    return task_embs


# -------------------------------------------------
# pkl -> episodes
# -------------------------------------------------
def load_task_episodes_from_pkls(
    task_dir: str,
    load_obs: bool = True,
    leading_keep: Optional[int] = None,
) -> List[Dict]:
    pkl_files = sorted(glob.glob(os.path.join(task_dir, "**/*.pkl"), recursive=True))
    print(f"[{os.path.basename(task_dir)}] Found pkl files: {len(pkl_files)}")

    episodes = []

    def keep_numeric_observation(obs):
        if not isinstance(obs, dict):
            return obs
        return {
            key: obs[key]
            for key in (
                "state",
                "left_force_history",
                "left_state_history",
                "right_force_history",
                "right_state_history",
            )
            if key in obs
        }

    def trim_zero_action_episode(ep: Dict, trailing_keep: int = 5) -> Optional[Dict]:
        """
        ep: 각 key가 길이 T인 list인 episode dict
        actions를 기준으로 앞/뒤 zero-action 구간 제거
        뒤쪽은 end_idx 이후 trailing_keep step 만큼 추가 유지
        """
        if "actions" not in ep or len(ep["actions"]) == 0:
            return None

        actions = [np.asarray(a) for a in ep["actions"]]
        T = len(actions)

        # 앞쪽 zero-action 제거
        start_idx = 0
        while start_idx < T and np.all(actions[start_idx] == 0):
            start_idx += 1

        # 전부 zero-action이면 버림
        if start_idx == T:
            return None

        # 뒤쪽 zero-action 제거
        end_idx = T - 1
        while end_idx >= 0 and np.all(actions[end_idx] == 0):
            end_idx -= 1

        # 뒤쪽은 약간 더 남김 (두 번째 코드와 동일한 방식)
        trim_end = min(T, end_idx + 1 + trailing_keep)

        trimmed_ep = {
            k: v[start_idx:trim_end]
            for k, v in ep.items()
        }

        return trimmed_ep

    def zero_like_episode_value(value):
        if isinstance(value, dict):
            return {k: zero_like_episode_value(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return np.zeros_like(value)
        if torch.is_tensor(value):
            return torch.zeros_like(value)
        if isinstance(value, (list, tuple)):
            return type(value)(zero_like_episode_value(v) for v in value)
        if isinstance(value, (bool, np.bool_)):
            return False
        if isinstance(value, (int, float, np.number)):
            return type(value)(0)
        return 0

    def copy_episode_value(value):
        if isinstance(value, dict):
            return {k: copy_episode_value(v) for k, v in value.items()}
        if isinstance(value, np.ndarray):
            return value.copy()
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, list):
            return [copy_episode_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(copy_episode_value(v) for v in value)
        return value

    def zero_state_force_torque(value):
        state = np.asarray(value).copy()
        flat_state = state.reshape(-1)
        if flat_state.shape[0] >= RIGHT_TORQUE_SLICE.stop:
            flat_state[RIGHT_FORCE_SLICE] = 0.0
            flat_state[RIGHT_TORQUE_SLICE] = 0.0
        return flat_state.reshape(state.shape)

    def zero_observation_force_torque(obs):
        if not isinstance(obs, dict):
            return obs
        obs = dict(obs)
        if "state" in obs:
            obs["state"] = zero_state_force_torque(obs["state"])
        for force_history_key in FORCE_HISTORY_KEYS:
            if force_history_key in obs:
                obs[force_history_key] = zero_like_episode_value(obs[force_history_key])
        return obs

    def zero_leading_observation_force_torque(ep: Dict, leading_keep: int) -> None:
        observations = ep.get("observations", [])
        for i in range(min(leading_keep, len(observations))):
            observations[i] = zero_observation_force_torque(observations[i])

    def leading_pad_episode_value(key, value):
        if key == "state":
            return zero_state_force_torque(value)
        if isinstance(value, dict):
            return {
                child_key: zero_state_force_torque(child_value)
                if child_key == "state"
                else zero_like_episode_value(child_value)
                for child_key, child_value in value.items()
            }
        return zero_like_episode_value(value)

    def force_initial_gripper_open(ep: Dict) -> None:
        observations = ep.get("observations", [])
        if len(observations) == 0:
            return

        obs0 = observations[0]
        if not isinstance(obs0, dict) or "state" not in obs0:
            return

        state = np.asarray(obs0["state"]).copy()
        flat_state = state.reshape(-1)
        if flat_state.shape[0] <= PKL_RIGHT_GRIPPER_STATE_IDX:
            return

        flat_state[PKL_RIGHT_GRIPPER_STATE_IDX] = 1.0
        obs0 = dict(obs0)
        obs0["state"] = flat_state.reshape(state.shape)
        observations[0] = obs0

    def trim_zero_action_episode_with_leading_keep(
        ep: Dict,
        leading_keep: int = 32,
        trailing_keep: int = 5,
    ) -> Optional[Dict]:
        """
        임시 함수: 앞쪽 zero-action은 정확히 leading_keep step 남김.
        부족하면 앞에 zero padding을 추가하고, 많으면 leading_keep step만 유지.
        뒤쪽은 trim_zero_action_episode와 동일하게 trailing_keep step 유지.
        """
        if "actions" not in ep or len(ep["actions"]) == 0:
            return None

        actions = [np.asarray(a) for a in ep["actions"]]
        T = len(actions)

        start_idx = 0
        while start_idx < T and np.all(actions[start_idx] == 0):
            start_idx += 1

        if start_idx == T:
            return None

        end_idx = T - 1
        while end_idx >= 0 and np.all(actions[end_idx] == 0):
            end_idx -= 1

        trim_start = max(0, start_idx - leading_keep)
        trim_end = min(T, end_idx + 1 + trailing_keep)
        pad_front = max(0, leading_keep - start_idx)

        trimmed_ep = {}
        for k, v in ep.items():
            trimmed_values = list(v[trim_start:trim_end])
            if pad_front > 0:
                trimmed_values = [
                    leading_pad_episode_value(k, v[0])
                    for _ in range(pad_front)
                ] + trimmed_values
            trimmed_ep[k] = trimmed_values

        zero_leading_observation_force_torque(trimmed_ep, leading_keep)
        return trimmed_ep
    
    for pkl_path in pkl_files:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if len(data) == 0:
            continue

        cur = defaultdict(list)
        keys = list(data[0].keys())

        for step in data:
            for k in keys:
                value = step.get(k)
                if not load_obs and k == "observations":
                    value = keep_numeric_observation(value)
                cur[k].append(value)
            done = step.get("dones", False)

            if done:
                ep = {k: cur[k] for k in keys}
                if leading_keep is None:
                    ep = trim_zero_action_episode(ep, trailing_keep=5)
                else:
                    ep = trim_zero_action_episode_with_leading_keep(
                        ep, leading_keep=leading_keep, trailing_keep=5
                    )
                if ep is not None and len(ep["actions"]) > 0:
                    force_initial_gripper_open(ep)
                    episodes.append(ep)
                cur = defaultdict(list)

        # 마지막 episode가 dones=True 없이 끝난 경우
        if len(cur) > 0 and len(next(iter(cur.values()))) > 0:
            ep = {k: cur[k] for k in keys}
            if leading_keep is None:
                ep = trim_zero_action_episode(ep, trailing_keep=5)
            else:
                ep = trim_zero_action_episode_with_leading_keep(
                    ep, leading_keep=leading_keep, trailing_keep=5
                )
            if ep is not None and len(ep["actions"]) > 0:
                force_initial_gripper_open(ep)
                episodes.append(ep)

        del data, step, value

    print(f"[{os.path.basename(task_dir)}] Total episodes: {len(episodes)}")
    return episodes


# -------------------------------------------------
# helper
# -------------------------------------------------
def pad_sequence_list(seq_list, target_len, pad_last=True):
    """
    seq_list: 길이 T의 list
    target_len: 맞추고 싶은 길이
    """
    if len(seq_list) >= target_len:
        return seq_list[:target_len]

    if not pad_last:
        raise ValueError("Sequence shorter than target_len and pad_last=False")

    if len(seq_list) == 0:
        raise ValueError("Cannot pad empty sequence")

    pad_item = seq_list[-1]
    padded = list(seq_list)
    while len(padded) < target_len:
        padded.append(pad_item)
    return padded


def frame_stack_list(seq_list, frame_stack):
    """
    예: [x0, x1, x2], frame_stack=2
    -> [
         [x0, x0],
         [x0, x1],
         [x1, x2],
       ]
    """
    stacked = []
    T = len(seq_list)
    for t in range(T):
        frames = []
        for k in range(frame_stack):
            idx = max(0, t - frame_stack + 1 + k)
            frames.append(seq_list[idx])
        stacked.append(frames)
    return stacked


def to_numpy_array_list(seq_list):
    return [np.asarray(x) for x in seq_list]


def load_task_descriptions(task_descriptions=None, instruction_path=None):
    if task_descriptions is not None:
        return task_descriptions
    if instruction_path is None:
        return None
    with open(instruction_path, "r") as f:
        return json.load(f)


def get_force_history_value(obs_t, key):
    if key not in FORCE_HISTORY_KEYS:
        raise KeyError(f"Unsupported force history key '{key}'. Expected one of {FORCE_HISTORY_KEYS}.")
    if key not in obs_t:
        raise KeyError(f"Observation is missing '{key}'. Available keys: {list(obs_t.keys())}")
    return obs_t[key]


def get_right_state_value(obs_t, remove_force=True):
    arr = np.asarray(obs_t["state"])
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    right_state = arr[19:].astype(np.float32)
    if remove_force:
        right_state = np.delete(right_state, RIGHT_STATE_FORCE_TORQUE_IDXS, axis=0)
    return right_state


def remove_state_history_force_torque(state_history):
    state_history = np.asarray(state_history, dtype=np.float32).squeeze()
    return np.delete(state_history, RIGHT_STATE_FORCE_TORQUE_IDXS, axis=-1).astype(np.float32)


def relabel_gripper_action(gripper_action, gripper_state):
    return np.where(
        np.abs(gripper_action) < 0.9,
        (gripper_state > 0.95) * 2.0 - 1.0,
        gripper_action,
    ).astype(np.float32)


def relabel_episode_gripper_actions_from_state(episode):
    observations = episode.get("observations", [])
    actions = episode.get("actions", [])
    for i, (action, obs) in enumerate(zip(actions, observations)):
        if "state" not in obs:
            continue
        state = np.asarray(obs["state"], dtype=np.float32).squeeze()
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.shape[0] <= PKL_RIGHT_GRIPPER_STATE_IDX or action_arr.shape[0] <= PKL_RIGHT_GRIPPER_ACTION_IDX:
            continue
        action_arr[PKL_RIGHT_GRIPPER_ACTION_IDX] = relabel_gripper_action(
            action_arr[PKL_RIGHT_GRIPPER_ACTION_IDX],
            state[PKL_RIGHT_GRIPPER_STATE_IDX],
        )
        actions[i] = action_arr


def has_obs_key(obs_t, key):
    if key in FORCE_HISTORY_KEYS:
        return key in obs_t
    if key in RIGHT_STATE_KEYS:
        return "state" in obs_t
    return key in obs_t


def get_obs_value_with_alias(obs_t, key):
    if key in FORCE_HISTORY_KEYS:
        return get_force_history_value(obs_t, key)
    if key in RIGHT_STATE_KEYS:
        return get_right_state_value(obs_t)
    if key in STATE_HISTORY_KEYS:
        return remove_state_history_force_torque(obs_t[key])
    return obs_t[key]


def smooth_force_history_sequence(force_history, ft_config):
    force_history = np.asarray(force_history, dtype=np.float32)
    original_shape = force_history.shape
    flat_force_history = force_history.reshape(-1, original_shape[-1])

    if ft_config["history_filter"] == "butterworth":
        flat_force_history = causal_butterworth_lowpass_sequence(
            flat_force_history,
            sample_rate_hz=ft_config["history_sample_rate_hz"],
            cutoff_hz=ft_config["history_cutoff_hz"],
            order=ft_config["history_filter_order"],
        )
    elif ft_config["history_filter"] not in (None, "none"):
        raise ValueError(f"Unsupported history_filter: {ft_config['history_filter']}")
    return flat_force_history.reshape(original_shape).astype(np.float32)


def threshold_mask_force_history_sequence(force_history, ft_config):
    force_history = np.asarray(force_history, dtype=np.float32)
    flat_force_history = force_history.reshape(-1, force_history.shape[-1])
    _, flat_mask = compute_mask_sequence(flat_force_history, config=ft_config)
    return flat_mask.reshape(force_history.shape).astype(np.float32)


def reduce_ft_history_sequence(force_history, ft_config):
    if ft_config["history_reduce"] in (None, "none"):
        return force_history.astype(np.float32)
    return np.stack([
        reduce_force_history(history, mode=ft_config["history_reduce"])
        for history in force_history
    ], axis=0).astype(np.float32)


def normalize_ft_sequence(ft, ft_stats):
    mean = np.asarray(ft_stats["mean"], dtype=np.float32)
    std = np.maximum(np.asarray(ft_stats["std"], dtype=np.float32), EPS)
    return ((ft - mean) / std).astype(np.float32)


def load_lowdim_stats(path):
    with open(path, "r") as f:
        raw_stats = json.load(f)
    stats = {}
    for key, value in raw_stats.items():
        if "mean" not in value or "std" not in value:
            raise KeyError(f"lowdim_stats entry '{key}' must contain 'mean' and 'std'.")
        stats[key] = {
            "mean": np.asarray(value["mean"], dtype=np.float32),
            "std": np.maximum(np.asarray(value["std"], dtype=np.float32), EPS),
        }
    return stats


def normalize_lowdim_value(value, key, lowdim_stats):
    if lowdim_stats is None or key not in lowdim_stats:
        return value.astype(np.float32)
    stats = lowdim_stats[key]
    mean = stats["mean"]
    std = stats["std"]
    if value.shape[-1] != mean.shape[-1]:
        raise ValueError(
            f"Lowdim stats for '{key}' have dim {mean.shape[-1]}, "
            f"but value shape is {value.shape}."
        )
    return ((value - mean) / std).astype(np.float32)


# -------------------------------------------------
# real-world sequence dataset
# -------------------------------------------------
class RealWorldSequenceDataset(Dataset):
    def __init__(
        self,
        episodes: List[Dict],
        obs_keys: List[str],
        dataset_keys: List[str] = ["actions"],
        seq_length: int = 1,
        obs_seq_length: int = 1,
        frame_stack: int = 1,
        pad_seq_length: bool = True,
        pad_frame_stack: bool = True,
        use_ft: bool = False,
        ft_stats: Optional[Dict] = None,
        ft_config: Optional[Dict] = None,
        ft_shift: int = 1,
        lowdim_stats: Optional[Dict] = None,
        load_obs: bool = True,
        action_target_mode: str = "actions",
        leading_keep: Optional[int] = None,
    ):
        self.episodes = episodes
        self.obs_keys = obs_keys
        self.dataset_keys = dataset_keys
        self.seq_length = seq_length
        self.obs_seq_length = obs_seq_length
        self.frame_stack = frame_stack
        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.use_ft = use_ft
        self.ft_shift = ft_shift
        self.load_obs = load_obs
        self.action_target_mode = action_target_mode
        self.leading_keep = leading_keep
        self.ft_config = {**DEFAULT_FT_CONFIG, **(ft_config or {})}
        self.lowdim_stats = lowdim_stats

        if self.action_target_mode not in ("actions", "action_force", "action_force_norm"):
            raise ValueError(f"Unsupported action_target_mode: {self.action_target_mode}")

        self.n_demos = len(episodes)

        for ep in self.episodes:
            relabel_episode_gripper_actions_from_state(ep)

        for ep_idx, ep in enumerate(self.episodes):
            actions = ep.get("actions", [])
            if len(actions) == 0:
                continue

            action0 = np.asarray(actions[0], dtype=np.float32).reshape(-1)
            if action0.shape[0] <= PKL_RIGHT_GRIPPER_ACTION_IDX:
                continue

            gripper_action = action0[PKL_RIGHT_GRIPPER_ACTION_IDX]
            if gripper_action < 0:
                raise ValueError(
                    f"-1 gripper action in step 0: ep_idx={ep_idx}, "
                    f"gripper_action={gripper_action}"
                )

        self.masked_ft_episodes = None
        self.ft_mask_episodes = None
        self.state_ft_episodes = None
        self.force_history_episodes = None
        self.smoothed_force_history_episodes = None
        needs_state_ft = self.use_ft and self.ft_config["ft_source"] == "state"
        force_history_keys = set()
        if self.load_obs:
            force_history_keys.update(key for key in self.obs_keys if key in FORCE_HISTORY_KEYS)
        if self.use_ft and self.ft_config["ft_source"] in FORCE_HISTORY_KEYS:
            force_history_keys.add(self.ft_config["ft_source"])
        if needs_state_ft:
            self.state_ft_episodes = [
                extract_ft_sequence(ep)
                for ep in self.episodes
            ]

        if force_history_keys:
            self.force_history_episodes = {}
            self.smoothed_force_history_episodes = {}
            for force_history_key in sorted(force_history_keys):
                raw_episodes = []
                smoothed_episodes = []
                for ep in self.episodes:
                    force_history = extract_force_history_sequence(ep, force_history_key=force_history_key)
                    raw_episodes.append(force_history)
                    normalized_force_history = normalize_lowdim_value(
                        force_history,
                        force_history_key,
                        self.lowdim_stats,
                    )
                    smoothed_episodes.append(
                        smooth_force_history_sequence(normalized_force_history, self.ft_config)
                    )
                self.force_history_episodes[force_history_key] = raw_episodes
                self.smoothed_force_history_episodes[force_history_key] = smoothed_episodes

        self.action_target_episodes = None
        if self.action_target_mode == "action_force":
            self.action_target_episodes = []
            for ep_idx, ep in enumerate(self.episodes):
                actions = ep.get("actions", [])
                observations = ep.get("observations", [])
                if len(actions) != len(observations):
                    raise ValueError(
                        f"Episode {ep_idx} has {len(actions)} actions but {len(observations)} observations."
                    )
                targets = []
                for action, obs in zip(actions, observations):
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[7:]
                    if "state" not in obs:
                        raise KeyError("action_force target requires observation key 'state'.")
                    state_force = extract_right_ft_from_state(obs["state"])
                    targets.append(np.concatenate([action_arr, state_force], axis=0).astype(np.float32))
                self.action_target_episodes.append(targets)

        if self.action_target_mode == "action_force_norm" and (
            not self.use_ft or self.ft_config["ft_source"] != "state"
        ):
            raise ValueError("action_force_norm target requires use_ft=True and ft_source='state'.")

        if self.use_ft:
            if ft_stats is None:
                ft_stats = compute_ft_stats_for_episodes(self.episodes, config=self.ft_config)
            self.masked_ft_episodes = []
            self.ft_mask_episodes = []
            for ep_idx, _ in enumerate(self.episodes):
                if self.ft_config["ft_source"] == "state":
                    ft = normalize_ft_sequence(self.state_ft_episodes[ep_idx], ft_stats)
                    ft = smooth_ft_sequence(ft, ema_alpha=self.ft_config["ema_alpha"])
                    if self.ft_config["use_threshold_mask"]:
                        _, ft_mask = compute_mask_sequence(ft, config=self.ft_config)
                        masked_ft = ft * ft_mask.astype(np.float32)
                    else:
                        ft_mask = np.ones_like(ft, dtype=np.float32)
                        masked_ft = ft
                elif self.ft_config["ft_source"] in FORCE_HISTORY_KEYS:
                    ft_source = self.ft_config["ft_source"]
                    ft = normalize_ft_sequence(
                        self.force_history_episodes[ft_source][ep_idx],
                        ft_stats,
                    )
                    ft = smooth_force_history_sequence(ft, self.ft_config)
                    if self.ft_config["use_threshold_mask"]:
                        ft_mask = threshold_mask_force_history_sequence(ft, self.ft_config)
                        masked_ft = ft * ft_mask
                    else:
                        ft_mask = np.ones_like(ft, dtype=np.float32)
                        masked_ft = ft
                    masked_ft = reduce_ft_history_sequence(masked_ft, self.ft_config)
                    if self.ft_config["history_reduce"] not in (None, "none"):
                        ft_mask = np.any(ft_mask, axis=1).astype(np.float32)
                else:
                    raise ValueError(f"Unsupported ft_source: {self.ft_config['ft_source']}")
                self.masked_ft_episodes.append(masked_ft)
                self.ft_mask_episodes.append(ft_mask.astype(np.float32))

        if self.action_target_mode == "action_force_norm":
            self.action_target_episodes = []
            for ep_idx, ep in enumerate(self.episodes):
                actions = ep.get("actions", [])
                force_targets = self.masked_ft_episodes[ep_idx]
                if self.leading_keep is not None:
                    leading_zero_len = min(self.leading_keep, len(force_targets))
                    if leading_zero_len > 0:
                        force_targets[:leading_zero_len] = 0.0
                if len(actions) != len(force_targets):
                    raise ValueError(
                        f"Episode {ep_idx} has {len(actions)} actions but {len(force_targets)} force targets."
                    )
                targets = []
                for action, force_target in zip(actions, force_targets):
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)[7:]
                    force_arr = np.asarray(force_target, dtype=np.float32).reshape(-1)
                    targets.append(np.concatenate([action_arr, force_arr], axis=0).astype(np.float32))
                self.action_target_episodes.append(targets)

        if not self.load_obs:
            for ep in self.episodes:
                ep["observations"] = []

        self.index_map = []
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = len(ep["actions"])
            if pad_seq_length:
                num_seq = ep_len
            else:
                num_seq = max(0, ep_len - seq_length + 1)

            for start_t in range(num_seq):
                self.index_map.append((ep_idx, start_t))

        self.total_num_sequences = len(self.index_map)

    def __len__(self):
        return self.total_num_sequences

    def _slice_with_padding(self, arr_list, start, length):
        sliced = arr_list[start:start + length]
        if len(sliced) < length:
            if not self.pad_seq_length:
                raise IndexError("Sequence too short and padding disabled")
            sliced = pad_sequence_list(sliced, length, pad_last=True)
        return sliced

    def __getitem__(self, idx):
        ep_idx, start_t = self.index_map[idx]
        ep = self.episodes[ep_idx]

        ret = {}

        # -------------------------
        # actions
        # -------------------------
        if "actions" in self.dataset_keys:
            action_list = []
            for x in ep["actions"]:
                arr = np.asarray(x).astype(np.float32)   # (14,)
                arr = arr[7:]                            # -> (7,)
                action_list.append(arr)

            action_seq = self._slice_with_padding(action_list, start_t, self.seq_length)
            ret["actions"] = np.stack(action_seq, axis=0).astype(np.float32)

        if self.action_target_episodes is not None:
            target_seq = self._slice_with_padding(
                self.action_target_episodes[ep_idx],
                start_t,
                self.seq_length,
            )
            ret["action_targets"] = np.stack(target_seq, axis=0).astype(np.float32)

        if self.use_ft:
            ft_len = len(self.masked_ft_episodes[ep_idx])
            ft_start_t = min(start_t + self.ft_shift, ft_len - 1)
            masked_ft_seq = self._slice_with_padding(
                self.masked_ft_episodes[ep_idx],
                ft_start_t,
                self.seq_length,
            )
            ft_mask_seq = self._slice_with_padding(
                self.ft_mask_episodes[ep_idx],
                ft_start_t,
                self.seq_length,
            )
            ret["masked_ft"] = np.stack(masked_ft_seq, axis=0).astype(np.float32)
            ret["ft_mask"] = np.stack(ft_mask_seq, axis=0).astype(np.float32)

        if not self.load_obs:
            return ret

        # -------------------------
        # observations
        # -------------------------
        observations = ep["observations"]
        if len(observations) == 0:
            raise ValueError(f"Episode {ep_idx} has empty observations")

        obs_dict = {}

        for key in self.obs_keys:
            if not has_obs_key(observations[0], key):
                raise KeyError(
                    f"Observation key '{key}' not found in observations[0]. "
                    f"Available keys: {list(observations[0].keys())}"
                )

            obs_list = []
            for t, obs_t in enumerate(observations):
                force_history_is_preprocessed = False
                if (
                    key in FORCE_HISTORY_KEYS
                    and self.smoothed_force_history_episodes is not None
                    and key in self.smoothed_force_history_episodes
                ):
                    arr = self.smoothed_force_history_episodes[key][ep_idx][t]
                    force_history_is_preprocessed = True
                else:
                    arr = np.asarray(get_obs_value_with_alias(obs_t, key))

                # history: keep (H, D) so sequence encoders can consume the temporal axis directly.
                if key in FORCE_HISTORY_KEYS:
                    arr = arr.squeeze().astype(np.float32)
                    if not force_history_is_preprocessed:
                        arr = normalize_lowdim_value(arr, key, self.lowdim_stats)
                elif key in STATE_HISTORY_KEYS:
                    arr = arr.squeeze().astype(np.float32)
                    arr = normalize_lowdim_value(arr, key, self.lowdim_stats)

                # image: (1, H, W, C) -> (H, W, C)
                elif key in IMAGE_KEYS:
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]   # (1, H, W, C) -> (H, W, C)
                    arr = arr.astype(np.uint8)

                else:
                    arr = arr.astype(np.float32)
                    arr = normalize_lowdim_value(arr, key, self.lowdim_stats)

                obs_list.append(arr)

            obs_seq = self._slice_with_padding(obs_list, start_t, self.obs_seq_length)

            if self.frame_stack > 1:
                obs_seq = frame_stack_list(obs_seq, self.frame_stack)
                obs_seq = [np.stack(x, axis=0) for x in obs_seq]

            obs_arr = np.stack(obs_seq, axis=0)

            # image: [T, H, W, C] -> [T, C, H, W]
            if key in IMAGE_KEYS:
                if obs_arr.ndim == 4 and obs_arr.shape[-1] in (1, 3):
                    obs_arr = np.transpose(obs_arr, (0, 3, 1, 2))   # [T, H, W, C] -> [T, C, H, W]
                elif obs_arr.ndim == 5 and obs_arr.shape[-1] in (1, 3):
                    obs_arr = np.transpose(obs_arr, (0, 1, 4, 2, 3)) # [T, F, H, W, C] -> [T, F, C, H, W]
                obs_arr = obs_arr.astype(np.uint8)
            else:
                obs_arr = obs_arr.astype(np.float32)

            obs_dict[key] = obs_arr

        ret["obs"] = obs_dict
        return ret


# -------------------------------------------------
# task wrapper (LIBERO와 동일 역할)
# -------------------------------------------------
class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb, task_id):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.task_id = task_id
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset[idx]
        return_dict["task_emb"] = self.task_emb
        return_dict["task_id"] = self.task_id
        return return_dict


# -------------------------------------------------
# main build_dataset for real-world
# -------------------------------------------------
def build_realworld_dataset(
    data_prefix,
    seq_len,
    frame_stack,
    shape_meta,
    task_embedding_format="clip",
    obs_seq_len=1,
    task_descriptions: Optional[Dict[str, str]] = None,
    instruction_path: Optional[str] = None,
    use_ft: bool = False,
    ft_config: Optional[Dict] = None,
    ft_shift: int = 1,
    lowdim_stats_path: Optional[str] = None,
    load_obs: bool = True,
    action_target_mode: str = "actions",
    leading_keep: Optional[int] = None,
):
    """
    data_prefix 예:
        demos/
          task_a/
            xxx.pkl
            subdir/yyy.pkl
          task_b/
            zzz.pkl
          task_c/
            ...
    """

    # task 폴더 찾기
    task_names = sorted([
        d for d in os.listdir(data_prefix)
        if os.path.isdir(os.path.join(data_prefix, d))
    ])

    if len(task_names) == 0:
        raise ValueError(f"No task directories found under {data_prefix}")

    task_descriptions = load_task_descriptions(
        task_descriptions=task_descriptions,
        instruction_path=instruction_path,
    )

    obs_keys = []
    obs_keys += list(shape_meta["observation"]["rgb"].keys())
    obs_keys += list(shape_meta["observation"]["lowdim"].keys())

    ft_config = {**DEFAULT_FT_CONFIG, **(ft_config or {})}
    if lowdim_stats_path is None:
        candidate = os.path.join(data_prefix, "lowdim_stats.json")
        if os.path.exists(candidate):
            lowdim_stats_path = candidate
    lowdim_stats = load_lowdim_stats(lowdim_stats_path) if lowdim_stats_path is not None else None
    if lowdim_stats_path is not None:
        print(f"[INFO] Loaded lowdim stats from {lowdim_stats_path}")

    task_episode_items = []
    descriptions = []
    for task_name in task_names:
        task_dir = os.path.join(data_prefix, task_name)
        episodes = load_task_episodes_from_pkls(
            task_dir, load_obs=load_obs, leading_keep=leading_keep
        )

        if len(episodes) == 0:
            print(f"[WARNING] Skip empty task: {task_name}")
            continue

        task_episode_items.append((task_name, episodes))

        if task_descriptions is not None and task_name in task_descriptions:
            desc = task_descriptions[task_name]
        else:
            # 기본은 폴더명을 언어 description으로 사용
            desc = task_name.replace("_", " ")
        descriptions.append(desc)

    if len(task_episode_items) == 0:
        raise ValueError("No valid task datasets found.")

    ft_stats = None
    if use_ft:
        norm_stats_path = ft_config.get("norm_stats_path")
        norm_stats_key = ft_config.get("norm_stats_key")
        if norm_stats_path is not None:
            if norm_stats_key is None:
                raise ValueError("ft_config.norm_stats_key must be set when norm_stats_path is set.")
            ft_stats = load_ft_stats_from_lowdim_json(norm_stats_path, norm_stats_key)
            print(f"[INFO] Loaded FT norm stats from {norm_stats_path} key={norm_stats_key}")
        else:
            stats_path = ft_config.get("stats_path")
            if stats_path is None:
                if ft_config["ft_source"] == "state":
                    candidate_name = "force_torque_stats.json"
                elif ft_config["ft_source"] in FORCE_HISTORY_KEYS:
                    candidate_name = "force_history_stats.json"
                else:
                    candidate_name = None
                candidate = (
                    os.path.join(data_prefix, candidate_name)
                    if candidate_name is not None
                    else None
                )
                if candidate is not None and os.path.exists(candidate):
                    stats_path = candidate

            if stats_path is not None:
                ft_stats = load_global_ft_stats_from_json(stats_path)
                print(f"[INFO] Loaded global force/torque stats from {stats_path}")
            else:
                all_episodes = [
                    ep
                    for _, episodes in task_episode_items
                    for ep in episodes
                ]
                ft_stats = compute_ft_stats_for_episodes(all_episodes, config=ft_config)
                print(f"[INFO] Computed global force/torque stats from {len(all_episodes)} episodes")

    manip_datasets = []
    for task_name, episodes in task_episode_items:
        ds = RealWorldSequenceDataset(
            episodes=episodes,
            obs_keys=obs_keys,
            dataset_keys=["actions"],
            seq_length=seq_len,
            obs_seq_length=obs_seq_len,
            frame_stack=frame_stack,
            pad_seq_length=True,
            pad_frame_stack=True,
            use_ft=use_ft,
            ft_stats=ft_stats,
            ft_config=ft_config,
            ft_shift=ft_shift,
            lowdim_stats=lowdim_stats,
            load_obs=load_obs,
            action_target_mode=action_target_mode,
            leading_keep=leading_keep,
        )
        manip_datasets.append(ds)

    if len(manip_datasets) == 0:
        raise ValueError("No valid task datasets found.")

    task_embs = get_task_embs(task_embedding_format, descriptions)

    datasets = [
        SequenceVLDataset(ds, emb, i)
        for i, (ds, emb) in enumerate(zip(manip_datasets, task_embs))
    ]

    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)

    print("\n===================  Real-World Benchmark Information  ===================")
    print(f" Root: {data_prefix}")
    print(f" # Tasks: {len(datasets)}")
    print(" Task names:", " | ".join([name for name, _ in task_episode_items]))
    print(" # demonstrations: " + " ".join(f"({x})" for x in n_demos))
    print(" # sequences: " + " ".join(f"({x})" for x in n_sequences))
    print("=========================================================================\n")

    return concat_dataset
