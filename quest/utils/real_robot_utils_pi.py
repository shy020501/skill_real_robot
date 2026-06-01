import os
import json
from typing import Dict, List, Optional

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from quest.utils.force_torque_utils import DEFAULT_FT_CONFIG
from quest.utils.real_robot_utils import (
    SequenceVLDataset,
    frame_stack_list,
    get_task_embs,
    load_task_descriptions,
    load_task_episodes_from_pkls,
    pad_sequence_list,
    relabel_episode_gripper_actions_from_state,
)


IMAGE_KEYS = {"front_cam", "right_wrist_cam"}
FORCE_HISTORY_KEYS = ("left_force_history", "right_force_history")
STATE_HISTORY_KEYS = ("left_state_history", "right_state_history")
RIGHT_STATE_KEYS = ("state",)

FORCE_HISTORY_LEN = 10
FORCE_DIM = 6
PI_NORM_EPS = 1e-6
PI_STATE_INDICES = (
    23, 24, 25, 26, 27, 28,
    19,
    20, 21, 22,
    29, 30, 31,
)
PI_RIGHT_STATE_INDICES = (
    4, 5, 6, 7, 8, 9,
    0,
    1, 2, 3,
    10, 11, 12,
)


def prepare_image(img_array):
    if not isinstance(img_array, np.ndarray):
        img_array = np.array(img_array)
    img_array = img_array.squeeze()
    if img_array.dtype in [np.float32, np.float64]:
        img_array = (img_array * 255).astype(np.uint8)
    if img_array.ndim == 3 and img_array.shape[0] == 3:
        img_array = np.transpose(img_array, (1, 2, 0))
    return img_array


def load_pi_stats(path):
    with open(path, "r") as f:
        raw_stats = json.load(f)
    raw_stats = raw_stats.get("norm_stats", raw_stats)

    stats = {}
    for key, value in raw_stats.items():
        if not isinstance(value, dict) or "mean" not in value or "std" not in value:
            continue
        stats[key] = {
            "mean": np.asarray(value["mean"], dtype=np.float32),
            "std": np.asarray(value["std"], dtype=np.float32),
        }

    required_keys = {"state", "actions", "force_history"}
    missing = required_keys - set(stats)
    if missing:
        raise KeyError(
            f"PI stats file '{path}' is missing keys: {sorted(missing)}. "
            "Expected keys under top-level 'norm_stats'."
        )
    return stats


def normalize_pi_value(value, key, pi_stats):
    if pi_stats is None or key not in pi_stats:
        return value.astype(np.float32)

    value = value.astype(np.float32)
    mean = pi_stats[key]["mean"]
    std = pi_stats[key]["std"]

    if value.shape[-1] == mean.shape[-1]:
        return ((value - mean) / (std + PI_NORM_EPS)).astype(np.float32)

    if key == "force_history" and mean.shape[-1] == FORCE_DIM and value.size % FORCE_DIM == 0:
        original_shape = value.shape
        value = value.reshape(-1, FORCE_DIM)
        normalized = (value - mean) / (std + PI_NORM_EPS)
        return normalized.reshape(original_shape).astype(np.float32)

    raise ValueError(
        f"PI stats for '{key}' have dim {mean.shape[-1]}, but value shape is {value.shape}."
    )


def normalize_pi_state_force(value, pi_stats):
    if pi_stats is None or "state" not in pi_stats:
        return value.astype(np.float32)
    mean = pi_stats["state"]["mean"][7:13]
    std = pi_stats["state"]["std"][7:13]
    return ((value.astype(np.float32) - mean) / (std + PI_NORM_EPS)).astype(np.float32)


def get_pi_state_value(obs_t):
    raw_state = np.asarray(obs_t["state"], dtype=np.float32).squeeze()
    if raw_state.shape[-1] <= max(PI_STATE_INDICES):
        raise ValueError(
            f"Cannot extract PI state from shape {raw_state.shape}; "
            f"need indices up to {max(PI_STATE_INDICES)}."
        )
    return raw_state[..., PI_STATE_INDICES].astype(np.float32)


def prepare_force_history(obs_t, key="right_force_history"):
    if key not in FORCE_HISTORY_KEYS:
        raise KeyError(f"Unsupported force history key '{key}'. Expected one of {FORCE_HISTORY_KEYS}.")
    if key not in obs_t:
        raise KeyError(f"Observation is missing '{key}'. Available keys: {list(obs_t.keys())}")

    force_history = np.asarray(obs_t[key]).squeeze()
    if force_history.shape != (FORCE_HISTORY_LEN, FORCE_DIM):
        raise ValueError(f"Unexpected force history shape for '{key}': {force_history.shape}")
    return force_history.flatten().astype(np.float32)


def prepare_force_history_sequence(obs_t, key="right_force_history"):
    if key not in FORCE_HISTORY_KEYS:
        raise KeyError(f"Unsupported force history key '{key}'. Expected one of {FORCE_HISTORY_KEYS}.")
    if key not in obs_t:
        raise KeyError(f"Observation is missing '{key}'. Available keys: {list(obs_t.keys())}")

    force_history = np.asarray(obs_t[key]).squeeze()
    if force_history.shape != (FORCE_HISTORY_LEN, FORCE_DIM):
        raise ValueError(f"Unexpected force history shape for '{key}': {force_history.shape}")
    return force_history.astype(np.float32)


def prepare_state_history(state_history):
    state_history = np.asarray(state_history, dtype=np.float32).squeeze()
    if state_history.shape[-1] > max(PI_STATE_INDICES):
        return state_history[..., PI_STATE_INDICES].astype(np.float32)
    if state_history.shape[-1] > max(PI_RIGHT_STATE_INDICES):
        return state_history[..., PI_RIGHT_STATE_INDICES].astype(np.float32)
    raise ValueError(
        f"Cannot extract PI state history from shape {state_history.shape}; "
        f"need full-state indices up to {max(PI_STATE_INDICES)} or "
        f"right-state indices up to {max(PI_RIGHT_STATE_INDICES)}."
    )


def has_obs_key(obs_t, key):
    if key in FORCE_HISTORY_KEYS:
        return key in obs_t
    if key in RIGHT_STATE_KEYS:
        return "state" in obs_t
    return key in obs_t


def get_obs_value_with_alias(obs_t, key):
    if key in FORCE_HISTORY_KEYS:
        return prepare_force_history(obs_t, key)
    if key in RIGHT_STATE_KEYS:
        return get_pi_state_value(obs_t)
    if key in STATE_HISTORY_KEYS:
        return prepare_state_history(obs_t[key])
    return obs_t[key]


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
        self.ft_config = {**DEFAULT_FT_CONFIG, **(ft_config or {})}
        self.pi_stats = lowdim_stats

        self.n_demos = len(episodes)

        for ep in self.episodes:
            relabel_episode_gripper_actions_from_state(ep)

        self.masked_ft_episodes = None
        self.ft_mask_episodes = None
        if self.use_ft:
            self.masked_ft_episodes = []
            self.ft_mask_episodes = []
            ft_source = self.ft_config["ft_source"]
            for ep in self.episodes:
                ft_values = []
                for obs_t in ep.get("observations", []):
                    if ft_source == "state":
                        ft_values.append(normalize_pi_state_force(get_pi_state_value(obs_t)[7:13], self.pi_stats))
                    elif ft_source in FORCE_HISTORY_KEYS:
                        ft_values.append(
                            normalize_pi_value(
                                prepare_force_history_sequence(obs_t, ft_source),
                                "force_history",
                                self.pi_stats,
                            )
                        )
                    else:
                        raise ValueError(f"Unsupported ft_source: {ft_source}")
                ft = np.stack(ft_values, axis=0).astype(np.float32)
                self.masked_ft_episodes.append(ft)
                self.ft_mask_episodes.append(np.ones_like(ft, dtype=np.float32))

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

        if "actions" in self.dataset_keys:
            action_list = []
            for x in ep["actions"]:
                arr = np.asarray(x).astype(np.float32)
                arr = arr[7:]
                arr = normalize_pi_value(arr, "actions", self.pi_stats)
                action_list.append(arr)

            action_seq = self._slice_with_padding(action_list, start_t, self.seq_length)
            ret["actions"] = np.stack(action_seq, axis=0).astype(np.float32)

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
            for obs_t in observations:
                arr = np.asarray(get_obs_value_with_alias(obs_t, key))

                if key in IMAGE_KEYS:
                    arr = prepare_image(arr).astype(np.uint8)
                else:
                    arr = arr.squeeze().astype(np.float32)
                    if key in FORCE_HISTORY_KEYS:
                        arr = normalize_pi_value(arr, "force_history", self.pi_stats)
                    elif key in RIGHT_STATE_KEYS or key in STATE_HISTORY_KEYS:
                        arr = normalize_pi_value(arr, "state", self.pi_stats)

                obs_list.append(arr)

            obs_seq = self._slice_with_padding(obs_list, start_t, self.obs_seq_length)

            if self.frame_stack > 1:
                obs_seq = frame_stack_list(obs_seq, self.frame_stack)
                obs_seq = [np.stack(x, axis=0) for x in obs_seq]

            obs_arr = np.stack(obs_seq, axis=0)

            if key in IMAGE_KEYS:
                if obs_arr.ndim == 4 and obs_arr.shape[-1] in (1, 3):
                    obs_arr = np.transpose(obs_arr, (0, 3, 1, 2))
                elif obs_arr.ndim == 5 and obs_arr.shape[-1] in (1, 3):
                    obs_arr = np.transpose(obs_arr, (0, 1, 4, 2, 3))
                obs_arr = obs_arr.astype(np.uint8)
            else:
                obs_arr = obs_arr.astype(np.float32)

            obs_dict[key] = obs_arr

        ret["obs"] = obs_dict
        return ret


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
    pi_stats_path: Optional[str] = None,
    load_obs: bool = True,
):
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
    if pi_stats_path is None:
        candidate = os.path.join(data_prefix, "pi_stats.json")
        if os.path.exists(candidate):
            pi_stats_path = candidate
        elif lowdim_stats_path is not None:
            pi_stats_path = lowdim_stats_path
    if pi_stats_path is None:
        raise FileNotFoundError(
            f"PI stats file not found. Expected {os.path.join(data_prefix, 'pi_stats.json')} "
            "or pass dataset.pi_stats_path."
        )
    pi_stats = load_pi_stats(pi_stats_path)
    print(f"[INFO] Loaded PI norm stats from {pi_stats_path}")

    task_episode_items = []
    descriptions = []
    for task_name in task_names:
        task_dir = os.path.join(data_prefix, task_name)
        episodes = load_task_episodes_from_pkls(task_dir, load_obs=load_obs)

        if len(episodes) == 0:
            print(f"[WARNING] Skip empty task: {task_name}")
            continue

        task_episode_items.append((task_name, episodes))

        if task_descriptions is not None and task_name in task_descriptions:
            desc = task_descriptions[task_name]
        else:
            desc = task_name.replace("_", " ")
        descriptions.append(desc)

    if len(task_episode_items) == 0:
        raise ValueError("No valid task datasets found.")

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
            ft_stats=None,
            ft_config=ft_config,
            ft_shift=ft_shift,
            lowdim_stats=pi_stats,
            load_obs=load_obs,
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
