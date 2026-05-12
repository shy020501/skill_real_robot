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
    butterworth_lowpass_sequence,
    compute_ft_stats_for_episodes,
    compute_mask_sequence,
    extract_force_history_sequence,
    extract_ft_sequence,
    load_global_ft_stats_from_json,
    reduce_force_history,
    smooth_ft_sequence,
)

# IMAGE_KEYS = {"front_cam", "head_cam", "left_wrist_cam", "right_wrist_cam"}
IMAGE_KEYS = {"front_cam", "right_wrist_cam"}
FORCE_HISTORY_KEYS = ("right_force_history", "force_history")
RIGHT_STATE_KEYS = ("state", "right_state")
RIGHT_STATE_FORCE_HISTORY_KEY = "right_state_force_history"
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
def load_task_episodes_from_pkls(task_dir: str, load_obs: bool = True) -> List[Dict]:
    pkl_files = sorted(glob.glob(os.path.join(task_dir, "**/*.pkl"), recursive=True))
    print(f"[{os.path.basename(task_dir)}] Found pkl files: {len(pkl_files)}")

    episodes = []

    def keep_numeric_observation(obs):
        if not isinstance(obs, dict):
            return obs
        return {
            key: obs[key]
            for key in ("state", "right_force_history", "force_history")
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
                ep = trim_zero_action_episode(ep, trailing_keep=5)
                if ep is not None and len(ep["actions"]) > 0:
                    episodes.append(ep)
                cur = defaultdict(list)

        # 마지막 episode가 dones=True 없이 끝난 경우
        if len(cur) > 0 and len(next(iter(cur.values()))) > 0:
            ep = {k: cur[k] for k in keys}
            ep = trim_zero_action_episode(ep, trailing_keep=5)
            if ep is not None and len(ep["actions"]) > 0:
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


def get_force_history_value(obs_t):
    for force_key in FORCE_HISTORY_KEYS:
        if force_key in obs_t:
            return obs_t[force_key]
    raise KeyError(
        f"None of force history keys {FORCE_HISTORY_KEYS} found. "
        f"Available keys: {list(obs_t.keys())}"
    )


def get_right_state_value(obs_t, remove_force=False):
    arr = np.asarray(obs_t["state"])
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    right_state = arr[19:].astype(np.float32)
    if remove_force:
        right_state = np.delete(right_state, RIGHT_STATE_FORCE_TORQUE_IDXS, axis=0)
    return right_state


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
        return any(force_key in obs_t for force_key in FORCE_HISTORY_KEYS)
    if key in RIGHT_STATE_KEYS:
        return "state" in obs_t
    if key == RIGHT_STATE_FORCE_HISTORY_KEY:
        return "state" in obs_t and any(force_key in obs_t for force_key in FORCE_HISTORY_KEYS)
    return key in obs_t


def get_obs_value_with_alias(obs_t, key):
    if key in FORCE_HISTORY_KEYS:
        return get_force_history_value(obs_t)
    if key in RIGHT_STATE_KEYS:
        return get_right_state_value(obs_t)
    if key == RIGHT_STATE_FORCE_HISTORY_KEY:
        right_state_no_force = get_right_state_value(obs_t, remove_force=True)
        force_history = np.asarray(get_force_history_value(obs_t)).squeeze().astype(np.float32).reshape(-1)
        return np.concatenate([right_state_no_force, force_history], axis=0)
    return obs_t[key]


def smooth_force_history_sequence(force_history, ft_config):
    if ft_config["history_filter"] == "butterworth":
        force_history = np.stack([
            butterworth_lowpass_sequence(
                history,
                sample_rate_hz=ft_config["history_sample_rate_hz"],
                cutoff_hz=ft_config["history_cutoff_hz"],
                order=ft_config["history_filter_order"],
            )
            for history in force_history
        ], axis=0)
    elif ft_config["history_filter"] not in (None, "none"):
        raise ValueError(f"Unsupported history_filter: {ft_config['history_filter']}")
    return force_history.astype(np.float32)


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

        self.n_demos = len(episodes)

        for ep in self.episodes:
            relabel_episode_gripper_actions_from_state(ep)

        self.masked_ft_episodes = None
        self.ft_mask_episodes = None
        self.smoothed_state_ft_episodes = None
        self.smoothed_force_history_episodes = None
        needs_lowdim_force_history = self.load_obs and any(
            key in FORCE_HISTORY_KEYS or key == RIGHT_STATE_FORCE_HISTORY_KEY
            for key in self.obs_keys
        )
        needs_state_ft = self.use_ft and self.ft_config["ft_source"] == "state"
        needs_force_history = (
            needs_lowdim_force_history
            or (self.use_ft and self.ft_config["ft_source"] == "force_history")
        )
        if needs_state_ft:
            self.smoothed_state_ft_episodes = []
            for ep in self.episodes:
                state_ft = extract_ft_sequence(ep)
                self.smoothed_state_ft_episodes.append(
                    smooth_ft_sequence(state_ft, ema_alpha=self.ft_config["ema_alpha"])
                )

        if needs_force_history:
            self.smoothed_force_history_episodes = []
            for ep in self.episodes:
                force_history = extract_force_history_sequence(ep)
                self.smoothed_force_history_episodes.append(
                    smooth_force_history_sequence(force_history, self.ft_config)
                )

        if self.use_ft:
            if ft_stats is None:
                ft_stats = compute_ft_stats_for_episodes(self.episodes, config=self.ft_config)
            self.masked_ft_episodes = []
            self.ft_mask_episodes = []
            for ep_idx, _ in enumerate(self.episodes):
                if self.ft_config["ft_source"] == "state":
                    ft = normalize_ft_sequence(self.smoothed_state_ft_episodes[ep_idx], ft_stats)
                    if self.ft_config["use_threshold_mask"]:
                        _, ft_mask = compute_mask_sequence(ft, config=self.ft_config)
                        masked_ft = ft * ft_mask.astype(np.float32)
                    else:
                        ft_mask = np.ones_like(ft, dtype=np.float32)
                        masked_ft = ft
                elif self.ft_config["ft_source"] == "force_history":
                    ft = normalize_ft_sequence(self.smoothed_force_history_episodes[ep_idx], ft_stats)
                    masked_ft = reduce_ft_history_sequence(ft, self.ft_config)
                    ft_mask = np.ones_like(masked_ft, dtype=np.float32)
                else:
                    raise ValueError(f"Unsupported ft_source: {self.ft_config['ft_source']}")
                self.masked_ft_episodes.append(masked_ft)
                self.ft_mask_episodes.append(ft_mask.astype(np.float32))

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
                if key in FORCE_HISTORY_KEYS and self.smoothed_force_history_episodes is not None:
                    arr = self.smoothed_force_history_episodes[ep_idx][t]
                elif key == RIGHT_STATE_FORCE_HISTORY_KEY and self.smoothed_force_history_episodes is not None:
                    right_state_no_force = get_right_state_value(obs_t, remove_force=True)
                    force_history = self.smoothed_force_history_episodes[ep_idx][t].reshape(-1)
                    arr = np.concatenate([right_state_no_force, force_history], axis=0)
                else:
                    arr = np.asarray(get_obs_value_with_alias(obs_t, key))

                # right_force_history: (1, 10, 6), (10, 6), (1, 60), or (60,) -> (60,)
                if key in FORCE_HISTORY_KEYS:
                    arr = arr.squeeze().astype(np.float32).reshape(-1)

                # image: (1, H, W, C) -> (H, W, C)
                elif key in IMAGE_KEYS:
                    if arr.ndim == 4 and arr.shape[0] == 1:
                        arr = arr[0]   # (1, H, W, C) -> (H, W, C)
                    arr = arr.astype(np.uint8)

                else:
                    arr = arr.astype(np.float32)

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
    load_obs: bool = True,
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
            # 기본은 폴더명을 언어 description으로 사용
            desc = task_name.replace("_", " ")
        descriptions.append(desc)

    if len(task_episode_items) == 0:
        raise ValueError("No valid task datasets found.")

    ft_stats = None
    if use_ft:
        stats_path = ft_config.get("stats_path")
        if stats_path is None and ft_config["ft_source"] == "state":
            candidate = os.path.join(data_prefix, "force_torque_stats.json")
            if os.path.exists(candidate):
                stats_path = candidate

        if stats_path is not None:
            if ft_config["ft_source"] != "state":
                raise ValueError(
                    f"ft_config.stats_path is for state force stats, but ft_source={ft_config['ft_source']}"
                )
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

