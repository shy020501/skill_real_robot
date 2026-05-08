import os
import glob
from typing import Dict, List, Optional

import numpy as np
import tensorflow_datasets as tfds

from torch.utils.data import Dataset, ConcatDataset

from transformers import AutoModel, AutoTokenizer, logging


IMAGE_KEYS = {
    "wrist_image_left",
    "exterior_image_1_left",
    "exterior_image_2_left",
}

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

def task_name_to_description(task_name: str) -> str:
    """
    Example:
        REASSEMBLE_pick_round_peg_2 -> "pick round peg 2"
    """
    if task_name.startswith("REASSEMBLE_"):
        task_name = task_name[len("REASSEMBLE_"):]
    return task_name.replace("_", " ").strip()


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

def load_task_episodes_from_rlds(
    task_dir: str,
    combine_lowdim_as_state: bool = True,
):
    """
    task_dir:
        directory containing TFDS-folder-style RLDS dataset
        e.g. /path/to/df_rlds/REASSEMBLE_pick_usb
    """
    builder = tfds.builder_from_directory(task_dir)
    ds = builder.as_dataset(split="train")

    episodes = []

    for ep in tfds.as_numpy(ds):
        action_list = []
        obs_list = []

        reward_list = []
        discount_list = []
        is_first_list = []
        is_last_list = []
        is_terminal_list = []

        for step in ep["steps"]:
            action = np.asarray(step["action"], dtype=np.float32)  # (7,)
            obs = step["observation"]

            cartesian_position = np.asarray(obs["cartesian_position"], dtype=np.float32)  # (6,)
            gripper_position = np.asarray(obs["gripper_position"], dtype=np.float32)      # (1,)

            wrist_img = np.asarray(obs["wrist_image_left"], dtype=np.uint8)
            ext1_img = np.asarray(obs["exterior_image_1_left"], dtype=np.uint8)
            ext2_img = np.asarray(obs["exterior_image_2_left"], dtype=np.uint8)

            if action.shape != (7,):
                raise ValueError(f"Expected action shape (7,), got {action.shape}")
            if cartesian_position.shape != (6,):
                raise ValueError(
                    f"Expected cartesian_position shape (6,), got {cartesian_position.shape}"
                )
            if gripper_position.shape != (1,):
                raise ValueError(
                    f"Expected gripper_position shape (1,), got {gripper_position.shape}"
                )

            if combine_lowdim_as_state:
                # state = eef pose (6) + gripper (1) => 7
                state = np.concatenate(
                    [cartesian_position, gripper_position],
                    axis=0,
                ).astype(np.float32)

                obs_t = {
                    "state": state,  # (7,)
                    "wrist_image_left": wrist_img,
                    "exterior_image_1_left": ext1_img,
                    "exterior_image_2_left": ext2_img,
                }
            else:
                obs_t = {
                    "cartesian_position": cartesian_position,
                    "gripper_position": gripper_position,
                    "wrist_image_left": wrist_img,
                    "exterior_image_1_left": ext1_img,
                    "exterior_image_2_left": ext2_img,
                }

            action_list.append(action)
            obs_list.append(obs_t)
            reward_list.append(np.float32(step.get("reward", 0.0)))
            discount_list.append(np.float32(step.get("discount", 1.0)))
            is_first_list.append(bool(step.get("is_first", False)))
            is_last_list.append(bool(step.get("is_last", False)))
            is_terminal_list.append(bool(step.get("is_terminal", False)))

        if len(action_list) == 0:
            continue

        episode = {
            "actions": action_list,
            "observations": obs_list,
            "reward": np.asarray(reward_list, dtype=np.float32),
            "discount": np.asarray(discount_list, dtype=np.float32),
            "is_first": np.asarray(is_first_list, dtype=np.bool_),
            "is_last": np.asarray(is_last_list, dtype=np.bool_),
            "is_terminal": np.asarray(is_terminal_list, dtype=np.bool_),
        }
        episodes.append(episode)

    print(f"[{os.path.basename(task_dir)}] Total episodes: {len(episodes)}")
    return episodes


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

class RealWorldRldsSequenceDataset(Dataset):
    def __init__(
        self,
        episodes: List[Dict],
        obs_keys: List[str],
        dataset_keys: List[str] = ["actions"],
        seq_length: int = 1,
        obs_seq_length: int = 1,
        frame_stack: int = 1,
        pad_seq_length: bool = True,
    ):
        self.episodes = episodes
        self.obs_keys = obs_keys
        self.dataset_keys = dataset_keys
        self.seq_length = seq_length
        self.obs_seq_length = obs_seq_length
        self.frame_stack = frame_stack
        self.pad_seq_length = pad_seq_length

        self.n_demos = len(episodes)

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
            action_list = [np.asarray(x, dtype=np.float32) for x in ep["actions"]]
            action_seq = self._slice_with_padding(action_list, start_t, self.seq_length)
            ret["actions"] = np.stack(action_seq, axis=0).astype(np.float32)

        observations = ep["observations"]
        if len(observations) == 0:
            raise ValueError(f"Episode {ep_idx} has empty observations")

        obs_dict = {}

        for key in self.obs_keys:
            if key not in observations[0]:
                raise KeyError(
                    f"Observation key '{key}' not found in observations[0]. "
                    f"Available keys: {list(observations[0].keys())}"
                )

            obs_list = []
            for obs_t in observations:
                arr = np.asarray(obs_t[key])
                if key in IMAGE_KEYS:
                    arr = arr.astype(np.uint8)
                else:
                    arr = arr.astype(np.float32)
                obs_list.append(arr)

            obs_seq = self._slice_with_padding(obs_list, start_t, self.obs_seq_length)

            if key in IMAGE_KEYS:
                if self.frame_stack > 1:
                    fs_list = []
                    for t in range(len(obs_seq)):
                        left = max(0, t - self.frame_stack + 1)
                        frames = obs_seq[left:t + 1]
                        if len(frames) < self.frame_stack:
                            pad = [frames[0]] * (self.frame_stack - len(frames))
                            frames = pad + frames
                        fs_list.append(np.stack(frames, axis=0))  # (F,H,W,C)
                    arr = np.stack(fs_list, axis=0)  # (T,F,H,W,C)
                    arr = arr.transpose(0, 1, 4, 2, 3)  # (T,F,C,H,W)
                else:
                    arr = np.stack(obs_seq, axis=0)  # (T,H,W,C)
                    arr = arr.transpose(0, 3, 1, 2)  # (T,C,H,W)
                obs_dict[key] = arr.astype(np.uint8)
            else:
                arr = np.stack(obs_seq, axis=0).astype(np.float32)
                obs_dict[key] = arr

        ret["obs"] = obs_dict
        ret["task_emb"] = None
        return ret


def build_realworld_rlds_dataset(
    data_prefix: str,
    seq_len: int,
    frame_stack: int,
    shape_meta: Dict,
    task_embedding_format: str = "clip",
    obs_seq_len: int = 1,
    task_descriptions: Optional[Dict[str, str]] = None,
    combine_lowdim_as_state: bool = True,
    min_episodes_per_task: int = 1,
):
    """
    Supported forms:
      1) data_prefix itself is one RLDS task dir
      2) data_prefix contains multiple RLDS task dirs

    Policy:
      - one task_dir = one task
      - no regrouping by instruction
      - task description comes from directory name
    """
    candidate_dirs = []

    if len(glob.glob(os.path.join(data_prefix, "*.tfrecord-*"))) > 0:
        candidate_dirs = [data_prefix]
    else:
        subdirs = sorted([
            os.path.join(data_prefix, d)
            for d in os.listdir(data_prefix)
            if os.path.isdir(os.path.join(data_prefix, d))
        ])
        for d in subdirs:
            if len(glob.glob(os.path.join(d, "*.tfrecord-*"))) > 0:
                candidate_dirs.append(d)

    if len(candidate_dirs) == 0:
        raise ValueError(f"No RLDS task directories found under {data_prefix}")

    obs_keys = []
    obs_keys += list(shape_meta["observation"]["rgb"].keys())
    obs_keys += list(shape_meta["observation"]["lowdim"].keys())

    manip_datasets = []
    descriptions = []
    kept_task_names = []

    for task_dir in candidate_dirs:
        task_name = os.path.basename(task_dir)

        episodes = load_task_episodes_from_rlds(
            task_dir,
            combine_lowdim_as_state=combine_lowdim_as_state,
        )
        episodes = episodes[:50]
        for ep in episodes:
            for obs in ep["observations"]:
                for k in list(obs.keys()):
                    if k in IMAGE_KEYS:
                        del obs[k]

        if len(episodes) == 0:
            print(f"[WARNING] Skip empty task dataset: {task_name}")
            continue

        if len(episodes) < min_episodes_per_task:
            print(
                f"[WARNING] Skip task '{task_name}' because "
                f"{len(episodes)} < min_episodes_per_task={min_episodes_per_task}"
            )
            continue

        ds = RealWorldRldsSequenceDataset(
            episodes=episodes,
            obs_keys=obs_keys,
            dataset_keys=["actions"],
            seq_length=seq_len,
            obs_seq_length=obs_seq_len,
            frame_stack=frame_stack,
            pad_seq_length=True,
        )
        manip_datasets.append(ds)

        if task_descriptions is not None and task_name in task_descriptions:
            desc = task_descriptions[task_name]
        else:
            desc = task_name_to_description(task_name)

        descriptions.append(desc)
        kept_task_names.append(task_name)

    if len(manip_datasets) == 0:
        raise ValueError("No valid RLDS task datasets found.")

    task_embs = get_task_embs(task_embedding_format, descriptions)

    datasets = [
        SequenceVLDataset(ds, emb, i)
        for i, (ds, emb) in enumerate(zip(manip_datasets, task_embs))
    ]

    n_demos = [data.n_demos for data in datasets]
    n_sequences = [data.total_num_sequences for data in datasets]
    concat_dataset = ConcatDataset(datasets)

    print("\n===================  RLDS Real-World Benchmark Information  ===================")
    print(f" Root: {data_prefix}")
    print(f" # Tasks: {len(datasets)}")
    print(" Task names:")
    for i, (name, demos, seqs, desc) in enumerate(zip(kept_task_names, n_demos, n_sequences, descriptions)):
        print(f"   [{i}] {name} | demos={demos} | sequences={seqs} | desc={desc}")
    print("================================================================================\n")

    return concat_dataset