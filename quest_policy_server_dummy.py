import argparse
import asyncio
import copy
import logging
import os
import pickle
import random
import traceback

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames
from hydra.utils import instantiate
from omegaconf import OmegaConf

import quest.utils.utils as utils


DEFAULT_DATASET_PATH = (
    "/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/demos/multi_usb/"
    "2026-04-15/10_demos_2026-04-15_18-30-01.pkl"
)
DEFAULT_CHECKPOINT_PATH = (
    "/NHNHOME/WORKSPACE/0226010443_A/seunghyo/real_robot/QueST/experiments/"
    "real_robot/REAL_ROBOT_MULTI/quest/lowdim_autoencoder_quest/"
    "unmasked_block_32_ds_4_quest/0/stage_0/multitask_model_epoch_0200.pth"
)
DEFAULT_RIGHT_ACTION_SLICE = "7:14"
PKL_RIGHT_GRIPPER_STATE_IDX = 19
PKL_RIGHT_GRIPPER_ACTION_IDX = 13


def parse_slice(text):
    if text in (None, "", "none"):
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError(f"Slice must look like 'start:end', got {text!r}")
    start = int(parts[0]) if parts[0] else None
    end = int(parts[1]) if parts[1] else None
    return slice(start, end)


def trim_zero_action_episode(ep, trailing_keep=5):
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

    trim_end = min(T, end_idx + 1 + trailing_keep)
    return {k: v[start_idx:trim_end] for k, v in ep.items()}


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


def finalize_episodes(episodes):
    episodes = [ep for ep in episodes if ep is not None and len(ep["actions"]) > 0]
    for ep in episodes:
        relabel_episode_gripper_actions_from_state(ep)
    return episodes


def load_episodes_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "episodes" in data:
        episodes = [trim_zero_action_episode(ep, trailing_keep=5) for ep in data["episodes"]]
        return finalize_episodes(episodes)

    if not isinstance(data, (list, tuple)):
        raise TypeError(f"Unsupported dataset type: {type(data)}")
    if len(data) == 0:
        raise ValueError(f"Dataset is empty: {pkl_path}")

    first = data[0]
    if isinstance(first, dict) and "actions" in first and np.asarray(first["actions"]).ndim >= 2:
        episodes = [trim_zero_action_episode(ep, trailing_keep=5) for ep in data]
        return finalize_episodes(episodes)

    episodes = []
    cur = {}
    keys = list(first.keys())
    for step in data:
        for key in keys:
            cur.setdefault(key, []).append(step.get(key))
        if bool(step.get("dones", False)):
            ep = trim_zero_action_episode(cur, trailing_keep=5)
            if ep is not None and len(ep["actions"]) > 0:
                episodes.append(ep)
            cur = {}

    if cur:
        ep = trim_zero_action_episode(cur, trailing_keep=5)
        if ep is not None and len(ep["actions"]) > 0:
            episodes.append(ep)
    return finalize_episodes(episodes)


def episode_actions(episode):
    if "actions" not in episode:
        raise KeyError(f"Episode missing 'actions'. Available keys: {list(episode.keys())}")
    return np.asarray(episode["actions"], dtype=np.float32)


def choose_action_slice(actions, action_dim, requested_slice):
    if actions.ndim != 2:
        actions = actions.reshape(actions.shape[0], -1)
    if actions.shape[-1] == action_dim:
        return actions
    if requested_slice is not None:
        sliced = actions[:, requested_slice]
        if sliced.shape[-1] != action_dim:
            raise ValueError(
                f"Requested action slice gives dim {sliced.shape[-1]}, "
                f"but checkpoint action_dim is {action_dim}"
            )
        return sliced
    if actions.shape[-1] >= action_dim:
        return actions[:, :action_dim]
    raise ValueError(f"Action dim {actions.shape[-1]} is smaller than model action_dim {action_dim}")


class ReconstructedActionPolicy:
    def __init__(
        self,
        dataset_path,
        checkpoint_path,
        device,
        seed=None,
        episode_index=None,
        action_slice=DEFAULT_RIGHT_ACTION_SLICE,
        loop=True,
    ):
        self.device = device if torch.cuda.is_available() or "cuda" not in device else "cpu"
        self.dataset_path = os.path.expanduser(dataset_path)
        self.loop = loop
        self.model, self.cfg, self.checkpoint_path = self._load_autoencoder_policy(checkpoint_path)
        self.action_dim = int(self.cfg.task.shape_meta.action_dim)
        shape_meta = self.cfg.task.shape_meta
        self.image_keys = list(shape_meta.observation.rgb.keys())
        self.lowdim_keys = ["state"]
        self.action_slice = parse_slice(action_slice)

        self.episodes = load_episodes_from_pkl(self.dataset_path)
        if not self.episodes:
            raise ValueError(f"No episodes found in {self.dataset_path}")

        rng = random.Random(seed)
        self.episode_index = (
            int(episode_index)
            if episode_index is not None
            else rng.randrange(len(self.episodes))
        )
        if not 0 <= self.episode_index < len(self.episodes):
            raise IndexError(
                f"episode_index={self.episode_index} out of range for {len(self.episodes)} episodes"
            )

        self.raw_actions = episode_actions(self.episodes[self.episode_index])
        self.actions = choose_action_slice(self.raw_actions, self.action_dim, self.action_slice)
        # Debug path: bypass the autoencoder reconstruction and replay original actions.
        # Keep the attribute name so the websocket serving path stays unchanged.
        self.reconstructed_actions = self._reconstruct_actions(self.actions)
        # self.reconstructed_actions = self._chunk_original_actions(self.actions)
        self.cursor = 0

    def _load_autoencoder_policy(self, checkpoint_path):
        checkpoint_path = utils.get_latest_checkpoint(os.path.expanduser(checkpoint_path))
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if "config" not in state_dict:
            raise KeyError("Checkpoint does not contain saved config. Use a checkpoint saved by train.py.")

        saved_cfg = copy.deepcopy(state_dict["config"])
        saved_cfg["device"] = self.device
        saved_cfg["algo"]["policy"]["device"] = self.device
        saved_cfg["algo"]["policy"]["autoencoder"]["fsq_level"] = [8, 5, 5, 5]
        if "policy_prior" in saved_cfg["algo"]["policy"]:
            saved_cfg["algo"]["policy"]["policy_prior"]["device"] = self.device

        saved_cfg = OmegaConf.create(saved_cfg)
        model = instantiate(saved_cfg.algo.policy, shape_meta=saved_cfg.task.shape_meta)
        model.to(self.device)
        if hasattr(model, "device"):
            model.device = self.device
        model.eval()
        utils.soft_load_state_dict(model, state_dict["model"])
        model.eval()
        model.autoencoder.eval()
        return model, saved_cfg, checkpoint_path

    @torch.no_grad()
    def _reconstruct_actions(self, actions):
        block_size = int(self.cfg.algo.skill_block_size)
        if actions.shape[0] == 0:
            raise ValueError("Selected episode has no actions")

        pad = (-actions.shape[0]) % block_size
        if pad:
            pad_values = np.repeat(actions[-1:], pad, axis=0)
            padded = np.concatenate([actions, pad_values], axis=0)
        else:
            padded = actions

        chunks = padded.reshape(-1, block_size, actions.shape[-1])
        x = torch.as_tensor(chunks, dtype=torch.float32, device=self.device)
        # print(x.shape)
        # print(x)
        pred, _, _, _, _ = self.model.autoencoder(x)
        pred = pred.detach().cpu().numpy().reshape(-1, actions.shape[-1])
        return pred[: actions.shape[0]].astype(np.float32)

    def _chunk_original_actions(self, actions):
        block_size = int(self.cfg.algo.skill_block_size)
        if actions.shape[0] == 0:
            raise ValueError("Selected episode has no actions")

        pad = (-actions.shape[0]) % block_size
        if pad:
            pad_values = np.repeat(actions[-1:], pad, axis=0)
            padded = np.concatenate([actions, pad_values], axis=0)
        else:
            padded = actions

        chunks = padded.reshape(-1, block_size, actions.shape[-1])
        return chunks.reshape(-1, actions.shape[-1])[: actions.shape[0]].astype(np.float32)

    def metadata(self):
        return {
            "policy": "quest_reconstructed_action_dummy",
            "dataset_path": self.dataset_path,
            "checkpoint_path": self.checkpoint_path,
            "image_keys": self.image_keys,
            "lowdim_keys": self.lowdim_keys,
            "episode_index": self.episode_index,
            "num_episodes": len(self.episodes),
            "raw_action_shape": tuple(self.raw_actions.shape),
            "action_shape": tuple(self.actions.shape),
            "reconstructed_action_shape": tuple(self.reconstructed_actions.shape),
            "action_dim": self.action_dim,
            "device": self.device,
            "loop": self.loop,
        }

    def infer(self, request):
        if bool(request.pop("_reset", False)):
            self.cursor = 0

        if self.cursor >= len(self.reconstructed_actions):
            if not self.loop:
                raise StopIteration("Reconstructed episode is exhausted")
            self.cursor = 0

        action = self.reconstructed_actions[self.cursor]
        self.cursor += 1
        return {"actions": np.asarray(action, dtype=np.float32)}


class PickleWebsocketPolicyServer:
    def __init__(self, policy, host, port):
        self.policy = policy
        self.host = host
        self.port = port
        self.logger = logging.getLogger("quest_policy_server_dummy")

    async def serve_forever(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ):
            self.logger.info("Dummy QueST policy server listening on %s:%s", self.host, self.port)
            await asyncio.Future()

    async def _handler(self, websocket):
        self.logger.info("Connection from %s opened", websocket.remote_address)
        await websocket.send(pickle.dumps(self.policy.metadata()))

        while True:
            try:
                request = pickle.loads(await websocket.recv())
                response = self.policy.infer(request)
                await websocket.send(pickle.dumps(response))
            except websockets.ConnectionClosed:
                self.logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                tb = traceback.format_exc()
                self.logger.error(tb)
                await websocket.send(pickle.dumps({"error": tb}))
                await websocket.close(code=websockets.frames.CloseCode.INTERNAL_ERROR)
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--checkpoint_path", type=str, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode_index", type=int, default=None)
    parser.add_argument("--action_slice", type=str, default=DEFAULT_RIGHT_ACTION_SLICE)
    parser.add_argument("--no_loop", action="store_true")
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    policy = ReconstructedActionPolicy(
        dataset_path=args.dataset_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device,
        seed=args.seed,
        episode_index=args.episode_index,
        action_slice=args.action_slice,
        loop=not args.no_loop,
    )
    logging.info("Loaded dummy policy metadata: %s", policy.metadata())
    server = PickleWebsocketPolicyServer(policy, args.host, args.port)
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
