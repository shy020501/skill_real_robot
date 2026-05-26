import argparse
import asyncio
import copy
import logging
import os
import pickle
import traceback

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi
import torch
import websockets.asyncio.server
import websockets.frames
from hydra.utils import instantiate
from omegaconf import OmegaConf

import quest.utils.utils as utils
from quest.utils.force_torque_utils import DEFAULT_FT_CONFIG
from quest.utils.real_robot_utils import (
    FORCE_HISTORY_KEYS,
    get_task_embs,
    load_lowdim_stats,
    normalize_lowdim_value,
)


class OnlineForceHistoryFilter:
    def __init__(self, ft_config):
        self.ft_config = {**DEFAULT_FT_CONFIG, **(ft_config or {})}
        self.enabled = self.ft_config["history_filter"] == "butterworth"
        self.states = {}

        if not self.enabled:
            if self.ft_config["history_filter"] not in (None, "none"):
                raise ValueError(f"Unsupported history_filter: {self.ft_config['history_filter']}")
            self.sos = None
            return

        sample_rate_hz = self.ft_config["history_sample_rate_hz"]
        cutoff_hz = self.ft_config["history_cutoff_hz"]
        nyquist = 0.5 * sample_rate_hz
        if not 0.0 < cutoff_hz < nyquist:
            raise ValueError(
                f"cutoff_hz must be in (0, Nyquist). Got cutoff_hz={cutoff_hz}, "
                f"sample_rate_hz={sample_rate_hz}."
            )
        self.sos = butter(
            self.ft_config["history_filter_order"],
            cutoff_hz / nyquist,
            btype="low",
            output="sos",
        )

    def reset(self):
        self.states.clear()

    def _initial_zi(self, first_sample):
        zi = sosfilt_zi(self.sos)
        return zi[:, :, None] * first_sample[None, None, :]

    def _filter_sequence_from_initial(self, key, history):
        zi = self._initial_zi(history[0])
        filtered, zi = sosfilt(self.sos, history, axis=0, zi=zi)
        self.states[key] = {
            "raw_history": history.copy(),
            "filtered_history": filtered.astype(np.float32),
            "zi": zi,
        }
        return self.states[key]["filtered_history"]

    @staticmethod
    def _is_one_step_roll(prev_history, history):
        return (
            prev_history.shape == history.shape
            and len(history) > 1
            and np.allclose(prev_history[1:], history[:-1])
        )

    def filter_history(self, key, history):
        history = np.asarray(history, dtype=np.float32)
        if not self.enabled or history.shape[0] == 0:
            return history

        state = self.states.get(key)
        if state is None:
            return self._filter_sequence_from_initial(key, history)

        prev_history = state["raw_history"]
        if prev_history.shape == history.shape and np.allclose(prev_history, history):
            return state["filtered_history"]

        if self._is_one_step_roll(prev_history, history):
            filtered_new, zi = sosfilt(self.sos, history[-1:], axis=0, zi=state["zi"])
            filtered_history = np.concatenate(
                [state["filtered_history"][1:], filtered_new.astype(np.float32)],
                axis=0,
            )
            self.states[key] = {
                "raw_history": history.copy(),
                "filtered_history": filtered_history,
                "zi": zi,
            }
            return filtered_history

        return self._filter_sequence_from_initial(key, history)


class QueSTPolicy:
    def __init__(self, checkpoint_path, device):
        self.device = device if torch.cuda.is_available() or "cuda" not in device else "cpu"
        self.model, self.cfg, self.checkpoint_path = self._load_policy(checkpoint_path)

        shape_meta = self.cfg.task.shape_meta
        self.image_keys = list(shape_meta.observation.rgb.keys())
        self.lowdim_keys = list(shape_meta.observation.lowdim.keys())
        self.task_type = shape_meta.task.type
        self.task_embedding_format = self.cfg.task.get("task_embedding_format", "clip")
        self._task_emb_cache = {}
        ft_config = OmegaConf.select(self.cfg, "algo.dataset.ft_config")
        if ft_config is not None:
            ft_config = OmegaConf.to_container(ft_config, resolve=True)
        self.ft_filter = OnlineForceHistoryFilter(ft_config)
        self.lowdim_stats = self._load_lowdim_stats()

    def _load_policy(self, checkpoint_path):
        checkpoint_path = utils.get_latest_checkpoint(checkpoint_path)
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if "config" not in state_dict:
            raise KeyError("Checkpoint does not contain saved config. Use a checkpoint saved by train.py.")

        saved_cfg = copy.deepcopy(state_dict["config"])
        saved_cfg["device"] = self.device
        saved_cfg["algo"]["policy"]["device"] = self.device
        if "policy_prior" in saved_cfg["algo"]["policy"]:
            saved_cfg["algo"]["policy"]["policy_prior"]["device"] = self.device

        saved_cfg = OmegaConf.create(saved_cfg)
        model = instantiate(saved_cfg.algo.policy, shape_meta=saved_cfg.task.shape_meta)
        model.to(self.device)
        if hasattr(model, "device"):
            model.device = self.device
        if hasattr(model, "policy_prior") and hasattr(model.policy_prior, "device"):
            model.policy_prior.device = self.device
        model.eval()
        model.load_state_dict(state_dict["model"])
        model.reset()
        return model, saved_cfg, checkpoint_path

    def _resolve_stats_path(self, path):
        if path is None:
            data_prefix = OmegaConf.select(self.cfg, "data_prefix")
            if data_prefix is None:
                return None
            candidate = os.path.join(os.path.expanduser(str(data_prefix)), "lowdim_stats.json")
            return candidate if os.path.exists(candidate) else None

        path = os.path.expanduser(str(path))
        if os.path.isabs(path):
            return path

        data_prefix = OmegaConf.select(self.cfg, "data_prefix")
        if data_prefix is not None:
            candidate = os.path.join(os.path.expanduser(str(data_prefix)), path)
            if os.path.exists(candidate):
                return candidate

        return path

    def _load_lowdim_stats(self):
        stats_path = OmegaConf.select(self.cfg, "algo.dataset.lowdim_stats_path")
        stats_path = self._resolve_stats_path(stats_path)
        if stats_path is None:
            return None
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"lowdim_stats_path does not exist: {stats_path}")
        return load_lowdim_stats(stats_path)

    def metadata(self):
        return {
            "policy": "quest",
            "checkpoint_path": self.checkpoint_path,
            "image_keys": self.image_keys,
            "lowdim_keys": self.lowdim_keys,
            "device": self.device,
        }

    def _get_task_emb(self, prompt):
        if self.task_type == "onehot":
            return None
        if prompt not in self._task_emb_cache:
            self._task_emb_cache[prompt] = get_task_embs(
                self.task_embedding_format, [prompt]
            ).to(self.device)
        return self._task_emb_cache[prompt]

    def infer(self, request):
        if bool(request.pop("_reset", False)):
            self.model.reset()
            self.ft_filter.reset()

        prompt = request.pop("prompt", "")
        task_id = int(np.asarray(request.pop("task_id", 0)).item())
        task_emb = self._get_task_emb(prompt)

        obs = {}
        for key in self.image_keys + self.lowdim_keys:
            if key not in request:
                raise KeyError(f"Request missing obs key '{key}'. Available keys: {list(request.keys())}")
            value = request[key]
            if key in self.lowdim_keys:
                value = self._preprocess_lowdim(key, value)
            obs[key] = value

        action = self.model.get_action(obs, task_id, task_emb=task_emb)
        return {"actions": np.asarray(action, dtype=np.float32)}

    def _preprocess_lowdim(self, key, value):
        value = np.asarray(value, dtype=np.float32)
        original_shape = value.shape

        if key in FORCE_HISTORY_KEYS:
            history = value.squeeze().reshape(-1, value.shape[-1])
            history = self.ft_filter.filter_history(key, history)
            history = normalize_lowdim_value(history, key, self.lowdim_stats)
            return history.reshape(original_shape).astype(np.float32)

        return normalize_lowdim_value(value, key, self.lowdim_stats)


class PickleWebsocketPolicyServer:
    def __init__(self, policy, host, port):
        self.policy = policy
        self.host = host
        self.port = port
        self.logger = logging.getLogger("quest_policy_server")

    async def serve_forever(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
        ):
            self.logger.info("QueST policy server listening on %s:%s", self.host, self.port)
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
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log_level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()), force=True)
    policy = QueSTPolicy(args.checkpoint_path, args.device)
    server = PickleWebsocketPolicyServer(policy, args.host, args.port)
    asyncio.run(server.serve_forever())


if __name__ == "__main__":
    main()
