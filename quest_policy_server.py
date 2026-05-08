import argparse
import asyncio
import copy
import logging
import pickle
import traceback

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames
from hydra.utils import instantiate
from omegaconf import OmegaConf

import quest.utils.utils as utils
from quest.utils.real_robot_utils import get_task_embs


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

        prompt = request.pop("prompt", "")
        task_id = int(np.asarray(request.pop("task_id", 0)).item())
        task_emb = self._get_task_emb(prompt)

        obs = {}
        for key in self.image_keys + self.lowdim_keys:
            if key not in request:
                raise KeyError(f"Request missing obs key '{key}'. Available keys: {list(request.keys())}")
            obs[key] = request[key]

        action = self.model.get_action(obs, task_id, task_emb=task_emb)
        return {"actions": np.asarray(action, dtype=np.float32)}


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
