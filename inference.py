import argparse
import copy
import datetime
import os
import pickle
import pickle as pkl
import sys
import time
from pathlib import Path

import numpy as np
import websockets.sync.client


POLICY_DIM = 7
ENV_DIM = 14
FORCE_HISTORY_KEYS = ("left_force_history", "right_force_history")
STATE_HISTORY_KEYS = ("left_state_history", "right_state_history")
RIGHT_STATE_KEYS = ("state",)
RIGHT_STATE_START_IDX = 19
RIGHT_STATE_FORCE_TORQUE_IDXS = (1, 2, 3, 10, 11, 12)


class PickleWebsocketClientPolicy:
    def __init__(self, host, port):
        self.uri = f"ws://{host}:{port}"
        self.ws = self._wait_for_server()
        self.metadata = pickle.loads(self.ws.recv())

    def _wait_for_server(self):
        print(f"Waiting for QueST server at {self.uri}...")
        while True:
            try:
                return websockets.sync.client.connect(self.uri, compression=None, max_size=None)
            except ConnectionRefusedError:
                print("Still waiting for QueST server...")
                time.sleep(5)

    def get_server_metadata(self):
        return self.metadata

    def infer(self, obs):
        self.ws.send(pickle.dumps(obs))
        response = pickle.loads(self.ws.recv())
        if "error" in response:
            raise RuntimeError(f"Error in QueST policy server:\n{response['error']}")
        return response

    def close(self):
        self.ws.close()


def add_python_root(path):
    if path is None:
        return
    path = str(Path(path).expanduser().resolve())
    if path not in sys.path:
        sys.path.append(path)


def pad_action_to_env(action):
    action = np.asarray(action, dtype=np.float32)
    action = np.squeeze(action)

    if action.ndim == 2 and action.shape[-1] != POLICY_DIM and action.shape[0] == POLICY_DIM:
        action = action.T

    if action.shape[-1] == ENV_DIM:
        return action
    if action.shape[-1] != POLICY_DIM:
        raise ValueError(
            f"Unexpected action shape {action.shape}; expected last dim {POLICY_DIM} or {ENV_DIM}"
        )

    pad = np.zeros((*action.shape[:-1], ENV_DIM - POLICY_DIM), dtype=np.float32)
    return np.concatenate([pad, action], axis=-1)


def ensure_bthwc(image):
    image = np.asarray(image)
    if image.ndim == 5:
        out = image
    elif image.ndim == 4:
        out = image[:, None] if image.shape[0] == 1 else image[None]
    elif image.ndim == 3:
        out = image[None, None]
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return out.astype(np.uint8, copy=False)


def extract_state(obs):
    for key in ("state", "robot_state", "proprio", "lowdim", "low_dim_state", "states"):
        if key in obs:
            state = np.asarray(obs[key], dtype=np.float32)
            if state.ndim == 2 and state.shape[0] == 1:
                state = state[0]
            return state.reshape(-1)
    raise KeyError(
        "Observation is missing state key. "
        f"Available keys: {list(obs.keys())}"
    )


def extract_right_state(obs, remove_force=True):
    if "right_state" in obs:
        right_state = np.asarray(obs["right_state"], dtype=np.float32).squeeze().reshape(-1)
    else:
        state = extract_state(obs)
        right_state = state[RIGHT_STATE_START_IDX:]

    if remove_force:
        right_state = np.delete(right_state, RIGHT_STATE_FORCE_TORQUE_IDXS, axis=0)
    return right_state.astype(np.float32)[None, None]


def extract_history(obs, key):
    if key in obs:
        history = np.asarray(obs[key]).squeeze().astype(np.float32)
        if key in STATE_HISTORY_KEYS:
            history = np.delete(history, RIGHT_STATE_FORCE_TORQUE_IDXS, axis=-1).astype(np.float32)
        return history[None, None]
    raise KeyError(
        f"Observation is missing history key '{key}'. "
        f"Available keys: {list(obs.keys())}"
    )


def format_obs_for_quest(obs, image_keys, lowdim_keys, prompt, task_id, reset=False):
    if not isinstance(obs, dict):
        raise TypeError("QueST inference expects dict observations from the robot environment.")

    formatted = {}
    for key in image_keys:
        if key not in obs:
            raise KeyError(f"Observation is missing image key '{key}'. Available keys: {list(obs.keys())}")
        formatted[key] = ensure_bthwc(obs[key])

    for key in lowdim_keys:
        if key in FORCE_HISTORY_KEYS or key in STATE_HISTORY_KEYS:
            formatted[key] = extract_history(obs, key)
        elif key in RIGHT_STATE_KEYS:
            formatted[key] = extract_right_state(obs, remove_force=True)
        else:
            raise KeyError(
                f"Unsupported lowdim key for real robot inference: {key}. "
                "Supported keys are state, left_force_history, left_state_history, "
                "right_force_history, and right_state_history."
            )

    formatted["prompt"] = prompt
    formatted["task_id"] = np.asarray(task_id, dtype=np.int64)
    formatted["_reset"] = bool(reset)
    return formatted


def save_traj(transitions, save_name, task_name):
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = os.path.join("./online_demos", save_name, task_name, date)
    os.makedirs(save_dir, exist_ok=True)
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = os.path.join(save_dir, f"{task_name}_quest_online_demos_{uuid}.pkl")
    with open(file_name, "wb") as f:
        pkl.dump(transitions, f)
    print(f"saved trajectory to {file_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_ip", type=str, required=True)
    parser.add_argument("--trainer_port", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=0, help="0=loop forever")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--env_path", type=str, required=True)
    parser.add_argument("--env_name", type=str, default="default")
    parser.add_argument("--real_robo_root", type=str, default=None)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="Insert USB connector into USB port")
    args = parser.parse_args()

    add_python_root(args.real_robo_root)
    if args.real_robo_root is None:
        add_python_root(Path(__file__).resolve().parents[1])

    from experiments.config import build_environment, load_config

    client = PickleWebsocketClientPolicy(host=args.trainer_ip, port=args.trainer_port)
    metadata = client.get_server_metadata()
    image_keys = metadata["image_keys"]
    lowdim_keys = metadata["lowdim_keys"]

    env_config_path = os.path.join(args.env_path, f"{args.env_name}.json")
    print(f"Loading environment config from: {env_config_path}")
    print(f"[Env Client] connected to QueST server {args.trainer_ip}:{args.trainer_port}")
    print(f"Server metadata: {metadata}")
    print(f"Using prompt: {args.prompt}")

    env = build_environment(load_config(env_config_path))

    ep = 0
    try:
        while True:
            raw_obs, info = env.reset()
            trajectory = []
            done = False
            step = 0
            reset_next_request = True

            while not done:
                request = format_obs_for_quest(
                    raw_obs,
                    image_keys=image_keys,
                    lowdim_keys=lowdim_keys,
                    prompt=args.prompt,
                    task_id=args.task_id,
                    reset=reset_next_request,
                )
                reset_next_request = False

                response = client.infer(request)
                action = np.squeeze(pad_action_to_env(response["actions"]))

                next_obs, reward, terminated, truncated, step_info = env.step(action)
                done = bool(terminated or truncated)

                if "action_intervene" in step_info:
                    action = np.squeeze(pad_action_to_env(step_info["action_intervene"]))
                    reset_next_request = True

                if args.save_name is not None:
                    trajectory.append(copy.deepcopy({
                        "observations": raw_obs,
                        "actions": action,
                        "next_observations": next_obs,
                        "rewards": reward,
                        "masks": 1.0 - float(done),
                        "dones": done,
                        "infos": step_info,
                    }))

                raw_obs = next_obs
                step += 1
                if args.max_steps is not None and step >= args.max_steps:
                    done = True

            print(f"[episode {ep}] finished after {step} steps")
            if args.save_name is not None and len(trajectory) > 0:
                save_traj(trajectory, args.save_name, args.env_name)

            ep += 1
            if args.episodes > 0 and ep >= args.episodes:
                break
    finally:
        env.close()
        client.close()


if __name__ == "__main__":
    main()
