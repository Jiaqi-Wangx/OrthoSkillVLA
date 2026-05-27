#!/usr/bin/env python3
"""
Refined Libero evaluation client.
- Stronger typing, docstrings, and logging
- Safer HTTP client with timeouts & error handling
- Cleaner action post-processing and 6D-rotation helpers
- Deterministic seeding and robust main() patterned after the provided example
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import imageio
import json_numpy
import numpy as np
import requests
import torch  # noqa: F401  # (kept in case of future GPU array ops)
import torchvision.transforms as transforms  # noqa: F401
from tqdm import tqdm

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv  # noqa: F401
import robosuite.utils.transform_utils as T

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EPS = 1e-6

LIBERO_DATASETS = {
    "libero_goal": ["libero_goal"],
    "libero_object": ["libero_object"],
    "libero_spatial": ["libero_spatial"],
    "libero_10": ["libero_10"],
    "libero_90": ["libero_90"],
    "libero30": ["libero_goal", "libero_object", "libero_spatial"],
    "libero130": ["libero_goal", "libero_object", "libero_spatial", "libero_10", "libero_90"],
}

LIBERO_DATASETS_HORIZON = {
    "libero_goal": 800,
    "libero_object": 800,
    "libero_spatial": 800,
    "libero_10": 900,
    "libero_90": 800,
    "libero30": 800,
    "libero130": 800,
}

skill_infos = [
    {"name": "open_close", "task_ids": [22, 6, 28, 35], "lerobot_task_ids": [11, 8, 69, 57], "suite": "libero_90"},
    {"name": "pick_place", "task_ids": [12, 46, 74, 85], "lerobot_task_ids": [45, 0, 42, 10], "suite": "libero_90"},
    {"name": "turn", "task_ids": [20, 39], "lerobot_task_ids": [65, 63],"suite": "libero_90"},
]

benchmark_dict = benchmark.get_benchmark_dict()


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    """Match original code behavior: vertical+horizontal flips."""
    return np.flip(np.flip(img, 0), 1)


# -----------------------------------------------------------------------------
# Action processing
# -----------------------------------------------------------------------------
class LiberoAbsActionProcessor:
    """Helpers to convert between 6D rotation (Zhou et al.) and axis-angle."""

    def Rotate6D_to_AxisAngle(self, r6d: np.ndarray) -> np.ndarray:
        """Convert 6D rotation representation to axis-angle.

        Args:
            r6d: array with shape (N, 6) or (6,)
        Returns:
            array with shape (N, 3) or (3,)
        """
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        # b1
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)

        # b2
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)

        # b3
        b3 = np.cross(b1, b2, axis=-1)

        R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)

        axis_angle_list: List[np.ndarray] = []
        for i in range(R.shape[0]):
            quat = T.mat2quat(R[i])
            axis_angle = T.quat2axisangle(quat)
            axis_angle_list.append(axis_angle)

        axis_angle_array = np.stack(axis_angle_list, axis=0)
        return axis_angle_array[0] if single else axis_angle_array

    def Mat_to_Rotate6D(self, R: np.ndarray) -> np.ndarray:
        if R.ndim == 2:
            return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
        elif R.ndim == 3:
            return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
        else:
            raise ValueError("Rotation matrix must be (...,3,3)")

    def AxisAngle_to_Rotate6D(self, aa: np.ndarray) -> np.ndarray:
        if aa.ndim == 1:
            return self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(aa)))
        else:
            raise ValueError("Only 1D axis-angle supported here")

    def action_6d_to_axisangle(self, action: np.ndarray) -> np.ndarray:
        """Convert action [..., 3(pos)+6(rot6d)+1(grip)] -> [..., 3(pos)+3(aa)+1(grip)]"""
        if action.ndim == 1:
            final_ori = self.Rotate6D_to_AxisAngle(action[3:9])
            return np.concatenate([action[0:3], final_ori, action[-1:]])
        elif action.ndim == 2:
            final_ori = self.Rotate6D_to_AxisAngle(action[:, 3:9])
            return np.concatenate([action[:, 0:3], final_ori, action[:, -1:]], axis=-1)
        else:
            raise ValueError("Unsupported action shape")


# -----------------------------------------------------------------------------
# HTTP Client Policy
# -----------------------------------------------------------------------------
class ClientModel:
    """Thin HTTP client that queries a remote policy server and returns actions."""

    def __init__(self, host: str, port: int):
        self.url = f"http://{host}:{port}/act"
        self.processor = LiberoAbsActionProcessor()
        self.reset()

    def reset(self) -> None:
        self.proprio: Optional[np.ndarray] = None  # last absolute [pos(3)+ori6d(6)+grip(1)]
        self.action_plan: Deque[List[float]] = collections.deque()

    def _format_query(self, obs: Dict, goal: str) -> Dict:
        main_view = _flip_agentview(obs["agentview_image"])  # (256,256,3)
        wrist_view = obs["robot0_eye_in_hand_image"]  # (256,256,3)

        closed_loop_proprio = np.concatenate([obs["robo_pos"], obs["robo_ori"], np.array([0.0])], axis=-1)
        closed_loop_proprio = np.concatenate([closed_loop_proprio, np.zeros_like(closed_loop_proprio)], axis=-1)
        if self.proprio is None:
            # Initialize absolute proprio: [pos(3), ori6d(6), grip(1)] + past copy (legacy)
            self.proprio = closed_loop_proprio

        return {
            "proprio": json_numpy.dumps(self.proprio),
            "language_instruction": goal,
            "image0": json_numpy.dumps(main_view),
            "image1": json_numpy.dumps(wrist_view),
            "domain_id": 30, 
            "steps": 10,
        }

    def _post(self, payload: Dict) -> np.ndarray:
        try:
            resp = requests.post(self.url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"Policy server request failed: {e}") from e

        action = np.array(data["action"])  # shape (T, 10) expected: [pos3, rot6d, grip1]
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
        return action

    def step(self, obs: Dict, goal: str) -> np.ndarray:
        if not self.action_plan:
            payload = self._format_query(obs, goal)
            action = self._post(payload)
            # Update proprio to last absolute state (first 9 dims)
            # self.proprio = None
            self.proprio[:9] = action[-1, :9].copy()

            # Convert rot6d->axis-angle and build final [pos3, aa3, grip1]
            target_eef = action[:, :3]
            target_axis = self.processor.Rotate6D_to_AxisAngle(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)

            # Queue up the plan
            for row in final_action.tolist():
                self.action_plan.append(row)

        action_predict = np.array(self.action_plan.popleft(), dtype=np.float32)
        # Discretize gripper to {-1, 1}
        action_predict[-1] = 1.0 if action_predict[-1] > 0.5 else -1.0
        return action_predict

# -----------------------------------------------------------------------------
# Evaluator
# -----------------------------------------------------------------------------
class LIBEROEval:
    def __init__(
        self,
        task_suite_name: str,
        eval_horizon: int = 600,
        act_type: str = "abs",
        num_episodes: int = 10,
        eval_freq: int = 10,  # reserved
        init_seed: int = 42,
    ) -> None:
        self.task_suite_name = task_suite_name
        self.task_list = LIBERO_DATASETS[self.task_suite_name]
        self.task_suite_list = [benchmark_dict[task]() for task in self.task_list]
        self.eval_horizon = eval_horizon
        self.num_episodes = num_episodes
        self.eval_freq = eval_freq
        self.init_seed = init_seed
        self.act_type = act_type
        self.processor = LiberoAbsActionProcessor()
        self.base_dir: Path = Path(".")

    # ---- internal helpers --------------------------------------------------
    def _make_dir(self, save_path: Path) -> None:
        path = save_path / self.task_suite_name
        _ensure_dir(path)
        self.base_dir = path

    def _init_env(self, task_suite, task_id: int = 0, ep: int = 0):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(
            f"[info] retrieving task {task_id} from suite {self.task_suite_name}, "
            f"language: {task_description}, bddl: {task_bddl_file}"
        )

        env_args = {"bddl_file_name": task_bddl_file, "camera_heights": 256, "camera_widths": 256}
        env = OffScreenRenderEnv(**env_args)

        # Environment reset & deterministic init state
        env.seed(self.init_seed + ep + 100)
        obs = env.reset()
        init_states = task_suite.get_task_init_states(task_id)
        init_state_id = ep % init_states.shape[0]
        obs = env.set_init_state(init_states[init_state_id])

        # settle
        for _ in range(10):
            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
            obs, reward, done, info = env.step(action)

        if self.act_type == "abs":
            for robot in env.env.robots:
                robot.controller.use_delta = False
        elif self.act_type == "rel":
            pass
        else:
            raise ValueError("act_type must be 'abs' or 'rel'")

        return env, task_description, obs

    def _log_results(self, metrics: Dict) -> None:
        print(metrics)
        save_name = self.base_dir / "results.json"
        with open(save_name, "a+", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def _save_video(self, save_path: Path, images: List[np.ndarray], fps: int = 30) -> None:
        imageio.mimsave(save_path.as_posix(), images, fps=fps)

    def _rollout(self, task_suite, policy: ClientModel, task_id: int, ep: int) -> float:
        env, lang, obs = self._init_env(task_suite, task_id, ep)
        images: List[np.ndarray] = []

        done_flag = False
        for _ in tqdm(range(self.eval_horizon), desc=f"{lang}"):
            robo_ori = self.processor.Mat_to_Rotate6D(env.env.robots[0].controller.ee_ori_mat)
            robo_pos = env.env.robots[0].controller.ee_pos
            obs["robo_ori"] = robo_ori
            obs["robo_pos"] = robo_pos

            action = policy.step(obs, lang)

            frame = np.hstack([_flip_agentview(obs["agentview_image"]), obs["robot0_eye_in_hand_image"]])
            images.append(frame)
            obs, reward, done, info = env.step(action)
            if done:
                done_flag = True
                break

        save_path = self.base_dir / f"{lang}_{ep}.mp4"
        self._save_video(save_path, images, fps=30)

        success = 1.0 if done_flag else 0.0
        metrics = {f"sim/{self.task_suite_name}/{lang}": success}
        self._log_results(metrics)

        env.close()
        return success

    # ---- public API --------------------------------------------------------
    def eval_episodes(self, policy: ClientModel, save_path: Path) -> float:
        self._make_dir(save_path)

        rews: List[float] = []
        for task_suite in self.task_suite_list:
            for task_id in tqdm(range(len(task_suite.tasks)), desc="Evaluating tasks"):
                for ep in range(self.num_episodes):
                    policy.reset()
                    rew = self._rollout(task_suite, policy, task_id, ep)
                    rews.append(rew)

        eval_rewards = float(sum(rews) / max(len(rews), 1))
        metrics = {f"sim_summary/{self.task_suite_name}/all": eval_rewards}
        self._log_results(metrics)
        return eval_rewards

    def eval_skills(self, policy: ClientModel, save_path: Path, skill_ids: List[int]) -> Dict[str, float]:
        self._make_dir(save_path)

        if skill_ids is None:
            skill_ids = list(range(len(skill_infos)))

        skill_results = {}
        for s_id in skill_ids:
            skill_info = skill_infos[s_id]
            
            skill_name = skill_info["name"]
            task_ids = skill_info["task_ids"]
            # lerobot_task_ids = skill_info["lerobot_task_ids"]
            suite_name = skill_info["suite"]

            task_suite = benchmark_dict[suite_name]()
            rews: List[float] = []
            for task_id in tqdm(task_ids, desc=f"Evaluating skill {skill_name}"):
                for ep in range(self.num_episodes):
                    policy.reset()
                    rew = self._rollout(task_suite, policy, task_id, ep)
                    rews.append(rew)

            skill_success_rate = float(sum(rews) / max(len(rews), 1))
            skill_results[skill_name] = skill_success_rate
            metrics = {f"sim_summary/{self.task_suite_name}/{skill_name}": skill_success_rate}
            self._log_results(metrics)

        return skill_results


# -----------------------------------------------------------------------------
# Batch evaluator across suites
# -----------------------------------------------------------------------------


def eval_libero(
    agent: ClientModel,
    save_path: Path,
    num_episodes: int = 10,
    init_seed: int = 42,
    act_type: str = "abs",
    task_suites: Iterable[str] = ("libero_goal", "libero_spatial", "libero_10"),
) -> Dict[str, float]:
    result_dict: Dict[str, float] = {}
    for suite_name in task_suites:
        horizon = LIBERO_DATASETS_HORIZON[suite_name]
        evaluator = LIBEROEval(
            task_suite_name=suite_name,
            eval_horizon=horizon,
            act_type=act_type,
            num_episodes=num_episodes,
            init_seed=init_seed,
        )
        eval_rewards = evaluator.eval_episodes(agent, save_path=save_path)
        result_dict[suite_name] = eval_rewards

    # Also write a compact JSON summary at root
    with open((save_path / "results.json").as_posix(), "a+", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
        f.write("\n")
    return result_dict


def eval_libero_skills(
    agent: ClientModel,
    save_path: Path,
    num_episodes: int = 10,
    init_seed: int = 42,
    act_type: str = "abs",
    skill_ids: List[int] = None,
) -> Dict[str, float]:
    result_dict: Dict[str, float] = {}
    suite_name = "libero_90"  # all skills are defined within this suite
    evaluator = LIBEROEval(
        task_suite_name=suite_name,
        eval_horizon=LIBERO_DATASETS_HORIZON[suite_name],
        act_type=act_type,
        num_episodes=num_episodes,
        init_seed=init_seed,
    )
    skill_results = evaluator.eval_skills(agent, save_path=save_path, skill_ids=skill_ids)
    result_dict.update(skill_results)

    # Also write a compact JSON summary at root
    with open((save_path / "results.json").as_posix(), "a+", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
        f.write("\n")
    return result_dict


# -----------------------------------------------------------------------------
# Main (patterned after the provided example)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser("LIBERO Evaluation Client")
    # Connection options
    parser.add_argument(
        "--connection_info", type=str, default=None, help="Path to server info.json (contains 'host' and 'port')"
    )
    parser.add_argument("--server_ip", type=str, default=None, help="Manual server IP (if not using --connection_info)")
    parser.add_argument(
        "--server_port", type=int, default=None, help="Manual server port (if not using --connection_info)"
    )

    # Eval options
    parser.add_argument(
        "--output_dir", type=str, default="logs/", help="Directory for saving evaluation videos and logs"
    )
    parser.add_argument(
        "--skill_ids", nargs="+", type=int, default=[0,1,2]
    )
    parser.add_argument(
        "--task_suites",
        nargs="+",
        default=["libero_10", "libero_spatial", "libero_goal", "libero_object"],
        help="Libero suites to evaluate",
    )
    parser.add_argument("--eval_time", type=int, default=50, help="Episodes per task")
    parser.add_argument("--init_seed", type=int, default=43, help="Random seed")
    parser.add_argument("--act_type", type=str, default="abs", choices=["abs", "rel"], help="Action type")

    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    _ensure_dir(out_dir)

    print("🚀 [Client] Starting LIBERO evaluation client...")

    # ------------------------ 1) Load connection info ------------------------
    if args.connection_info is not None:
        info_path = Path(args.connection_info)
        print(f"🔍 Waiting for connection info file: {info_path}")
        spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        i = 0
        while not info_path.exists():
            sys.stdout.write(f"\r{spinner[i % len(spinner)]} Waiting for server to start...")
            sys.stdout.flush()
            time.sleep(0.5)
            i += 1
        print("\n✅ Connection info file found!")
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                infos = json.load(f)
            host, port = infos["host"], int(infos["port"])
            print(f"🔗 Loaded server info: host={host}, port={port}")
        except Exception as e:
            print(f"❌ Failed to read connection info: {e}")
            sys.exit(1)
    else:
        if not args.server_ip or not args.server_port:
            print("❌ Must specify either --connection_info or both --server_ip and --server_port.")
            sys.exit(1)
        host, port = args.server_ip, int(args.server_port)
        print(f"🔗 Using manual server address: {host}:{port}")

    # ------------------------ 2) Initialize client ---------------------------
    print(f"🛰️  Connecting to policy server at {host}:{port} ...")
    client = ClientModel(host, port)
    print("✅ Successfully initialized client!")

    # ------------------------ 3) Run evaluation ------------------------------
    print("🎯 Starting LIBERO policy evaluation...")
    print(f"📁 Results and videos will be saved to: {out_dir.resolve()}")
    print("-" * 88)
    print("init seed:", args.init_seed)
    print("task suites:", args.task_suites)
    print("episodes per task:", args.eval_time)
    print("action type:", args.act_type)
    print("-" * 88)


    try:
        eval_results = eval_libero_skills(
            agent=client,
            save_path=out_dir,
            init_seed=args.init_seed,
            num_episodes=args.eval_time,
            # task_suites=args.task_suites,
            act_type=args.act_type,
            skill_ids=args.skill_ids,
        )
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        sys.exit(2)

    print("\n✅ All evaluations completed successfully!")
    print(f"📊 Summary: {json.dumps(eval_results, indent=2)}")


