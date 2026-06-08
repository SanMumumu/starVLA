"""JointFlow LIBERO eval client (task-sharded, sends state + PIL).

所属：JointFlow LIBERO eval（client 端）。
复用（import，不改源）：
- deployment.model_server.tools.websocket_policy_client.WebsocketClientPolicy
- libero.libero.benchmark / get_libero_path（构建 task suite）

env 工具（_get_libero_env / _quat2axisangle / LIBERO_DUMMY_ACTION）原从
examples.LIBERO.eval_files.eval_libero import，但该文件用了 3.10 联合注解 + 硬 import
tyro/ModelClient，在本地 libero_env(py3.8) 会崩；故原样内联到本文件（语义一致：同样的
180° 旋转、等待步、动作拆分）。gripper 处理**不沿用** examples 的 _binarize_gripper_open
（它对本数据集 {0,1} 的 gripper 是错的，见 _run_task 内说明 + demo 回放验证）。

为何另写 client（不直接用 examples 的 eval_libero.py / ModelClient）：
1) **任务分片**：要把 (suite, task_id) 摊到 8/4 个 client 进程并行跑，原 client 只能
   串行整套 suite，无法分片。
2) **传 state**：JointFlow 用 state.inject_mode=token 训练，必须把 LIBERO proprio
   传给 server 归一化后注入；原 client 的 example 不含 state。
3) **传 PIL**：framework live DINO 的 prepare_dino_input 需要 PIL（.convert），原
   client 发的是 np.uint8，会在 server 端崩。这里统一转 PIL。
均通过新文件实现，不改 examples/ 源码。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy


######### // code // ##########
# 中文注释：以下 4 个 helper 原本从 examples/LIBERO/eval_files/eval_libero.py import。
# 但那个文件顶部硬 import tyro + model2libero_interface.ModelClient，并使用 3.10 才支持的
# `np.ndarray | float` 联合注解；本地 libero_env 是 Python 3.8 且没装 tyro，直接 import 会崩。
# client 实际只需要这 4 个小工具，故原样内联到 jointflow 本地（语义与 eval_libero.py 完全一致：
# 同样的 dummy action、同样的 180° 处理所依赖的 axisangle、同样的 gripper 二值化、同样的 env 构建），
# 既不改 examples 源，又解除对 tyro/ModelClient/3.10 语法的依赖。
# 本文件已 `from __future__ import annotations`，联合注解会被当作字符串，3.8 下安全。
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# 中文注释:原 examples 的 _binarize_gripper_open(把 gripper 映射成 {-1,+1})对本数据是错的——
# 本数据集 gripper 是 {0,1},env 只有 action≈0 才闭合,故已删除该 helper,夹爪在 _run_task 里
# 按 {0,1} 阈值直接发(见那里的说明 + demo 回放验证)。


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """Copied from robosuite transform_utils.quat2axisangle (xyzw quaternion)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
######### // code // ##########


logger = logging.getLogger(__name__)

LIBERO_ENV_RESOLUTION = 256  # 中文注释：与训练数据渲染分辨率一致；DINO 内部再 Resize 到 224。

# 中文注释：与 examples/LIBERO/eval_files/eval_libero.py 完全一致的每套件最大步数。
SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


######### // code // ##########
# 中文注释：与 server 的薄连接层。维护 action chunk 缓存：每 action_chunk_size 步
# 才向 server 请求一次新的动作块（server 返回的是已反归一化的 [B,T,D]）。
class PolicyConnection:
    def __init__(self, host: str, port: int, unnorm_key: Optional[str], use_ddim: bool, num_ddim_steps: int):
        self.client = WebsocketClientPolicy(host, port)
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.raw_actions: Optional[np.ndarray] = None
        logger.info("Connected %s:%d action_chunk_size=%d meta=%s", host, port, self.action_chunk_size, meta)

    def reset(self) -> None:
        self.raw_actions = None

    def get_action(self, example: dict, step: int) -> np.ndarray:
        if self.raw_actions is None or step % self.action_chunk_size == 0:
            query = {
                "examples": [example],
                "unnorm_key": self.unnorm_key,
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(query)
            if response.get("status") != "ok" or "data" not in response:
                raise RuntimeError(f"Server inference error: {response.get('error', response)}")
            actions_batch = np.asarray(response["data"]["actions"])  # (B,T,D), un-normalized
            self.raw_actions = actions_batch[0]  # (T,D)
        return self.raw_actions[step % self.action_chunk_size]
######### // code // ##########


######### // code // ##########
# 中文注释：构建 (suite, task_id) 的全局扁平列表，再按 idx % num_shards == shard_id 取本分片。
# 按 suite 分组返回，避免重复初始化 benchmark。返回 {suite: [task_id, ...]}。
def build_shard_assignment(task_suites: List[str], num_shards: int, shard_id: int) -> "dict[str, list[int]]":
    benchmark_dict = benchmark.get_benchmark_dict()
    flat = []
    for suite in task_suites:
        if suite not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO suite: {suite}. Known: {sorted(benchmark_dict.keys())}")
        n_tasks = benchmark_dict[suite]().n_tasks
        for task_id in range(n_tasks):
            flat.append((suite, task_id))

    assigned: "dict[str, list[int]]" = {}
    for idx, (suite, task_id) in enumerate(flat):
        if idx % num_shards == shard_id:
            assigned.setdefault(suite, []).append(task_id)
    return assigned
######### // code // ##########


######### // code // ##########
# 中文注释：把 env 观测转成发给 server 的 example：
# - image: [primary, wrist] 两路 np.uint8（180° 旋转，匹配训练预处理）。
#   注意：发送 np（不是 PIL）——websocket 用 msgpack_numpy 序列化，只认 np，不认 PIL；
#   np->PIL 的转换放在 server 端 wrapper 里做（live DINO 需要 PIL）。
# - lang: 任务描述。
# - state: 8 维 proprio（eef_pos3 + axisangle3 + gripper_qpos2），原始值，server 端归一化。
def _build_example(obs, task_description: str):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            _quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    example = {
        "image": [img, wrist_img],
        "lang": str(task_description),
        "state": state,
    }
    return example, img
######### // code // ##########


######### // code // ##########
# 中文注释：跑单个 (suite, task_id) 的全部 trial，返回 (successes, episodes)。
def _run_task(conn, task_suite, task_id, args, video_dir: Optional[Path]) -> "tuple[int, int]":
    import imageio

    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    max_steps = SUITE_MAX_STEPS[args.current_suite]

    successes, episodes = 0, 0
    for episode_idx in range(args.num_trials_per_task):
        conn.reset()
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        t, step, done = 0, 0, False
        replay_images = []
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            example, vis_img = _build_example(obs, task_description)
            if video_dir is not None:
                replay_images.append(vis_img)

            raw = np.asarray(conn.get_action(example, step), dtype=np.float32).reshape(-1)
            world_vector = raw[:3]
            rotation_delta = raw[3:6]
            # 中文注释：本数据集的 gripper 是 {0,1}(0=合,1=开)且训练不归一化,模型直接回归 {0,1}。
            # 旧的 _binarize_gripper_open 把它映射成 {-1,+1},但这个 LIBERO env 实测只有 action≈0
            # 才闭合,{-1,+1} 都不闭合 → 夹爪永不抓取、pick 任务必 0%。已用 demo 回放验证:
            # 直接按 {0,1} 阈值发(raw / thr01)夹爪能闭合抓取(qpos→0.001),binarize 不抓取(qpos≈0.05)。
            gripper = np.array([1.0 if float(raw[6]) > 0.5 else 0.0], dtype=np.float32)
            delta_action = np.concatenate([world_vector, rotation_delta, gripper], axis=0)

            obs, _, done, _ = env.step(delta_action.tolist())
            if done:
                successes += 1
                break
            t += 1
            step += 1

        episodes += 1
        if video_dir is not None and replay_images:
            suffix = "success" if done else "failure"
            seg = str(task_description).replace(" ", "_")[:60]
            video_dir.mkdir(parents=True, exist_ok=True)
            imageio.mimwrite(video_dir / f"{args.current_suite}_t{task_id}_ep{episode_idx}_{suffix}.mp4",
                             [np.asarray(x) for x in replay_images], fps=10)
        logger.info("[%s task=%d ep=%d] done=%s running SR=%.3f",
                    args.current_suite, task_id, episode_idx, done, successes / max(episodes, 1))
    return successes, episodes
######### // code // ##########


def eval_libero_shard(args) -> dict:
    np.random.seed(args.seed)
    conn = PolicyConnection(args.host, args.port, args.unnorm_key, args.use_ddim, args.num_ddim_steps)

    task_suites = [s.strip() for s in args.task_suites.split(",") if s.strip()]
    assigned = build_shard_assignment(task_suites, args.num_shards, args.shard_id)
    logger.info("Shard %d/%d assignment: %s", args.shard_id, args.num_shards,
                {k: len(v) for k, v in assigned.items()})

    benchmark_dict = benchmark.get_benchmark_dict()
    video_root = Path(args.video_out_path) / f"shard{args.shard_id}" if args.save_videos else None

    results = {"shard_id": args.shard_id, "num_shards": args.num_shards, "per_suite": {}}
    total_s, total_e = 0, 0
    for suite, task_ids in assigned.items():
        args.current_suite = suite
        task_suite = benchmark_dict[suite]()
        suite_s, suite_e, per_task = 0, 0, {}
        for task_id in task_ids:
            s, e = _run_task(conn, task_suite, task_id, args, video_root)
            per_task[str(task_id)] = {"successes": s, "episodes": e}
            suite_s += s
            suite_e += e
        results["per_suite"][suite] = {"successes": suite_s, "episodes": suite_e, "per_task": per_task}
        total_s += suite_s
        total_e += suite_e
        logger.info("[suite %s] SR=%.3f (%d/%d)", suite, suite_s / max(suite_e, 1), suite_s, suite_e)

    results["total"] = {"successes": total_s, "episodes": total_e,
                        "success_rate": total_s / max(total_e, 1)}

    if args.results_json:
        out = Path(args.results_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote shard results -> %s", out)
    logger.info("Shard %d total SR=%.3f (%d/%d)", args.shard_id, results["total"]["success_rate"], total_s, total_e)
    return results


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=6500)
    p.add_argument("--task_suites", type=str, default="libero_goal,libero_object,libero_spatial,libero_10")
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_trials_per_task", type=int, default=50)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--unnorm_key", type=str, default=None)
    p.add_argument("--use_ddim", action="store_true")
    p.add_argument("--num_ddim_steps", type=int, default=10)
    p.add_argument("--save_videos", action="store_true")
    p.add_argument("--video_out_path", type=str, default="experiments/jointflow_libero/videos")
    p.add_argument("--results_json", type=str, default="")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s", force=True)
    args = build_argparser().parse_args()
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError(f"shard_id={args.shard_id} out of range [0,{args.num_shards}).")
    eval_libero_shard(args)
