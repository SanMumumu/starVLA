#!/usr/bin/env python3
import argparse
import json
import logging
import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Shell 中 CUDA_VISIBLE_DEVICES=i 后，当前进程内部只有一张逻辑 GPU，编号为 0。
# 新 mujoco 必须用逻辑 EGL 0 初始化。
# 旧 robosuite 却要求 MUJOCO_EGL_DEVICE_ID 与 CUDA_VISIBLE_DEVICES 的物理编号一致。
# 因此先用 0 初始化 mujoco，再恢复物理编号供 robosuite 的旧检查使用。
_physical_gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
import mujoco as _mujoco  # noqa: F401

os.environ["MUJOCO_EGL_DEVICE_ID"] = _physical_gpu_id

import pathlib
import re
import sys
import time
from typing import Any

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"

from examples.LIBERO.eval_files.model2libero_interface import ModelClient

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--shard-id", type=int, required=True)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--expected-total", type=int, default=10030)
    args = p.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"invalid shard id {args.shard_id}/{args.num_shards}")
    return args


def find_classification_file() -> pathlib.Path:
    candidates = []
    libero_home = os.environ.get("LIBERO_HOME")
    if libero_home:
        candidates.append(pathlib.Path(libero_home) / "libero/libero/benchmark/task_classification.json")
    candidates.append(pathlib.Path(get_libero_path("benchmark_root")) / "benchmark/task_classification.json")
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("task_classification.json not found:\n" + "\n".join(map(str, candidates)))


def load_task_meta() -> dict[str, dict[int, dict[str, Any]]]:
    path = find_classification_file()
    logging.info("classification file: %s", path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        suite: {int(item["id"]): item for item in raw[suite]}
        for suite in SUITES
    }


def load_completed(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("skip corrupted JSONL line %d in %s", line_no, path)
                continue
            completed[record["key"]] = record
    return completed


def append_record(path: pathlib.Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:160]


def binarize_gripper(open_gripper: np.ndarray) -> np.ndarray:
    value = float(np.asarray(open_gripper, dtype=np.float32).reshape(-1)[0])
    return np.asarray([1.0 - 2.0 * (value > 0.5)], dtype=np.float32)


def make_env(task: Any, seed: int) -> OffScreenRenderEnv:
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def save_video(path: pathlib.Path, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, [np.asarray(x) for x in frames], fps=25)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    np.random.seed(args.seed)

    out = pathlib.Path(args.output_dir)
    shard_file = out / "shards" / f"shard_{args.shard_id:02d}.jsonl"
    completed = load_completed(shard_file)
    logging.info("resume: %d completed records", len(completed))

    task_meta = load_task_meta()
    bench = benchmark.get_benchmark_dict()
    suite_objs = {suite: bench[suite]() for suite in SUITES}

    all_jobs: list[tuple[str, int]] = []
    for suite in SUITES:
        n = suite_objs[suite].n_tasks
        logging.info("%s: %d tasks", suite, n)
        all_jobs.extend((suite, task_id) for task_id in range(n))

    if len(all_jobs) != args.expected_total:
        raise RuntimeError(f"LIBERO-Plus task count mismatch: got {len(all_jobs)}, expected {args.expected_total}")

    assigned = all_jobs[args.shard_id::args.num_shards]
    logging.info(
        "shard %d/%d: %d tasks (global=%d)",
        args.shard_id,
        args.num_shards,
        len(assigned),
        len(all_jobs),
    )

    client = ModelClient(
        host=args.host,
        port=args.port,
        image_size=[224, 224],
    )

    for local_idx, (suite, task_id) in enumerate(assigned, 1):
        key = f"{suite}:{task_id}:0"
        if key in completed:
            continue

        task_suite = suite_objs[suite]
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if len(initial_states) < 1:
            raise RuntimeError(f"no initial state: {suite} task {task_id}")

        meta = task_meta[suite].get(task_id + 1, {})
        category = str(meta.get("category", "unknown"))
        task_name = str(meta.get("name", task.language))
        task_description = str(task.language)

        logging.info(
            "[%d/%d] %s task=%d category=%s name=%s",
            local_idx,
            len(assigned),
            suite,
            task_id,
            category,
            task_name,
        )

        env = make_env(task, args.seed)
        frames: list[np.ndarray] = []
        success = False
        policy_step = 0
        start_time = time.time()

        try:
            client.reset(task_description=task_description)
            env.reset()
            obs = env.set_init_state(initial_states[0])

            for t in range(MAX_STEPS[suite] + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                    continue

                image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                if args.save_video:
                    frames.append(image)

                response = client.step(
                    example={"image": [image, wrist], "lang": task_description},
                    step=policy_step,
                )
                raw_action = response["raw_action"]
                world = np.asarray(raw_action["world_vector"], dtype=np.float32).reshape(-1)
                rotation = np.asarray(raw_action["rotation_delta"], dtype=np.float32).reshape(-1)
                grip_raw = np.asarray(raw_action["open_gripper"], dtype=np.float32).reshape(-1)
                if world.size != 3 or rotation.size != 3 or grip_raw.size != 1:
                    raise ValueError(
                        f"bad action shape world={world.shape}, rotation={rotation.shape}, gripper={grip_raw.shape}"
                    )
                action = np.concatenate([world, rotation, binarize_gripper(grip_raw)], axis=0)
                obs, _, done, _ = env.step(action.tolist())
                policy_step += 1
                if done:
                    success = True
                    break
        finally:
            env.close()

        elapsed = time.time() - start_time
        record = {
            "key": key,
            "shard_id": args.shard_id,
            "suite": suite,
            "task_id": task_id,
            "episode_id": 0,
            "category": category,
            "task_name": task_name,
            "task_description": task_description,
            "success": bool(success),
            "elapsed_seconds": elapsed,
        }
        append_record(shard_file, record)
        completed[key] = record
        logging.info("result: success=%s elapsed=%.1fs", success, elapsed)

        if args.save_video:
            status = "success" if success else "failure"
            video_path = out / "videos" / suite / f"shard_{args.shard_id:02d}" / (
                f"task_{task_id:05d}_{safe_name(task_name)}_{status}.mp4"
            )
            save_video(video_path, frames)

    successes = sum(int(x["success"]) for x in completed.values())
    logging.info("shard done: %d/%d successes", successes, len(completed))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("fatal error")
        sys.exit(1)
