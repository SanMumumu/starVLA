"""Generic RoboDojo rollout loop for the repository-owned StarVLA adapter.

The policy server deliberately returns one cached chunk element per request.
This keeps the simulator observation current at every 25 Hz control step while
the repository adapter replans its checkpoint-defined StarVLA chunk at the
configured execution interval.
"""

import os


def _eventmem_capture_enabled():
    return os.environ.get("ROBODOJO_EVENTMEM_CAPTURE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _finish_eventmem_rollout(TASK_ENV, model_client):
    if not _eventmem_capture_enabled():
        return
    model_client.call(
        func_name="trial_end",
        obs={
            "success_by_env": {
                str(env_idx): bool(success)
                for env_idx, success in enumerate(TASK_ENV.success)
            },
            "steps_by_env": {
                str(env_idx): int(steps)
                for env_idx, steps in enumerate(TASK_ENV.take_action_cnt)
            },
        },
    )


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        if len(actions) != 1:
            raise RuntimeError(f"StarVLA server must return one cached action per control step, got {len(actions)}.")
        TASK_ENV.take_action(actions[0])
    _finish_eventmem_rollout(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        env_idx_list = TASK_ENV.get_running_env_idx_list()
        if not env_idx_list:
            break
        model_client.call(
            func_name="update_obs_batch",
            obs=TASK_ENV.get_obs_batch(env_idx_list),
        )
        actions = model_client.call(func_name="get_action_batch", obs=env_idx_list)
        if any(len(env_actions) != 1 for env_actions in actions):
            sizes = [len(env_actions) for env_actions in actions]
            raise RuntimeError(f"StarVLA server must return one cached action per env, got chunk sizes {sizes}.")
        TASK_ENV.take_action_batch([env_actions[0] for env_actions in actions], env_idx_list)
    _finish_eventmem_rollout(TASK_ENV, model_client)


__all__ = ["eval_one_episode", "eval_one_episode_batch"]
