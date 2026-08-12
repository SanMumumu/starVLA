"""Generic RoboDojo rollout loop for the repository-owned StarVLA adapter.

The policy server deliberately returns one cached chunk element per request.
This keeps the simulator observation current at every 25 Hz control step while
the repository adapter replans its checkpoint-defined StarVLA chunk at the
configured execution interval.
"""


def eval_one_episode(TASK_ENV, model_client):
    model_client.call(func_name="reset")
    while not TASK_ENV.is_episode_end():
        model_client.call(func_name="update_obs", obs=TASK_ENV.get_obs())
        actions = model_client.call(func_name="get_action")
        if len(actions) != 1:
            raise RuntimeError(f"StarVLA server must return one cached action per control step, got {len(actions)}.")
        TASK_ENV.take_action(actions[0])


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


__all__ = ["eval_one_episode", "eval_one_episode_batch"]
