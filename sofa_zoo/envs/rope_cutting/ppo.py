import numpy as np
from stable_baselines3 import PPO

from sofa_env.scenes.rope_cutting.rope_cutting_env import RenderMode, ObservationType, RopeCuttingEnv, ActionType

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # obs, ropes, to_cut, depth
    parameters = ["STATE", 10, 3, "flat"]

    # parameters[3]
    board_depth = {
        "flat": 20,
        "deep": 50,
    }


    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]
    render_mode = RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.1,
        "frame_skip": 1,
        "settle_steps": 10,
        "settle_step_dt": 0.01,
        "render_mode": render_mode,
        "reward_amount_dict": {
            "distance_cauter_active_rope": -0.0,
            "delta_distance_cauter_active_rope": -5.0,
            "cut_active_rope": 5.0,
            "cut_inactive_rope": -5.0,
            "worspace_violation": -0.0,
            "state_limits_violation": -0.0,
            "successful_task": 10.0,
            "failed_task": -20.0,
        },
        "on_reset_callbacks": None,
        "num_ropes": int(parameters[1]),
        "num_ropes_to_cut": int(parameters[2]),
        "create_scene_kwargs": {
            "depth": board_depth[parameters[3]],
        },
    }

    config = {"max_episode_steps": max(400, env_kwargs["num_ropes_to_cut"] * 200), **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    info_keywords = [
        "ret_dis_cau_act_rop",
        "ret_del_dis_cau_act_rop",
        "ret_cut_act_rop",
        "ret_cut_ina_rop",
        "ret_wor_vio",
        "ret_sta_lim_vio",
        "ret_suc_tas",
        "ret_fai_tas",
        "ret",
        "num_cut_ropes",
        "num_cut_inactive_ropes",
        "successful_task",
        "failed_task",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=RopeCuttingEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False if image_based else True,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        reward_clip=reward_clip,
        normalize_reward=normalize_reward,
        use_watchdog_vec_env=True,
        watchdog_vec_env_timeout=30.0,
        reset_process_on_env_reset=True,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name= f"PPO_{observation_type.name}_{parameters[1]}ropes_{parameters[2]}cut_{parameters[3]}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
