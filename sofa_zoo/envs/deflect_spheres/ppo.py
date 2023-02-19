import numpy as np
from stable_baselines3 import PPO

from sofa_env.scenes.deflect_spheres.deflect_spheres_env import Mode, RenderMode, ObservationType, DeflectSpheresEnv, ActionType

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # obs, multi_agent, num_deflect
    parameters = ["RGB", "False", 1]

    # parameters[1]
    multi_agent = {"True": True, "False": False}

    # parameters[2]
    deflections_to_win = {
        1: 1,
        2: 2,
        -1: 5,
    }

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.1,
        "frame_skip": 1,
        "settle_steps": 10,
        "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        "reward_amount_dict": {
            "action_violated_cartesian_workspace": -0.0,
            "action_violated_state_limits": -0.0,
            "tool_collision": -0.0,
            "distance_to_active_sphere": -0.0,
            "delta_distance_to_active_sphere": -5.0,
            "deflection_of_inactive_spheres": -0.005,
            "deflection_of_active_sphere": 0.0,
            "delta_deflection_of_active_sphere": 1.0,
            "done_with_active_sphere": 10.0,
            "successful_task": 100.0,
        },
        "single_agent": not multi_agent[parameters[1]],
        "individual_agents": False,
        "mode": Mode.WITHOUT_REPLACEMENT,
        "min_deflection_distance": 3.0,
        "num_objects": 5,
        "num_deflect_to_win": deflections_to_win[int(parameters[2])],
        "allow_deflection_with_instrument_shaft": False,
    }

    config = {"max_episode_steps": 500 * env_kwargs["num_deflect_to_win"], **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    info_keywords = [
        "num_deflections",
        "ret_act_vio_car_wor",
        "ret_act_vio_sta_lim",
        "ret_too_col",
        "ret_dis_to_act_sph",
        "ret_del_dis_to_act_sph",
        "ret_def_of_ina_sph",
        "ret_def_of_act_sph",
        "ret_del_def_of_act_sph",
        "ret_don_wit_act_sph",
        "ret_suc_tas",
        "distance_to_active_sphere",
        "deflection_of_active_sphere",
        "successful_task",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=DeflectSpheresEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False if image_based else True,
        normalize_reward=normalize_reward,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        reward_clip=reward_clip,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name = f"PPO_{observation_type.name}_{continuous_actions=}_{parameters[0]}_{parameters[2]}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
