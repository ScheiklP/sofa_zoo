import numpy as np
from torch import nn
from stable_baselines3 import PPO

from sofa_env.scenes.ligating_loop.ligating_loop_env import RenderMode, ObservationType, LigatingLoopEnv, ActionType

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # observation_type, stiffness, loop_size
    parameters = ["RGB", "soft", "large"]

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.05,
        "frame_skip": 2,
        "settle_steps": 50,
        "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        "num_tracking_points_cavity": {
            "height": 3,
            "radius": 1,
            "angle": 3,
        },
        "num_tracking_points_marking": {
            "height": 1,
            "radius": 1,
            "angle": 3,
        },
        "num_tracking_points_loop": 6,
        "target_loop_closed_ratio": 0.5 if parameters[2] == "stiff" else 0.8,
        "target_loop_overlap_ratio": 0.1,
        "randomize_marking_position": True,
        "with_gripper": False,
        "individual_agents": False,
        "band_width": 6.0,
        "reward_amount_dict": {
            "distance_loop_to_marking_center": -0.05,
            "delta_distance_loop_to_marking_center": -100.0,
            "loop_center_in_cavity": 0.0,
            "instrument_not_in_cavity": -0.0,
            "instrument_shaft_collisions": -0.05,
            "loop_marking_overlap": 1.0,
            "loop_closed_around_marking": 6.0,
            "loop_closed_in_thin_air": -0.1,
            "successful_task": 100.0,
        },
        "create_scene_kwargs": {
            "stiff_loop": parameters[1] == "stiff",
            "num_rope_points": 50 if parameters[2] == "small" else 90,
            "loop_radius": 18.0 if parameters[2] == "small" else 30.0,
        },
    }

    config = {"max_episode_steps": 500, **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    info_keywords = [
        "distance_loop_to_marking_center",
        "instrument_not_in_cavity",
        "instrument_shaft_collisions",
        "loop_center_in_cavity",
        "loop_closed_around_marking",
        "loop_marking_overlap",
        "ret",
        "ret_del_dis_loo_to_mar_cen",
        "ret_dis_loo_to_mar_cen",
        "ret_ins_not_in_cav",
        "ret_ins_sha_col",
        "ret_loo_cen_in_cav",
        "ret_loo_clo_aro_mar",
        "ret_loo_mar_ove",
        "ret_suc_tas",
        "ret_loo_clo_in_thi_air",
        "successful_task",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=LigatingLoopEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False if image_based else True,
        normalize_reward=normalize_reward,
        reward_clip=reward_clip,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        use_watchdog_vec_env=True,
        watchdog_vec_env_timeout=1.0,
        reset_process_on_env_reset=False,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name=f"PPO_{observation_type.name}_{continuous_actions=}_{parameters[1]}_{parameters[2]}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
