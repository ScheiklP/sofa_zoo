import numpy as np
from stable_baselines3 import PPO

from sofa_env.scenes.rope_threading.rope_threading_env import RenderMode, ObservationType, RopeThreadingEnv

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # observations, bimanual, randomized, eyes
    parameters = ["STATE", "True", "True", "2"]

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    eye_configs = {
        "1": [
            (60, 10, 0, 90),
        ],
        "2": [
            (60, 10, 0, 90),
            (10, 10, 0, 90),
        ],
    }

    bimanual_grasp = parameters[1] == "True"
    randomized_eye = parameters[2] == "True"
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "time_step": 0.01,
        "frame_skip": 10,
        "settle_steps": 20,
        "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        "reward_amount_dict": {
            "passed_eye": 10.0,
            "lost_eye": -20.0,  # more than passed_eye
            "goal_reached": 100.0,
            "distance_to_active_eye": -0.0,
            "lost_grasp": -0.1,
            "collision": -0.1,
            "floor_collision": -0.1,
            "bimanual_grasp": 0.0,
            "moved_towards_eye": 200.0,
            "moved_away_from_eye": -200.0,
            "workspace_violation": -0.01,
            "state_limit_violation": -0.01,
            "distance_to_lost_rope": -0.0,
            "delta_distance_to_lost_rope": -0.0,
            "fraction_rope_passed": 0.0,
            "delta_fraction_rope_passed": 200.0,
        },
        "create_scene_kwargs": {
            "eye_config": eye_configs[parameters[3]],
            "randomize_gripper": True,
            "start_grasped": True,
            "randomize_grasp_index": True,
        },
        "on_reset_callbacks": None,
        "color_eyes": True,
        "individual_agents": False,
        "only_right_gripper": not bimanual_grasp,
        "fraction_of_rope_to_pass": 0.05,
        "num_rope_tracking_points": 10,
    }

    if bimanual_grasp:
        env_kwargs["reward_amount_dict"]["bimanual_grasp"] = 100.0
        env_kwargs["reward_amount_dict"]["distance_to_bimanual_grasp"] = -0.0
        env_kwargs["reward_amount_dict"]["delta_distance_to_bimanual_grasp"] = -200.0

    if randomized_eye:
        env_kwargs["create_scene_kwargs"]["eye_reset_noise"] = {
            "low": np.array([-20.0, -20.0, 0.0, -15]),
            "high": np.array([20.0, 20.0, 0.0, 15]),
        }

    config = {"max_episode_steps": 200 + 150 * (len(eye_configs[parameters[3]]) - 1), **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]


    info_keywords = [
        "distance_to_active_eye",
        "lost_grasps",
        "recovered_lost_grasps",
        "passed_eyes",
        "lost_eyes",
        "collisions",
        "floor_collisions",
        "successful_task",
        "rew_delta_distance",
        "rew_absolute_distance",
        "rew_losing_eyes",
        "rew_losing_grasp",
        "rew_collisions",
        "rew_floor_collisions",
        "rew_workspace_violation",
        "rew_state_limit_violation",
        "rew_dist_to_lost_rope",
        "rew_delt_dist_to_lost_rope",
        "rew_passed_eyes",
        "rew_bimanual_grasp",
        "rew_dist_to_bimanual_grasp",
        "rew_delt_dist_to_bimanual_grasp",
        "rew_fraction_passed",
        "rew_delta_fraction_passed",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=RopeThreadingEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False if image_based else True,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        normalize_reward=normalize_reward,
        reward_clip=reward_clip,
        use_watchdog_vec_env=True,
        watchdog_vec_env_timeout=20.0,
        reset_process_on_env_reset=False,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name= f"{parameters[0]}_{parameters[1]}Biman_{parameters[2]}Random_{parameters[3]}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
