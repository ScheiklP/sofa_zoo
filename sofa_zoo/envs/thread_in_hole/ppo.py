import numpy as np
from stable_baselines3 import PPO

from sofa_env.scenes.thread_in_hole.thread_in_hole_env import RenderMode, ObservationType, ThreadInHoleEnv, ActionType

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # mechanical_config, observation_type, randomize camera
    parameters = ["flexible", "RGB", "False"]

    mechanical_configs = {
        "normal": {
            "thread_config": {
                "length": 50.0,
                "radius": 2.0,
                "total_mass": 1.0,
                "young_modulus": 4000.0,
                "poisson_ratio": 0.3,
                "beam_radius": 5.0,
                "mechanical_damping": 0.2,
            },
            "hole_config": {
                "inner_radius": 5.0,
                "outer_radius": 25.0,
                "height": 30.0,
                "young_modulus": 5000.0,
                "poisson_ratio": 0.3,
                "total_mass": 10.0,
            },
        },
        "flexible": {
            "thread_config": {
                "length": 80.0,
                "radius": 2.0,
                "total_mass": 1.0,
                "young_modulus": 1000.0,
                "poisson_ratio": 0.3,
                "beam_radius": 3.0,
                "mechanical_damping": 0.2,
            },
            "hole_config": {
                "inner_radius": 6.0,
                "outer_radius": 25.0,
                "height": 30.0,
                "young_modulus": 5000.0,
                "poisson_ratio": 0.3,
                "total_mass": 10.0,
            },
        },
        "inverted": {
            "thread_config": {
                "length": 50.0,
                "radius": 2.0,
                "total_mass": 10.0,
                "young_modulus": 1e5,
                "poisson_ratio": 0.3,
                "beam_radius": 5.0,
                "mechanical_damping": 0.2,
            },
            "hole_config": {
                "inner_radius": 6.0,
                "outer_radius": 15.0,
                "height": 60.0,
                "young_modulus": 5e2,
                "poisson_ratio": 0.3,
                "total_mass": 1.0,
            },
        },
    }

    camera_noise = {
        "True": np.array([20, 20, 20, 20, 20, 20]),
        "False": None,
    }

    mechanical_parameters = mechanical_configs[parameters[0]]
    camera_reset_noise = camera_noise[parameters[2]]

    observation_type = ObservationType[parameters[1]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.01,
        "frame_skip": 10,
        "settle_steps": 50,
        "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        "camera_reset_noise": camera_reset_noise,
        "reward_amount_dict": {
            "thread_tip_distance_to_hole": -0.1,
            "delta_thread_tip_distance_to_hole": -0.1,
            "thread_center_distance_to_hole": -0.0,
            "delta_thread_center_distance_to_hole": -0.0,
            "thread_points_distance_to_hole": -0.0,
            "delta_thread_points_distance_to_hole": -0.0,
            "unstable_deformation": -0.0,
            "thread_velocity": -0.0,
            "gripper_velocity": -0.0,
            "action_violated_cartesian_workspace": -0.0,
            "action_violated_state_limits": -0.0,
            "ratio_rope_in_hole": 0.1,
            "delta_ratio_rope_in_hole": 1.0,
            "successful_task": 100.0,
            "gripper_collisions": -1.0,
        },
        "create_scene_kwargs": {
            "randomize_gripper": True,
            "gripper_config": {
                "cartesian_workspace": {
                    "low": np.array([-100.0] * 2 + [0.0]),
                    "high": np.array([100.0] * 2 + [200.0]),
                },
                "state_reset_noise": np.array([15.0, 15.0, 0.0, 20.0]),
                "rcm_reset_noise": np.array([10.0, 10.0, 10.0, 5.0, 5.0, 5.0]),
                "gripper_ptsd_state": np.array([60.0, 0.0, 180.0, 90.0]),
                "gripper_rcm_pose": np.array([100.0, 0.0, 150.0, 0.0, 180.0, 0.0]),
            },
            "camera_config": {
                "placement_kwargs": {
                    "position": [0.0, -135.0, 100.0],
                    "lookAt": [0.0, 0.0, 45.0],
                },
                "vertical_field_of_view": 62.0,
            },
            **mechanical_parameters,
        },
        "on_reset_callbacks": None,
        "num_thread_tracking_points": 4,
    }

    config = {"max_episode_steps": 300, **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    info_keywords = [
        "thread_center_distance_to_hole",
        "thread_points_distance_to_hole",
        "thread_tip_distance_to_hole",
        "ratio_rope_in_hole",
        "ret",
        "ret_act_vio_car_wor",
        "ret_act_vio_sta_lim",
        "ret_del_thr_cen_dis_to_hol",
        "ret_del_thr_poi_dis_to_hol",
        "ret_del_thr_tip_dis_to_hol",
        "ret_gri_vel",
        "ret_gri_col",
        "ret_thr_cen_dis_to_hol",
        "ret_thr_poi_dis_to_hol",
        "ret_thr_tip_dis_to_hol",
        "ret_thr_vel",
        "ret_del_rat_rop_in_hol",
        "ret_rat_rop_in_hol",
        "ret_suc_tas",
        "ret_uns_def",
        "successful_task",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=ThreadInHoleEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False if image_based else True,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        normalize_reward=normalize_reward,
        reward_clip=reward_clip,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name= f"PPO_{observation_type.name}_{continuous_actions=}_{parameters[0]}_{parameters[2]}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
