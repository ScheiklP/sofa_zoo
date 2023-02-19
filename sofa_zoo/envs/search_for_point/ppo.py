import numpy as np
from stable_baselines3 import PPO

from sofa_env.scenes.search_for_point.search_for_point_env import RenderMode, SearchForPointEnv, ActionType, ObservationType, ActiveVision

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # obs, active vision,
    parameters = ["STATE", "True"]

    observation_type = ObservationType[parameters[0]]
    active_vision = eval(parameters[1])
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    reward_amount_dict = {
        ActiveVision.DEACTIVATED: {
            "poi_is_in_frame": 0.01,
            "relative_camera_distance_error_to_poi": -0.01,
            "delta_relative_camera_distance_error_to_poi": -0.01,
            "relative_pixel_distance_poi_to_image_center": -0.001,
            "delta_relative_pixel_distance_poi_to_image_center": -0.001,
            "successful_task": 100.0,
        },
        ActiveVision.CAUTER: {
            "collision_cauter": -0.001,
            "relative_distance_cauter_target": -0.0005,
            "relative_delta_distance_cauter_target": -5.0,
            "cauter_touches_target": 0.0,
            "successful_task": 100.0,
            "cauter_action_violated_state_limits": -0.0,
        },
    }

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        "time_step": 0.1,
        "frame_skip": 1,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "reward_amount_dict": reward_amount_dict,
        "active_vision_mode": ActiveVision.CAUTER if active_vision else ActiveVision.DEACTIVATED,
    }

    config = {"max_episode_steps": 500, **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    if active_vision:
        info_keywords = [
            "ret_col_cau",
            "ret_rel_dis_cau_tar",
            "ret_rel_del_dis_cau_tar",
            "ret_cau_tou_tar",
            "ret_cau_act_vio_sta_lim",
            "ret_suc_tas",
            "ret",
            "collision_cauter",
            "relative_distance_cauter_target",
            "relative_delta_distance_cauter_target",
            "cauter_touches_target",
            "cauter_action_violated_state_limits",
            "successful_task",
        ]
    else:
        info_keywords = [
            "ret_poi_is_in_fra",
            "ret_rel_cam_dis_err_to_poi",
            "ret_del_rel_cam_dis_err_to_poi",
            "ret_rel_pix_dis_poi_to_ima_cen",
            "ret_del_rel_pix_dis_poi_to_ima_cen",
            "ret_suc_tas",
            "ret",
            "poi_is_in_frame",
            "relative_camera_distance_error_to_poi",
            "relative_pixel_distance_poi_to_image_center",
            "successful_task",
        ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=SearchForPointEnv,
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
        tb_log_name= f"PPO_{observation_type.name}{'_AV' if active_vision else ''}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
