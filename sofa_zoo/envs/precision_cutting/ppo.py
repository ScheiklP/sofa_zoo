import numpy as np
from typing import Callable
from functools import partial
from stable_baselines3 import PPO

import sofa_env.scenes.precision_cutting.sofa_objects.cloth_cut as cloth_cut
from sofa_env.scenes.precision_cutting.precision_cutting_env import RenderMode, ObservationType, PrecisionCuttingEnv, ActionType

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # observation_type, function, control, randomized function
    parameters = ["RGB", "line", "state", "False"]

    randomize_function = parameters[3] == "True"

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]
    render_mode = RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE

    def line_cutting_path_generator(rng: np.random.Generator) -> Callable:
        if randomize_function:
            position = rng.uniform(low=0.3, high=0.7)
            depth = rng.uniform(low=0.5, high=0.7)
            slope = rng.uniform(low=-0.5, high=0.5)
        else:
            position = 0.5
            depth = 0.5
            slope = 0.0
        cutting_path = partial(cloth_cut.linear_cut, slope=slope, position=position, depth=depth)
        return cutting_path

    def sine_cutting_path_generator(rng: np.random.Generator) -> Callable:
        if randomize_function:
            position = rng.uniform(low=0.3, high=0.7)
            depth = rng.uniform(low=0.3, high=0.7)
            frequency = rng.uniform(low=0.5, high=1.5) / 75
            amplitude = rng.uniform(low=10.0, high=20.0)
        else:
            position = 0.6
            depth = 0.5
            frequency = 1.0 / 75
            amplitude = 15.0
        cutting_path = partial(cloth_cut.sine_cut, frequency=frequency, amplitude=amplitude, position=position, depth=depth)
        return cutting_path

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.025,
        "frame_skip": 4,
        "settle_steps": 10,
        "render_mode": render_mode,
        "reward_amount_dict": {
            "unstable_deformation": -0.0001,
            "distance_scissors_cutting_path": -1.0,
            "delta_distance_scissors_cutting_path": -500.0,
            "cuts_on_path": 0.0,
            "cuts_off_path": -0.1,
            "cut_ratio": 0.0,
            "delta_cut_ratio": 10.0,
            "workspace_violation": -0.0,
            "state_limits_violation": -0.0,
            "rcm_violation_xyz": -0.0,
            "rcm_violation_rpy": -0.0,
            "jaw_angle_violation": -0.0,
            "successful_task": 50.0,
        },
        "camera_reset_noise": None,
        "create_scene_kwargs": {
            "show_closest_point_on_path": False,
        },
        "cartesian_control": parameters[2] == "cartesian",
        "cloth_cutting_path_func_generator": sine_cutting_path_generator if parameters[1] == "sine" else line_cutting_path_generator,
        "ratio_to_cut": 0.85,
    }

    config = {"max_episode_steps": 500, **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    info_keywords = [
        "ret",
        "total_cut_on_path",
        "total_cut_off_path",
        "total_unstable_deformation",
        "ret_uns_def",
        "ret_cut_on_pat",
        "ret_cut_rat",
        "ret_del_cut_rat",
        "ret_cut_off_pat",
        "ret_dis_sci_cut_pat",
        "ret_del_dis_sci_cut_pat",
        "ret_suc_tas",
        "cut_ratio",
        "successful_task",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=PrecisionCuttingEnv,
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
        tb_log_name= f"PPO_{observation_type.name}_{parameters[1]}_{parameters[2]}{'_RF' if randomize_function else ''}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
