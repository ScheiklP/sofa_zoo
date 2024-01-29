import gymnasium as gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from sofa_env.scenes.tissue_retraction.tissue_retraction_env import CollisionPunishmentMode, ObservationType, TissueRetractionEnv, ActionType, RenderMode

from stable_baselines3 import PPO
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


class CollisionPunishmentCurriculumCallback(BaseCallback):
    """

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """

    def __init__(self, end_coll: float, end_ws: float, steps: int, verbose: int = 0):
        super(CollisionPunishmentCurriculumCallback, self).__init__(verbose)

        self.end_coll = end_coll
        self.end_ws = end_ws
        self.steps = steps

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.model is not None
        assert isinstance(self.training_env, VecEnv)

        progress = self.model.num_timesteps / self.steps

        current_collision_cost = self.end_coll * progress
        current_workspace_cost = self.end_ws * progress

        self.training_env.env_method("set_reward_coefficients", collision_cost_factor=current_collision_cost, workspace_violation_cost=current_workspace_cost)


if __name__ == "__main__":
    add_render_callback = True
    image_based = True
    continuous_actions = True
    normalize_reward = True

    env_kwargs = {
        "render_mode": RenderMode.HEADLESS if image_based or add_render_callback else RenderMode.NONE,
        "image_shape": (64, 64),
        "window_size": (300, 300),
        "observation_type": ObservationType.RGB if image_based else ObservationType.STATE,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "observe_phase_state": False if image_based else True,
        "time_step": 0.1,
        "frame_skip": 3,
        "settle_steps": 20,
        "maximum_robot_velocity": 3.0,
        "discrete_action_magnitude": 6.0,
        "maximum_grasp_height": 0.009,
        "grasping_threshold": 0.003,
        "grasping_position": (-0.0485583, 0.0085, 0.0356076),
        "end_position": (-0.019409, 0.062578, -0.00329643),
        "end_position_threshold": 0.003,
        "collision_tolerance": 0.006,  # collisions less than 6 mm distance to the grasping point are ignored
        "collision_punishment_mode": CollisionPunishmentMode.CONTACTDISTANCE,
        "maximum_in_tissue_travel": 0.003,
        "reward_amount_dict": {
            "one_time_reward_grasped": 1.0,
            "one_time_reward_goal": 1.0,
            "time_step_cost_scale_in_grasp_phase": 1.2,
            "target_visible_scaling": 0,
            "control_cost_factor": 0.0,
            "workspace_violation_cost": 0.0,
            "collision_cost_factor": 0.0,
        },
        "action_space": gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32) if continuous_actions else gym.spaces.Discrete(7),
        "create_scene_kwargs": {
            "show_floor": True,
            "texture_objects": False,
            "camera_position": (0.0, 0.135365, 0.233074),
            "camera_orientation": (-0.24634687, -0.0, -0.0, 0.96918173),
            "camera_field_of_view_vertical": 42,
            "workspace_height": 0.09,
            "workspace_width": 0.07,
            "workspace_depth": 0.09,
            "workspace_floor": 0.007,  # minimum height of the gripper (5 mm board + 2 mm tolerance for the tissue -> maximum 6 mm in the tissue)
            "remote_center_of_motion": (0.09036183, 0.15260103, 0.01567807),
        },
    }

    config = {"max_episode_steps": 1000, **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs

    info_keywords = [
        "phase",
        "distance_to_grasping_position",
        "distance_to_end_position",
        "control_cost",
        "episode_control_cost",
        "episode_collision_cost",
        "episode_workspace_violation_cost",
        "steps_in_grasping_phase",
        "steps_in_retraction_phase",
        "steps_in_collision",
        "steps_in_workspace_violation",
        "return_from_grasping",
        "return_from_retracting",
    ]

    model, callback = configure_learning_pipeline(
        env_class=TissueRetractionEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        extra_callbacks=[
            CollisionPunishmentCurriculumCallback(
                end_coll=10.0,
                end_ws=0.1,
                steps=config["total_timesteps"],
            )
        ],
        normalize_reward=normalize_reward,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name=f"PPO_{image_based=}_{continuous_actions=}",
    )

    log_path = model.logger.dir
    if not isinstance(log_path, str):
        log_path = "./"
    model.save(log_path + "saved_model.pth")
