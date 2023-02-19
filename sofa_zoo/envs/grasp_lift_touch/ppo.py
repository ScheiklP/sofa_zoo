import numpy as np
from torch import nn
from stable_baselines3 import PPO

from sofa_env.scenes.grasp_lift_touch.grasp_lift_touch_env import CollisionEffect, GraspLiftTouchEnv, Phase, RenderMode, ObservationType, ActionType

from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS


if __name__ == "__main__":

    add_render_callback = True
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf

    # observation_type, [start phase, end phase]
    parameters = ["STATE", ["GRASP", "DONE"]]

    if isinstance(parameters[1], str):
        start_name = eval(parameters[1])[0]
        end_name = eval(parameters[1])[1]
    else:
        start_name = parameters[1][0]
        end_name = parameters[1][1]
    start_phase = Phase[start_name]
    end_phase = Phase[end_name]

    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]

    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "time_step": 0.05,
        "frame_skip": 2,
        "settle_steps": 50,
        "render_mode": RenderMode.HEADLESS,
        "start_in_phase": start_phase,
        "end_in_phase": end_phase,
        "tool_collision_distance": 5.0,
        "goal_tolerance": 5.0,
        "individual_agents": False,
        "individual_rewards": True,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "on_reset_callbacks": None,
        "reward_amount_dict": {
            Phase.ANY: {
                "collision_cauter_gripper": -0.1,
                "collision_cauter_gallbladder": -0.1,
                "collision_cauter_liver": -0.1,
                "collision_gripper_liver": -0.01,
                "distance_cauter_target": -0.5,
                "delta_distance_cauter_target": -1.0,
                "target_visible": 0.0,
                "gallbladder_is_grasped": 20.0,
                "new_grasp_on_gallbladder": 10.0,
                "lost_grasp_on_gallbladder": -10.0,
                "active_grasping_springs": 0.0,
                "delta_active_grasping_springs": 0.0,
                "gripper_pulls_gallbladder_out": 0.005,
                "dynamic_force_on_gallbladder": -0.003,
                "successful_task": 200.0,
                "failed_task": -0.0,
                "cauter_action_violated_state_limits": -0.0,
                "cauter_action_violated_cartesian_workspace": -0.0,
                "gripper_action_violated_state_limits": -0.0,
                "gripper_action_violated_cartesian_workspace": -0.0,
                "phase_change": 10.0,
                "overlap_gallbladder_liver": -0.1,
                "delta_overlap_gallbladder_liver": -0.01,
            },
            Phase.GRASP: {
                "distance_gripper_graspable_region": -0.2,
                "delta_distance_gripper_graspable_region": -10.0,
            },
            Phase.TOUCH: {
                "cauter_activation_in_target": 0.0,
                "cauter_delta_activation_in_target": 1.0,
                "cauter_touches_target": 0.0,
                "delta_distance_cauter_target": -5.0,
            },
        },
        "collision_punish_mode": CollisionEffect.CONSTANT,
        "losing_grasp_ends_episode": False,
    }
    config = {"max_episode_steps": 400 + 100 * (end_phase.value - start_phase.value), **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]

    info_keywords = [
        "ret_col_cau_gri",
        "ret_col_cau_gal",
        "ret_col_cau_liv",
        "ret_col_gri_liv",
        "ret_cau_act_vio_sta_lim",
        "ret_cau_act_vio_car_wor",
        "ret_gri_act_vio_sta_lim",
        "ret_gri_act_vio_car_wor",
        "ret_dis_cau_tar",
        "ret_del_dis_cau_tar",
        "ret_cau_tou_tar",
        "ret_cau_act_in_tar",
        "ret_tar_vis",
        "ret_dis_gri_gra_reg",
        "ret_del_dis_gri_gra_reg",
        "ret_gal_is_gra",
        "ret_new_gra_on_gal",
        "ret_los_gra_on_gal",
        "ret_act_gra_spr",
        "ret_del_act_gra_spr",
        "ret_gri_dis_to_tro",
        "ret_gri_pul_gal_out",
        "ret_ove_gal_liv",
        "ret_del_ove_gal_liv",
        "ret_dyn_for_on_gal",
        "ret_suc_tas",
        "ret_fai_tas",
        "ret_pha_cha",
        "ret",
        "distance_cauter_target",
        "cauter_touches_target",
        "cauter_activation_in_target",
        "target_visible",
        "distance_gripper_graspable_region",
        "gripper_distance_to_trocar",
        "gripper_pulls_gallbladder_out",
        "successful_task",
        "final_phase",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=GraspLiftTouchEnv,
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
        tb_log_name=f"PPO_{observation_type.name}_{continuous_actions=}_start={start_phase.name}_end={end_phase.name}",
    )

    log_path = str(model.logger.dir)
    model.save(log_path + "saved_model.pth")
