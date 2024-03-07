"""https://pytorch.org/rl/tutorials/coding_ppo.html"""

from collections import defaultdict

import matplotlib.pyplot as plt
from sofa_env.scenes.reach.reach_env import ActionType, RenderMode, ObservationType, ReachEnv
import torch
import torch.multiprocessing as multiprocessing

from gymnasium.wrappers.time_limit import TimeLimit

from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.record import VideoRecorder
from torchrl.record.loggers.csv import CSVLogger
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm

PRINT_ALL = False


if __name__ == "__main__":
    ### Hyperparameters
    is_fork = multiprocessing.get_start_method() == "fork"
    device = torch.device(0) if torch.cuda.is_available() and not is_fork else torch.device("cpu")
    num_cells = 256  # number of cells in each layer i.e. output dim.
    lr = 3e-4
    max_grad_norm = 1.0

    frames_per_batch = 1000
    # For a complete training, bring the number of frames up to 10k
    total_frames = 10_000

    ### PPO Hyperparameters
    sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
    num_epochs = 10  # optimisation steps per batch of data collected
    clip_epsilon = 0.2  # clip value for PPO loss: see the equation in the intro for more context.
    gamma = 0.99
    lmbda = 0.95
    entropy_eps = 1e-4

    ### Environment
    env_kwargs = {
        "image_shape": (64, 64),
        "render_mode": RenderMode.HEADLESS,
        "observation_type": ObservationType.STATE,
        "action_type": ActionType.CONTINUOUS,
        "distance_to_target_threshold": 0.003,  # m
        "time_step": 0.1,
        "frame_skip": 1,
        "observe_target_position": True,
        "reward_amount_dict": {
            "distance_to_target": -1.0,
            "delta_distance_to_target": -10.0,
            "successful_task": 100.0,
            "time_step_cost": -0.0,
            "worspace_violation": -0.0,
        },
        "on_reset_callbacks": None,
        "create_scene_kwargs": {
            "show_bounding_boxes": True,
        },
        "sphere_radius": 0.008,
    }
    logger = CSVLogger(
        exp_name="reach",
        log_dir="rollout_videos",
        video_format="mp4",
        video_fps=10,
    )

    org_env = ReachEnv(**env_kwargs)
    org_env = TimeLimit(org_env, max_episode_steps=500)
    base_env = GymWrapper(
        org_env,
        device=device,
        from_pixels=True,
    )
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ObservationNorm(in_keys=["observation"]),
            DoubleToFloat(),
            StepCounter(),
            VideoRecorder(
                logger=logger,
                tag="run_video",
                skip=1,
            ),
        ),
    )

    ### Calculate parameters of the normalization wrapper
    env.transform[0].init_stats(num_iter=1000, reduce_dim=0, cat_dim=0)

    if PRINT_ALL:
        print("normalization constant shape:", env.transform[0].loc.shape)
        print("observation_spec:", env.observation_spec)
        print("reward_spec:", env.reward_spec)
        print("input_spec:", env.input_spec)
        print("action_spec (as defined by input_spec):", env.action_spec)

    ### Check environment specs
    check_env_specs(env)

    ### Define the model
    actor_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(2 * env.action_spec.shape[-1], device=device),
        NormalParamExtractor(),
    )

    policy_module = TensorDictModule(actor_net, in_keys=["observation"], out_keys=["loc", "scale"])

    policy_module = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "min": env.action_spec.space.low,
            "max": env.action_spec.space.high,
        },
        return_log_prob=True,
        # we'll need the log-prob for the numerator of the importance weights
    )

    value_net = nn.Sequential(
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(num_cells, device=device),
        nn.Tanh(),
        nn.LazyLinear(1, device=device),
    )

    value_module = ValueOperator(
        module=value_net,
        in_keys=["observation"],
    )

    # Pass a batch of data through the networks to initialize the LazyLinear layers
    reset_obs = env.reset()
    policy_output = policy_module(reset_obs)
    value_output = value_module(reset_obs)
    if PRINT_ALL:
        print("Running policy:", policy_output)
        print("Running value:", value_output)

    ### Data collector
    collector = SyncDataCollector(
        env,
        policy_module,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
    )

    ### Replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )

    ### Loss
    advantage_module = GAE(gamma=gamma, lmbda=lmbda, value_network=value_module, average_gae=True)

    loss_module = ClipPPOLoss(
        actor_network=policy_module,
        critic_network=value_module,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coef=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, total_frames // frames_per_batch, 0.0)

    ### Training loop
    logs = defaultdict(list)
    pbar = tqdm(total=total_frames)
    eval_str = ""

    # We iterate over the collector until it reaches the total number of frames it was
    # designed to collect:
    for i, tensordict_data in enumerate(collector):
        # we now have a batch of data to work with. Let's learn something from it.
        for _ in range(num_epochs):
            # We'll need an "advantage" signal to make PPO work.
            # We re-compute it at each epoch as its value depends on the value
            # network which is updated in the inner loop.
            advantage_module(tensordict_data)
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())
            for _ in range(frames_per_batch // sub_batch_size):
                subdata = replay_buffer.sample(sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                loss_value = loss_vals["loss_objective"] + loss_vals["loss_critic"] + loss_vals["loss_entropy"]

                # Optimization: backward, grad clipping and optimization step
                loss_value.backward()
                # this is not strictly mandatory but it's good practice to keep
                # your gradient norm bounded
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                optim.step()
                optim.zero_grad()

        logs["reward"].append(tensordict_data["next", "reward"].mean().item())
        pbar.update(tensordict_data.numel())
        cum_reward_str = f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
        logs["step_count"].append(tensordict_data["step_count"].max().item())
        stepcount_str = f"step count (max): {logs['step_count'][-1]}"
        logs["lr"].append(optim.param_groups[0]["lr"])
        lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
        if i % 10 == 0:
            # We evaluate the policy once every 10 batches of data.
            # Evaluation is rather simple: execute the policy without exploration
            # (take the expected value of the action distribution) for a given
            # number of steps (1000, which is our ``env`` horizon).
            # The ``rollout`` method of the ``env`` can take a policy as argument:
            # it will then execute this policy at each step.
            with set_exploration_type(ExplorationType.MEAN), torch.no_grad():
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(1000, policy_module)
                logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                logs["eval reward (sum)"].append(eval_rollout["next", "reward"].sum().item())
                logs["eval step_count"].append(eval_rollout["step_count"].max().item())
                eval_str = f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} " f"(init: {logs['eval reward (sum)'][0]: 4.4f}), " f"eval step-count: {logs['eval step_count'][-1]}"
                env.transform.dump()
                del eval_rollout
        pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

        # We're also using a learning rate scheduler. Like the gradient clipping,
        # this is a nice-to-have but nothing necessary for PPO to work.
        scheduler.step()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.plot(logs["reward"])
    plt.title("training rewards (average)")
    plt.subplot(2, 2, 2)
    plt.plot(logs["step_count"])
    plt.title("Max step count (training)")
    plt.subplot(2, 2, 3)
    plt.plot(logs["eval reward (sum)"])
    plt.title("Return (test)")
    plt.subplot(2, 2, 4)
    plt.plot(logs["eval step_count"])
    plt.title("Max step count (test)")
    plt.show()

    env.close()
