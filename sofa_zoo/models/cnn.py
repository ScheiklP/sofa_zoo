from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gymnasium as gym
import torch as th
import numpy as np
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


class DoubleNatureCNN(BaseFeaturesExtractor):
    """SB3's NatureCNN, but with separate networks for policy and value function."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(DoubleNatureCNN, self).__init__(observation_space, 2 * features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), "You should use NatureCNN " f"only with images not with {observation_space}\n" "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n" "If you are using a custom environment,\n" "please check it using our env checker:\n" "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        n_input_channels = observation_space.shape[0]
        self.policy_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.value_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.policy_cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.policy_linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self.value_linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return th.cat((self.policy_linear(self.policy_cnn(observations)), self.value_linear(self.value_cnn(observations))), dim=-1)


class SplitDoubleNatureCnn(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 512,
        last_layer_dim_vf: int = 512,
    ):
        super(SplitDoubleNatureCnn, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.feature_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        assert feature_dim / 2 == last_layer_dim_pi
        assert feature_dim / 2 == last_layer_dim_vf

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return features[..., : self.latent_dim_pi], features[..., self.latent_dim_pi :]

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features[..., : self.latent_dim_pi]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return features[..., self.latent_dim_pi :]


class ActorCriticDoubleNatureCnnPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = [],
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class=DoubleNatureCNN,
        *args,
        **kwargs,
    ):
        super(ActorCriticDoubleNatureCnnPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitDoubleNatureCnn(self.features_dim)


def get_conv_layer_output_size(input_size, kernel_size, stride, padding):
    return int(np.ceil((input_size - kernel_size + 2 * padding) / stride + 1))


class SofaCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        output_features: Optional[int] = None,
        conv_maps_layer_1: int = 16,
        batch_norm: bool = True,
    ):
        initial_output_features = output_features or 1
        super(SofaCNN, self).__init__(observation_space, initial_output_features)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        assert observation_space.shape[1] == observation_space.shape[2]

        image_width = observation_space.shape[1]
        current_feature_size = image_width
        previous_conv_maps = n_input_channels
        current_conv_maps = conv_maps_layer_1

        layer_i = 1

        layers = []

        # input width and height have to be a power of two, otherwise the 4, 2, 1 does not work (always halves)
        assert np.log2(image_width).is_integer()

        while current_feature_size > 1:
            layers.append(nn.Conv2d(previous_conv_maps, current_conv_maps, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm2d(current_conv_maps))
            current_feature_size = get_conv_layer_output_size(current_feature_size, 4, 2, 1)

            previous_conv_maps = current_conv_maps
            current_conv_maps *= 2
            layer_i += 1

        assert current_feature_size == 1

        layers.append(nn.Flatten())

        self.cnn = nn.Sequential(*layers)

        # Compute shape by doing one forward pass
        with th.no_grad():
            # deactivate for batch_norm
            self.train(False)
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
            self.train(True)

        if output_features is not None:
            self.linear = nn.Linear(n_flatten, output_features)
            self._features_dim = output_features

        else:
            self.linear = lambda x: x
            self._features_dim = n_flatten

        print(f"Created {layer_i} convolutional layers with a final size of {current_feature_size}x{current_feature_size}x{previous_conv_maps}, flattened to a vector of size {n_flatten}{', passed to a linear layer with output size ' + str(output_features) + '.' if output_features is not None else '.'}")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class DoubleSofaCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        output_features: Optional[int] = 512,
        conv_maps_layer_1: int = 16,
        batch_norm: bool = True,
    ):
        initial_output_features = output_features or 1
        super(DoubleSofaCNN, self).__init__(observation_space, initial_output_features)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        assert observation_space.shape[1] == observation_space.shape[2]

        image_width = observation_space.shape[1]
        current_feature_size = image_width
        previous_conv_maps = n_input_channels
        current_conv_maps = conv_maps_layer_1

        layer_i = 1

        policy_layers = []
        critic_layers = []

        # input width and height have to be a power of two, otherwise the 4, 2, 1 does not work (always halves)
        assert np.log2(image_width).is_integer()

        while current_feature_size > 1:
            policy_layers.append(nn.Conv2d(previous_conv_maps, current_conv_maps, kernel_size=4, stride=2, padding=1))
            policy_layers.append(nn.ReLU())
            if batch_norm:
                policy_layers.append(nn.BatchNorm2d(current_conv_maps))
            critic_layers.append(nn.Conv2d(previous_conv_maps, current_conv_maps, kernel_size=4, stride=2, padding=1))
            critic_layers.append(nn.ReLU())
            if batch_norm:
                critic_layers.append(nn.BatchNorm2d(current_conv_maps))
            current_feature_size = get_conv_layer_output_size(current_feature_size, 4, 2, 1)

            previous_conv_maps = current_conv_maps
            current_conv_maps *= 2
            layer_i += 1

        assert current_feature_size == 1

        policy_layers.append(nn.Flatten())
        critic_layers.append(nn.Flatten())

        self.policy_cnn = nn.Sequential(*policy_layers)
        self.critic_cnn = nn.Sequential(*critic_layers)

        # Compute shape by doing one forward pass
        with th.no_grad():
            # deactivate for batch_norm
            self.train(False)
            n_flatten = self.policy_cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]
            self.train(True)

        if output_features is not None:
            self.policy_linear = nn.Linear(n_flatten, output_features)
            self.critic_linear = nn.Linear(n_flatten, output_features)
            self._features_dim = output_features * 2

        else:
            self.policy_linear = lambda x: x
            self.critic_linear = lambda x: x
            self._features_dim = n_flatten

        print(f"Created {layer_i} convolutional layers with a final size of {current_feature_size}x{current_feature_size}x{previous_conv_maps}, flattened to a vector of size {n_flatten}{', passed to a linear layer with output size ' + str(output_features) + '.' if output_features is not None else '.'}")

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return th.cat((self.policy_linear(self.policy_cnn(observations)), self.critic_linear(self.critic_cnn(observations))), dim=-1)


class DreamerCnn(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, share_encoder: bool = False):
        super(DreamerCnn, self).__init__(observation_space, 600)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False), "You should use NatureCNN " f"only with images not with {observation_space}\n" "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n" "If you are using a custom environment,\n" "please check it using our env checker:\n" "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html"
        n_input_channels = observation_space.shape[0]
        self.policy_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.policy_cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.policy_linear = nn.Sequential(
            nn.Linear(n_flatten, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
        )
        self.value_linear = nn.Sequential(
            nn.Linear(n_flatten, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
        )

        if share_encoder:
            self.forward = self.shared_forward
            self.value_cnn = lambda x: x
        else:
            self.forward = self.separate_forward
            self.value_cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Flatten(),
            )

    def shared_forward(self, observations: th.Tensor) -> th.Tensor:
        h = self.policy_cnn(observations)
        return th.cat(
            (
                self.policy_linear(h),
                self.value_linear(h),
            ),
            dim=-1,
        )

    def separate_forward(self, observations: th.Tensor) -> th.Tensor:
        return th.cat(
            (
                self.policy_linear(self.policy_cnn(observations)),
                self.value_linear(self.value_cnn(observations)),
            ),
            dim=-1,
        )


class SplitDreamerCnn(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 300,
        last_layer_dim_vf: int = 300,
    ):
        super(SplitDreamerCnn, self).__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.feature_dim = feature_dim
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        assert feature_dim / 2 == last_layer_dim_pi
        assert feature_dim / 2 == last_layer_dim_vf

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return features[..., : self.latent_dim_pi], features[..., self.latent_dim_pi :]

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return features[..., : self.latent_dim_pi]

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return features[..., self.latent_dim_pi :]


class ActorCriticDreamerCnnPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = [],
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class=DreamerCnn,
        *args,
        **kwargs,
    ):
        super(ActorCriticDreamerCnnPolicy, self).__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = SplitDreamerCnn(self.features_dim)
