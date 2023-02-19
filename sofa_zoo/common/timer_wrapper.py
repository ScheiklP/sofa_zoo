import time
from typing import Any, Dict, Tuple

import gym
import numpy as np


class TimerWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.episode_length = 0

    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        start = time.perf_counter()
        obs, reward, done, info = super().step(action)
        end = time.perf_counter()
        info["step_duration"] = end - start
        self.episode_length += 1
        return obs, reward, done, info

    def reset(self, **kwargs) -> Any:
        self.episode_length = 0
        return super().reset(**kwargs)
