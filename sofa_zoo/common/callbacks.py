import numpy as np
from typing import List
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeInfoLoggerCallback(BaseCallback):
    """
    Logs additional keys from the episode info buffer.
    OnPolicyAlgorithm's collect_rollouts logs the mean
    reward and length of finished episodes. This callback
    tries to log the mean of all the other keys in the
    episode info dictionaries. This can be useful to log
    information added through a Monitor environment wrapper.
    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    :param exclude_keys: (List[str]) keys from episode infos to exclude. By default, excludes reward and legth (already logged by OnPolicyAlgorithm) and time
    """

    def __init__(self, verbose: int = 0, exclude_keys: List[str] = ["r", "l", "t"]):
        super(EpisodeInfoLoggerCallback, self).__init__(verbose)
        self.exclude_keys = exclude_keys

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        iteration = (self.model.num_timesteps - self.model._num_timesteps_at_start) // self.model.n_envs // self.model.n_steps

        if self.locals["log_interval"] is not None and (iteration) % self.locals["log_interval"] == 0:

            if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[0]) > 0:

                logging_keys = self.model.ep_info_buffer[0].keys()

                for key in logging_keys:
                    if key in self.exclude_keys:
                        continue
                    else:
                        try:
                            upcast_key_data = np.array([ep_info[key] for ep_info in self.model.ep_info_buffer], dtype=np.float64)
                            safe_mean = np.nan if len(upcast_key_data) == 0 else np.nanmean(upcast_key_data)
                            self.logger.record(f"trajectory/ep_{key}_mean", safe_mean)
                        except TypeError:
                            if self.verbose > 0:
                                print(f"Episode info key {key} can not be averaged by np.nanmean. Will not try to log the key in the future.")
                                self.exclude_keys.append(key)


class AdjustLoggingWindow(BaseCallback):
    def __init__(self, window_length: int, verbose=0):
        super(AdjustLoggingWindow, self).__init__(verbose)
        self.window_length = window_length

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        self.model.ep_info_buffer = deque(maxlen=self.window_length)
        self.model.ep_success_buffer = deque(maxlen=self.window_length)
        return super()._on_training_start()


class RenderCallback(BaseCallback):
    def _on_step(self) -> bool:
        self.model._vec_normalize_env.render(mode="human")
        return super()._on_step()
