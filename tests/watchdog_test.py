import gymnasium as gym
import time

from sofa_zoo.common.reset_process_vec_env import WatchdogVecEnv


class SlowEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=float)
        self.life_time_counter = 0

    def step(self, step_duration: float):
        time.sleep(step_duration)
        obs = self.observation_space.sample()
        self.life_time_counter += 1
        obs[:] = self.life_time_counter
        return obs, 0, False, False, {}

    def reset(self, **kwargs):
        obs = self.observation_space.sample()
        obs[:] = -1.0
        return obs, {}


class TestWatchdogVecEnv:
    def test_slow_env(self):
        env = SlowEnv()
        env.reset()

        start = time.time()
        env.step(1)
        end = time.time()

        assert end - start > 0.9 and end - start < 1.1

    def test_watchdog_vec_env(self):
        make_functions = [SlowEnv] * 5
        vec_env = WatchdogVecEnv(
            env_fns=make_functions,
            step_timeout_sec=1.0,
            reset_process_on_env_reset=False,
        )
        reset_obs = vec_env.reset()
        assert reset_obs.shape == (5, 1)
        assert all(reset_obs == -1.0)

        start = time.time()
        obs, _, _, _ = vec_env.step([0.1, 0.2, 1.5, 0.3, 0.1])
        end = time.time()
        assert end - start > 0.9 and end - start < 1.1
        passing_envs = [0, 1, 3, 4]
        hanging_envs = [2]
        # Env 2 should be killed and restarted -> life_time_counter == -1
        # All other envs should do one step -> life_time_counter == 1
        assert all(obs[passing_envs] == 1.0)
        assert all(obs[hanging_envs] == -1.0)

        # Now no env hangs -> life_time_counter increases by one for each
        start = time.time()
        obs, _, _, _ = vec_env.step([0.1, 0.2, 0.5, 0.3, 0.1])
        end = time.time()
        assert all(obs[passing_envs] == 2.0)
        assert all(obs[hanging_envs] == 1.0)

        # Resetting the envs does not change the life_time_counter, but returns an obs of -1
        reset_obs = vec_env.reset()
        assert all(reset_obs == -1.0)

        # So the next step should increase the life_time_counter regularly
        obs, _, _, _ = vec_env.step([0.1, 0.2, 0.5, 0.3, 0.1])
        assert all(obs[passing_envs] == 3.0)
        assert all(obs[hanging_envs] == 2.0)

    def test_watchdog_vec_env_reset_process(self):
        make_functions = [SlowEnv] * 5
        vec_env = WatchdogVecEnv(
            env_fns=make_functions,
            step_timeout_sec=1.0,
            reset_process_on_env_reset=True,
        )
        reset_obs = vec_env.reset()
        assert reset_obs.shape == (5, 1)
        assert all(reset_obs == -1.0)

        # Same setup as test before, that Env 2 was alive for one step, and all others for 2.
        passing_envs = [0, 1, 3, 4]
        hanging_envs = [2]
        obs, _, _, _ = vec_env.step([0.1, 0.2, 1.5, 0.3, 0.1])
        obs, _, _, _ = vec_env.step([0.1, 0.2, 0.5, 0.3, 0.1])
        assert all(obs[passing_envs] == 2.0)
        assert all(obs[hanging_envs] == 1.0)

        # Resetting the env should kill all processes and restart them -> life_time_counter == 1 after one step
        # Normal reset does not reset the process or the life_time_counter
        reset_obs = vec_env.reset()
        assert reset_obs.shape == (5, 1)
        assert all(reset_obs == -1.0)
        obs, _, _, _ = vec_env.step([0.1, 0.2, 0.5, 0.3, 0.1])
        assert all(obs[passing_envs] == 1.0)
        assert all(obs[hanging_envs] == 1.0)
