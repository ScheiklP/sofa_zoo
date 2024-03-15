import selectors
import time
import gymnasium as gym
import numpy as np
import multiprocessing as mp

from datetime import datetime
from typing import Callable, List, Optional
from collections import defaultdict

from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv, VecEnvObs, VecEnvStepReturn
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv, _flatten_obs, _worker


class WatchdogVecEnv(SubprocVecEnv):
    """Variant of Stable Baseline 3's SubprocVecEnv that closes and restarts environment processes that hang during a step.

    This VecEnv features a watchdog in its asynchronous step function to reset environments that take longer than
    ``step_timeout_sec`` for one step. This might happen when unstable deformations cause SOFA to hang.

    Resetting a SOFA scene that features topological changes such as removing/cutting tetrahedral elements does not
    restore the initial number of elements in the meshes. Manually removing and adding elements to SOFA's simulation
    tree technically works, but is sometimes quite unreliable and prone to memory leaks. This VecEnv avoids this
    problem by creating a completely new environment with simulation, if a reset signal is sent to the environment.

    Notes:
        If an environment is reset by the step watchdog, the returned values for this environment will be:
        ``reset_obs, 0.0, True, defaultdict(float), reset_info``. Meaning it returs the reset observation, a reward of 0.0, a done signal,
        an empty info dict that defaults to returning 0.0, if a key is accessed and the reset_info dict. The ``defaultdict`` is used to prevent
        crashing the ``VecMonitor`` when accessing the info dict.

    Args:
        env_fns (List[Callable[[], gymnasium.Env]]): List of environment constructors.
        step_timeout_sec (Optional[float]): Timeout in seconds for a single step. If a step takes longer than this
            timeout, the environment will be reset. If ``None``, no timeout is used.
        reset_process_on_env_reset (bool): Additionally to hanging envs, close and restart the process of envs at every reset.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        step_timeout_sec: Optional[float] = None,
        reset_process_on_env_reset: bool = False,
    ) -> None:
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        # forkserver is way too slow since we need to start a new process on
        # every reset
        ctx = mp.get_context("fork")

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            # daemon=True: if the main process crashes, we should not cause things to hang
            process = ctx.Process(target=_worker, args=args, daemon=True)  # pytype:disable=attribute-error
            process.start()
            self.processes.append(process)
            # do not close work_remote to prevent it being garbage collected

        self.ctx = ctx
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

        self.env_fns = env_fns
        self.step_timeout = step_timeout_sec
        self.reset_process_on_env_reset = reset_process_on_env_reset

    def step_wait(self) -> VecEnvStepReturn:
        hanging_envs = []
        if self.step_timeout is not None:
            # wait for all remotes to finish
            successes = wait_all(self.remotes, timeout=self.step_timeout)
            if len(successes) != len(self.remotes):
                hanging_envs = [i for i, remote in enumerate(self.remotes) if remote not in successes]
                for i in hanging_envs:
                    print(f"Environment {i} is hanging and will be terminated and restarted " f"({datetime.now().strftime('%H:%M:%S')})")
                    self.processes[i].terminate()  # terminate worker
                    # clear any data in the pipe
                    while self.remotes[i].poll():
                        self.remotes[i].recv()
                    # start new worker, seed, and reset it
                    self._restart_process(i)

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        # any environments that were just reset will send an extra message that
        # must be consumed.
        # in addition, the observation and done state and reset info must be updated
        for i in hanging_envs:
            # Return of reset is (obs, reset_info)
            reset_obs, reset_info = results[i]
            # Result order: obs, reward, done, info, reset_info: See class SubProcVecEnv
            results[i] = (reset_obs, 0.0, True, defaultdict(float), reset_info)

        obs, rews, dones, infos, self.reset_infos = zip(*results)

        if self.reset_process_on_env_reset:
            obs = list(obs)  # convert to list to allow modification
            self.reset_infos = list(self.reset_infos)  # convert to list to allow modification
            for i, (done, remote, process) in enumerate(zip(dones, self.remotes, self.processes)):
                if done and i not in hanging_envs:  # do not double-reset environments that were hanging
                    remote.send(("close", None))  # command worker to stop
                    process.join()  # wait for worker to stop
                    # start new worker, seed, and reset it
                    self._restart_process(i)
                    obs[i], self.reset_infos[i] = remote.recv()  # collect reset observation

        return _flatten_obs(obs, self.observation_space), np.stack(rews), np.stack(dones), infos

    def _restart_process(self, i: int) -> None:
        """Restarts the worker process ``i`` with its original ``env_fn``. The
        original pipe is reused. The new environment is seeded and reset, but
        the reset observation is *not* yet collected from the pipe.
        """
        work_remote, remote = self.work_remotes[i], self.remotes[i]
        # start new worker
        args = (work_remote, remote, CloudpickleWrapper(self.env_fns[i]))
        process = self.ctx.Process(target=_worker, args=args, daemon=True)
        process.start()

        # reseed and reset new env
        remote.send(("reset", (self._seeds[i], self._options[i])))

        self.processes[i] = process

    def reset(self) -> VecEnvObs:
        if self.reset_process_on_env_reset:
            # command environments to shut down
            for remote in self.remotes:
                remote.send(("close", None))  # command worker to stop
            for process in self.processes:
                process.join()  # wait for worker to stop
            # start new workers, seed, and reset them
            for i in range(len(self.processes)):
                self._restart_process(i)
            results = [remote.recv() for remote in self.remotes]
            obs, self.reset_infos = zip(*results)
            # Seeds and options are only used once
            self._reset_seeds()
            self._reset_options()
            return _flatten_obs(obs, self.observation_space)
        else:
            return super().reset()

    def close(self) -> None:
        super().close()
        for remote in self.remotes:
            remote.close()  # close pipe


# poll/select have the advantage of not requiring any extra file
# descriptor, contrarily to epoll/kqueue (also, they require a single
# syscall).
if hasattr(selectors, "PollSelector"):
    _WaitSelector = selectors.PollSelector
else:
    _WaitSelector = selectors.SelectSelector


def wait_all(object_list, timeout: Optional[float] = None):
    """
    Wait till all objects in ``object_list`` are ready/readable, or the timeout expires.

    Adapted from ``multiprocessing.connection.wait`` in the standard library.

    Args:
        object_list (list): list of objects to wait for. E.g. a list of pipes.
        timeout (float): timeout in seconds. If ``None``, wait forever.

    Returns:
        list: list of objects in ``object_list`` which are ready/readable.
    """
    with _WaitSelector() as selector:
        for obj in object_list:
            selector.register(obj, selectors.EVENT_READ)

        if timeout is not None:
            deadline = time.monotonic() + timeout

        all_ready = []
        while True:
            ready = selector.select(timeout)
            ready = [key.fileobj for (key, events) in ready]

            all_ready.extend(ready)
            for obj in ready:
                selector.unregister(obj)

            if len(all_ready) == len(object_list):
                return all_ready
            else:
                if timeout is not None:
                    timeout = deadline - time.monotonic()
                    if timeout < 0:
                        return all_ready
