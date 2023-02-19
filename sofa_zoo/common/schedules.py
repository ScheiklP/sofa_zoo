from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        progress_remaining will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def exponential_schedule(initial_value: float, gamma: float, iterations: int) -> Callable[[float], float]:
    """
    Exponential learning rate schedule.

    :param initial_value: Initial learning rate.
    :param gamma: decay factor
    :param iterations: total number of iterations to convert progress_remaining into a step

    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        progress_remaining will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """

        current_iteration = int((1.0 - progress_remaining) * iterations)

        return initial_value * gamma ** current_iteration

    return func
