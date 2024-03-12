import asyncio
import functools
import math
import random
from typing import Iterable, TypeVar
from collections.abc import Sequence

from abm1d.utils import history  # noqa: F401

T = TypeVar("T")

_background_tasks: set[asyncio.Task] = set()


def background_task(task: asyncio.Task) -> asyncio.Task:
    if task not in _background_tasks:
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    return task


def current_time() -> float:
    return asyncio.get_running_loop().time()


@functools.cache
def machine_epsilon() -> float:
    eps = 1.0
    while True:
        eps_half = eps * 0.5
        if 1.0 + eps_half > 1.0:
            eps = eps_half
        else:
            break
    return eps


def check_probability(prob: float) -> float:
    if prob < 0.0 or prob > 1.0:
        raise RuntimeError(f"Invalid probability: {prob}")
    return prob


def cum_weights(weights: Iterable[float]) -> list[float]:
    result = list(weights)
    if not result:
        raise RuntimeError("Empty weights not allowed")
    for i, w in enumerate(result):
        if w < 0.0:
            raise RuntimeError(f"Invalid weight: {w}")
        if i > 0:
            result[i] += result[i - 1]
    return result


def weighted_choice(choices: Sequence[T], cum_weights: Sequence[float]) -> T:
    return random.choices(choices, cum_weights=cum_weights, k=1)[0]


def weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    product_sum = 0.0
    weight_sum = 0.0
    for value, weight in zip(values, weights):
        product_sum += value * weight
        weight_sum += weight
    if math.isclose(weight_sum, 0.0):
        raise RuntimeError("Weights must not sum to zero")
    return product_sum / weight_sum


def fundamental_value(dividends: Sequence[float], risk_free: float) -> float:
    if not dividends:
        raise RuntimeError("Empty dividends not allowed")

    result = 0.0
    discount = 1.0
    discount_mult = 1.0 + risk_free

    # Known dividend payments
    for i in range(len(dividends) - 1):
        dividend = dividends[i]
        discount *= discount_mult
        result += dividend / discount

    # Perpetual dividend payments
    result += dividends[-1] / discount / risk_free
    return result
