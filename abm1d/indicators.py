import math
import warnings
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats

from abm1d import utils
from abm1d.common import HistoryIndicator, Indicator, ScalarIndicator


class MarketPrices(Indicator):
    def fields(self) -> list[str]:
        return ["bid", "ask", "mid", "fun"]

    def values(self) -> dict[str, float]:
        bid_price = self.sim.exchange.bid_price()
        ask_price = self.sim.exchange.ask_price()
        return {
            "bid": bid_price,
            "ask": ask_price,
            "mid": (bid_price + ask_price) * 0.5,
            "fun": self.sim.environment.fundamental_value,
        }


class MarketDepth(Indicator):
    def fields(self) -> list[str]:
        return ["bid", "ask", "total"]

    def values(self) -> dict[str, float]:
        bid_depth = self.sim.exchange.bid_depth()
        ask_depth = self.sim.exchange.ask_depth()
        return {
            "bid": bid_depth,
            "ask": ask_depth,
            "total": bid_depth + ask_depth,
        }


class MarketReturn(ScalarIndicator):
    def __init__(self, *, prices: HistoryIndicator, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prices = prices

    def values(self) -> dict[str, float]:
        history = self.prices.history()
        if len(history) > 1:
            return self._wrap(history.iloc[-1]["mid"] - history.iloc[-2]["mid"])
        return self._wrap(0.0)


class SentimentIndex(ScalarIndicator):
    def values(self) -> dict[str, float]:
        if self.sim.sentiment_limit > 0.0:
            return self._wrap(self.sim.sentiment / self.sim.sentiment_limit)
        return self._wrap(0.0)


class WindowIndicator(Indicator):
    def __init__(
        self, *, target: HistoryIndicator, window: float, strict: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.target = target
        self.window = max(window, utils.machine_epsilon())
        self.strict = strict

    def window_values(self, window: pd.DataFrame) -> dict[str, float] | None:
        raise NotImplementedError()

    def values(self) -> dict[str, float] | None:
        history = self.target.history()
        if len(history) == 0:
            return None
        window = utils.history.slice_relative(history, -self.window, None)
        if len(window) == 0:
            return None  # should never happen
        if self.strict:
            span = window.iloc[-1]["timestamp"] - window.iloc[0]["timestamp"]
            if span < self.window:
                return None
        return self.window_values(window)


class ChandeMomentum(ScalarIndicator, WindowIndicator):
    def __init__(self, *, prices: HistoryIndicator, **kwargs) -> None:
        super().__init__(target=prices, **kwargs)

    def window_values(self, window: pd.DataFrame) -> dict[str, float]:
        prices = window["mid"]
        returns = prices.array[1:] - prices.array[:-1]
        total = np.sum(np.abs(returns))
        if math.isclose(total, 0.0):
            return self._wrap(0.0)
        return self._wrap(np.sum(returns) / total)


class PearsonCorrelation(ScalarIndicator, WindowIndicator):
    def __init__(self, *, prices: HistoryIndicator, **kwargs) -> None:
        super().__init__(target=prices, **kwargs)

    def window_values(self, window: pd.DataFrame) -> dict[str, float]:
        if len(window) < 2:
            return self._wrap(0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", stats.DegenerateDataWarning)
            result = stats.pearsonr(window["timestamp"], window["mid"]).statistic
        return self._wrap(result if math.isfinite(result) else 0.0)


class HistoricalVolatility(ScalarIndicator, WindowIndicator):
    def __init__(self, *, target: HistoryIndicator, field: str, **kwargs) -> None:
        super().__init__(target=target, **kwargs)
        self.field = field

    def window_values(self, window: pd.DataFrame) -> dict[str, float]:
        return self._wrap(window[self.field].std())


class ExponentialSmoothing(ScalarIndicator):
    def __init__(
        self,
        *,
        target: Indicator,
        field: str,
        alpha: float,
        step: float = 1.0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.target = target
        self.field = field
        self.alpha = min(max(alpha, 0.0), 1.0)
        self.step = max(step, utils.machine_epsilon())
        self._timestamp = math.inf
        self._value = 0.0

    def values(self) -> dict[str, float] | None:
        values = self.target.values()
        if not values:
            return None

        value = values[self.field]
        curtime = utils.current_time()
        timedelta = curtime - self._timestamp
        if timedelta < 0.0 or self.alpha >= 1.0:
            self._value = value
        else:
            momentum = (1.0 - self.alpha) ** (timedelta / self.step)
            self._value = self._value * momentum + value * (1.0 - momentum)

        self._timestamp = curtime
        return self._wrap(self._value)


class ScalarFunction(ScalarIndicator):
    def __init__(
        self, *, target: Callable[..., float], args: Iterable[ScalarIndicator], **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.target = target
        self.args = list(args)

    def values(self) -> dict[str, float] | None:
        args = [x.value() for x in self.args]
        if None in args:
            return None
        return self._wrap(self.target(*args))
