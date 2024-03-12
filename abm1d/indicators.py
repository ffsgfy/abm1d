import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from abm1d import utils
from abm1d.common import Indicator, ScalarIndicator, HistoryIndicator


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
        return self._wrap(np.sum(returns) / np.sum(np.abs(returns)))


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
    def __init__(self, *, prices: HistoryIndicator, **kwargs) -> None:
        super().__init__(target=prices, **kwargs)

    def window_values(self, window: pd.DataFrame) -> dict[str, float]:
        return self._wrap(window["mid"].std())
