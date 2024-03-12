from __future__ import annotations

import enum
import math
import random
from typing import assert_never

from abm1d import utils
from abm1d.common import PeriodicMarketAgent, ScalarIndicator, TraderAgent
from abm1d.exchange import OrderSide


class ChartistAgent(TraderAgent, PeriodicMarketAgent):
    class Action(enum.Enum):
        MARKET = enum.auto()
        LIMIT = enum.auto()
        CANCEL = enum.auto()
        NOOP = enum.auto()

    class Position(enum.Enum):
        INSIDE_SPREAD = enum.auto()
        OUTSIDE_SPREAD = enum.auto()

    def __init__(
        self,
        *,
        action_weights: dict[Action, float],
        position_weights: dict[Position, float],
        indicator_weights: dict[ScalarIndicator, float],
        price_delta_std: float,
        min_amount: float,
        max_amount: float,
        sentiment_p0: float,  # probability for score=0
        sentiment_p1: float,  # probability for score=1
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._actions = list(action_weights.keys())
        self._action_cum_weights = utils.cum_weights(action_weights.values())
        self._positions = list(position_weights.keys())
        self._position_cum_weights = utils.cum_weights(position_weights.values())
        # NOTE: all indicators must return values in range [-1, 1]
        self.indicator_weights = indicator_weights.copy()
        self.price_delta_std = price_delta_std
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.sentiment_p0 = utils.check_probability(sentiment_p0)
        self.sentiment_p0 = max(self.sentiment_p0, utils.machine_epsilon())
        self.sentiment_p1 = utils.check_probability(sentiment_p1)
        self.sentiment_p1 = max(self.sentiment_p1, self.sentiment_p0)
        self.sentiment = random.choice((-1.0, 1.0))

    async def on_attach(self) -> None:
        await super().on_attach()
        self.sim.sentiment += self.sentiment
        self.sim.sentiment_limit += 1.0

    async def on_detach(self) -> None:
        await super().on_detach()
        self.sim.sentiment -= self.sentiment
        self.sim.sentiment_limit -= 1.0

    def _sample_side(self) -> OrderSide:
        if self.sentiment > 0.0:
            return OrderSide.BID
        if self.sentiment < 0.0:
            return OrderSide.ASK
        return random.choice(tuple(OrderSide))  # should never happen

    def _sample_amount(self) -> float:
        return random.uniform(self.min_amount, self.max_amount)

    def _sample_price(
        self, side: OrderSide, best_bid_price: float, best_ask_price: float
    ) -> float:
        match side:
            case OrderSide.BID:
                best_price = best_bid_price
                sign = -1.0
            case OrderSide.ASK:
                best_price = best_ask_price
                sign = 1.0
            case _ as never_side:
                assert_never(never_side)

        position = utils.weighted_choice(self._positions, self._position_cum_weights)
        match position:
            case self.Position.INSIDE_SPREAD:
                price = random.uniform(best_bid_price, best_ask_price)
            case self.Position.OUTSIDE_SPREAD:
                delta = random.expovariate(1.0 / self.price_delta_std)
                price = best_price + sign * delta
            case _ as never_position:
                assert_never(never_position)

        return price * (1.0 + sign * self.exchange.commission)

    async def _update_sentiment(self) -> None:
        indicator_values: list[float] = []
        for indicator in self.indicator_weights.keys():
            value = indicator.value()
            if value is None:
                return
            indicator_values.append(value)

        score = utils.weighted_mean(indicator_values, self.indicator_weights.values())
        score *= math.copysign(1.0, -self.sentiment)

        # Map score (in range [-1, 1]) to probability (in range [0, sentiment_p1])
        log_p0 = math.log(self.sentiment_p0)
        log_p1 = math.log(self.sentiment_p1)
        c2 = log_p0 - log_p1
        c1 = -2.0 * c2
        prob = math.exp(c2 * score * score + c1 * score + log_p0)

        if random.random() < prob:
            self.sentiment *= -1.0
            self.sim.sentiment += 2.0 * self.sentiment

    async def _action_market(self) -> None:
        self.exchange.market_order(
            self.account, self._sample_side(), self._sample_amount()
        )

    async def _action_limit(self) -> None:
        side = self._sample_side()
        self.exchange.limit_order(
            self.account,
            side,
            self._sample_price(
                side, self.exchange.bid_price(), self.exchange.ask_price()
            ),
            self._sample_amount(),
        )

    async def _action_cancel(self) -> None:
        orders = list(self.account.orders)
        if orders:
            self.exchange.cancel_order(random.choice(orders))

    async def run(self) -> None:
        if not self.exchange.bids or not self.exchange.asks:
            return

        await self._update_sentiment()

        action = utils.weighted_choice(self._actions, self._action_cum_weights)
        match action:
            case self.Action.MARKET:
                await self._action_market()
            case self.Action.LIMIT:
                await self._action_limit()
            case self.Action.CANCEL:
                await self._action_cancel()
            case self.Action.NOOP:
                pass
            case _ as never:
                assert_never(never)
