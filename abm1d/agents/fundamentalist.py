from __future__ import annotations

import enum
import random
from typing import assert_never

from abm1d import utils
from abm1d.common import PeriodicMarketAgent, TraderAgent
from abm1d.exchange import OrderSide


class FundamentalistAgent(TraderAgent, PeriodicMarketAgent):
    class Action(enum.Enum):
        ORDER = enum.auto()
        CANCEL = enum.auto()

    def __init__(
        self,
        *,
        action_weights: dict[Action, float],
        price_delta_mean: float,
        amount_gamma: float,
        min_amount: float,
        max_amount: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._actions = list(action_weights.keys())
        self._action_cum_weights = utils.cum_weights(action_weights.values())
        self.price_delta_mean = price_delta_mean
        self.amount_gamma = amount_gamma
        self.min_amount = min_amount
        self.max_amount = max_amount

    def _sample_side(self) -> OrderSide:
        return random.choice(tuple(OrderSide))

    def _sample_price(self, side: OrderSide, price_fun: float) -> float:
        delta = random.expovariate(1.0 / self.price_delta_mean)
        match side:
            case OrderSide.BID:
                return (price_fun - delta) * (1.0 - self.exchange.commission)
            case OrderSide.ASK:
                return (price_fun + delta) * (1.0 + self.exchange.commission)
            case _ as never:
                assert_never(never)

    async def _action_order(self) -> None:
        price_mid = (self.exchange.bid_price() + self.exchange.ask_price()) * 0.5
        price_fun = self.environment.fundamental_value
        amount = abs(price_fun - price_mid) / price_mid / self.amount_gamma
        amount = min(max(amount, self.min_amount), self.max_amount)
        target_bid = self.exchange.bid_price() * (1.0 - self.exchange.commission)
        target_ask = self.exchange.ask_price() * (1.0 + self.exchange.commission)
        side = self._sample_side()

        if (price_fun >= target_ask and side == OrderSide.BID) or (
            price_fun <= target_bid and side == OrderSide.ASK
        ):
            self.exchange.market_order(self.account, side, amount)
        else:
            self.exchange.limit_order(
                self.account, side, self._sample_price(side, price_fun), amount
            )

    async def _action_cancel(self) -> None:
        orders = list(self.account.orders)
        if orders:
            self.exchange.cancel_order(random.choice(orders))

    async def run(self) -> None:
        if not self.exchange.bids or not self.exchange.asks:
            return

        action = utils.weighted_choice(self._actions, self._action_cum_weights)
        match action:
            case self.Action.ORDER:
                await self._action_order()
            case self.Action.CANCEL:
                await self._action_cancel()
            case _ as never:
                assert_never(never)
