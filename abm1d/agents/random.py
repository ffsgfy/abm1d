from __future__ import annotations

import enum
import random
from typing import assert_never

from abm1d import utils
from abm1d.exchange import OrderSide
from abm1d.common import PeriodicMarketAgent, TraderAgent


class RandomAgent(TraderAgent, PeriodicMarketAgent):
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
        price_delta_std: float,
        min_amount: float,
        max_amount: float,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._actions = list(action_weights.keys())
        self._action_cum_weights = utils.cum_weights(action_weights.values())
        self._positions = list(position_weights.keys())
        self._position_cum_weights = utils.cum_weights(position_weights.values())
        self.price_delta_std = price_delta_std
        self.min_amount = min_amount
        self.max_amount = max_amount

    def _sample_side(self) -> OrderSide:
        return random.choice(tuple(OrderSide))

    def _sample_amount(self) -> float:
        return random.uniform(self.min_amount, self.max_amount)

    def _sample_price(
        self, side: OrderSide, best_bid_price: float, best_ask_price: float
    ) -> float:
        position = utils.weighted_choice(self._positions, self._position_cum_weights)
        match position:
            case self.Position.INSIDE_SPREAD:
                return random.uniform(best_bid_price, best_ask_price)
            case self.Position.OUTSIDE_SPREAD:
                delta = random.expovariate(1.0 / self.price_delta_std)
                match side:
                    case OrderSide.BID:
                        return best_bid_price - delta
                    case OrderSide.ASK:
                        return best_ask_price + delta
                    case _ as never:
                        assert_never(never)
            case _ as never:
                assert_never(never)

    async def _action_market(self) -> None:
        if not self.exchange.bids or not self.exchange.asks:
            return
        self.exchange.market_order(
            self.account, self._sample_side(), self._sample_amount()
        )

    async def _action_limit(self) -> None:
        if self.exchange.bids and self.exchange.asks:
            side = self._sample_side()
            best_bid_price = self.exchange.bids[0].price
            best_ask_price = self.exchange.asks[0].price
        elif self.exchange.bids:
            side = OrderSide.ASK
            best_bid_price = self.exchange.bids[0].price + utils.machine_epsilon()
            best_ask_price = best_bid_price
        elif self.exchange.asks:
            side = OrderSide.BID
            best_ask_price = self.exchange.asks[0].price - utils.machine_epsilon()
            best_bid_price = best_ask_price
        else:
            return
        self.exchange.limit_order(
            self.account,
            side,
            self._sample_price(side, best_bid_price, best_ask_price),
            self._sample_amount(),
        )

    async def _action_cancel(self) -> None:
        orders = list(self.account.orders)
        if orders:
            self.exchange.cancel_order(random.choice(orders))

    async def run(self) -> None:
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
