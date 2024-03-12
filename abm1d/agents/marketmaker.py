from __future__ import annotations

import math

from abm1d import utils
from abm1d.common import TraderAgent, PeriodicMarketAgent, ScalarIndicator


class MarketMakerAgent(TraderAgent, PeriodicMarketAgent):
    def __init__(
        self, *, amount_limit: float, volatility: ScalarIndicator, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if amount_limit < 0.0:
            raise RuntimeError("Amount limit must not be negative")

        self.amount_limit = amount_limit
        self.volatility = volatility

    async def run(self) -> None:
        for order in list(self.account.orders):
            self.exchange.cancel_order(order)

        if not self.exchange.bids or not self.exchange.asks:
            return

        amount_current = self.account.base
        if (abs(amount_current) > self.amount_limit) or math.isclose(
            abs(amount_current), self.amount_limit
        ):
            if amount_current < 0.0:
                self.exchange.market_bid(self.account, -amount_current)
            else:
                self.exchange.market_ask(self.account, amount_current)
        else:
            volatility = self.volatility.value()
            if volatility is None:
                return

            best_bid_price = self.exchange.bid_price()
            best_ask_price = self.exchange.ask_price()
            mid_price = (best_bid_price + best_ask_price) * 0.5
            offset = max(volatility, utils.machine_epsilon())
            anchor = mid_price - offset * (amount_current / self.amount_limit)
            self.exchange.limit_bid(
                self.account, anchor - offset, self.amount_limit - amount_current
            )
            self.exchange.limit_ask(
                self.account, anchor + offset, self.amount_limit + amount_current
            )
