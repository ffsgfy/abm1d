from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Sequence

from sortedcontainers import SortedList  # type: ignore[import-untyped]

from abm1d import utils
from abm1d.des import Entity


@dataclass
class Account:
    # Amount of base asset held
    base: float = 0.0
    # Amount of base asset locked in limit asks
    base_locked: float = 0.0
    # Amount of quote asset held
    quote: float = 0.0
    # Amount of quote asset locked in limit bids
    quote_locked: float = 0.0
    # Open limit orders (only)
    orders: set[Order] = field(default_factory=set)


class OrderSide(enum.Enum):
    BID = enum.auto()
    ASK = enum.auto()


@dataclass
class OrderData:
    account: Account
    side: OrderSide
    price: float
    amount: float  # requested amount
    capacity: float  # remaining (unfilled) amount

    def is_bid(self) -> bool:
        return self.side == OrderSide.BID

    def is_ask(self) -> bool:
        return self.side == OrderSide.ASK


class Order(Entity, OrderData):
    pass


class Exchange:
    def __init__(self, *, commission: float = 0.0) -> None:
        self.bids: SortedList[Order] = SortedList(key=lambda x: -x.price)
        self.asks: SortedList[Order] = SortedList(key=lambda x: x.price)
        self.commission = commission  # fraction of transaction value
        self.account = Account(base=math.inf, quote=math.inf)

    def _cleanup_order(self, order: Order) -> None:
        if order.is_bid():
            self.bids.discard(order)

        if order.is_ask():
            self.asks.discard(order)

        order.account.orders.discard(order)

    def _create_order(
        self, account: Account, side: OrderSide, price: float, amount: float
    ) -> Order:
        if amount < 0.0:
            raise RuntimeError("Order amount must not be negative")

        order = Order(account, side, price, amount, amount)

        if order.is_bid():
            value = amount * price
            account.quote -= value
            account.quote_locked += value

        if order.is_ask():
            account.base -= amount
            account.base_locked += amount

        return order

    def _fill_order(self, order: Order, price: float, amount: float) -> None:
        if amount > order.capacity:
            raise RuntimeError("Fill amount must not exceed order capacity")

        account = order.account
        value = amount * price

        if order.is_bid():
            locked_value = amount * order.price
            account.base += amount
            account.quote += locked_value - value * (1.0 + self.commission)
            account.quote_locked -= locked_value

        if order.is_ask():
            account.base_locked -= amount
            account.quote += value * (1.0 - self.commission)

        order.capacity -= amount

    def cancel_order(self, order: Order) -> None:
        account = order.account

        if order.is_bid():
            locked_value = order.capacity * order.price
            account.quote += locked_value
            account.quote_locked -= locked_value

        if order.is_ask():
            account.base += order.capacity
            account.base_locked -= order.capacity

        order.capacity = 0.0
        self._cleanup_order(order)

    def market_order(self, account: Account, side: OrderSide, amount: float) -> Order:
        if amount < 0.0:
            raise RuntimeError("Order amount must not be negative")

        book: Sequence[Order] = []
        if side == OrderSide.BID:
            book = self.asks
        if side == OrderSide.ASK:
            book = self.bids

        cleanup: list[Order] = []
        capacity = amount
        value = 0.0

        for order in book:
            part = min(capacity, order.capacity)
            self._fill_order(order, order.price, part)
            if order.capacity <= 0.0:
                cleanup.append(order)

            capacity -= part
            value += part * order.price
            if capacity <= 0.0:
                break

        for x in cleanup:
            self._cleanup_order(x)

        filled = amount - capacity
        if filled == 0.0:
            price = 0.0
        else:
            price = value / filled

        order = self._create_order(account, side, price, filled)
        self._fill_order(order, price, filled)
        return order

    def market_bid(self, account: Account, amount: float) -> Order:
        return self.market_order(account, OrderSide.BID, amount)

    def market_ask(self, account: Account, amount: float) -> Order:
        return self.market_order(account, OrderSide.ASK, amount)

    def _match_orders(self, primary: Order, secondary: Order) -> None:
        if primary.side == secondary.side:
            raise RuntimeError("Cannot match orders on the same side")

        # NOTE: primary order sets the price
        price = primary.price
        amount = min(primary.capacity, secondary.capacity)
        self._fill_order(primary, price, amount)
        self._fill_order(secondary, price, amount)

    def limit_order(
        self, account: Account, side: OrderSide, price: float, amount: float
    ) -> Order:
        order = self._create_order(account, side, price, amount)
        cleanup: list[Order] = []

        if order.is_bid():
            for ask in self.asks:
                if ask.price <= order.price:
                    self._match_orders(ask, order)
                    if ask.capacity <= 0.0:
                        cleanup.append(ask)
                    if order.capacity <= 0.0:
                        break
                else:
                    break

            if order.capacity > 0.0:
                self.bids.add(order)

        if order.is_ask():
            for bid in self.bids:
                if bid.price >= order.price:
                    self._match_orders(bid, order)
                    if bid.capacity <= 0.0:
                        cleanup.append(bid)
                    if order.capacity <= 0.0:
                        break
                else:
                    break

            if order.capacity > 0.0:
                self.asks.add(order)

        for x in cleanup:
            self._cleanup_order(x)

        if order.capacity > 0.0:
            account.orders.add(order)

        return order

    def limit_bid(self, account: Account, price: float, amount: float) -> Order:
        return self.limit_order(account, OrderSide.BID, price, amount)

    def limit_ask(self, account: Account, price: float, amount: float) -> Order:
        return self.limit_order(account, OrderSide.ASK, price, amount)

    def random_orders(
        self,
        count: int,
        price_mean: float,
        price_std: float,
        min_amount: float,
        max_amount: float,
    ) -> list[Order]:
        orders = []

        for i in range(count):
            sign = (i % 2) * 2 - 1
            amount = random.uniform(min_amount, max_amount)
            price = random.normalvariate(price_mean + sign * price_std, price_std)
            price = max(price, utils.machine_epsilon())
            if price > price_mean:
                order = self.limit_ask(self.account, price, amount)
            else:
                order = self.limit_bid(self.account, price, amount)
            orders.append(order)

        return orders

    def bid_price(self) -> float:
        if not self.bids:
            raise RuntimeError("No bids in the order book")
        return self.bids[0].price

    def ask_price(self) -> float:
        if not self.asks:
            raise RuntimeError("No asks in the order book")
        return self.asks[0].price

    def bid_depth(self) -> float:
        return float(sum(x.capacity for x in self.bids))

    def ask_depth(self) -> float:
        return float(sum(x.capacity for x in self.asks))
