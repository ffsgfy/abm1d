from abm1d.common import MarketSimulation
from abm1d.exchange import Order, OrderSide


async def fundamental_value_shock(sim: MarketSimulation, price_delta: float) -> None:
    dividend_delta = price_delta * sim.environment.risk_free
    dividends = sim.environment.dividends
    for i, dividend in enumerate(dividends):
        dividends[i] = dividend + dividend_delta


async def market_price_shock(sim: MarketSimulation, price_delta: float) -> None:
    order_capacities: dict[Order, float] = {}

    for order in list(sim.exchange.bids) + list(sim.exchange.asks):
        order_capacities[order] = order.capacity
        sim.exchange.cancel_order(order)

    for order, capacity in order_capacities.items():
        sim.exchange.limit_order(
            order.account, order.side, order.price + price_delta, capacity
        )


async def liquidity_shock(sim: MarketSimulation, amount: float) -> None:
    side = OrderSide.ASK if amount < 0.0 else OrderSide.BID
    sim.exchange.market_order(sim.exchange.account, side, abs(amount))


async def dividend_access_change(sim: MarketSimulation, access: int) -> None:
    access = max(access, 1)
    sim.environment.dividend_access = access
    sim.environment.pregenerate_dividends(access)
    while len(sim.environment.dividends) > access:
        sim.environment.dividends.pop()


async def commission_change(sim: MarketSimulation, commission: float) -> None:
    sim.exchange.commission = commission
