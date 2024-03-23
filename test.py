import random
import time

import matplotlib.pyplot as plt
import numpy as np

from abm1d import events
from abm1d.agents import (
    ChartistAgent,
    FundamentalistAgent,
    MarketMakerAgent,
    RandomAgent,
)
from abm1d.common import Agent, EnvironmentAgent, MarketSimulation
from abm1d.exchange import Account, Exchange
from abm1d.indicators import (
    ChandeMomentum,
    HistoricalVolatility,
    MarketDepth,
    MarketPrices,
    MarketReturn,
    PearsonCorrelation,
    SentimentIndex,
)
from abm1d.utils import history

seed = hash(time.time())
seed = 1186796360058043791
print("Seed:", seed)
random.seed(seed)

risk_free = 0.0005
initial_price = 100.0
exchange = Exchange()
exchange.random_orders(1000, initial_price, 25.0, 1.0, 5.0)
environment = EnvironmentAgent(
    risk_free=risk_free,
    dividend_initial=initial_price * risk_free,
    dividend_mult_std=0.005,
    dividend_access=1,
)
environment.pregenerate_dividends(500)
sim = MarketSimulation(exchange=exchange, environment=environment)
agent: Agent

hist_depth = sim.track_indicator(MarketDepth(sim=sim), 1.0)
hist_prices = sim.track_indicator(MarketPrices(sim=sim), 1.0)
hist_return = sim.track_indicator(
    MarketReturn(prices=hist_prices, sim=sim), hist_prices
)
hist_return_vol = sim.track_indicator(
    HistoricalVolatility(
        target=hist_return, field="value", window=8.0, strict=False, sim=sim
    ),
    hist_return,
)

# Random
random_action_weights = {
    RandomAgent.Action.MARKET: 0.2,
    RandomAgent.Action.LIMIT: 0.4,
    RandomAgent.Action.CANCEL: 0.2,
    RandomAgent.Action.NOOP: 0.2,
}
random_position_weights = {
    RandomAgent.Position.INSIDE_SPREAD: 0.3,
    RandomAgent.Position.OUTSIDE_SPREAD: 0.7,
}
random_count = 5
for _ in range(random_count):
    account = Account(quote=1000.0)
    agent = RandomAgent(
        action_weights=random_action_weights,
        position_weights=random_position_weights,
        price_delta_mean=2.5,
        min_amount=0.0,
        max_amount=3.0,
        account=account,
        period=0.5,
        jitter=0.25,
    )
    sim.attach_agent(agent)
    last_random = agent

# Fundamentalist
fundamentalist_action_weights = {
    FundamentalistAgent.Action.ORDER: 0.6,
    FundamentalistAgent.Action.CANCEL: 0.4,
}
fundamentalist_count = 5
for _ in range(fundamentalist_count):
    account = Account(quote=1000.0)
    agent = FundamentalistAgent(
        action_weights=fundamentalist_action_weights,
        price_delta_mean=2.5,
        amount_gamma=0.005,
        min_amount=0.0,
        max_amount=5.0,
        account=account,
        period=1.0,
        jitter=0.5,
    )
    sim.attach_agent(agent)
    last_fundamentalist = agent

# Chartist
chartist_action_weights = {
    ChartistAgent.Action.MARKET: 0.2,
    ChartistAgent.Action.LIMIT: 0.4,
    ChartistAgent.Action.CANCEL: 0.2,
    ChartistAgent.Action.NOOP: 0.2,
}
chartist_position_weights = {
    ChartistAgent.Position.INSIDE_SPREAD: 0.3,
    ChartistAgent.Position.OUTSIDE_SPREAD: 0.7,
}
inst_sentiment = SentimentIndex(sim=sim)
hist_sentiment = sim.track_indicator(inst_sentiment, 1.0)
hist_chande = sim.track_indicator(
    ChandeMomentum(prices=hist_prices, window=5.0, sim=sim), hist_prices
)
hist_correlation = sim.track_indicator(
    PearsonCorrelation(prices=hist_prices, window=13.0, strict=False, sim=sim),
    hist_prices,
)
chartist_indicator_weights = {
    inst_sentiment: 1.0,
    hist_chande: 1.5,
    hist_correlation: 0.8,
}
chartist_count = 5
for _ in range(chartist_count):
    account = Account(quote=1000.0)
    agent = ChartistAgent(
        action_weights=chartist_action_weights,
        position_weights=chartist_position_weights,
        indicator_weights=chartist_indicator_weights,
        price_delta_mean=2.5,
        min_amount=0.0,
        max_amount=5.0,
        sentiment_p0=0.4,
        sentiment_p1=0.9,
        account=account,
        period=1.0,
        jitter=0.5,
    )
    sim.attach_agent(agent)
    last_chartist = agent

mm_period = 2.0
mm_hist_prices = sim.track_indicator(MarketPrices(sim=sim), mm_period * 0.5)
mm_hist_volatility = sim.track_indicator(
    HistoricalVolatility(
        target=mm_hist_prices,
        field="mid",
        window=mm_period * 5.0,
        strict=False,
        sim=sim,
    ),
    mm_hist_prices,
)
marketmaker_count = 7
for _ in range(marketmaker_count):
    account = Account(quote=1000.0)
    agent = MarketMakerAgent(
        amount_limit=2.0,
        volatility=mm_hist_volatility,
        account=account,
        period=mm_period,
        jitter=mm_period * 0.5,
    )
    sim.attach_agent(agent)
    last_marketmaker = agent

duration = 500.0
event_timestamp = 250.0
sim.schedule_absolute(events.market_price_shock(sim, -10.0), event_timestamp)
sim.run(duration)

# shock_price_dt_pct = -0.10
# shock_target = 0.0
# shock_price = 0.0
#
#
# async def price_shock() -> None:
#     global sim, shock_price, shock_target, shock_price_dt_pct
#     vol_mean = hist_return_vol.history()["value"].mean()
#     vol_std = hist_return_vol.history()["value"].std()
#     shock_target = vol_mean + vol_std
#     shock_price = hist_prices.values()["mid"]
#     await events.market_price_shock(sim, shock_price * shock_price_dt_pct)
#
#
# sim.schedule_absolute(price_shock(), event_timestamp)
# sim.run(duration)
#
# eq_time = 0.0
# for _, row in history.slice_absolute(
#     hist_return_vol.history(), event_timestamp, None
# ).iterrows():
#     if row["timestamp"] <= event_timestamp:
#         continue
#     if row["value"] <= shock_target:
#         eq_time = row["timestamp"]
#         break
#
# if eq_time >= event_timestamp:
#     eq_price_index = history.index_absolute(hist_prices.history(), eq_time)
#     eq_price = hist_prices.history().iloc[eq_price_index]["mid"]
#     eq_price_dt_pct = (eq_price - shock_price) / shock_price
#     eq_time_dt = eq_time - event_timestamp
#     print(f"{eq_time_dt},{eq_price_dt_pct}")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, height_ratios=(5.0, 1.5, 1.0))
ax1.plot(
    hist_prices.history()["timestamp"],
    hist_prices.history()["mid"],
    label="Mid price",
    color="k",
)
ax1.fill_between(
    hist_prices.history()["timestamp"],
    hist_prices.history()["bid"],
    hist_prices.history()["ask"],
    color="gray",
    alpha=0.3,
)
ax1.plot(
    hist_prices.history()["timestamp"],
    hist_prices.history()["fun"],
    label="Fundamental value",
    color="blue",
    alpha=0.5,
)
ax1.axvline(event_timestamp, color="k", linestyle="--", alpha=0.5)
ax1.legend()

ax2.axhline(y=0.0, color="k", linestyle="--")
ax2.plot(
    hist_sentiment.history()["timestamp"],
    hist_sentiment.history()["value"],
    label="Sentiment",
)
ax2.plot(
    hist_chande.history()["timestamp"], hist_chande.history()["value"], label="Chande"
)
ax2.plot(
    hist_correlation.history()["timestamp"],
    hist_correlation.history()["value"],
    label="Correlation",
)
ax2.legend()

ax3.plot(
    hist_return_vol.history()["timestamp"],
    hist_return_vol.history()["value"],
    label="Return volatility"
)
# ax3.bar(
#     hist_return.history()["timestamp"],
#     hist_return.history()["value"].abs(),
#     label="Absolute returns",
# )
ax3.legend()

plt.show()
