import time
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from abm1d.exchange import Exchange, Account
from abm1d.common import (
    MarketSimulation,
    EnvironmentAgent,
)
from abm1d.indicators import (
    SentimentIndex,
    ChandeMomentum,
    PearsonCorrelation,
    HistoricalVolatility,
    MarketPrices,
    MarketDepth,
)
from abm1d.agents import (
    FundamentalistAgent,
    ChartistAgent,
    MarketMakerAgent,
    RandomAgent,
)

seed = hash(time.time())
# seed = 125738502195212317
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
environment.pregenerate_dividends(1000)
sim = MarketSimulation(exchange=exchange, environment=environment)

hist_prices = sim.track_indicator(MarketPrices(sim=sim), 1.0)
hist_depth = sim.track_indicator(MarketDepth(sim=sim), 1.0)

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
random_count = 10
for _ in range(random_count):
    account = Account(quote=1000.0)
    agent = RandomAgent(
        action_weights=random_action_weights,
        position_weights=random_position_weights,
        price_delta_std=2.5,
        min_amount=1.0,
        max_amount=5.0,
        account=account,
        period=1.0,
        jitter=0.5,
    )
    sim.attach_agent(agent)
    last_random = agent

# Fundamentalist
fundamentalist_action_weights = {
    FundamentalistAgent.Action.ORDER: 0.6,
    FundamentalistAgent.Action.CANCEL: 0.4,
}
fundamentalist_count = 20
for _ in range(fundamentalist_count):
    account = Account(quote=1000.0)
    agent = FundamentalistAgent(
        action_weights=fundamentalist_action_weights,
        price_delta_std=2.5,
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
    ChandeMomentum(prices=hist_prices, window=5.0, sim=sim), 1.0
)
hist_correlation = sim.track_indicator(
    PearsonCorrelation(prices=hist_prices, window=13.0, strict=False, sim=sim), 1.0
)
chartist_indicator_weights = {
    inst_sentiment: 1.0,
    hist_chande: 2.0,
    hist_correlation: 0.8,
}
chartist_count = 15
for _ in range(chartist_count):
    account = Account(quote=1000.0)
    agent = ChartistAgent(
        action_weights=chartist_action_weights,
        position_weights=chartist_position_weights,
        indicator_weights=chartist_indicator_weights,
        price_delta_std=2.5,
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

hist_volatility = sim.track_indicator(
    HistoricalVolatility(prices=hist_prices, window=8.0, strict=False, sim=sim), 1.0
)
marketmaker_count = 5
for _ in range(marketmaker_count):
    account = Account(quote=1000.0)
    agent = MarketMakerAgent(
        amount_limit=5.0,
        volatility=hist_volatility,
        account=account,
        period=2.0,
        jitter=1.0,
    )
    sim.attach_agent(agent)
    last_marketmaker = agent

duration = 500.0
sim.run(duration)

_, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, height_ratios=(5.0, 1.5, 1.0))
ax1.plot(
    hist_prices.history()["timestamp"],
    hist_prices.history()["mid"],
    label="Mid",
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
    label="Fun",
    color="blue",
    alpha=0.5,
)
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
    hist_depth.history()["timestamp"],
    hist_depth.history()["total"],
    label="Market depth",
)
ax3.legend()

plt.show()
