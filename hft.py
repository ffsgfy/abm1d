import datetime
import random
import itertools

from tqdm import tqdm

from abm1d import events
from abm1d.agents import (
    ChartistAgent,
    FundamentalistAgent,
    MarketMakerAgent,
    RandomAgent,
)
from abm1d.common import EnvironmentAgent, MarketSimulation
from abm1d.exchange import Account, Exchange
from abm1d.indicators import (
    ChandeMomentum,
    HistoricalVolatility,
    MarketPrices,
    MarketReturn,
    PearsonCorrelation,
    SentimentIndex,
)
from abm1d.utils import history

seed = 1186796360058043791
outpath = f"out/{datetime.datetime.now().replace(microsecond=0).isoformat()}.csv"

with open(outpath, "w") as file:
    file.write("advantage,time_dt,price_dt\n")


def simulate(
    n_random: int,
    n_fund: int,
    n_chart: int,
    n_mm_slow: int,
    n_mm_fast: int,
    advantage: float,
):
    global seed, outpath
    random.seed(seed)

    simulation_time = 500.0
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
    environment.pregenerate_dividends(int(simulation_time))
    sim = MarketSimulation(exchange=exchange, environment=environment)

    for _ in range(n_random):
        sim.attach_agent(RandomAgent(
            action_weights={
                RandomAgent.Action.MARKET: 0.2,
                RandomAgent.Action.LIMIT: 0.4,
                RandomAgent.Action.CANCEL: 0.2,
                RandomAgent.Action.NOOP: 0.2,
            },
            position_weights={
                RandomAgent.Position.INSIDE_SPREAD: 0.3,
                RandomAgent.Position.OUTSIDE_SPREAD: 0.7,
            },
            price_delta_mean=2.5,
            min_amount=0.0,
            max_amount=3.0,
            account=Account(quote=1000.0),
            period=0.5,
            jitter=0.25,
        ))

    for _ in range(n_fund):
        sim.attach_agent(FundamentalistAgent(
            action_weights={
                FundamentalistAgent.Action.ORDER: 0.6,
                FundamentalistAgent.Action.CANCEL: 0.4,
            },
            price_delta_mean=2.5,
            amount_gamma=0.005,
            min_amount=0.0,
            max_amount=5.0,
            account=Account(quote=1000.0),
            period=1.0,
            jitter=0.5,
        ))

    ind_prices = sim.track_indicator(MarketPrices(sim=sim), 1.0)
    ind_sentiment = SentimentIndex(sim=sim)
    ind_chande = sim.track_indicator(
        ChandeMomentum(prices=ind_prices, window=5.0, sim=sim), ind_prices
    )
    ind_correlation = sim.track_indicator(
        PearsonCorrelation(prices=ind_prices, window=13.0, strict=False, sim=sim),
        ind_prices,
    )

    for _ in range(n_chart):
        sim.attach_agent(ChartistAgent(
            action_weights={
                ChartistAgent.Action.MARKET: 0.2,
                ChartistAgent.Action.LIMIT: 0.4,
                ChartistAgent.Action.CANCEL: 0.2,
                ChartistAgent.Action.NOOP: 0.2,
            },
            position_weights={
                ChartistAgent.Position.INSIDE_SPREAD: 0.3,
                ChartistAgent.Position.OUTSIDE_SPREAD: 0.7,
            },
            indicator_weights={
                ind_sentiment: 1.0,
                ind_chande: 1.5,
                ind_correlation: 0.8,
            },
            price_delta_mean=2.5,
            min_amount=0.0,
            max_amount=5.0,
            sentiment_p0=0.4,
            sentiment_p1=0.9,
            account=Account(quote=1000.0),
            period=1.0,
            jitter=0.5,
        ))

    slow_period = 2.0
    slow_prices = sim.track_indicator(MarketPrices(sim=sim), slow_period * 0.5)
    slow_volatility = sim.track_indicator(
        HistoricalVolatility(
            target=slow_prices,
            field="mid",
            window=slow_period * 5.0,
            strict=False,
            sim=sim,
        ),
        slow_prices,
    )

    for _ in range(n_mm_slow):
        sim.attach_agent(MarketMakerAgent(
            amount_limit=2.0,
            volatility=slow_volatility,
            account=Account(quote=1000.0),
            period=slow_period,
            jitter=slow_period * 0.5,
        ))

    fast_period = slow_period / advantage
    fast_prices = sim.track_indicator(MarketPrices(sim=sim), fast_period * 0.5)
    fast_volatility = sim.track_indicator(
        HistoricalVolatility(
            target=fast_prices,
            field="mid",
            window=fast_period * 5.0,
            strict=False,
            sim=sim,
        ),
        fast_prices,
    )

    for _ in range(n_mm_fast):
        sim.attach_agent(MarketMakerAgent(
            amount_limit=2.0,
            volatility=fast_volatility,
            account=Account(quote=1000.0),
            period=fast_period,
            jitter=fast_period * 0.5,
        ))

    ind_return = sim.track_indicator(
        MarketReturn(prices=ind_prices, sim=sim), ind_prices
    )
    ind_return_volatility = sim.track_indicator(
        HistoricalVolatility(target=ind_return, field="value", window=8.0, sim=sim),
        ind_return,
    )

    shock_time = simulation_time * 0.5
    shock_price_dt_pct = -0.10

    shock_target = 0.0
    shock_price = 0.0

    async def price_shock() -> None:
        nonlocal sim, shock_price, shock_target, shock_price_dt_pct
        vol_mean = ind_return_volatility.history()["value"].mean()
        vol_std = ind_return_volatility.history()["value"].std()
        shock_target = vol_mean + vol_std
        shock_price = ind_prices.values()["mid"]
        await events.market_price_shock(sim, shock_price * shock_price_dt_pct)

    sim.schedule_absolute(price_shock(), shock_time)
    sim.run(simulation_time)

    eq_time = 0.0
    for _, row in history.slice_absolute(
        ind_return_volatility.history(), shock_time, None
    ).iterrows():
        if row["timestamp"] <= shock_time:
            continue
        if row["value"] <= shock_target:
            eq_time = row["timestamp"]
            break

    if eq_time <= shock_time:
        return

    eq_price_index = history.index_absolute(ind_prices.history(), eq_time)
    eq_price = ind_prices.history().iloc[eq_price_index]["mid"]
    eq_price_dt_pct = (eq_price - shock_price) / shock_price
    eq_time_dt = eq_time - shock_time

    with open(outpath, "a") as outfile:
        outfile.write(f"{advantage},{eq_time_dt},{eq_price_dt_pct}\n")


rng_random = range(4, 10)
rng_fund = range(4, 10)
rng_chart = range(4, 10)
rng_mm_slow = range(2, 4)
n_mm_fast = 5

combos = list(itertools.product(rng_random, rng_fund, rng_chart, rng_mm_slow))
for advantage in (1.0, 2.0, 4.0, 7.0, 10.0):
    for n_random, n_fund, n_chart, n_mm_slow in tqdm(combos, desc=str(advantage)):
        simulate(n_random, n_fund, n_chart, n_mm_slow, n_mm_fast, advantage)
