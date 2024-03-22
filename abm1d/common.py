from __future__ import annotations

import asyncio
import itertools
import math
import random
from collections import deque
from collections.abc import Awaitable
from typing import cast, overload

import pandas as pd

from abm1d import utils
from abm1d.des import Agent, Entity, PeriodicAgent, Simulation, VirtualEventLoop
from abm1d.exchange import Account, Exchange


class MarketSimulation(Simulation):
    def __init__(
        self,
        *,
        exchange: Exchange,
        environment: EnvironmentAgent,
        loop: VirtualEventLoop | None = None,
    ) -> None:
        super().__init__(loop=loop)

        self.exchange = exchange
        self.environment = environment
        self.attach_agent(self.environment)

        self.history: dict[Indicator, pd.DataFrame] = {}
        self.history_agents: dict[Indicator, HistoryAgent] = {}
        self.sentiment = 0.0
        self.sentiment_limit = 0.0

    @overload
    def track_indicator(
        self, indicator: ScalarIndicator, period: float | HistoryIndicator
    ) -> ScalarHistoryIndicator:
        ...

    @overload
    def track_indicator(
        self, indicator: Indicator, period: float | HistoryIndicator
    ) -> HistoryIndicator:
        ...

    def track_indicator(
        self, indicator: Indicator, period: float | HistoryIndicator,
    ) -> HistoryIndicator:
        if indicator in self.history:
            raise RuntimeError("Indicator already being tracked")

        if isinstance(period, HistoryIndicator):
            agent = self.history_agents[period.indicator]
        else:
            agent = HistoryAgent(period=float(period))
            self.attach_agent(agent)

        agent.indicators.append(indicator)
        self.history[indicator] = utils.history.make(indicator.fields())
        self.history_agents[indicator] = agent

        if isinstance(indicator, ScalarIndicator):
            return ScalarHistoryIndicator(indicator=indicator, sim=self)
        else:
            return HistoryIndicator(indicator=indicator, sim=self)

    def schedule_relative(self, event: Awaitable, delay: float) -> asyncio.Task:
        async def wrapper():
            try:
                await asyncio.sleep(delay)
                await event
            except asyncio.CancelledError:
                pass

        return utils.background_task(self.loop.create_task(wrapper()))

    def schedule_absolute(self, event: Awaitable, when: float) -> asyncio.Task:
        delay = when - self.loop.time()
        if delay < 0.0:
            raise ValueError("Cannot schedule event in the past")
        return self.schedule_relative(event, delay)


class MarketAgent(Agent):
    def _set_sim(self, sim: Simulation | None) -> None:
        if sim is not None:
            if not isinstance(sim, MarketSimulation):
                raise RuntimeError(f"Invalid simulation type: {type(sim).__name__}")
        super()._set_sim(sim)

    @property
    def sim(self) -> MarketSimulation:
        return cast(MarketSimulation, super().sim)

    @property
    def exchange(self) -> Exchange:
        return self.sim.exchange

    @property
    def environment(self) -> EnvironmentAgent:
        return self.sim.environment


class PeriodicMarketAgent(PeriodicAgent, MarketAgent):
    pass


class EnvironmentAgent(PeriodicMarketAgent):
    def __init__(
        self,
        *,
        risk_free: float,
        dividend_initial: float,
        dividend_mult_std: float,
        dividend_access: int,
        locked_dividends: bool = True,  # pay out dividends on locked base assets
        locked_interest: bool = True,  # pay out interest on locked quote assets
        **kwargs,
    ) -> None:
        # NOTE: always exactly one tick per unit of time
        super().__init__(period=1.0, jitter=0.0, eager=False, **kwargs)

        if dividend_initial < utils.machine_epsilon():
            raise RuntimeError(f"Invalid initial dividend: {dividend_initial}")

        self.risk_free = risk_free
        self.dividends = deque([dividend_initial])
        self.dividend_mult_std = dividend_mult_std
        self.dividend_access = dividend_access
        self.locked_dividends = locked_dividends
        self.locked_interest = locked_interest

        self.fundamental_value = 0.0
        self.pregenerate_dividends(self.dividend_access)
        self._update_fundamental_value()

    def _sample_next_dividend(self) -> float:
        last = self.dividends[-1]
        mult = math.exp(random.normalvariate(0.0, self.dividend_mult_std))
        return max(last * mult, utils.machine_epsilon())

    def _update_fundamental_value(self) -> None:
        dividends = list(itertools.islice(self.dividends, self.dividend_access))
        self.fundamental_value = utils.fundamental_value(dividends, self.risk_free)

    def pregenerate_dividends(self, count: int) -> None:
        while len(self.dividends) < count:
            self.dividends.append(self._sample_next_dividend())

    async def run(self) -> None:
        # Pay out interest and dividends
        dividend = self.dividends[0]
        for agent in self.sim.agents:
            if not isinstance(agent, TraderAgent):
                continue

            account = agent.account
            assets_base = account.base
            if self.locked_dividends:
                assets_base += account.base_locked
            assets_quote = account.quote
            if self.locked_interest:
                assets_quote += account.quote_locked

            account.quote += assets_base * dividend
            account.quote += assets_quote * self.risk_free

        # Shift the dividend window
        self.dividends.append(self._sample_next_dividend())
        self.dividends.popleft()
        self._update_fundamental_value()


class Indicator(Entity):
    def __init__(self, *, sim: MarketSimulation, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sim = sim

    def fields(self) -> list[str]:
        raise NotImplementedError()

    def values(self) -> dict[str, float] | None:
        raise NotImplementedError()


class ScalarIndicator(Indicator):
    def fields(self) -> list[str]:
        return ["value"]

    def value(self) -> float | None:
        if values := self.values():
            return values["value"]
        return None

    def _wrap(self, value: float) -> dict[str, float]:
        return {"value": value}


class HistoryIndicator(Indicator):
    def __init__(self, *, indicator: Indicator, **kwargs) -> None:
        super().__init__(**kwargs)
        self.indicator = indicator

    def history(self) -> pd.DataFrame:
        return self.sim.history[self.indicator]

    def fields(self) -> list[str]:
        return self.indicator.fields()

    def values(self) -> dict[str, float] | None:
        history = self.history()
        if len(history) > 0:
            return self.history().iloc[-1].to_dict()
        return None


class ScalarHistoryIndicator(ScalarIndicator, HistoryIndicator):
    def __init__(self, *, indicator: ScalarIndicator, **kwargs) -> None:
        super().__init__(indicator=indicator, **kwargs)


class HistoryAgent(PeriodicMarketAgent):
    def __init__(self, *, period: float, **kwargs) -> None:
        # NOTE: no jitter in history
        super().__init__(period=period, jitter=0.0, eager=True, **kwargs)
        self.indicators: list[Indicator] = []

    async def run(self) -> None:
        for indicator in self.indicators:
            if values := indicator.values():
                utils.history.append(
                    self.sim.history[indicator], utils.current_time(), values
                )


class TraderAgent(Agent):
    def __init__(self, *, account: Account, **kwargs) -> None:
        super().__init__(**kwargs)
        self.account = account
