import asyncio
import random

from abm1d.des.base import Agent


class PeriodicAgent(Agent):
    def __init__(
        self, *, period: float, jitter: float = 0.0, eager: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        if period <= 0.0:
            raise RuntimeError(f"Invalid period: {period}")

        self.period = period
        self.jitter = jitter
        self.eager = eager
        self._clock_task: asyncio.Task | None = None

    async def on_attach(self) -> None:
        await super().on_attach()
        await self._stop_clock()
        await self._start_clock()

    async def on_resume(self) -> None:
        await super().on_resume()
        await self._start_clock()

    async def on_detach(self) -> None:
        await super().on_detach()
        await self._stop_clock()

    async def _start_clock(self) -> None:
        if self._clock_task is not None:
            if not self._clock_task.done():
                return
        self._clock_task = asyncio.create_task(self._clock())

    async def _stop_clock(self) -> None:
        if self._clock_task is not None:
            if not self._clock_task.done():
                self._clock_task.cancel()
                try:
                    await self._clock_task
                except asyncio.CancelledError:
                    pass
        self._clock_task = None

    async def _clock(self) -> None:
        if self.eager:
            await self.run()

        while True:
            jitter = abs(self.jitter)
            jitter = random.uniform(-jitter, jitter)
            await asyncio.sleep(max(self.period + jitter, 0.0))
            await self.run()

    async def run(self) -> None:
        raise NotImplementedError()
