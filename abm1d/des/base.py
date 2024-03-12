from __future__ import annotations

import asyncio
import uuid
from typing import Any, Callable, ClassVar, Hashable

from abm1d.des.loop import VirtualEventLoop


class Entity:
    uid_factory: ClassVar[Callable[[], Hashable]] = uuid.uuid4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.uid = type(self).uid_factory()

    def __hash__(self) -> int:
        return hash(self.uid)

    def __eq__(self, other: Any) -> bool:
        return self.uid == other.uid


class Agent(Entity):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._sim: Simulation | None = None

    def _set_sim(self, sim: Simulation | None) -> None:
        self._sim = sim

    @property
    def sim(self) -> Simulation:
        if self._sim is None:
            raise RuntimeError("Not attached to a simulation")
        return self._sim

    async def on_attach(self) -> None:
        pass

    async def on_resume(self) -> None:
        pass

    async def on_detach(self) -> None:
        pass


class Simulation:
    def __init__(self, *, loop: VirtualEventLoop | None = None) -> None:
        self.loop = loop or VirtualEventLoop()  # type: ignore[abstract]
        self.agents: set[Agent] = set()
        self._pending_agents: dict[Agent, tuple[asyncio.Task, bool]] = {}

    def attach_agent(self, agent: Agent) -> asyncio.Task:
        if agent in self.agents:
            raise RuntimeError("Agent already attached")
        return self._schedule_pending_agent(agent, True)

    def detach_agent(self, agent: Agent) -> asyncio.Task:
        if agent not in self.agents:
            raise RuntimeError("Agent already detached")
        return self._schedule_pending_agent(agent, False)

    def _schedule_pending_agent(self, agent: Agent, flag: bool) -> asyncio.Task:
        if (pending := self._pending_agents.get(agent)) is not None:
            task = pending[0]
            if not task.done():
                self._pending_agents[agent] = (task, flag)
                return task

        task = self.loop.create_task(self._handle_pending_agent(agent))
        self._pending_agents[agent] = (task, flag)
        return task

    async def _handle_pending_agent(self, agent: Agent) -> None:
        while agent in self._pending_agents:
            current_state = agent in self.agents
            desired_state = self._pending_agents[agent][1]
            if current_state == desired_state:
                self._pending_agents.pop(agent)
            elif desired_state:
                agent._set_sim(self)
                await agent.on_attach()
                self.agents.add(agent)
                await agent.on_resume()
            else:
                self.agents.discard(agent)
                await agent.on_detach()
                agent._set_sim(None)

    def run(self, duration: float) -> None:
        for agent in self.agents:
            self.loop.create_task(agent.on_resume())
        self.loop.run_until_complete(asyncio.sleep(duration))
