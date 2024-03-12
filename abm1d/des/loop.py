import contextvars
import heapq
from asyncio import AbstractEventLoop, Future, Task, TimerHandle, ensure_future, events
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeAlias, TypeVarTuple, Unpack

ExceptionHandlerT: TypeAlias = Callable[[AbstractEventLoop, dict[str, Any]], Any]
ArgsT = TypeVarTuple("ArgsT")


class VirtualEventLoop(AbstractEventLoop):
    def __init__(self) -> None:
        self._time = 0.0
        self._running = False
        self._debug = False
        self._queue: list[TimerHandle] = []
        self._exception: BaseException | None = None
        self._exception_handler: ExceptionHandlerT | None = None

    def run_forever(self) -> None:
        try:
            self._running = True
            self._exception = None
            events._set_running_loop(self)

            while self._running and self._queue:
                if self._exception is not None:
                    raise self._exception

                handle = heapq.heappop(self._queue)
                self._time = handle.when()
                if not handle.cancelled():
                    handle._run()
        finally:
            self._running = False
            self._exception = None
            events._set_running_loop(None)

    def run_until_complete(self, future: Awaitable[Any]) -> Any:  # type: ignore[override]
        future = ensure_future(future, loop=self)
        future.add_done_callback(self._stop_callback)
        try:
            self.run_forever()
        finally:
            future.remove_done_callback(self._stop_callback)
        return future.result()

    def _stop_callback(self, *_, **__) -> None:
        self.stop()

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running

    def is_closed(self) -> bool:
        return not self.is_running()

    def close(self) -> None:
        self.stop()

    async def shutdown_asyncgens(self) -> None:
        pass

    async def shutdown_default_executor(self) -> None:
        pass

    def _timer_handle_cancelled(self, handle: TimerHandle) -> None:
        pass

    def call_later(
        self,
        delay: float,
        callback: Callable[[Unpack[ArgsT]], Any],
        *args: Unpack[ArgsT],
        context: contextvars.Context | None = None,
    ) -> TimerHandle:
        return self.call_at(self._time + delay, callback, *args, context=context)

    def call_at(
        self,
        when: float,
        callback: Callable[[Unpack[ArgsT]], Any],
        *args: Unpack[ArgsT],
        context: contextvars.Context | None = None,
    ) -> TimerHandle:
        if when < self._time:
            raise ValueError("Cannot schedule callback in the past")
        handle = TimerHandle(when, callback, args, loop=self, context=context)
        heapq.heappush(self._queue, handle)
        return handle

    def time(self) -> float:
        return self._time

    def create_future(self) -> Future:
        return Future(loop=self)

    def create_task(  # type: ignore[override]
        self,
        coro: Coroutine,
        *,
        name: Any = None,
        context: contextvars.Context | None = None,
    ) -> Task:
        async def wrapper() -> Any:
            try:
                return await coro
            except BaseException as e:
                self._exception = e

        return Task(wrapper(), loop=self, name=name, context=context)

    def get_exception_handler(self) -> ExceptionHandlerT | None:
        return self._exception_handler

    def set_exception_handler(self, handler: ExceptionHandlerT | None) -> None:
        self._exception_handler = handler

    def default_exception_handler(self, context: dict[str, Any]) -> None:
        self._exception = context.get("exception")

    def call_exception_handler(self, context: dict[str, Any]) -> None:
        if self._exception_handler:
            self._exception_handler(self, context)
        else:
            self.default_exception_handler(context)

    def get_debug(self) -> bool:
        return self._debug

    def set_debug(self, enabled: bool) -> None:
        self._debug = enabled
