import asyncio
from typing import List, Tuple, Any, Callable
from implementations import CrossEncoderRerankerBatcherServiceImpl


class CrossEncoderRerankerBatcherService(CrossEncoderRerankerBatcherServiceImpl):
    def __init__(
        self,
        callFn: Callable[[List[Tuple[str, str]]], List[float]],
        maxBatchSize: int = 50,
        maxDelayMs: int = 5,
    ):
        self.callFn = callFn
        self.maxBatchSize = maxBatchSize
        self.maxDelay = maxDelayMs / 1000.0
        self.queue: Any = asyncio.Queue()
        self.task = None

    async def start(self) -> None:
        if self.task is None:
            self.task = asyncio.create_task(self._runLoop())

    async def submit(self, pairs: List[Tuple[str, str]]) -> List[Any]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((pairs, future))
        return await future

    async def _runLoop(self) -> None:
        while True:
            allPairs: List[Tuple[str, str]] = []
            allFutures: List[Any] = []

            pairs, future = await self.queue.get()
            allPairs.extend(pairs)
            allFutures.append((len(pairs), future))

            try:
                while len(allPairs) < self.maxBatchSize:
                    pairs, future = await asyncio.wait_for(
                        self.queue.get(), timeout=self.maxDelay
                    )
                    allPairs.extend(pairs)
                    allFutures.append((len(pairs), future))
            except asyncio.TimeoutError:
                pass

            try:
                results = await asyncio.to_thread(self.callFn, allPairs)
                index = 0
                for count, future in allFutures:
                    part = results[index : index + count]
                    index += count
                    future.set_result(part)
            except Exception as error:
                for _, future in allFutures:
                    future.set_exception(error)
