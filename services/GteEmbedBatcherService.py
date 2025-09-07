import asyncio
from typing import List, Any, Callable

class GteEmbedBatcherService:
    def __init__(
        self,
        callFn: Callable[[List[str]], List[Any]],
        maxBatchSize: int = 16,
        maxDelayMs: int = 10,
    ):
        self.callFn = callFn
        self.maxBatchSize = maxBatchSize
        self.maxDelay = maxDelayMs / 1000.0
        self.queue: Any = asyncio.Queue()
        self.task = None 

    async def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._runLoop())

    async def submit(self, texts: List[str]) -> List[Any]:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self.queue.put((texts, future))
        return await future

    async def _runLoop(self):
        while True:
            allTexts: List[str] = []
            allFutures: List[Any] = []

            texts, future = await self.queue.get()
            allTexts.extend(texts)
            allFutures.append((len(texts), future))

            try:
                while len(allTexts) < self.maxBatchSize:
                    texts, future = await asyncio.wait_for(
                        self.queue.get(), timeout=self.maxDelay
                    )
                    allTexts.extend(texts)
                    allFutures.append((len(texts), future))
            except asyncio.TimeoutError:
                pass

            try:
                results = await asyncio.to_thread(self.callFn, allTexts)
                index = 0
                for count, future in allFutures:
                    part = results[index : index + count]
                    index += count
                    future.set_result(part)
            except Exception as error:
                for _, future in allFutures:
                    future.set_exception(error)