import os
import asyncio
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
import torch
from services import GteEmbedService
from controllers import EmbedController
from fastapi.middleware.cors import CORSMiddleware


port = int(os.getenv("PORT", "8000"))

service = GteEmbedService()
controller = EmbedController(service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(service.LoadModel)

    try:
        if (
            torch.cuda.is_available()
            and getattr(__import__("services").GteEmbedService, "keepaliveTask", None)
            is None
        ):
            loop = asyncio.get_running_loop()
            __import__("services").GteEmbedService.keepaliveTask = loop.create_task(
                service.GpuKeepAlive()
            )
    except Exception:
        pass

    try:
        yield
    finally:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)
app.include_router(controller.router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        loop="uvloop",
        http="httptools",
    )
