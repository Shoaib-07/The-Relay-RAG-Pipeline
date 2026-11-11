
import asyncio
import os
import random
import time
from collections import deque
from typing import Deque

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="The Relay Stand-in Model Service")

# Reproducible configuration through environment variables
BASE_LATENCY_MS = int(os.getenv("STANDIN_BASE_LATENCY_MS", "450"))
JITTER_MS = int(os.getenv("STANDIN_JITTER_MS", "150"))
FAILURE_RATE = float(os.getenv("STANDIN_FAILURE_RATE", "0.02"))
TIMEOUT_RATE = float(os.getenv("STANDIN_TIMEOUT_RATE", "0.01"))
RPM_LIMIT = int(os.getenv("STANDIN_RPM_LIMIT", "60"))
MAX_IN_FLIGHT = int(os.getenv("STANDIN_MAX_IN_FLIGHT", "8"))
LOAD_SLOWDOWN_MS = int(os.getenv("STANDIN_LOAD_SLOWDOWN_MS", "120"))
SIMULATED_DAILY_LIMIT = int(os.getenv("STANDIN_DAILY_LIMIT", "1000"))

request_times: Deque[float] = deque()
request_count_today = 0
in_flight = 0
lock = asyncio.Lock()


class ChatRequest(BaseModel):
    question: str
    stream: bool = False


def cleanup_minute(now: float):
    while request_times and now - request_times[0] >= 60:
        request_times.popleft()


async def release_slot():
    global in_flight
    async with lock:
        in_flight = max(0, in_flight - 1)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rpm_limit": RPM_LIMIT,
        "max_in_flight": MAX_IN_FLIGHT,
        "requests_today": request_count_today,
        "daily_limit": SIMULATED_DAILY_LIMIT,
        "in_flight": in_flight,
    }


@app.post("/chat")
async def chat(body: ChatRequest):
    global request_count_today, in_flight

    now = time.time()

    async with lock:
        cleanup_minute(now)

        if request_count_today >= SIMULATED_DAILY_LIMIT:
            raise HTTPException(
                status_code=429, detail="simulated_daily_limit"
            )

        if len(request_times) >= RPM_LIMIT:
            raise HTTPException(
                status_code=429, detail="simulated_rpm_limit"
            )

        if in_flight >= MAX_IN_FLIGHT:
            raise HTTPException(
                status_code=429, detail="simulated_concurrency_limit"
            )

        request_times.append(now)
        request_count_today += 1
        in_flight += 1
        current_load = in_flight

    slot_released = False

    try:
        # Simulated upstream failure: reject before response starts.
        if random.random() < FAILURE_RATE:
            await asyncio.sleep(0.03)
            raise HTTPException(
                status_code=503, detail="simulated_upstream_error"
            )

        # Simulated timeout. A client with a shorter timeout may
        # disconnect before this server-side timeout is returned.
        if random.random() < TIMEOUT_RATE:
            await asyncio.sleep(30)
            raise HTTPException(
                status_code=504, detail="simulated_upstream_timeout"
            )

        delay_ms = (
            BASE_LATENCY_MS
            + random.randint(0, max(0, JITTER_MS))
            + max(0, current_load - 1) * LOAD_SLOWDOWN_MS
        )

        await asyncio.sleep(delay_ms / 1000)

        answer = (
            "Stand-in response: this endpoint simulates model service "
            "latency, failures, rate limits, overload slowdown and "
            "streaming. It does not perform real language-model reasoning."
        )

        if body.stream:

            async def generate():
                try:
                    for token in answer.split():
                        yield token + " "
                        await asyncio.sleep(0.025)
                finally:
                    await release_slot()

            slot_released = True
            return StreamingResponse(
                generate(), media_type="text/plain"
            )

        return PlainTextResponse(answer)

    finally:
        if not slot_released:
            await release_slot()


@app.post("/admin/reset")
async def reset():
    global request_count_today

    async with lock:
        request_count_today = 0
        request_times.clear()

    return {"status": "reset"}
