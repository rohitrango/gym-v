from __future__ import annotations

import asyncio
import base64
from collections.abc import Iterable
import json
import threading
import time
from typing import Any

import requests


def run_coroutine(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return run_coroutine_in_thread(coro)


def run_coroutine_in_thread(coro):
    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except BaseException as exc:
            error.append(exc)
        finally:
            loop.close()

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result.get("value")


def decode_data_url(value: str) -> bytes:
    _, _, b64 = value.partition(",")
    return base64.b64decode(b64)

class BaseNetworkClient:
    def __init__(
        self,
        *,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
        backoff_factor: float = 1.0,
        concurrency: int = 1,
        session: requests.Session | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.headers = headers
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.concurrency = concurrency
        self._session = session or requests.Session()

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        attempts = max(self.max_retries, 1)
        last_exc: Exception | None = None
        headers = self.headers
        for attempt in range(attempts):
            try:
                if headers:
                    resp = self._session.post(
                        self.endpoint,
                        json=payload,
                        headers=headers,
                        timeout=self.timeout_s,
                    )
                else:
                    resp = self._session.post(
                        self.endpoint, json=payload, timeout=self.timeout_s
                    )
                resp.raise_for_status()
                data = resp.json()
                if headers and "choices" not in data:
                    raise RuntimeError(f"No choices in response: {data}")
                return data
            except Exception as exc:
                last_exc = exc
                if attempt + 1 >= attempts:
                    raise
                sleep_s = self.backoff_factor * (2**attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
        raise last_exc or RuntimeError("Request failed")

    async def request_async(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.request, payload)

    async def request_many(
        self,
        payloads: Iterable[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        payload_list = list(payloads)

        concurrency = self.concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def _run(payload: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                return await self.request_async(payload)

        tasks = [_run(payload) for payload in payload_list]
        return await asyncio.gather(*tasks)

    def close(self) -> None:
        self._session.close()
