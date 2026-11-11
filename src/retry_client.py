
import asyncio
import httpx


class RetryClient:
    def __init__(
        self,
        base_url="http://127.0.0.1:8001",
        max_retries=2,
        base_backoff=0.5,
        timeout=8.0,
    ):
        self.url = f"{base_url}/chat"
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        self.timeout = timeout

    async def request(self, question):
        attempts = 0
        history = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            while True:
                try:
                    response = await client.post(
                        self.url,
                        json={"question": question, "stream": False},
                    )

                    if response.status_code == 200:
                        return {
                            "status": "ok",
                            "answer": response.text,
                            "retries": attempts,
                            "history": history,
                        }

                    history.append(response.status_code)

                    if response.status_code not in (429, 503):
                        return {
                            "status": (
                                "rate_limited"
                                if response.status_code == 429
                                else "error"
                            ),
                            "answer": "",
                            "retries": attempts,
                            "history": history,
                        }

                    if attempts >= self.max_retries:
                        return {
                            "status": (
                                "rate_limited"
                                if response.status_code == 429
                                else "error"
                            ),
                            "answer": "",
                            "retries": attempts,
                            "history": history,
                        }

                except httpx.TimeoutException:
                    history.append("timeout")
                    return {
                        "status": "timeout",
                        "answer": "",
                        "retries": attempts,
                        "history": history,
                    }

                except httpx.HTTPError as exc:
                    history.append(str(exc))
                    return {
                        "status": "error",
                        "answer": "",
                        "retries": attempts,
                        "history": history,
                    }

                delay = self.base_backoff * (2 ** attempts)
                await asyncio.sleep(delay)
                attempts += 1
