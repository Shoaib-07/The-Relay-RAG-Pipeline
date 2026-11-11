import httpx

class StandInClient:
    def __init__(self, base_url="http://127.0.0.1:8001", timeout=8.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def answer(self, question: str, stream=False):
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat",
                json={"question": question, "stream": stream},
            )
            if response.status_code == 429:
                return {"status": "rate_limited", "text": "", "tokens_in": 0, "tokens_out": 0}
            response.raise_for_status()
            return {
                "status": "ok",
                "text": response.text,
                "tokens_in": 0,
                "tokens_out": 0,
            }
