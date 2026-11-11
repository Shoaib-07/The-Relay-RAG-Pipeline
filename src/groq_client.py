import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

class GroqModel:
    def __init__(self, model=None):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is missing. Check your .env file.")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.client = Groq(api_key=api_key)

    def answer(self, messages, max_tokens=1200, temperature=0.2):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            reasoning_effort="low",
            reasoning_format="hidden",
        )
        choice = response.choices[0]
        message = choice.message
        usage = getattr(response, "usage", None)

        # Keep diagnostic metadata: an empty content field must not be mistaken
        # for a valid answer. Reasoning text is not used as the user-facing answer.
        return {
            "text": message.content or "",
            "reasoning": getattr(message, "reasoning", None),
            "tokens_in": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "tokens_out": getattr(usage, "completion_tokens", 0) if usage else 0,
            "model": self.model,
            "finish_reason": choice.finish_reason,
        }
