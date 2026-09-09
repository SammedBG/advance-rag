import logging
from dataclasses import dataclass

from groq import Groq

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GroqLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
    ) -> None:
        if not api_key:
            raise ValueError("LLM API key is not configured.")

        if not model:
            raise ValueError("LLM model is not configured.")

        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> LLMResponse:
        if not system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")

        if not user_prompt.strip():
            raise ValueError("User prompt cannot be empty.")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        choice = response.choices[0]
        usage = response.usage

        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        return LLMResponse(
            answer=choice.message.content or "",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    def generate_stream(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ):
        if not system_prompt.strip():
            raise ValueError("System prompt cannot be empty.")

        if not user_prompt.strip():
            raise ValueError("User prompt cannot be empty.")

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.warning("Streaming fallback due to LLM error: %s", exc)
            fallback = self.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            for word in fallback.answer.split(" "):
                yield word + " "

