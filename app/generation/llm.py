from dataclasses import dataclass

from groq import Groq

from app.core.config import settings


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


def get_llm() -> GroqLLM:
    return GroqLLM(
        api_key=settings.llm_api_key,
        model=settings.llm_model,
    )