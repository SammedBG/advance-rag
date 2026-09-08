from dataclasses import dataclass

from app.context.compressor import CompressedContext
from app.generation.llm import GroqLLM
from app.generation.prompt_builder import PromptBuilder


@dataclass
class GeneratedAnswer:
    answer: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int


class GenerationService:
    def __init__(
        self,
        llm: GroqLLM,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.llm = llm
        self.prompt_builder = prompt_builder

    def generate(
        self,
        query: str,
        contexts: list[CompressedContext],
    ) -> GeneratedAnswer:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not contexts:
            raise ValueError("No contexts available for generation.")

        system_prompt, user_prompt = self.prompt_builder.build(
            query=query,
            contexts=contexts,
        )

        response = self.llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.0,
            max_tokens=1000,
        )

        return GeneratedAnswer(
            answer=response.answer,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )