from app.context.compressor import CompressedContext


class PromptBuilder:
    SYSTEM_PROMPT = """You are a documentation question-answering assistant.

Answer the user's question using ONLY the provided documentation context.

Rules:
1. Do not use outside knowledge.
2. Do not invent or assume information.
3. If the answer is not supported by the context, say:
   "I could not find this information in the provided documentation."
4. Give a concise and technically accurate answer.
5. When the context contains commands, configuration, error messages, or code, preserve them accurately.
6. Cite the relevant context sources using [1], [2], etc.
"""

    def build(
        self,
        query: str,
        contexts: list[CompressedContext],
    ) -> tuple[str, str]:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if not contexts:
            raise ValueError("At least one context is required.")

        context_parts: list[str] = []

        for index, context in enumerate(contexts, start=1):
            heading = " > ".join(context.heading_path)

            source = (
                f"[{index}]\n"
                f"Document: {context.title}\n"
                f"Section: {heading}\n"
                f"Content:\n{context.content}"
            )

            context_parts.append(source)

        context_text = "\n\n".join(context_parts)

        user_prompt = f"""Documentation Context:

{context_text}

User Question:
{query}

Answer the question using only the documentation context above.
Include source citations such as [1] or [2] for the claims you make.
"""

        return self.SYSTEM_PROMPT, user_prompt