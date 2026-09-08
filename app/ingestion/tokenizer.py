import tiktoken


class TokenCounter:

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
    ):
        self.encoder = tiktoken.get_encoding(
            encoding_name
        )

    def count(
        self,
        text: str,
    ) -> int:

        return len(
            self.encoder.encode(text)
        )