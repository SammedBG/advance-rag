from dataclasses import dataclass, field


@dataclass
class StructuredQuery:
    original_query: str

    intent: str

    keywords: list[str] = field(
        default_factory=list
    )

    technical_terms: list[str] = field(
        default_factory=list
    )

    entities: list[str] = field(
        default_factory=list
    )

    filters: dict[str, str] = field(
        default_factory=dict
    )

    language: str = "en"