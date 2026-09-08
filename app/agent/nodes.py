import re

from app.agent.state import AgentState
from app.context.compressor import ContextCompressor
from app.context.parent_expander import ParentExpander
from app.context.selector import ContextSelector
from app.generation.citation import CitationValidator
from app.generation.grounding import GroundingValidator
from app.generation.service import GenerationService
from app.mcp.service import MCPService
from app.retrieval.hybrid import HybridSearch


class AgentDependencies:
    def __init__(
        self,
        search_service: HybridSearch,
        parent_expander: ParentExpander,
        context_selector: ContextSelector,
        context_compressor: ContextCompressor,
        generation_service: GenerationService,
        grounding_validator: GroundingValidator,
        mcp_service: MCPService,
    ) -> None:
        self.search_service = search_service
        self.parent_expander = parent_expander
        self.context_selector = context_selector
        self.context_compressor = context_compressor
        self.generation_service = generation_service
        self.grounding_validator = grounding_validator
        self.mcp_service = mcp_service


def route_node(state: AgentState) -> AgentState:
    query = state["query"].strip().lower()

    if not query:
        raise ValueError("Query cannot be empty.")

    mcp_patterns = [
        r"\bget\b.*\bpod\b.*\bstatus\b",
        r"\bcurrent\b.*\bpod\b.*\bstatus\b",
        r"\bcheck\b.*\bpod\b",
        r"\bshow\b.*\bpod\b.*\bstatus\b",
        r"\bpod\b.*\bstatus\b",
        r"\bpod\b.*\blogs?\b",
        r"\brunning\b.*\bpods?\b",
        r"\bdeployment\b.*\bstatus\b",
    ]

    for pattern in mcp_patterns:
        if re.search(pattern, query):
            state["route"] = "mcp"
            return state

    state["route"] = "rag"
    return state


def retrieve_node(
    state: AgentState,
    dependencies: AgentDependencies,
) -> AgentState:
    query = state["query"]

    reranked_results = dependencies.search_service.search(
        query=query,
        limit=5,
        retrieval_limit=20,
    )

    expanded_results = dependencies.parent_expander.expand(
        reranked_results
    )

    selected_contexts = dependencies.context_selector.select(
        expanded_results
    )

    compressed_contexts = dependencies.context_compressor.compress(
        query=query,
        contexts=selected_contexts,
    )

    state["compressed_contexts"] = compressed_contexts

    state["context_stats"] = {
        "contexts_selected": len(selected_contexts),
        "contexts_compressed": len(compressed_contexts),
        "selected_tokens": sum(
            context.token_count
            for context in selected_contexts
        ),
        "compressed_tokens": sum(
            context.compressed_token_count
            for context in compressed_contexts
        ),
    }

    return state


def generate_node(
    state: AgentState,
    dependencies: AgentDependencies,
) -> AgentState:
    query = state["query"]
    contexts = state.get("compressed_contexts", [])

    if not contexts:
        state["answer"] = (
            "I could not find this information "
            "in the provided documentation."
        )

        state["citations"] = []

        state["citation_validation"] = {
            "valid": False,
            "invalid_citations": [],
            "missing_citations": True,
        }

        state["grounding"] = {
            "score": 0.0,
            "grounded": False,
            "matched_terms": [],
            "unmatched_terms": [],
        }

        return state

    generated_answer = dependencies.generation_service.generate(
        query=query,
        contexts=contexts,
    )

    citation_validator = CitationValidator()

    citation_validation = citation_validator.validate(
        answer=generated_answer.answer,
        context_count=len(contexts),
    )

    grounding_validation = (
        dependencies.grounding_validator.validate(
            answer=generated_answer.answer,
            contexts=[
                context.content
                for context in contexts
            ],
        )
    )

    state["answer"] = generated_answer.answer

    state["citations"] = citation_validation.citations

    state["citation_validation"] = {
        "valid": citation_validation.valid,
        "invalid_citations": (
            citation_validation.invalid_citations
        ),
        "missing_citations": (
            citation_validation.missing_citations
        ),
    }

    state["grounding"] = {
        "score": grounding_validation.score,
        "grounded": grounding_validation.grounded,
        "matched_terms": (
            grounding_validation.matched_terms
        ),
        "unmatched_terms": (
            grounding_validation.unmatched_terms
        ),
    }

    return state


def mcp_node(
    state: AgentState,
    dependencies: AgentDependencies,
) -> AgentState:
    query = state["query"].strip()

    pod_name = _extract_pod_name(query)

    if not pod_name:
        pod_name = "example-pod"

    result = dependencies.mcp_service.get_pod_status(
        pod_name=pod_name,
        namespace="default",
    )

    if not result.success:
        state["error"] = (
            result.error
            or "MCP tool failed."
        )

        state["answer"] = (
            "The MCP tool could not retrieve "
            "the requested data."
        )

        state["mcp_result"] = {
            "success": False,
            "tool": result.tool_name,
            "error": result.error,
        }

        return state

    data = result.content

    if not isinstance(data, dict):
        state["error"] = (
            "MCP tool returned an unexpected "
            "response format."
        )

        state["answer"] = (
            "The MCP tool returned an unexpected "
            "response format."
        )

        state["mcp_result"] = {
            "success": False,
            "tool": result.tool_name,
            "error": state["error"],
            "raw_data": data,
        }

        return state

    pod_name = str(
        data.get(
            "pod_name",
            pod_name,
        )
    )

    namespace = str(
        data.get(
            "namespace",
            "default",
        )
    )

    status = str(
        data.get(
            "status",
            "unknown",
        )
    )

    ready = data.get(
        "ready",
        False,
    )

    restarts = data.get(
        "restarts",
        0,
    )

    state["mcp_result"] = {
        "success": True,
        "tool": result.tool_name,
        "data": data,
    }

    state["answer"] = (
        f"Pod '{pod_name}' in namespace "
        f"'{namespace}' is currently "
        f"{status}. "
        f"Ready: {ready}. "
        f"Restarts: {restarts}."
    )

    state["citations"] = []
    state["citation_validation"] = {
        "valid": True,
        "invalid_citations": [],
        "missing_citations": False,
    }

    state["grounding"] = {
        "score": 1.0,
        "grounded": True,
        "matched_terms": [],
        "unmatched_terms": [],
    }

    return state


def _extract_pod_name(
    query: str,
) -> str | None:
    patterns = [
        r"\bpod\s+status\s+for\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bpod\s+status\s+of\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bstatus\s+for\s+pod\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bstatus\s+of\s+pod\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bpod\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)\s+status\b",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            query,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(1)

    return None