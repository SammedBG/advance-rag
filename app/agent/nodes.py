import logging
import re
from typing import Any

from app.agent.state import AgentState
from app.context.compressor import ContextCompressor
from app.context.parent_expander import ParentExpander
from app.context.selector import ContextSelector
from app.generation.citation import CitationValidator
from app.generation.grounding import GroundingValidator
from app.generation.service import GenerationService
from app.mcp.service import MCPService
from app.retrieval.hybrid import HybridSearch

logger = logging.getLogger(__name__)


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

    trace = state.get("reasoning_trace", [])
    state["original_query"] = state.get("original_query", state["query"])
    state["iteration_count"] = state.get("iteration_count", 0)
    state["max_iterations"] = state.get("max_iterations", 2)

    # 1. Check for Pod MCP route
    mcp_pod_patterns = [
        r"\bget\b.*\bpod\b.*\bstatus\b",
        r"\bcurrent\b.*\bpod\b.*\bstatus\b",
        r"\bcheck\b.*\bpod\b",
        r"\bshow\b.*\bpod\b.*\bstatus\b",
        r"\bpod\b.*\bstatus\b",
        r"\bpod\b.*\blogs?\b",
        r"\brunning\b.*\bpods?\b",
        r"\bdeployment\b.*\bstatus\b",
    ]
    for pattern in mcp_pod_patterns:
        if re.search(pattern, query):
            state["route"] = "mcp"
            trace.append(f"Query routed to MCP tool 'get_pod_status' based on pattern match.")
            state["reasoning_trace"] = trace
            return state

    # 2. Check for Cluster Health MCP route
    mcp_cluster_patterns = [
        r"\bcluster\b.*\bhealth\b",
        r"\bcluster\b.*\bstatus\b",
        r"\bnode\b.*\bstatus\b",
        r"\bcluster\b.*\bnodes\b",
    ]
    for pattern in mcp_cluster_patterns:
        if re.search(pattern, query):
            state["route"] = "mcp"
            trace.append(f"Query routed to MCP tool 'get_cluster_health' based on pattern match.")
            state["reasoning_trace"] = trace
            return state

    state["route"] = "rag"
    trace.append(f"Query routed to Advanced RAG knowledge retrieval pipeline.")
    state["reasoning_trace"] = trace
    return state


def retrieve_node(
    state: AgentState,
    dependencies: AgentDependencies,
) -> AgentState:
    query = state["query"]
    trace = state.get("reasoning_trace", [])
    iteration = state.get("iteration_count", 0)

    trace.append(f"Executing hybrid retrieval (iteration {iteration}) for query: '{query}'")

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
        "selected_tokens": sum(c.token_count for c in selected_contexts),
        "compressed_tokens": sum(c.compressed_token_count for c in compressed_contexts),
    }

    trace.append(
        f"Retrieved {len(reranked_results)} candidates, expanded to {len(selected_contexts)} contexts, compressed to {len(compressed_contexts)} items ({state['context_stats']['compressed_tokens']} tokens)."
    )
    state["reasoning_trace"] = trace
    return state


def generate_node(
    state: AgentState,
    dependencies: AgentDependencies,
) -> AgentState:
    query = state["query"]
    contexts = state.get("compressed_contexts", [])
    trace = state.get("reasoning_trace", [])

    if not contexts:
        state["answer"] = (
            "I could not find this information in the provided documentation."
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
        trace.append("No relevant contexts found. Returned standard no-answer response.")
        state["reasoning_trace"] = trace
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

    grounding_validation = dependencies.grounding_validator.validate(
        answer=generated_answer.answer,
        contexts=[context.content for context in contexts],
    )

    state["answer"] = generated_answer.answer
    state["citations"] = citation_validation.citations
    state["citation_validation"] = {
        "valid": citation_validation.valid,
        "invalid_citations": citation_validation.invalid_citations,
        "missing_citations": citation_validation.missing_citations,
    }
    state["grounding"] = {
        "score": grounding_validation.score,
        "grounded": grounding_validation.grounded,
        "matched_terms": grounding_validation.matched_terms,
        "unmatched_terms": grounding_validation.unmatched_terms,
    }

    trace.append(
        f"Generated answer with {len(citation_validation.citations)} citations. Grounding score: {grounding_validation.score:.2f} (grounded={grounding_validation.grounded})."
    )
    state["reasoning_trace"] = trace
    return state


def refine_node(state: AgentState) -> AgentState:
    """
    Self-reflection node: Reformulates the query when retrieval/grounding is weak.
    """
    trace = state.get("reasoning_trace", [])
    iteration = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration

    current_query = state.get("query", "")
    grounding = state.get("grounding", {})
    unmatched = grounding.get("unmatched_terms", [])

    # Query refinement logic: focus on unmatched keywords or simplify query
    if unmatched:
        refined_query = f"{current_query} {' '.join(unmatched[:3])}".strip()
    else:
        # Strip common punctuation/noise
        words = re.findall(r"\w+", current_query)
        refined_query = " ".join([w for w in words if len(w) > 2])

    state["query"] = refined_query
    trace.append(
        f"Self-reflection triggered query refinement (attempt {iteration}/{state.get('max_iterations', 2)}): '{refined_query}'"
    )
    state["reasoning_trace"] = trace
    return state


def mcp_node(
    state: AgentState,
    dependencies: AgentDependencies,
) -> AgentState:
    query = state["query"].strip().lower()
    trace = state.get("reasoning_trace", [])

    # Check if cluster health or pod status
    if "cluster" in query or "node" in query:
        result = dependencies.mcp_service.get_cluster_health()
        if result.success and isinstance(result.content, dict):
            data = result.content
            cluster_name = data.get("cluster_name", "k8s-cluster")
            status = data.get("status", "Healthy")
            nodes_ready = data.get("nodes_ready", 0)
            nodes_total = data.get("nodes_total", 0)
            cpu = data.get("cpu_utilization_pct", 0)
            mem = data.get("memory_utilization_pct", 0)

            state["mcp_result"] = {
                "success": True,
                "tool": result.tool_name,
                "data": data,
            }
            state["answer"] = (
                f"Cluster '{cluster_name}' is currently {status}. "
                f"Nodes: {nodes_ready}/{nodes_total} Ready. "
                f"CPU: {cpu}%, Memory: {mem}%."
            )
            state["citations"] = []
            state["citation_validation"] = {"valid": True, "invalid_citations": [], "missing_citations": False}
            state["grounding"] = {"score": 1.0, "grounded": True, "matched_terms": [], "unmatched_terms": []}
            trace.append(f"MCP tool '{result.tool_name}' executed successfully.")
            state["reasoning_trace"] = trace
            return state

    # Pod status route
    pod_name = _extract_pod_name(state["query"]) or "example-pod"
    result = dependencies.mcp_service.get_pod_status(
        pod_name=pod_name,
        namespace="default",
    )

    if not result.success:
        state["error"] = result.error or "MCP tool failed."
        state["answer"] = "The MCP tool could not retrieve the requested data."
        state["mcp_result"] = {"success": False, "tool": result.tool_name, "error": result.error}
        trace.append(f"MCP tool '{result.tool_name}' failed: {result.error}")
        state["reasoning_trace"] = trace
        return state

    data = result.content
    if not isinstance(data, dict):
        state["error"] = "MCP tool returned an unexpected response format."
        state["answer"] = "The MCP tool returned an unexpected response format."
        state["mcp_result"] = {"success": False, "tool": result.tool_name, "error": state["error"]}
        state["reasoning_trace"] = trace
        return state

    pod_name = str(data.get("pod_name", pod_name))
    namespace = str(data.get("namespace", "default"))
    status = str(data.get("status", "unknown"))
    ready = data.get("ready", False)
    restarts = data.get("restarts", 0)

    state["mcp_result"] = {
        "success": True,
        "tool": result.tool_name,
        "data": data,
    }
    state["answer"] = (
        f"Pod '{pod_name}' in namespace '{namespace}' is currently {status}. "
        f"Ready: {ready}. Restarts: {restarts}."
    )
    state["citations"] = []
    state["citation_validation"] = {"valid": True, "invalid_citations": [], "missing_citations": False}
    state["grounding"] = {"score": 1.0, "grounded": True, "matched_terms": [], "unmatched_terms": []}
    trace.append(f"MCP tool '{result.tool_name}' returned status for pod '{pod_name}'.")
    state["reasoning_trace"] = trace
    return state


def _extract_pod_name(query: str) -> str | None:
    patterns = [
        r"\bpod\s+status\s+for\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bpod\s+status\s+of\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bstatus\s+for\s+pod\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bstatus\s+of\s+pod\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
        r"\bpod\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)\s+status\b",
        r"\bcheck\s+pod\s+([a-zA-Z0-9][a-zA-Z0-9.-]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None