"""AD 698 Financial RAG — section-scoped retrieval over SEC 10-K filings."""

from .rag import (
    CONFIG,
    build_index,
    retrieve,
    rag_answer,
    evaluate,
)

__all__ = ["CONFIG", "build_index", "retrieve", "rag_answer", "evaluate"]
