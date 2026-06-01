"""
RAG Answer Gate
===============

Shows how to use ``guard_output`` as a post-generation gate in a
Retrieval-Augmented Generation pipeline.

The pattern:
  1. Retrieve context chunks from a vector store.
  2. Generate an answer with your LLM.
  3. Gate the answer through ``guard_output`` before returning it to the user.

If the routing action is BLOCK or HUMAN_REVIEW the pipeline can fall back
to a safe default ("I don't know") instead of surfacing a risky answer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hallucination_sentinel.calibration import load_calibration
from hallucination_sentinel.integrations.middleware import (
    PolicyAction,
    RoutingDecision,
    TaskCriticality,
    guard_output,
)


# ---------------------------------------------------------------------------
# Stub components (replace with real implementations)
# ---------------------------------------------------------------------------


def retrieve_chunks(query: str, top_k: int = 3) -> list[str]:
    """Retrieve the top-k context chunks for *query*.

    Replace this with your actual vector-store retriever.
    """
    return [
        f"Context chunk {i}: relevant information about '{query}'."
        for i in range(top_k)
    ]


def generate_answer(query: str, context: list[str]) -> str:
    """Generate an answer from *query* and *context*.

    Replace this with your actual LLM call.
    """
    context_block = "\n".join(context)
    return (
        f"Based on the following context:\n{context_block}\n\n"
        f"Answer: The answer to '{query}' is derived from the provided context."
    )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def rag_answer_gate(
    query: str,
    calibration_path: str,
    criticality: TaskCriticality = TaskCriticality.MEDIUM,
) -> dict:
    """Run the full RAG pipeline with a hallucination gate.

    Args:
        query: User query.
        calibration_path: Path to a calibration artifact JSON file.
        criticality: How critical the downstream task is.

    Returns:
        A dict with the answer, routing decision, and context chunks.
    """
    # 1. Retrieve
    chunks = retrieve_chunks(query)

    # 2. Generate
    answer = generate_answer(query, chunks)

    # 3. Gate
    calibration = load_calibration(calibration_path)
    decision = guard_output(
        query,
        answer,
        calibration=calibration,
        provider="openai",
        policy=criticality,
    )

    # 4. Act on the decision
    safe_answer = answer
    if decision.action in (PolicyAction.BLOCK, PolicyAction.HUMAN_REVIEW):
        safe_answer = (
            "I'm not confident enough in this answer to share it. "
            "Please consult a primary source or a human expert."
        )

    return {
        "query": query,
        "answer": safe_answer,
        "original_answer": answer if safe_answer != answer else None,
        "decision": decision.to_dict(),
        "chunks_used": chunks,
    }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python rag_answer_gate.py <calibration.json>")
        print()
        print("Runs a demo RAG pipeline with a hallucination gate.")
        sys.exit(1)

    calibration_path = sys.argv[1]
    query = "What is the CES algorithm used for?"

    result = rag_answer_gate(
        query,
        calibration_path,
        criticality=TaskCriticality.MEDIUM,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
