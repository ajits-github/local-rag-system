"""Bounded-execution constants shared by the orchestrator and the Knowledge Agent.

Split into their own module (rather than living in `orchestrator.py` or
`knowledge_agent.py`) specifically so neither specialist module needs to
import the other, or the orchestrator, just to read a shared integer --
see `knowledge_agent.py`/`business_agent.py`'s module docstrings for why
that import boundary is the whole point of the tool-isolation design (and
`tests/test_tool_isolation.py`, which statically asserts on it).
"""

from __future__ import annotations

#: `knowledge_agent`'s default `RetrievalPipeline.retrieve(candidate_k=...)`.
DEFAULT_KNOWLEDGE_TOP_K = 5

#: Widened `candidate_k` for the one bounded retry `evidence_critic` may request.
RETRY_KNOWLEDGE_TOP_K = 10

#: How many times `coordinator` may run in one thread (round 0 plus at
#: most one retry round). Prevents an unbounded coordinator<->critic loop
#: -- see `orchestrator.evidence_critic`'s docstring.
MAX_COORDINATOR_ROUNDS = 2

#: How many times `evidence_critic` may request a knowledge-specialist retry.
MAX_KNOWLEDGE_RETRIES = 1
