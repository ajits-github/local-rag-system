from __future__ import annotations

from fastapi import APIRouter, Depends

from rag.api.deps import get_llm, get_vectorstore
from rag.generation.base import LLM
from rag.vectorstore.base import VectorStore

router = APIRouter()


@router.get("/health")
def health(
    vectorstore: VectorStore = Depends(get_vectorstore),
    llm: LLM = Depends(get_llm),
) -> dict:
    db_ok = vectorstore.health_check()
    llm_ok = llm.health_check()
    return {
        "status": "ok" if (db_ok and llm_ok) else "degraded",
        "dependencies": {
            "vectorstore": "ok" if db_ok else "unreachable",
            "llm": "ok" if llm_ok else "unreachable",
        },
    }
