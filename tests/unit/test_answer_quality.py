from __future__ import annotations

from rag.eval.answer_quality import KeywordOverlapScorer


def test_keyword_overlap_full_match():
    scorer = KeywordOverlapScorer()
    score = scorer.score(
        question="what is pgvector",
        generated_answer="pgvector is a postgres extension for vectors",
        reference_answer="pgvector is a postgres extension",
    )
    assert score == 1.0


def test_keyword_overlap_no_reference_returns_zero():
    scorer = KeywordOverlapScorer()
    score = scorer.score(question="q", generated_answer="anything", reference_answer=None)
    assert score == 0.0


def test_keyword_overlap_partial_match():
    scorer = KeywordOverlapScorer()
    score = scorer.score(
        question="q",
        generated_answer="postgres extension",
        reference_answer="postgres extension vectors",
    )
    assert 0.0 < score < 1.0
