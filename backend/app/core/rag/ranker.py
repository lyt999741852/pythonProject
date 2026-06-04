"""Pure RRF (Reciprocal Rank Fusion) algorithm.

No side effects — operates on ``SearchResult`` rank fields and returns
a score dict that the caller enriches with parent recipe text.
"""

from __future__ import annotations

from app.models.recipe import SearchResult


def rrf_fuse(
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    rrf_k: int = 60,
    final_top_n: int = 3,
) -> dict[str, dict]:
    """Reciprocal Rank Fusion — combine two ranked result lists.

    Formula
    -------
    ``score(p) = 1/(k + rank_vector(p)) + 1/(k + rank_bm25(p))``

    Returns
    -------
    dict mapping ``parent_id`` → ``{score, rank_vector, rank_bm25}``,
    sorted by fused score descending and truncated to *final_top_n*.
    """
    score_map: dict[str, float] = {}
    rank_v_map: dict[str, int] = {}
    rank_b_map: dict[str, int] = {}

    for res in vector_results:
        if res.rank_vector is None:
            continue
        pid = res.parent_id
        score_map[pid] = score_map.get(pid, 0.0) + 1.0 / (rrf_k + res.rank_vector)
        rank_v_map[pid] = res.rank_vector

    for res in bm25_results:
        if res.rank_bm25 is None:
            continue
        pid = res.parent_id
        score_map[pid] = score_map.get(pid, 0.0) + 1.0 / (rrf_k + res.rank_bm25)
        rank_b_map[pid] = res.rank_bm25

    sorted_pids = sorted(score_map.keys(), key=lambda p: score_map[p], reverse=True)
    top_pids = sorted_pids[:final_top_n]

    return {
        pid: {
            "score": score_map[pid],
            "rank_vector": rank_v_map.get(pid),
            "rank_bm25": rank_b_map.get(pid),
        }
        for pid in top_pids
    }
