from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np

from src.frec.data.schema import Post, Session

def build_post_stats(posts: List[Post], train_sessions: List[Session], now_ts: int):
    """
    For baseline: popularity_24h from training impressions/clicks, plus freshness.
    """
    n_posts = len(posts)
    post_created = np.array([p.created_ts for p in posts], dtype=np.int32)

    # count clicks/impressions in last 24h window of "now_ts" within training
    window = 24 * 60
    imp = np.zeros(n_posts, dtype=np.int32)
    clk = np.zeros(n_posts, dtype=np.int32)

    for s in train_sessions:
        if now_ts - window <= s.ts <= now_ts:
            for pid in s.impression_post_ids:
                imp[pid] += 1
            for pid in s.clicked_post_ids:
                clk[pid] += 1

    # smoothed CTR-ish popularity
    popularity = (clk + 1.0) / (imp + 50.0)

    # freshness: newer posts higher
    age = np.maximum(0, now_ts - post_created).astype(np.float32)
    freshness = -age / (24 * 60)

    return popularity, freshness

def retrieve_baseline(
    user_id: int,
    posts: List[Post],
    follows: Dict[int, List[int]],
    popularity: np.ndarray,
    freshness: np.ndarray,
    retrieval_k: int,
):
    """
    Baseline retrieval: return top-K by (popularity + freshness), plus boosting followed authors.
    """
    w_follow = 1.0
    follow_set = set(follows.get(user_id, []))

    author_ids = np.array([p.author_id for p in posts], dtype=np.int32)
    follow_bonus = np.array([1.0 if a in follow_set else 0.0 for a in author_ids], dtype=np.float32)

    score = (0.6 * popularity) + (0.4 * freshness) + (w_follow * follow_bonus * 0.05)
    topk = np.argpartition(-score, kth=min(retrieval_k, len(score)-1))[:retrieval_k]
    # sort the topk
    topk_sorted = topk[np.argsort(-score[topk])]
    return topk_sorted.tolist(), score

def rank_candidates(
    candidate_ids: List[int],
    score: np.ndarray,
    w_pop: float,
    w_fresh: float,
    popularity: np.ndarray,
    freshness: np.ndarray,
    follows_bonus: np.ndarray,
    rank_k: int,
):
    s = w_pop * popularity[candidate_ids] + w_fresh * freshness[candidate_ids] + 0.1 * follows_bonus[candidate_ids]
    order = np.argsort(-s)[:rank_k]
    return [candidate_ids[i] for i in order], s
