from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from src.frec.data.schema import Post, Session


def build_post_arrays(posts: List[Post]):
    n = len(posts)
    topic = np.zeros(n, dtype=np.int32)
    created = np.zeros(n, dtype=np.int32)
    for p in posts:
        topic[p.post_id] = p.topic_id
        created[p.post_id] = p.created_ts
    return topic, created


def user_topic_profile_from_history(
    clicked_history: List[int],
    post_topic: np.ndarray,
    n_topics: int,
    max_history_items: int = 50,
) -> np.ndarray:
    """
    Simple user-topic preference vector from recent clicks.
    """
    hist = clicked_history[:max_history_items]
    if not hist:
        return np.ones(n_topics, dtype=np.float32) / n_topics

    counts = np.zeros(n_topics, dtype=np.float32)
    for pid in hist:
        counts[int(post_topic[pid])] += 1.0
    counts = counts / (counts.sum() + 1e-12)
    return counts


def retrieve_by_topic(
    user_history: List[int],
    posts: List[Post],
    post_topic: np.ndarray,
    post_created: np.ndarray,
    now_ts: int,
    n_topics: int,
    retrieval_k: int = 500,
    max_history_items: int = 50,
    topic_boost: float = 1.0,
    recency_boost: float = 0.15,
    popularity_boost: float = 0.15,
    popularity_24h: np.ndarray | None = None,
    top_topics: int = 3,
    exploration_frac: float = 0.10,
) -> List[int]:
    pref = user_topic_profile_from_history(user_history, post_topic, n_topics, max_history_items=max_history_items)

    # pick top topics
    topT = int(top_topics)
    top_topic_ids = np.argsort(-pref)[:topT]
    top_topic_set = set(int(t) for t in top_topic_ids)

    # candidate pool: posts in top topics (plus small exploration)
    in_top = np.array([t in top_topic_set for t in post_topic], dtype=bool)
    pool_ids = np.where(in_top)[0]

    # exploration bucket (random from outside top topics)
    n_explore = int(retrieval_k * float(exploration_frac))
    if n_explore > 0:
        outside_ids = np.where(~in_top)[0]
        if outside_ids.size > 0:
            n_explore = min(n_explore, outside_ids.size)
            explore = np.random.choice(outside_ids, size=n_explore, replace=False)
            pool_ids = np.unique(np.concatenate([pool_ids, explore]))

    if pool_ids.size == 0:
        pool_ids = np.arange(len(posts))

    # scoring inside pool
    age = np.maximum(0, now_ts - post_created[pool_ids]).astype(np.float32)
    freshness = -age / (24 * 60)

    pop = popularity_24h[pool_ids] if popularity_24h is not None else np.zeros(pool_ids.size, dtype=np.float32)

    topic_score = pref[post_topic[pool_ids]]
    score = topic_boost * topic_score + recency_boost * freshness + popularity_boost * pop

    # take top retrieval_k from pool
    k = min(retrieval_k, pool_ids.size)
    topk = np.argpartition(-score, kth=min(k - 1, score.size - 1))[:k]
    topk = topk[np.argsort(-score[topk])]
    return pool_ids[topk].tolist()


# def retrieve_by_topic(
#     user_history: List[int],
#     posts: List[Post],
#     post_topic: np.ndarray,
#     post_created: np.ndarray,
#     now_ts: int,
#     n_topics: int,
#     retrieval_k: int = 500,
#     max_history_items: int = 50,
#     topic_boost: float = 1.0,
#     recency_boost: float = 0.15,
#     popularity_boost: float = 0.15,
#     popularity_24h: np.ndarray | None = None,
# ) -> List[int]:
#     """
#     Candidate retrieval using topic alignment.
#     Score(post) = topic_boost * P_user(topic(post)) + recency_boost * freshness + popularity_boost * popularity
#     """
#     pref = user_topic_profile_from_history(user_history, post_topic, n_topics, max_history_items=max_history_items)

#     # freshness in [-1, 0]
#     age = np.maximum(0, now_ts - post_created).astype(np.float32)
#     freshness = -age / (24 * 60)

#     topic_score = pref[post_topic]  # vectorized lookup

#     pop = popularity_24h if popularity_24h is not None else np.zeros(len(posts), dtype=np.float32)

#     score = topic_boost * topic_score + recency_boost * freshness + popularity_boost * pop

#     topk = np.argpartition(-score, kth=min(retrieval_k, len(score) - 1))[:retrieval_k]
#     topk = topk[np.argsort(-score[topk])]
#     return topk.tolist()
