from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable
import numpy as np

from src.frec.data.schema import Session


@dataclass
class CovisitIndex:
    """
    Stores item->neighbors as a compact numpy structure.
    neighbors[item] = array of (neighbor_id, score) sorted desc by score
    """
    neighbors: Dict[int, np.ndarray]  # item_id -> shape (M,2) float/int packed as float64
    n_items: int


def build_covisitation_index(
    sessions: List[Session],
    n_items: int,
    max_neighbors_per_item: int = 200,
    use_clicked_only: bool = True,
    max_items_per_session: int = 60,
    session_pair_window: int = 10,
    downsample_impressions: bool = True,
    seed: int = 42,
) -> CovisitIndex:
    """
    Efficient item-item co-visitation builder.

    Key idea: avoid O(m^2) per session by:
      - limiting items per session (max_items_per_session)
      - pairing within a small window in the session list (session_pair_window)
    """
    rng = np.random.RandomState(seed)

    counts: Dict[int, Dict[int, float]] = {}

    for s in sessions:
        items = s.clicked_post_ids if use_clicked_only else s.impression_post_ids
        if not items:
            continue

        # de-duplicate while keeping order
        items = list(dict.fromkeys(items))

        # limit items to keep runtime sane
        if len(items) > max_items_per_session:
            if downsample_impressions:
                # sample a subset (random) to max_items_per_session
                items = rng.choice(items, size=max_items_per_session, replace=False).tolist()
            else:
                # take the most recent max_items_per_session
                items = items[:max_items_per_session]

        L = len(items)
        if L < 2:
            continue

        # pair only within a window: for i, pair with i+1..i+W (and symmetric)
        W = min(session_pair_window, L - 1)

        for i in range(L):
            a = items[i]
            if a not in counts:
                counts[a] = {}
            ca = counts[a]

            jmax = min(L, i + W + 1)
            for j in range(i + 1, jmax):
                b = items[j]
                if b == a:
                    continue

                # simple count update (symmetric)
                ca[b] = ca.get(b, 0.0) + 1.0

                if b not in counts:
                    counts[b] = {}
                cb = counts[b]
                cb[a] = cb.get(a, 0.0) + 1.0

    # Convert to top-N neighbor arrays
    neighbors: Dict[int, np.ndarray] = {}
    for i, nbrs in counts.items():
        if not nbrs:
            continue
        items_scores = sorted(nbrs.items(), key=lambda x: x[1], reverse=True)[:max_neighbors_per_item]
        arr = np.zeros((len(items_scores), 2), dtype=np.float64)
        for k, (j, sc) in enumerate(items_scores):
            arr[k, 0] = float(j)
            arr[k, 1] = float(sc)
        neighbors[i] = arr

    return CovisitIndex(neighbors=neighbors, n_items=n_items)


def retrieve_covisitation(
    user_history: List[int],
    index: CovisitIndex,
    topk_per_history_item: int = 100,
    max_history_items: int = 20,
    retrieval_k: int = 500,
) -> List[int]:
    """
    Given a user's recent clicked history items, retrieve candidates by unioning
    neighbors of each history item, with score aggregation.
    """
    if not user_history:
        return []

    # most recent first assumed; we only take max_history_items
    hist = user_history[:max_history_items]

    # score aggregation
    scores: Dict[int, float] = {}

    for h in hist:
        arr = index.neighbors.get(h)
        if arr is None:
            continue
        # take top neighbors for this history item
        m = min(topk_per_history_item, arr.shape[0])
        for t in range(m):
            j = int(arr[t, 0])
            sc = float(arr[t, 1])
            if j == h:
                continue
            scores[j] = scores.get(j, 0.0) + sc

    if not scores:
        return []

    # remove items already in history (avoid repeats in retrieval set)
    hist_set = set(hist)
    for h in hist_set:
        scores.pop(h, None)

    # take top retrieval_k by aggregated score
    items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:retrieval_k]
    return [i for i, _ in items]


def candidate_recall_at_k(
    candidates: List[int],
    positives: List[int],
    k: int,
) -> float:
    """
    Recall@k = (# positives in top-k candidates) / (# positives)
    """
    pos = set(positives)
    if len(pos) == 0:
        return 0.0
    topk = set(candidates[:k])
    return float(len(pos & topk) / len(pos))


def evaluate_candidate_recall(
    sessions_sorted: List[Session],
    index: CovisitIndex,
    history_by_user: Dict[int, List[int]],
    topk_per_history_item: int,
    max_history_items: int,
    retrieval_k: int,
    k_list: List[int],
) -> Dict[str, float]:
    """
    Evaluate candidate recall on a set of sessions with *online* history:
    history_by_user contains clicks from prior sessions only (no leakage).
    sessions_sorted must be sorted by ts.
    """
    recalls = {k: [] for k in k_list}

    for s in sessions_sorted:
        if not s.clicked_post_ids:
            # skip no-positive sessions for recall
            continue

        hist = history_by_user.get(s.user_id, [])
        cand = retrieve_covisitation(
            user_history=hist,
            index=index,
            topk_per_history_item=topk_per_history_item,
            max_history_items=max_history_items,
            retrieval_k=retrieval_k,
        )

        for k in k_list:
            recalls[k].append(candidate_recall_at_k(cand, s.clicked_post_ids, k))

        # update history AFTER evaluating (prevents leakage)
        if s.clicked_post_ids:
            # prepend new clicks as most recent
            history_by_user.setdefault(s.user_id, [])
            history_by_user[s.user_id] = (s.clicked_post_ids + history_by_user[s.user_id])[:max_history_items * 5]

    out = {}
    for k in k_list:
        out[f"cand_recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
    out["n_eval_sessions"] = int(sum(1 for s in sessions_sorted if s.clicked_post_ids))
    return out
