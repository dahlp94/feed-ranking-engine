from __future__ import annotations
from typing import Dict, List
import numpy as np

def _dcg(rels: np.ndarray) -> float:
    # rels are 0/1
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))

def ndcg_at_k(y_true: List[int], ranked_ids: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    yset = set(y_true)
    topk = ranked_ids[:k]
    rels = np.array([1.0 if pid in yset else 0.0 for pid in topk], dtype=np.float32)
    dcg = _dcg(rels)
    ideal_rels = np.sort(rels)[::-1]
    idcg = _dcg(ideal_rels)
    return 0.0 if idcg == 0.0 else dcg / idcg

def recall_at_k(y_true: List[int], ranked_ids: List[int], k: int) -> float:
    yset = set(y_true)
    if len(yset) == 0:
        return 0.0
    topk = set(ranked_ids[:k])
    return float(len(yset & topk) / len(yset))

def map_at_k(y_true: List[int], ranked_ids: List[int], k: int) -> float:
    yset = set(y_true)
    if len(yset) == 0:
        return 0.0
    hits = 0
    s = 0.0
    for i, pid in enumerate(ranked_ids[:k], start=1):
        if pid in yset:
            hits += 1
            s += hits / i
    return s / min(len(yset), k)

def evaluate_sessions(rank_fn, sessions, k_list: List[int]) -> Dict[str, float]:
    out = {}
    for k in k_list:
        ndcgs, recs, maps = [], [], []
        for s in sessions:
            if not s.clicked_post_ids:
                continue
            ranked = rank_fn(s)
            ndcgs.append(ndcg_at_k(s.clicked_post_ids, ranked, k))
            recs.append(recall_at_k(s.clicked_post_ids, ranked, k))
            maps.append(map_at_k(s.clicked_post_ids, ranked, k))
        out[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        out[f"recall@{k}"] = float(np.mean(recs)) if recs else 0.0
        out[f"map@{k}"] = float(np.mean(maps)) if maps else 0.0
    return out
