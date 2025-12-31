import json
from pathlib import Path
import yaml
import numpy as np

from src.frec.retrieval.baseline import build_post_stats
from src.frec.data.simulator import simulate_data, split_sessions_by_time, MINUTES_PER_DAY
from src.frec.retrieval.topic import build_post_arrays, retrieve_by_topic
from src.frec.retrieval.covisit import candidate_recall_at_k


# def evaluate_topic_retrieval(sessions_sorted, posts, post_topic, post_created, n_topics, popularity, cfg_topic, history_by_user, k_list):
#     recalls = {k: [] for k in k_list}
#     for s in sessions_sorted:
#         if not s.clicked_post_ids:
#             continue

#         hist = history_by_user.get(s.user_id, [])
#         # cand = retrieve_by_topic(
#         #     user_history=hist,
#         #     posts=posts,
#         #     post_topic=post_topic,
#         #     post_created=post_created,
#         #     now_ts=s.ts,
#         #     n_topics=n_topics,
#         #     retrieval_k=int(cfg_topic["retrieval_k"]),
#         #     max_history_items=int(cfg_topic["max_history_items"]),
#         #     topic_boost=float(cfg_topic["topic_boost"]),
#         #     recency_boost=float(cfg_topic["recency_boost"]),
#         #     popularity_boost=float(cfg_topic["popularity_boost"]),
#         #     popularity_24h=popularity,
#         # )

#         cand = retrieve_by_topic(
#             user_history=hist,
#             posts=posts,
#             post_topic=post_topic,
#             post_created=post_created,
#             now_ts=s.ts,
#             n_topics=n_topics,
#             retrieval_k=int(cfg_topic["retrieval_k"]),
#             max_history_items=int(cfg_topic["max_history_items"]),
#             topic_boost=float(cfg_topic["topic_boost"]),
#             recency_boost=float(cfg_topic["recency_boost"]),
#             popularity_boost=float(cfg_topic["popularity_boost"]),
#             popularity_24h=popularity,
#             top_topics=int(cfg_topic["top_topics"]),
#             exploration_frac=float(cfg_topic["exploration_frac"]),
#         )





#         for k in k_list:
#             recalls[k].append(candidate_recall_at_k(cand, s.clicked_post_ids, k))

#         # update history after eval (no leakage)
#         history_by_user.setdefault(s.user_id, [])
#         history_by_user[s.user_id] = (s.clicked_post_ids + history_by_user[s.user_id])[: int(cfg_topic["max_history_items"]) * 5]

#     out = {f"cand_recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in k_list}
#     out["n_eval_sessions"] = int(sum(1 for s in sessions_sorted if s.clicked_post_ids))
#     return out

def evaluate_topic_retrieval(
    sessions_sorted,
    posts,
    post_topic,
    post_created,
    n_topics,
    popularity,
    freshness,
    cfg_topic,
    history_by_user,
    k_list,
    cold_start_min_clicks: int = 5,
):
    # buckets
    recalls_all = {k: [] for k in k_list}
    recalls_warm = {k: [] for k in k_list}
    recalls_cold = {k: [] for k in k_list}

    n_all = n_warm = n_cold = 0

    for s in sessions_sorted:
        if not s.clicked_post_ids:
            continue

        hist = history_by_user.get(s.user_id, [])
        is_warm = len(hist) >= cold_start_min_clicks

        # fallback if cold-start
        if not is_warm:
            # popularity+freshness fallback (top-K global)
            score = 0.7 * popularity + 0.3 * freshness
            topk = np.argpartition(-score, kth=min(int(cfg_topic["retrieval_k"]), len(score)-1))[: int(cfg_topic["retrieval_k"])]
            topk = topk[np.argsort(-score[topk])]
            cand = topk.tolist()
        else:
            cand = retrieve_by_topic(
                user_history=hist,
                posts=posts,
                post_topic=post_topic,
                post_created=post_created,
                now_ts=s.ts,
                n_topics=n_topics,
                retrieval_k=int(cfg_topic["retrieval_k"]),
                max_history_items=int(cfg_topic["max_history_items"]),
                topic_boost=float(cfg_topic["topic_boost"]),
                recency_boost=float(cfg_topic["recency_boost"]),
                popularity_boost=float(cfg_topic["popularity_boost"]),
                popularity_24h=popularity,
                top_topics=int(cfg_topic["top_topics"]),
                exploration_frac=float(cfg_topic["exploration_frac"]),
            )

        # update metrics
        n_all += 1
        if is_warm:
            n_warm += 1
        else:
            n_cold += 1

        for k in k_list:
            r = candidate_recall_at_k(cand, s.clicked_post_ids, k)
            recalls_all[k].append(r)
            if is_warm:
                recalls_warm[k].append(r)
            else:
                recalls_cold[k].append(r)

        # update history AFTER eval
        history_by_user.setdefault(s.user_id, [])
        history_by_user[s.user_id] = (s.clicked_post_ids + history_by_user[s.user_id])[: int(cfg_topic["max_history_items"]) * 5]

    def pack(recalls, n):
        out = {f"cand_recall@{k}": float(np.mean(recalls[k])) if recalls[k] else 0.0 for k in k_list}
        out["n_eval_sessions"] = int(n)
        return out

    return {
        "all": pack(recalls_all, n_all),
        "warm": pack(recalls_warm, n_warm),
        "cold": pack(recalls_cold, n_cold),
        "cold_start_min_clicks": cold_start_min_clicks,
    }


def main():
    cfg = yaml.safe_load(open("configs/dev.yaml", "r"))
    data = simulate_data(cfg)

    train_days = int(cfg["split"]["train_days"])
    val_days = int(cfg["split"]["val_days"])
    train_end = train_days * MINUTES_PER_DAY
    val_end = (train_days + val_days) * MINUTES_PER_DAY

    train, val, test = split_sessions_by_time(data.sessions, train_end, val_end)
    train_sorted = sorted(train, key=lambda s: s.ts)
    val_sorted = sorted(val, key=lambda s: s.ts)
    test_sorted = sorted(test, key=lambda s: s.ts)

    # post arrays
    post_topic, post_created = build_post_arrays(data.posts)
    n_topics = int(cfg["sim"]["n_topics"])

    # build popularity proxy from TRAIN (simple)
    #popularity = np.zeros(len(data.posts), dtype=np.float32)
    # optional: you can reuse your baseline popularity function later; this is fine for now.

    # popularity proxy from TRAIN at end of training window
    popularity, freshness = build_post_stats(data.posts, train_sorted, now_ts=train_end)



    # build user history from TRAIN clicks
    history_by_user = {}
    for s in train_sorted:
        if s.clicked_post_ids:
            history_by_user.setdefault(s.user_id, [])
            history_by_user[s.user_id] = (s.clicked_post_ids + history_by_user[s.user_id])[: 500]

    k_list = cfg["eval"]["k_list"]
    tcfg = cfg["topic_retrieval"]

    # report_val = evaluate_topic_retrieval(
    #     val_sorted, data.posts, post_topic, post_created, n_topics, popularity, tcfg,
    #     {u: h.copy() for u, h in history_by_user.items()}, k_list
    # )

    # report_test = evaluate_topic_retrieval(
    #     test_sorted, data.posts, post_topic, post_created, n_topics, popularity, tcfg,
    #     {u: h.copy() for u, h in history_by_user.items()}, k_list
    # )

    report_val = evaluate_topic_retrieval(
        val_sorted, data.posts, post_topic, post_created, n_topics,
        popularity, freshness, tcfg,
        {u: h.copy() for u, h in history_by_user.items()}, k_list,
        cold_start_min_clicks=5
    )

    report_test = evaluate_topic_retrieval(
        test_sorted, data.posts, post_topic, post_created, n_topics,
        popularity, freshness, tcfg,
        {u: h.copy() for u, h in history_by_user.items()}, k_list,
        cold_start_min_clicks=5
    )


    out = {
        "counts": {"n_train": len(train), "n_val": len(val), "n_test": len(test)},
        "topic_retrieval_config": tcfg,
        "candidate_recall_val": report_val,
        "candidate_recall_test": report_test,
    }

    out_path = Path("outputs/topic_candidate_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nWrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()
