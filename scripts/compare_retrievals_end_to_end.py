import json
from pathlib import Path
import yaml
import numpy as np

from src.frec.data.simulator import simulate_data, split_sessions_by_time, MINUTES_PER_DAY
from src.frec.retrieval.baseline import build_post_stats, retrieve_baseline
from src.frec.retrieval.topic import build_post_arrays, retrieve_by_topic
from src.frec.eval.ranking_metrics import evaluate_sessions


def main():
    cfg = yaml.safe_load(open("configs/dev.yaml", "r"))
    data = simulate_data(cfg)

    # split
    train_days = int(cfg["split"]["train_days"])
    val_days = int(cfg["split"]["val_days"])
    train_end = train_days * MINUTES_PER_DAY
    val_end = (train_days + val_days) * MINUTES_PER_DAY

    train, val, test = split_sessions_by_time(data.sessions, train_end, val_end)

    # arrays
    post_topic, post_created = build_post_arrays(data.posts)
    n_topics = int(cfg["sim"]["n_topics"])
    author_ids = np.array([p.author_id for p in data.posts], dtype=np.int32)

    # baseline stats computed at train_end (consistent comparison)
    popularity, freshness = build_post_stats(data.posts, train, now_ts=train_end)

    # configs
    bcfg = cfg["baseline"]
    tcfg = cfg["topic_retrieval"]

    retrieval_k = int(bcfg["retrieval_k"])
    rank_k = int(bcfg["rank_k"])
    w_pop = float(bcfg["w_popularity_24h"])
    w_fresh = float(bcfg["w_freshness"])
    w_follow = float(bcfg["w_follow"])

    def baseline_rank_fn(session):
        u = session.user_id
        follow_set = set(data.follows.get(u, []))
        follows_bonus = np.array([1.0 if a in follow_set else 0.0 for a in author_ids], dtype=np.float32)

        candidates, _ = retrieve_baseline(
            user_id=u,
            posts=data.posts,
            follows=data.follows,
            popularity=popularity,
            freshness=freshness,
            retrieval_k=retrieval_k,
        )

        cand = np.array(candidates, dtype=np.int32)
        scores = w_pop * popularity[cand] + w_fresh * freshness[cand] + w_follow * follows_bonus[cand]
        order = np.argsort(-scores)[:rank_k]
        return [int(cand[i]) for i in order]

    # Build user history from TRAIN clicks
    train_sorted = sorted(train, key=lambda s: s.ts)
    history_by_user = {}
    for s in train_sorted:
        if s.clicked_post_ids:
            history_by_user.setdefault(s.user_id, [])
            history_by_user[s.user_id] = (s.clicked_post_ids + history_by_user[s.user_id])[: int(tcfg["max_history_items"]) * 5]

    def topic_rank_fn_factory(sessions_sorted):
        # local copy so val/test don't leak into each other
        hist = {u: h.copy() for u, h in history_by_user.items()}

        def _rank_fn(session):
            u = session.user_id
            follow_set = set(data.follows.get(u, []))
            follows_bonus = np.array([1.0 if a in follow_set else 0.0 for a in author_ids], dtype=np.float32)

            user_hist = hist.get(u, [])

            # topic retrieval candidates
            candidates = retrieve_by_topic(
                user_history=user_hist,
                posts=data.posts,
                post_topic=post_topic,
                post_created=post_created,
                now_ts=session.ts,
                n_topics=n_topics,
                retrieval_k=int(tcfg["retrieval_k"]),
                max_history_items=int(tcfg["max_history_items"]),
                topic_boost=float(tcfg["topic_boost"]),
                recency_boost=float(tcfg["recency_boost"]),
                popularity_boost=float(tcfg["popularity_boost"]),
                popularity_24h=popularity,
                top_topics=int(tcfg["top_topics"]),
                exploration_frac=float(tcfg["exploration_frac"]),
            )

            cand = np.array(candidates, dtype=np.int32)

            # same ranker scoring as baseline (apples-to-apples)
            scores = w_pop * popularity[cand] + w_fresh * freshness[cand] + w_follow * follows_bonus[cand]
            order = np.argsort(-scores)[:rank_k]
            ranked = [int(cand[i]) for i in order]

            # update history after evaluation (no leakage across time within split)
            if session.clicked_post_ids:
                hist.setdefault(u, [])
                hist[u] = (session.clicked_post_ids + hist[u])[: int(tcfg["max_history_items"]) * 5]

            return ranked

        return _rank_fn

    # Evaluate
    k_list = cfg["eval"]["k_list"]

    val_topic_rank_fn = topic_rank_fn_factory(sorted(val, key=lambda s: s.ts))
    test_topic_rank_fn = topic_rank_fn_factory(sorted(test, key=lambda s: s.ts))

    report = {
        "counts": {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        "baseline_metrics_val": evaluate_sessions(baseline_rank_fn, val, k_list),
        "baseline_metrics_test": evaluate_sessions(baseline_rank_fn, test, k_list),
        "topic_retrieval_metrics_val": evaluate_sessions(val_topic_rank_fn, val, k_list),
        "topic_retrieval_metrics_test": evaluate_sessions(test_topic_rank_fn, test, k_list),
        "notes": "Topic retrieval changes candidates only; ranking function is held fixed for apples-to-apples comparison."
    }

    out_path = Path("outputs/week2_retrieval_comparison.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote: {out_path.resolve()}")

if __name__ == "__main__":
    main()
