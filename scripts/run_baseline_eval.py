import json
from pathlib import Path
import yaml
import numpy as np

from src.frec.data.simulator import simulate_data, split_sessions_by_time, MINUTES_PER_DAY
from src.frec.retrieval.baseline import build_post_stats, retrieve_baseline
from src.frec.eval.ranking_metrics import evaluate_sessions

def main():
    cfg = yaml.safe_load(open("configs/dev.yaml", "r"))
    data = simulate_data(cfg)

    # time split
    train_days = int(cfg["split"]["train_days"])
    val_days = int(cfg["split"]["val_days"])
    train_end = train_days * MINUTES_PER_DAY
    val_end = (train_days + val_days) * MINUTES_PER_DAY

    train, val, test = split_sessions_by_time(data.sessions, train_end, val_end)

    # compute baseline stats at end of train
    now_ts = train_end
    popularity, freshness = build_post_stats(data.posts, train, now_ts)

    # precompute follow bonus vector once
    author_ids = np.array([p.author_id for p in data.posts], dtype=np.int32)

    bcfg = cfg["baseline"]
    retrieval_k = int(bcfg["retrieval_k"])
    rank_k = int(bcfg["rank_k"])
    w_pop = float(bcfg["w_popularity_24h"])
    w_fresh = float(bcfg["w_freshness"])
    w_follow = float(bcfg["w_follow"])


    def rank_fn(session):
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
        # rank candidates
        cand = np.array(candidates, dtype=np.int32)
        scores = w_pop * popularity[cand] + w_fresh * freshness[cand] + w_follow * follows_bonus[cand]
        order = np.argsort(-scores)[:rank_k]
        return [int(cand[i]) for i in order]
    

    report = {
        "counts": {
            "n_users": len(data.users),
            "n_posts": len(data.posts),
            "n_sessions_total": len(data.sessions),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        "metrics_val": evaluate_sessions(rank_fn, val, cfg["eval"]["k_list"]),
        "metrics_test": evaluate_sessions(rank_fn, test, cfg["eval"]["k_list"]),
    }

    out_path = Path(cfg["eval"]["report_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
