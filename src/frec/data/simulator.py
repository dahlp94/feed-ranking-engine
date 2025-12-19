from __future__ import annotations
import math
import random
from dataclasses import asdict
from typing import List, Dict, Tuple
import numpy as np

from src.frec.data.schema import User, Post, Session, SimData

MINUTES_PER_DAY = 24 * 60

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)

def simulate_data(cfg: dict) -> SimData:
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)

    scfg = cfg["sim"]
    n_users = int(scfg["n_users"])
    n_posts = int(scfg["n_posts"])
    n_authors = int(scfg["n_authors"])
    n_topics = int(scfg["n_topics"])
    days = int(scfg["days"])

    # ---- users ----
    users = [User(u) for u in range(n_users)]

    # ---- follows (each user follows ~5-30 authors) ----
    follows: Dict[int, List[int]] = {}
    for u in range(n_users):
        k = np.random.randint(5, 31)
        follows[u] = np.random.choice(n_authors, size=k, replace=False).tolist()

    # ---- posts arrive over time ----
    total_minutes = days * MINUTES_PER_DAY
    created_ts = np.random.randint(0, total_minutes, size=n_posts)
    created_ts.sort()

    posts: List[Post] = []
    for i in range(n_posts):
        author_id = int(np.random.randint(0, n_authors))
        topic_id = int(np.random.randint(0, n_topics))
        posts.append(Post(post_id=i, author_id=author_id, topic_id=topic_id, created_ts=int(created_ts[i])))

    # ---- user tastes (topic preference vectors) with drift ----
    taste = np.random.normal(size=(n_users, n_topics))
    taste = taste / (np.linalg.norm(taste, axis=1, keepdims=True) + 1e-12)

    taste_drift = float(scfg["taste_drift"])

    # click model weights
    base_ctr = float(scfg["base_ctr"])
    w_topic = float(scfg["w_topic_match"])
    w_fresh = float(scfg["w_freshness"])
    w_pop = float(scfg["w_popularity"])
    w_follow = float(scfg["w_follow_author"])

    # impressions
    sessions_per_user_per_day = int(scfg["sessions_per_user_per_day"])
    candidate_pool_size = int(scfg["candidate_pool_size"])

    # index posts by created time for "available posts at time t"
    post_created = np.array([p.created_ts for p in posts], dtype=np.int32)

    sessions: List[Session] = []

    def popularity_24h(ts_min: int) -> np.ndarray:
        """
        Proxy popularity: some posts are inherently more popular + time decay.
        For Week 1, simulate popularity as random 'quality' * exp(-age/half_life).
        """
        # deterministic per post "quality"
        rng = np.random.RandomState(seed)  # fixed
        quality = rng.lognormal(mean=0.0, sigma=0.8, size=n_posts)
        age = np.maximum(0, ts_min - post_created)
        half_life = 12 * 60  # 12h
        decay = np.exp(-age / (half_life + 1e-12))
        return quality * decay

    for day in range(days):
        day_start = day * MINUTES_PER_DAY

        # drift taste once per day
        noise = np.random.normal(scale=taste_drift, size=taste.shape)
        taste = taste + noise
        taste = taste / (np.linalg.norm(taste, axis=1, keepdims=True) + 1e-12)

        for u in range(n_users):
            for s in range(sessions_per_user_per_day):
                # session timestamp within the day
                ts = int(day_start + np.random.randint(0, MINUTES_PER_DAY))

                # available posts: created <= ts
                avail_mask = post_created <= ts
                avail_ids = np.where(avail_mask)[0]
                if len(avail_ids) < candidate_pool_size:
                    continue

                # sample impressions from available pool
                imp_ids = np.random.choice(avail_ids, size=candidate_pool_size, replace=False)

                # compute click prob per impression
                pop = popularity_24h(ts)[imp_ids]
                pop = (pop - pop.mean()) / (pop.std() + 1e-9)

                # freshness: newer is better
                age = np.maximum(0, ts - post_created[imp_ids]).astype(np.float32)
                freshness = -age / (24 * 60)  # normalized negative age

                # topic match: dot(user taste, one-hot topic)
                topics = np.array([posts[i].topic_id for i in imp_ids], dtype=np.int32)
                topic_match = taste[u, topics]  # picks the topic coordinate

                # follow author bonus
                follow_set = set(follows[u])
                authors = np.array([posts[i].author_id for i in imp_ids], dtype=np.int32)
                follow_bonus = np.array([1.0 if a in follow_set else 0.0 for a in authors], dtype=np.float32)

                # score -> probability
                logits = (
                    math.log(base_ctr / (1.0 - base_ctr + 1e-12))
                    + w_topic * topic_match
                    + w_fresh * freshness
                    + w_pop * pop
                    + w_follow * follow_bonus
                )
                p = 1.0 / (1.0 + np.exp(-logits))

                # sample clicks (sparse)
                clicks = (np.random.rand(len(imp_ids)) < p).nonzero()[0]
                clicked_ids = [int(imp_ids[i]) for i in clicks]

                sessions.append(Session(
                    user_id=u,
                    ts=ts,
                    impression_post_ids=[int(x) for x in imp_ids.tolist()],
                    clicked_post_ids=clicked_ids
                ))

    return SimData(users=users, posts=posts, follows=follows, sessions=sessions)


def split_sessions_by_time(sessions: List[Session], train_end_ts: int, val_end_ts: int):
    train, val, test = [], [], []
    for s in sessions:
        if s.ts < train_end_ts:
            train.append(s)
        elif s.ts < val_end_ts:
            val.append(s)
        else:
            test.append(s)
    return train, val, test
